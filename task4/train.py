"""Task 4: fine-tune a pretrained timm image model on game screenshots (one or more CV folds).

Used to train the models in the final Task 4 blend (see task4.ipynb and train_all.sh).
Augmentation uses generic geometry and image-quality corruptions (ImageNet-C style families such as blur, noise,
pixelation and JPEG, which match the datacard's sources: live streams, forum posts and support tickets) plus a
mild standard colour jitter. An earlier heavy colour augmentation (inversion, hue rotation, channel swaps) was
removed because it had been designed after viewing test images.
Saves, per fold: clean OOF probabilities, corrupted-proxy OOF probabilities (fixed seed), a second
geometric proxy, and test probabilities (horizontal-flip TTA). The checkpoint with the best
degraded-proxy macro F1 on the validation fold is used for the test prediction.

Requires a CUDA GPU. Settings come from environment variables:
  T4_MODEL  timm model name               T4_SIZE   H,W input size
  T4_EPOCHS number of epochs              T4_LR     peak learning rate (OneCycle)
  T4_BS     batch size                    T4_FOLDS  comma-separated fold ids (0-4)
  T4_TAG    output file prefix            T4_AUG    5 = generic robustness augmentation (used), 6 = stronger variant (tried on one fold only)
  T4_DATA   folder with train.csv, test.csv, train/, test/
  T4_OUT    output folder                 T4_WORKERS dataloader workers
"""
import copy
import os
import time

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

MODEL = os.environ.get('T4_MODEL', 'convnext_base.fb_in22k_ft_in1k')
SIZE = tuple(int(v) for v in os.environ.get('T4_SIZE', '160,320').split(','))
EPOCHS = int(os.environ.get('T4_EPOCHS', 15))
LR = float(os.environ.get('T4_LR', 3e-4))
BS = int(os.environ.get('T4_BS', 64))
FOLDS = [int(f) for f in os.environ.get('T4_FOLDS', '0').split(',')]
TAG = os.environ.get('T4_TAG', 'run')
DATA = os.environ.get('T4_DATA', '../data/04_image_classification')
OUT = os.environ.get('T4_OUT', 'preds_raw')
SUBSET = int(os.environ.get('T4_SUBSET', 0))  # smoke test: use only this many train images
DROP_PATH = float(os.environ.get('T4_DROP_PATH', 0.1))
NW = int(os.environ.get('T4_WORKERS', 8))
EMA_DECAY = float(os.environ.get('T4_EMA', 0.999))
AUG = int(os.environ.get('T4_AUG', 5))
os.makedirs(OUT, exist_ok=True)
dev = 'cuda'


class GaussNoise:
    def __call__(self, x):
        return (x + torch.randn_like(x) * 0.08 * torch.rand(1).item()).clamp(0, 1) if torch.rand(1).item() < 0.5 else x


# standard mild photometric augmentation (common default ranges) and Gaussian blur
std_photo = [
    T.RandomApply([T.ColorJitter(0.3, 0.3, 0.3, 0.05)], p=0.5),
    T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.3),
]
import torchvision.transforms.functional as TF


norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
tf_eval = T.Compose([T.Resize(SIZE), T.ToTensor(), norm])


import io, random
from PIL import Image as _Im, ImageFilter as _IF


class JPEG:
    def __init__(self, p=0.35): self.p = p
    def __call__(self, im):
        if random.random() > self.p: return im
        b = io.BytesIO(); im.save(b, "JPEG", quality=random.randint(8, 60)); b.seek(0)
        return _Im.open(b).convert("RGB")


class Pixelate:
    def __init__(self, p=0.3): self.p = p
    def __call__(self, im):
        if random.random() > self.p: return im
        w, h = im.size; f = random.uniform(1.5, 6)
        rs = random.choice([_Im.NEAREST, _Im.BILINEAR, _Im.BICUBIC])
        return im.resize((max(8, int(w / f)), max(8, int(h / f))), _Im.BILINEAR).resize((w, h), rs)


class ZoomOut:
    def __init__(self, p=0.25): self.p = p
    def __call__(self, im):
        if random.random() > self.p: return im
        w, h = im.size; s = random.uniform(0.55, 0.95)
        nw, nh = int(w * s), int(h * s); bg = tuple(random.randint(0, 255) for _ in range(3))
        c = _Im.new("RGB", (w, h), bg); c.paste(im.resize((nw, nh)), (random.randint(0, w - nw), random.randint(0, h - nh)))
        return c


class MotionBlur:
    def __init__(self, p=0.2): self.p = p
    def __call__(self, x):
        if random.random() > self.p: return x
        k = random.choice([5, 7, 9, 11]); ker = torch.zeros(k, k)
        if random.random() < 0.5: ker[k // 2, :] = 1
        else: ker[:, k // 2] = 1
        ker = TF.rotate(ker[None], random.uniform(0, 180))[0]; ker = ker / ker.sum().clamp_min(1e-6)
        return torch.nn.functional.conv2d(x[:, None], ker[None, None], padding=k // 2)[:, 0].clamp(0, 1)


class MedianOrSharpen:
    def __init__(self, p=0.15): self.p = p
    def __call__(self, im):
        if random.random() > self.p: return im
        return im.filter(random.choice([_IF.MedianFilter(3), _IF.SHARPEN, _IF.EDGE_ENHANCE, _IF.SMOOTH_MORE]))


class Haze:
    def __init__(self, p=0.25): self.p = p
    def __call__(self, x):
        if random.random() > self.p: return x
        a = random.uniform(0.2, 0.6); c = torch.rand(3, 1, 1) * 0.5 + random.uniform(0.3, 0.5)
        return (x * (1 - a) + c * a).clamp(0, 1)


class ShotNoise:
    def __init__(self, p=0.25): self.p = p
    def __call__(self, x):
        if random.random() > self.p: return x
        lam = random.uniform(8, 60)
        return (torch.poisson(x * lam) / lam).clamp(0, 1)


class Defocus:
    def __init__(self, p=0.2): self.p = p
    def __call__(self, x):
        if random.random() > self.p: return x
        r = random.randint(2, 5); k = 2 * r + 1
        yy, xx = torch.meshgrid(torch.arange(k) - r, torch.arange(k) - r, indexing="ij")
        ker = ((xx ** 2 + yy ** 2) <= r * r).float(); ker = ker / ker.sum()
        return torch.nn.functional.conv2d(x[:, None], ker[None, None], padding=r)[:, 0].clamp(0, 1)


if AUG == 5:  # generic robustness: geometry + image-quality corruptions
    tf_train = T.Compose([
        T.Resize(SIZE), ZoomOut(), T.RandomApply([T.RandomRotation(25)], p=0.5),
        T.RandomPerspective(0.3, p=0.3),
        T.RandomResizedCrop(SIZE, scale=(0.3, 1.0), ratio=(1.2, 2.8)), T.RandomHorizontalFlip(),
        *std_photo, MedianOrSharpen(), Pixelate(), JPEG(), T.ToTensor(), MotionBlur(), GaussNoise(),
        T.RandomErasing(p=0.3, scale=(0.02, 0.15), value="random"), norm])
if AUG == 6:  # AUG 5 with higher rates and extra ImageNet-C style families (defocus, fog-like haze, shot noise)
    tf_train = T.Compose([
        T.Resize(SIZE), ZoomOut(0.35), T.RandomApply([T.RandomRotation(30)], p=0.5),
        T.RandomPerspective(0.35, p=0.35),
        T.RandomResizedCrop(SIZE, scale=(0.25, 1.0), ratio=(1.1, 3.0)), T.RandomHorizontalFlip(),
        *std_photo, MedianOrSharpen(0.2), Pixelate(0.4), JPEG(0.5), T.ToTensor(),
        MotionBlur(0.25), Defocus(0.2), Haze(0.25), ShotNoise(0.25), GaussNoise(),
        T.RandomErasing(p=0.4, scale=(0.02, 0.2), value="random"), norm])
# validation proxies (fixed seeds): image-quality degradations, and geometric degradations
tf_corrupt = T.Compose([T.Resize(SIZE), Pixelate(0.4), JPEG(0.5), T.ToTensor(), MotionBlur(0.3), Defocus(0.2),
                        ShotNoise(0.3), GaussNoise(), norm])
tf_corrupt_g = T.Compose([T.Resize(SIZE), ZoomOut(0.4), T.RandomPerspective(0.3, p=0.4),
                          T.RandomResizedCrop(SIZE, scale=(0.4, 1.0), ratio=(1.2, 2.8)), Pixelate(0.4), JPEG(0.5),
                          T.ToTensor(), MotionBlur(0.3), norm])


class Imgs(Dataset):
    def __init__(self, folder, names, labels, tf):
        self.folder, self.names, self.labels, self.tf = folder, names, labels, tf

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        im = Image.open(os.path.join(self.folder, self.names[i])).convert('RGB')
        return self.tf(im), (int(self.labels[i]) if self.labels is not None else -1)


def make_model():
    kw = dict(pretrained=True, num_classes=5)
    if 'vit' in MODEL or 'eva' in MODEL or 'dinov2' in MODEL:
        kw.update(img_size=SIZE)
    try:
        return timm.create_model(MODEL, drop_path_rate=DROP_PATH, **kw).to(dev)
    except TypeError:
        return timm.create_model(MODEL, **kw).to(dev)


@torch.inference_mode()
def predict(model, dl, flip=True):
    model.eval(); out = []
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for x, _ in dl:
            x = x.to(dev, non_blocking=True)
            views = [x, torch.flip(x, dims=[3])] if flip else [x]
            p = sum(model(v).float().softmax(1) for v in views) / len(views)
            out.append(p.cpu().numpy())
    return np.concatenate(out)


def main():
    train = pd.read_csv(f'{DATA}/train.csv'); test = pd.read_csv(f'{DATA}/test.csv')
    y = train['label'].values
    if SUBSET:
        train = train.sample(SUBSET, random_state=0).reset_index(drop=True); y = train['label'].values
    names = train['file_name'].values
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    oof = np.full((len(train), 5), np.nan); oofc = np.full((len(train), 5), np.nan)
    test_probs = np.zeros((len(test), 5)); logs = []
    test_dl = DataLoader(Imgs(f'{DATA}/test', test['file_name'].values, None, tf_eval), batch_size=128, num_workers=NW)
    for fold, (tr_i, va_i) in enumerate(skf.split(train, y)):
        if fold not in FOLDS:
            continue
        tr_dl = DataLoader(Imgs(f'{DATA}/train', names[tr_i], y[tr_i], tf_train), batch_size=BS, shuffle=True,
                           num_workers=NW, drop_last=True, persistent_workers=True)
        va_dl = DataLoader(Imgs(f'{DATA}/train', names[va_i], y[va_i], tf_eval), batch_size=128, num_workers=NW)
        vc_dl = DataLoader(Imgs(f'{DATA}/train', names[va_i], y[va_i], tf_corrupt), batch_size=128, num_workers=NW)
        vg_dl = DataLoader(Imgs(f'{DATA}/train', names[va_i], y[va_i], tf_corrupt_g), batch_size=128, num_workers=NW)
        oofg = np.full((len(train), 5), np.nan) if 'oofg' not in dir() else oofg
        model = make_model()
        ema = copy.deepcopy(model).eval()
        for p in ema.parameters():
            p.requires_grad_(False)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * len(tr_dl), pct_start=0.1)
        crit = nn.CrossEntropyLoss(label_smoothing=0.1)
        best = -1; step = 0
        for ep in range(EPOCHS):
            model.train(); t0 = time.time()
            for x, t in tr_dl:
                x, t = x.to(dev, non_blocking=True), t.to(dev, non_blocking=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = crit(model(x), t)
                opt.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step(); sched.step(); step += 1
                d = min(EMA_DECAY, (1 + step) / (10 + step))
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.mul_(d).add_(pm.detach(), alpha=1 - d)
                    for be, bm in zip(ema.buffers(), model.buffers()):
                        be.copy_(bm)
            pv = predict(ema, va_dl)
            torch.manual_seed(1234)
            pc = predict(ema, vc_dl)
            random.seed(99); np.random.seed(99); torch.manual_seed(99)
            pg = predict(ema, vg_dl)
            f_g = f1_score(y[va_i], pg.argmax(1), average='macro')
            f_clean = f1_score(y[va_i], pv.argmax(1), average='macro'); f_cor = f1_score(y[va_i], pc.argmax(1), average='macro')
            print(f'[{TAG}] fold {fold} ep {ep} loss {loss.item():.3f} clean {f_clean:.4f} corrupted {f_cor:.4f} geo {f_g:.4f} '
                  f'{time.time() - t0:.0f}s', flush=True)
            logs.append(dict(fold=fold, ep=ep, f1_clean=f_clean, f1_corrupt=f_cor, f1_geo=f_g))
            if f_cor > best:
                best = f_cor; oof[va_i] = pv; oofc[va_i] = pc; oofg[va_i] = pg
                torch.save(ema.state_dict(), f'{OUT}/{TAG}_f{fold}.pt')
        ema.load_state_dict(torch.load(f'{OUT}/{TAG}_f{fold}.pt'))
        tp = predict(ema, test_dl)
        np.save(f'{OUT}/{TAG}_f{fold}_test.npy', tp)
        np.save(f'{OUT}/{TAG}_f{fold}_oof.npy', oof); np.save(f'{OUT}/{TAG}_f{fold}_oofc.npy', oofc); np.save(f'{OUT}/{TAG}_f{fold}_oofg.npy', oofg)
        np.save(f'{OUT}/{TAG}_f{fold}_idx.npy', va_i)
        test_probs += tp
        os.remove(f'{OUT}/{TAG}_f{fold}.pt')  # quota: keep only predictions
        print(f'[{TAG}] fold {fold} best corrupted {best:.4f}', flush=True)
    pd.DataFrame(logs).to_csv(f'{OUT}/{TAG}_log_{"-".join(map(str, FOLDS))}.csv', index=False)


if __name__ == '__main__':
    main()
