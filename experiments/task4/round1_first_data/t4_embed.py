"""Task 4: extract frozen pretrained image embeddings (CPU) for a linear-probe baseline."""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, 'data', 'task4_raw', 'task4_image_classification')
OUT = os.path.join(ROOT, 'preds')
MODEL = sys.argv[1] if len(sys.argv) > 1 else 'convnext_tiny.fb_in22k'
SIZE = (128, 240)  # H, W, close to native 120x230

torch.set_num_threads(16)
model = timm.create_model(MODEL, pretrained=True, num_classes=0).eval()
cfg = timm.data.resolve_data_config({}, model=model)
mean = torch.tensor(cfg['mean']).view(3, 1, 1); std = torch.tensor(cfg['std']).view(3, 1, 1)


class Imgs(Dataset):
    def __init__(self, folder, ids):
        self.folder, self.ids = folder, ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        im = Image.open(os.path.join(self.folder, self.ids[i] + '.jpg')).convert('RGB').resize(SIZE[::-1])
        x = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255
        return (x - mean) / std


tag = MODEL.split('.')[0]
for split in ('train', 'test'):
    ids = pd.read_csv(os.path.join(IMG, f'{split}.csv'))['id'].tolist()
    dl = DataLoader(Imgs(os.path.join(IMG, split), ids), batch_size=128, num_workers=0)
    feats, t0 = [], time.time()
    with torch.inference_mode():
        for b, x in enumerate(dl):
            f = model(x)
            # also add horizontally flipped view (TTA)
            f = (f + model(torch.flip(x, dims=[3]))) / 2
            feats.append(f.numpy().astype(np.float32))
            if b % 20 == 0:
                print(split, b * 128, '/', len(ids), f'{time.time() - t0:.0f}s', flush=True)
    np.save(os.path.join(OUT, f't4_emb_{tag}_{split}.npy'), np.concatenate(feats))
print('done')
