import numpy as np, pandas as pd, os, hashlib
from sklearn.metrics import fbeta_score, roc_auc_score
D='<project_root>/MLTest/cpe342-kaggle-main'
D='<project_root>/MLTest/cpe342-kaggle-main'
P='<project_root>/MLKaggleChallenge/preds'
OD='<project_root>/MLKaggleChallenge/data/cpe342-kaggle-main/01_binary_classification'
tr=pd.read_csv(D+'/01_binary_classification/train.csv'); te=pd.read_csv(D+'/01_binary_classification/test.csv')
print(tr.shape, te.shape, tr.is_cheater.isna().sum(), tr.is_cheater.mean())
for f in ['train.csv','test.csv']:
    print(f, hashlib.md5(open(D+'/01_binary_classification/'+f,'rb').read()).hexdigest(), hashlib.md5(open(OD+'/'+f,'rb').read()).hexdigest() if os.path.exists(OD+'/'+f) else None)
lab=tr.dropna(subset=['is_cheater']).reset_index(drop=True); y=lab.is_cheater.astype(int).values
oof=np.load(P+'/t1_seedsbox_oof.npy'); tp=np.load(P+'/t1_seedsbox_test.npy')
print(len(oof), len(y), len(tp))
for t in [0.36,0.37,0.38]: print(t, fbeta_score(y,oof>t,beta=2,average='macro'))
print('auc',roc_auc_score(y,oof))
s=[np.load(P+f'/t1_{m}_s{s}_oof.npy') for m in ['lgb','xgb','cat'] for s in [42,1,2,3]]
print('seeds mean equals t1_seeds:', np.abs(np.mean(s,0)-np.load(P+'/t1_seeds_oof.npy')).max())
sub=pd.read_csv(D+'/work/out/sub38_t4new.csv')
print(sub.shape, (sub.task1.values==(tp>0.37)).mean(), (sub.task1.values==(tp>0.37)).sum(), sub.task1.mean())
print('ids', te.player_id.head(3).tolist(), sub.id.head(3).tolist())
