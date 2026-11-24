import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
R='<project_root>/'
P=R+'MLKaggleChallenge/preds_v2/'
OUT=R+'MLTest/cpe342-kaggle-main/work/push/t2/'
def load():
    tr=pd.read_csv(R+'MLTest/cpe342-kaggle-main/02_multiclass_classification/train.csv')
    te=pd.read_csv(R+'MLTest/cpe342-kaggle-main/02_multiclass_classification/test.csv')
    return tr,te
def folds(y,seed=42): return list(StratifiedKFold(5,shuffle=True,random_state=seed).split(np.zeros(len(y)),y))
def mf1(y,p,K=4):
    cm=np.bincount(y*K+p,minlength=K*K).reshape(K,K); tp=np.diag(cm)
    return np.mean(2*tp/np.maximum(cm.sum(0)+cm.sum(1),1))
def fit_w(y,p,grid=np.arange(0.5,2.5,0.05)):
    w=np.ones(4); lp=p
    for _ in range(3):
        for k in range(4):
            best=(-1,1)
            for v in grid:
                w2=w.copy(); w2[k]=v; s=mf1(y,(lp*w2).argmax(1))
                if s>best[0]+1e-12: best=(s,v)
            w[k]=best[1]
    return w/w[3]
def nested_pred(y,p,fl):
    """labels with class weights fitted on the other folds' OOF (nested)"""
    out=np.zeros(len(y),int)
    for a,b in fl:
        w=fit_w(y[a],p[a]); out[b]=(p[b]*w).argmax(1)
    return out
def baseline():
    return np.load(P+'t2_blendnn_oof.npy'),np.load(P+'t2_blendnn_test.npy'),np.load(P+'t2_blendnn_w.npy')
def logloss(y,p): return -np.mean(np.log(np.clip(p[np.arange(len(y)),y],1e-12,1)))
def report(name,y,p,fl,ref_lab=None,nb=1000,rng=0):
    raw=mf1(y,p.argmax(1)); w=fit_w(y,p); ins=mf1(y,(p*w).argmax(1)); lab=nested_pred(y,p,fl); nst=mf1(y,lab)
    s=f'{name}: ll {logloss(y,p):.4f} raw {raw:.4f} insampleW {ins:.4f} nestedW {nst:.4f}'
    if ref_lab is not None:
        r=np.random.default_rng(rng); d=[]
        for _ in range(nb):
            i=r.integers(0,len(y),len(y)); d.append(mf1(y[i],lab[i])-mf1(y[i],ref_lab[i]))
        d=np.array(d); s+=f' | diff vs ref {nst-mf1(y,ref_lab):+.4f} (boot sd {d.std():.4f}, P(diff>0) {(d>0).mean():.2f})'
    print(s,flush=True); return lab,w
