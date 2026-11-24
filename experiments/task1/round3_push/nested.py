"""Nested-CV threshold (choose th on 4/5 of OOF rows, apply to the held-out 1/5), averaged over 20 split seeds, plus paired bootstrap vs production."""
import sys, glob, numpy as np
from sklearn.metrics import roc_auc_score
from c1 import *
X,y,U,T=load(); ob=box_outside(X,y,T); outX=ob(X)
prod=np.load(P+'/t1_seedsbox_oof.npy')
def avg(pat): return np.where(outX,0,np.mean([np.load(f) for f in sorted(glob.glob(f'{OUT}/oof_{pat}_s*.npy'))],0))
def nested(p,reps=20):
    r=[]
    for s in range(reps):
        pred=np.zeros(len(y),bool)
        for tr,va in folds(y,1000+s):
            t=TH[curve(y[tr],p[tr]).argmax()]; pred[va]=p[va]>t
        r.append(fbeta_score(y,pred,beta=2,average='macro'))
    return np.mean(r),np.std(r)
C={'prod seedsbox':prod}
for pat in sys.argv[1:]: C[pat+' (seed-avg)']=avg(pat)
if len(sys.argv)>2: pass
nn=avg('lgb_nonoise'); C['0.5 prod + 0.5 lgb_nonoise']=np.where(outX,0,0.5*prod+0.5*nn)
rng=np.random.default_rng(0); B=[rng.integers(0,len(y),len(y)) for _ in range(200)]
for k,p in C.items():
    b,t=best(y,p); m,s=nested(p)
    d=[mf2(y[i],p[i],t)-mf2(y[i],prod[i],0.37) for i in B]
    print(f'{k:30s} AUC {roc_auc_score(y,p):.5f} best {b:.5f}@{t} nested {m:.5f}(+-{s:.5f}) paired-bootstrap diff vs prod {np.mean(d):+.5f} sd {np.std(d):.5f}')
