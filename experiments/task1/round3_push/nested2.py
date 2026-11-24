"""Blend production OOF with LGB seed-averages (seeds 1,2,3 only, same seeds for every variant) and compare vs prod:
nested-CV threshold + paired bootstrap (rows) at each candidate's own best threshold vs prod at 0.37."""
import sys, numpy as np
from sklearn.metrics import roc_auc_score
from c1 import *
X,y,U,T=load(); outX=box_outside(X,y,T)(X)
prod=np.load(P+'/t1_seedsbox_oof.npy')
def avg(n,seeds=(1,2,3)): return np.mean([np.load(f'{OUT}/oof_lgb_{n}_s{s}.npy') for s in seeds],0)
def nested(p,reps=10):
    r=[]
    for s in range(reps):
        pred=np.zeros(len(y),bool)
        for tr,va in folds(y,1000+s):
            t=TH[curve(y[tr],p[tr]).argmax()]; pred[va]=p[va]>t
        r.append(fbeta_score(y,pred,beta=2,average='macro'))
    return np.mean(r)
rng=np.random.default_rng(0); B=[rng.integers(0,len(y),len(y)) for _ in range(200)]
pn,_=nested(prod),0
for n in sys.argv[1:]:
    for w in (1.0,0.5):
        p=np.where(outX,0,w*avg(n)+(1-w)*prod); b,t=best(y,p); m=nested(p)
        d=np.array([mf2(y[i],p[i],t)-mf2(y[i],prod[i],0.37) for i in B])
        print(f'{w} {n:16s} + {1-w} prod: AUC {roc_auc_score(y,p):.5f} best {b:.5f}@{t} nested {m:.5f} (prod nested {pn:.5f}, diff {m-pn:+.5f}) bootstrap diff {d.mean():+.5f} sd {d.std():.5f}',flush=True)
