import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from c1 import *
X,y,U,T=load(); out=box_outside(X,y,T)(X)
L=lambda m,s: np.load(f'{P}/t1_{m}_s{s}_oof.npy')
S=[42,1,2,3]
comp={m:np.mean([L(m,s) for s in S],0) for m in ['lgb','xgb','cat']}
base=np.load(P+'/t1_seedsbox_oof.npy')
def rep(name,p):
    p=np.where(out,0,p); b,t=best(y,p); print(f'{name:28s} AUC {roc_auc_score(y,p):.5f} best {b:.5f}@{t} @0.37 {mf2(y,p,0.37):.5f}')
rep('seedsbox (prod)',base)
for m in comp: rep(m+' x4 seeds',comp[m])
for s in S: rep(f'lgb+xgb+cat s{s}',np.mean([L(m,s) for m in comp],0))
for w in [(1,1,0),(1,0,1),(0,1,1),(2,1,1),(1,2,1),(1,1,2)]:
    rep(f'w lgb,xgb,cat={w}',sum(wi*comp[m] for wi,m in zip(w,comp))/sum(w))
r=np.mean([rankdata(L(m,s))/len(y) for m in comp for s in S],0)
# rank average: threshold scale differs; best over quantile thresholds
c=[mf2(y,r,q) for q in np.arange(0.4,0.7,0.002)]; print('rank-avg best',max(c))
# baseline noise: bootstrap SD of macroF2 at 0.37, and nested threshold
rng=np.random.default_rng(0); bs=[]
for _ in range(200):
    i=rng.integers(0,len(y),len(y)); bs.append(mf2(y[i],base[i],0.37))
print('bootstrap SD of macroF2 (absolute level)',np.std(bs))
# nested threshold: choose th on 4 folds of OOF, apply to 5th
for fs in [42,7,11]:
    pred=np.zeros(len(y),bool); ths=[]
    for tr,va in folds(y,fs):
        c=curve(y[tr],base[tr]); t=TH[c.argmax()]; ths.append(t); pred[va]=base[va]>t
    print('nested-threshold macroF2',fbeta_score(y,pred,beta=2,average='macro'),ths)
print('curve 0.33..0.42', dict(zip(TH[13:23],np.round(curve(y,base)[13:23],5))))
