from lib import *; from nb2 import *
from scipy.optimize import minimize_scalar
tr,te=load(); y=tr.segment.values; fl=folds(y)
bo,_,_=baseline(); blab=np.load(OUT+'base_nested_lab.npy')
feats=[c for c in tr.columns if c not in('player_id','segment')]
F2=[c for c in feats if c not in DROP0+NOINFO]
o=np.zeros((len(y),4)); tp=np.zeros((len(te),4))
for a,b in fl:
    m=NB2(nbins=100,smooth=2.5).fit(tr[F2].iloc[a],y[a]); o[b]=m.predict_proba(tr[F2].iloc[b]); tp+=np.log(m.predict_proba(te[F2]))/5
np.save(OUT+'oof_nb.npy',o); np.save(OUT+'test_nb_logmean.npy',tp)
L=np.log(o)
def temp(t):
    z=L*t; z-=z.max(1,keepdims=True); p=np.exp(z); return p/p.sum(1,keepdims=True)
r=minimize_scalar(lambda t: logloss(y,temp(t)),bounds=(0.2,2),method='bounded'); print('best temp',r.x,'ll',r.fun)
# per-row loss diagnostic
lnb=-np.log(o[np.arange(len(y)),y]); lbl=-np.log(bo[np.arange(len(y)),y])
d=lnb-lbl; print('mean diff',d.mean(),'share of diff from top 1% rows',np.sort(d)[::-1][:1040].sum()/d.sum())
conf=o.max(1)
for q in [0.5,0.7,0.9,0.97,0.99,1.01]:
    pass
bins=pd.cut(conf,[0,.4,.5,.6,.7,.8,.9,.95,.99,1.0])
print(pd.DataFrame({'bin':bins,'d':d,'nbacc':o.argmax(1)==y,'blacc':bo.argmax(1)==y,'nbconf':conf,'blconf':bo.max(1)}).groupby('bin').agg(['mean','count']).round(4).to_string())
