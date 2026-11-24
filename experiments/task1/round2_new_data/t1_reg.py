from common import *
from sklearn.metrics import roc_auc_score
tr,te=load(1)
tr=tr.dropna(subset=['is_cheater']).reset_index(drop=True)
F=[c for c in tr.columns if c not in('player_id','is_cheater')]
y=tr.is_cheater.astype(int).values
p=np.load('t1_qda_oof.npy')[:,2]
a=tr.account_age_days.values
m=(a>200)&(a<900)
print('region rows',m.sum(),'rate',y[m].mean(),'lgb auc in region',roc_auc_score(y[m],p[m]))
mm=m&(p>0.02)&(p<0.98); print('ambiguous',mm.sum(),y[mm].mean(),roc_auc_score(y[mm],p[mm]))
s=tr[m]; ys=y[m]
out=[]
for c in F:
    v=s[c]; k=v.notna()
    out.append((c,round(roc_auc_score(ys[k],v[k]),3)))
print(sorted(out,key=lambda t:-abs(t[1]-.5)))
# within region, per class corr among informative
inf=[c for c,u in out if abs(u-.5)>0.01]
for k in (0,1):
    R=s.loc[ys==k,inf].rank(pct=True).corr().values; iu=np.triu_indices(len(inf),1)
    o=np.argsort(-np.abs(R[iu]))[:14]
    print(k,[(inf[iu[0][i]][:9],inf[iu[1][i]][:9],round(R[iu][i],2)) for i in o])
