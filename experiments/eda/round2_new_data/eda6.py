from common import *
import warnings; warnings.filterwarnings('ignore')
tr,_=load(2)
num=[c for c in tr.columns if tr[c].dtype!=object and str(tr[c].dtype)!='str' and c not in('player_id','segment')]
C=np.zeros((len(num),len(num)))
res=tr[num].copy()
for s in range(4):
    res.loc[tr.segment==s]=tr.loc[tr.segment==s,num]-tr.loc[tr.segment==s,num].mean()
cm=res.corr().abs().values.copy(); np.fill_diagonal(cm,0)
idx=np.dstack(np.unravel_index(np.argsort(-cm.ravel()),cm.shape))[0][:40:2]
for i,j in idx: print(num[i],num[j],round(res[num[i]].corr(res[num[j]]),3))
# class-conditional means standardized
g=tr.groupby('segment')[num].mean(); z=(g-tr[num].mean())/tr[num].std()
print(z.T.round(2).to_string())
