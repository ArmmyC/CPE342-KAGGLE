import numpy as np
from sklearn.metrics import roc_auc_score
from c1 import *
X,y,U,T=load(); out=box_outside(X,y,T)(X)
g=np.load('<project_root>/MLKaggleChallenge/work/deep/t1_lin_oof.npy')
print([roc_auc_score(y,r) for r in g]); base=np.load(P+'/t1_seedsbox_oof.npy')
if g.ndim>1: print('cols',g.shape)
gg=g[1]
print('gam auc',roc_auc_score(y,gg))
for w in [0,0.1,0.2,0.3]:
    p=np.where(out,0,(1-w)*base+w*gg); b,t=best(y,p); print(w,round(roc_auc_score(y,p),5),round(b,5),t)
