import numpy as np, pandas as pd
from c1 import *
X,y,U,T=load(); outT=box_outside(X,y,T)(T)
prod=np.load(P+'/t1_seedsbox_test.npy'); sub=pd.read_csv(D+'/work/out/sub38_t4new.csv').task1.values
for n,w,t in [('pseudo_hard',0.5,0.40),('base',0.5,0.40),('pseudo_hard',1.0,0.38)]:
    p=np.where(outT,0,w*np.mean([np.load(f'{OUT}/test_lgb_{n}_s{s}.npy') for s in (1,2,3)],0)+(1-w)*prod)
    print(n,w,t,'rows changed vs sub38:',int(((p>t).astype(int)!=sub).sum()),'pos rate',(p>t).mean().round(4),'sub38 pos rate',sub.mean().round(4))
