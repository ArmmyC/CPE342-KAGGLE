import numpy as np, pandas as pd
from scipy import stats
R='<project_root>/'
tr=pd.read_csv(R+'MLTest/cpe342-kaggle-main/02_multiclass_classification/train.csv')
y=tr.segment.values
num=[c for c in tr.columns if tr[c].dtype!=object and str(tr[c].dtype)!='str' and c!='segment' and not c.startswith('random')]
for c in num:
    out=[]
    for k in [0,3]:
        x=tr[c][y==k].dropna().values
        lo,hi=x.min(),x.max(); xi=x[(x>lo)&(x<hi)]
        s=stats.skew(xi); ku=stats.kurtosis(xi)
        ls=stats.skew(np.log(xi[xi>0])) if (xi>0).all() else np.nan
        lk=stats.kurtosis(np.log(xi[xi>0])) if (xi>0).all() else np.nan
        q=np.quantile(x,[0,.01,.05,.25,.5,.75,.95,.99,1])
        out.append(f'k{k} sk {s:.2f} ku {ku:.2f} logsk {ls:.2f} logku {lk:.2f} q {np.round(q,2)}')
    print(c); [print('  ',o) for o in out]
