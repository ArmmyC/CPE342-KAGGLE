import numpy as np, pandas as pd
from scipy import stats
R='<project_root>/'
tr=pd.read_csv(R+'MLTest/cpe342-kaggle-main/02_multiclass_classification/train.csv')
y=tr.segment.values
cols=[c for c in tr.columns if c not in('player_id','segment')]
# discretize each column within class: numeric -> 8 quantile bins (within class), cat -> codes; NaN -> own level
B={}
for c in cols:
    v=tr[c]
    if v.dtype==object or str(v.dtype)=='str':
        B[c]=pd.factorize(v.fillna('NA'))[0]
    else:
        b=np.full(len(v),-1)
        for k in range(4):
            m=(y==k)&v.notna().values
            r=v[m].rank(pct=True,method='average').values
            b[m]=np.minimum((r*8).astype(int),7)
        B[c]=b+1  # 0 = NaN
res=[]
for i,a in enumerate(cols):
    for b in cols[i+1:]:
        G=0;df=0
        for k in range(4):
            m=y==k
            ct=pd.crosstab(B[a][m],B[b][m]).values
            g,p,d,_=stats.chi2_contingency(ct,lambda_='log-likelihood')
            G+=g;df+=d
        z=(G-df)/np.sqrt(2*df)
        res.append((z,a,b,G,df))
res.sort(reverse=True)
for r in res[:40]: print('%.1f %s %s G=%.0f df=%d'%r)
print('n pairs',len(res),'z>4:',sum(r[0]>4 for r in res))
