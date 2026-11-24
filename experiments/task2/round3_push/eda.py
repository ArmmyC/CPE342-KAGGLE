import numpy as np, pandas as pd
R='<project_root>/'
tr=pd.read_csv(R+'MLTest/cpe342-kaggle-main/02_multiclass_classification/train.csv')
y=tr.segment.values
num=[c for c in tr.columns if tr[c].dtype!=object and str(tr[c].dtype)!='str' and c!='segment']
pd.set_option('display.width',250)
rows=[]
for c in num:
    x=tr[c]; nu=x.nunique(); isint=(x.dropna()%1==0).mean()
    r=dict(c=c,nu=nu,int=round(isint,2),min=x.min(),max=x.max(),fmin=(x==x.min()).mean().round(3),fmax=(x==x.max()).mean().round(3))
    for k in range(4):
        xk=x[y==k]; r[f'm{k}']=round(xk.mean(),2); r[f's{k}']=round(xk.std(),2)
    rows.append(r)
print(pd.DataFrame(rows).to_string())
cat=[c for c in tr.columns if c not in num and c not in('player_id','segment')]
for c in cat:
    print(c, pd.crosstab(tr[c].fillna('NA'),y,normalize='columns').round(3).to_string())
# missingness vs class
M=tr[num+cat].isna()
print('nmiss by class', pd.crosstab(M.sum(1).clip(0,8),y,normalize='columns').round(3).to_string())
for c in num+cat:
    r=M[c].groupby(y).mean().values
    if r.max()-r.min()>0.006: print('miss dep',c,r.round(4))
