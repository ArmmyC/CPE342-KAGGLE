from common import *
tr,te=load(2)
print('label nan',tr.segment.isna().sum(), tr.segment.value_counts().to_dict())
tr=tr.dropna(subset=['segment']).reset_index(drop=True)
y=tr.segment.astype(int).values
P=np.load('push/t2fix/preds/t2_blendnn_oof.npy'); print(P.shape)
from sklearn.metrics import f1_score; print('oof f1',f1_score(y,P.argmax(1),average='macro'))
F=[c for c in tr.columns if c not in('player_id','segment')]
def binz(s):
    if s.dtype==object or s.nunique()<=10: return s.astype(str).where(s.notna(),'nan').astype('category').cat.codes.values
    return pd.qcut(s.rank(method='first'),8,labels=False).fillna(-1).astype(int).values
Q={c:binz(tr[c]) for c in F}
rows=[]
for i,a in enumerate(F):
    for b in F[i+1:]:
        r=[a,b]
        for k in range(4):
            ok=(y==k)&(Q[a]>=0)&(Q[b]>=0)
            t=pd.crosstab(Q[a][ok],Q[b][ok]).values.astype(float)
            e=t.sum(1,keepdims=True)*t.sum(0,keepdims=True)/t.sum(); dof=max((t.shape[0]-1)*(t.shape[1]-1),1)
            r.append(((t-e)**2/np.maximum(e,1e-9)).sum()/dof)
        rows.append(r)
d=pd.DataFrame(rows,columns=['a','b','c0','c1','c2','c3']); d['mx']=d[['c0','c1','c2','c3']].max(1)
print(d.sort_values('mx',ascending=False).head(25).round(2).to_string()); print(d.mx.describe().round(2))
d.to_csv('tmp_t2_dep.csv',index=False)
