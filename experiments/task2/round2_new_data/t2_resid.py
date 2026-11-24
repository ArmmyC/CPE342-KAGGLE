from common import *
tr,te=load(2)
y=tr.segment.astype(int).values
P=np.load('push/t2fix/preds/t2_blendnn_oof.npy')
Y=np.eye(4)[y]; R=Y-P; V=P*(1-P)
F=[c for c in tr.columns if c not in('player_id','segment')]
def binz(s,n):
    if s.dtype==object or s.nunique()<=12: return s.astype(str).where(s.notna(),'nan').astype('category').cat.codes.values
    return pd.qcut(s.rank(method='first'),n,labels=False).fillna(-1).astype(int).values+1
one=[]
for c in F:
    b=binz(tr[c],40); g=pd.DataFrame(R).groupby(b).sum().values/np.sqrt(pd.DataFrame(V).groupby(b).sum().values+1e-9)
    one.append((c,round((g**2).mean(),2),round(np.abs(g).max(),2)))
print(sorted(one,key=lambda t:-t[1])[:12])
Q={c:binz(tr[c],10) for c in F}
rows=[]
for i,a in enumerate(F):
    for b in F[i+1:]:
        k=Q[a]*100+Q[b]
        g=pd.DataFrame(R).groupby(k).sum().values/np.sqrt(pd.DataFrame(V).groupby(k).sum().values+1e-9)
        rows.append((a,b,(g**2).mean(),np.abs(g).max()))
d=pd.DataFrame(rows,columns=['a','b','chi','zmax']).sort_values('chi',ascending=False)
print(d.head(15).round(2).to_string()); print(d.chi.describe().round(3))
