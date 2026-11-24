from common import *
tr,te=load(3)
y=tr.spending_30d.values
o=np.load('push/t3/final_m42_oof.npy'); print(len(y),len(o),np.isnan(y).sum())
nmae=np.abs(y-o).mean()/np.abs(y).mean(); print('nmae',nmae,'score',1/(1+nmae))
s=np.sign(y-o)
F=[c for c in tr.columns if c not in('player_id','spending_30d')]
def binz(v,n):
    if v.nunique()<=12: return v.astype(str).where(v.notna(),'nan').astype('category').cat.codes.values
    return pd.qcut(v.rank(method='first'),n,labels=False).fillna(-1).astype(int).values+1
w=np.abs(y-o)
one=[]
for c in F:
    b=binz(tr[c],40); g=pd.Series(s).groupby(b); z=g.sum()/np.sqrt(g.count())
    one.append((c,round((z**2).mean(),2),round(z.abs().max(),1)))
print(sorted(one,key=lambda t:-t[1])[:12])
Q={c:binz(tr[c],10) for c in F}
rows=[]
for i,a in enumerate(F):
    for b in F[i+1:]:
        k=Q[a]*100+Q[b]; g=pd.Series(s).groupby(k); z=g.sum()/np.sqrt(g.count())
        rows.append((a,b,(z**2).mean(),z.abs().max()))
d=pd.DataFrame(rows,columns=['a','b','chi','zmax']).sort_values('chi',ascending=False)
print(d.head(15).round(2).to_string()); print(d.chi.describe().round(3))
