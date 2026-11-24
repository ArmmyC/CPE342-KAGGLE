from common import *
tr,te=load(1)
tr=tr.dropna(subset=['is_cheater']).reset_index(drop=True)
F=[c for c in tr.columns if c not in('player_id','is_cheater')]
y=tr.is_cheater.astype(int).values
p=np.load('t1_qda_oof.npy')[:,2]
r=y-p; v=p*(1-p)
B={c:pd.qcut(tr[c].rank(method='first'),20,labels=False).fillna(-1).astype(int).values for c in F}
# 1D fine bins (50)
res=[]
for c in F:
    b=pd.qcut(tr[c].rank(method='first'),50,labels=False).fillna(-1).astype(int).values
    z=pd.Series(r).groupby(b).sum()/np.sqrt(pd.Series(v).groupby(b).sum())
    res.append((c,np.round((z**2).sum()/len(z),2),round(z.abs().max(),2)))
print(sorted(res,key=lambda t:-t[1])[:10])
rows=[]
for i,a in enumerate(F):
    for c in F[i+1:]:
        k=B[a]*21+B[c]
        g=pd.DataFrame({'k':k,'r':r,'v':v}).groupby('k').sum()
        z=g.r/np.sqrt(g.v+1e-9)
        rows.append((a,c,(z**2).mean(),z.abs().max()))
d=pd.DataFrame(rows,columns=['a','b','chi','zmax']).sort_values('chi',ascending=False)
print(d.head(20).round(2).to_string()); print(d.chi.describe())
