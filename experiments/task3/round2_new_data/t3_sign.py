from common import *
tr,te=load(3)
y=tr.spending_30d.values; o=np.load('push/t3/final_m42_oof.npy')
s=np.sign(y-o); m=o>0
print('pred>0 rows',m.sum(),'mean sign there',s[m].mean(),' pred==0 rows: frac y>0',(y[~m]>0).mean())
d=pd.DataFrame({'y':y,'o':o,'s':s})[m]
d['q']=pd.qcut(d.o.rank(method='first'),20,labels=False)
print(d.groupby('q').agg(o=('o','median'),y0=('y',lambda v:(v==0).mean()),sign=('s','mean')).round(3).to_string())
F=[c for c in tr.columns if c not in('player_id','spending_30d')]
def binz(v,n):
    if v.nunique()<=12: return v.astype(str).where(v.notna(),'nan').astype('category').cat.codes.values
    return pd.qcut(v.rank(method='first'),n,labels=False).fillna(-1).astype(int).values+1
T=tr[m].reset_index(drop=True); sm=s[m]
one=[]
for c in F:
    b=binz(T[c],30); g=pd.Series(sm).groupby(b); z=g.sum()/np.sqrt(g.count())
    one.append((c,round((z**2).mean(),2),round(z.abs().max(),1)))
print(sorted(one,key=lambda t:-t[1])[:12])
Q={c:binz(T[c],8) for c in F}
rows=[]
for i,a in enumerate(F):
    for b in F[i+1:]:
        k=Q[a]*100+Q[b]; g=pd.Series(sm).groupby(k); z=g.sum()/np.sqrt(g.count())
        rows.append((a,b,(z**2).mean(),z.abs().max()))
dd=pd.DataFrame(rows,columns=['a','b','chi','zmax']).sort_values('chi',ascending=False)
print(dd.head(12).round(2).to_string()); print(dd.chi.describe().round(3))
# pred==0 rows: is there a subgroup where y>0 more than half (should predict >0)?
z0=tr[~m].reset_index(drop=True); pos=(y[~m]>0).astype(float)
best=[]
for c in F:
    b=binz(z0[c],20); g=pd.Series(pos).groupby(b).agg(['mean','size']); best.append((c,round(g['mean'].max(),3),int(g['size'][g['mean'].idxmax()])))
print('pred0 max P(y>0) per feature bin',sorted(best,key=lambda t:-t[1])[:8])
