from common import *
tr,_=load(1); print("nan labels",tr.is_cheater.isna().sum()); tr=tr[tr.is_cheater.notna()]; y=tr.is_cheater.values.astype(int)
f=[c for c in tr.columns if c not in('player_id','is_cheater')]
for c in (0,1):
    d=tr[y==c][f].rank()
    C=d.corr().values.copy(); np.fill_diagonal(C,0)
    pairs=[(abs(C[i,j]),f[i],f[j],round(C[i,j],3)) for i in range(len(f)) for j in range(i+1,len(f)) if abs(C[i,j])>0.03]
    print('class',c,sorted(pairs,reverse=True)[:25])
