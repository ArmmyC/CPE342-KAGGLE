from common import *
tr,te=load(1)
F=[c for c in tr.columns if c not in('player_id','is_cheater')]
for c in F:
    a=tr[c].dropna(); b=te[c].dropna()
    if a.nunique()<2000: continue
    inter=np.intersect1d(a.values,b.values); w=np.intersect1d(a.values,a.values)
    print(c[:22].ljust(22),'train uniq',a.nunique(),'/',len(a),' test-in-train',len(inter))
