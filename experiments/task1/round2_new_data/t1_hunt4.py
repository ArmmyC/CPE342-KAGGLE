from common import *
tr,te=load(1)
tr=tr.dropna(subset=['is_cheater'])
y=tr.is_cheater.values
F=[c for c in tr.columns if c not in('player_id','is_cheater')]
a=tr.account_age_days
for k,(lo,hi) in [(0,(0,115)),(0,(300,800)),(0,(950,1500)),(1,(0,190)),(1,(200,1000))]:
    s=tr[(y==k)&(a>lo)&(a<hi)]
    R=s[F].rank(pct=True).corr().values; iu=np.triu_indices(len(F),1)
    o=np.argsort(-np.abs(R[iu]))[:8]
    print(k,lo,hi,len(s),[(F[iu[0][i]][:10],F[iu[1][i]][:10],round(R[iu][i],3)) for i in o])
# fine age edges
for k in (0,1):
    s=np.sort(a[(y==k)].dropna().values)
    g=np.diff(s); i=np.argsort(-g)[:6]
    print(k,[(round(s[j],2),round(s[j+1],2)) for j in sorted(i)])
