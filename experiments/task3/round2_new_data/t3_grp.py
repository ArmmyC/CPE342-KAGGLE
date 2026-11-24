from common import *
tr,te=load(3)
y=tr.spending_30d.values; o=np.load('push/t3/final_m42_oof.npy')
s=np.sign(y-o); m=o>0
g=tr.vip_status.astype(str)+'_'+tr.is_premium_member.astype(str)
print(pd.DataFrame({'g':g[m],'s':s[m],'y':y[m],'o':o[m]}).groupby('g').agg(n=('s','size'),sign=('s','mean'),o=('o','median'),y=('y','median')).round(3).to_string())
for c in ['tournament_participation','segment','guild_membership','owns_limited_edition','is_premium_member','platform']:
    print(c,pd.DataFrame({'g':tr[c].values[m],'s':s[m]}).groupby('g').s.agg(['size','mean']).round(3).T.to_string())
