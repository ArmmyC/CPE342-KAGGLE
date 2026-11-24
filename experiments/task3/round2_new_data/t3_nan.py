from common import *
tr,te=load(3)
y=tr.spending_30d.values; o=np.load('push/t3/final_m42_oof.npy')
s=np.sign(y-o); m=o>0
for c in ['tournament_participation','segment','guild_membership','owns_limited_edition','is_premium_member','platform','vip_status','payment_methods_used','games_played','primary_game']:
    na=tr[c].isna().values
    print(c.ljust(26),'nan rows(pred>0)',(na&m).sum(),'sign nan',round(s[na&m].mean(),3),'sign obs',round(s[~na&m].mean(),3),' |  pred==0 nan: P(y>0)',round((y[na&~m]>0).mean(),3),'obs',round((y[~na&~m]>0).mean(),3))
