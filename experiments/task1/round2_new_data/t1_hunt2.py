from common import *
tr,te=load(1)
tr=tr.dropna(subset=['is_cheater'])
y=tr.is_cheater.values
for c in ['account_age_days','reports_received','kill_death_ratio','headshot_percentage','game_sense_score','level_progression_speed','friend_network_size']:
    for k in (0,1):
        q=tr.loc[y==k,c].quantile([0,.01,.05,.1,.25,.5,.75,.9,.95,.99,1]).values
        print(c[:18].ljust(18),k,np.round(q,3))
# within-class conditional means by age decile
for k in (0,1):
    s=tr[y==k].copy(); s['ad']=pd.qcut(s.account_age_days,10,labels=False)
    print('class',k); print(s.groupby('ad')[['account_age_days','reports_received','kill_death_ratio','headshot_percentage','game_sense_score','friend_network_size']].mean().round(3).to_string())
