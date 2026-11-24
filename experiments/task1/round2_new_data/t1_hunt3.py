from common import *
tr,te=load(1)
tr=tr.dropna(subset=['is_cheater'])
y=tr.is_cheater.values
# legit: look for age cut points: fine bins of age, mean of reports/friends
for k in (0,1):
    s=tr[y==k]; b=pd.cut(s.account_age_days,[0,30,60,90,100,110,120,130,140,150,160,180,200,250,300,800,900,950,1000,1050,1100,1200,1500])
    print('class',k); print(s.groupby(b,observed=True)[['reports_received','kill_death_ratio','headshot_percentage','game_sense_score','friend_network_size','level']].agg(['mean']).round(2).assign(n=s.groupby(b,observed=True).size()).to_string())
