from common import *
import warnings; warnings.filterwarnings('ignore')
exec(open('t5_gap2.py').read().split("def gaps")[0].replace("[un]",""))
un=~G.any(axis=1).values
rows=[]
Vl=V[[c for c in V if c.endswith('_m1234') or c in('account_age_days','behavior_stability')]]
T=te[['password_changed','vpn_usage','suspicious_login_time','mass_item_sale']]
Vl=pd.concat([Vl,T],axis=1)
out={}
for g in ['act','comm','B','bot','skill']:
    s=Vl[G[g].values]; r=Vl[un]
    out[g]=((s.mean()-r.mean())/r.std())
D=pd.DataFrame(out); print(D[(D.abs()>0.25).any(axis=1)].round(2).to_string())
