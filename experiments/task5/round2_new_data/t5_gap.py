from common import *
import warnings; warnings.filterwarnings('ignore')
_,te=load(5)
base=['login_count','location_changes','device_count','ip_hash_entropy','login_time_variance','session_duration_mean','session_duration_std','play_pattern_consistency','activity_level','skill_rating','skill_trajectory','kda_mean','kda_std','accuracy_mean','accuracy_std','reaction_time_mean','reaction_time_std','performance_stability','purchase_frequency','purchase_amount_mean','purchase_amount_std','refund_count','currency_transfers','friend_adds','friend_removes','communication_frequency','guild_changes']
V={}
for b in base:
    M=te[[f'{b}_{i}' for i in range(1,4)]]   # use p1-3 only (p4 shifts for anomalies)
    V[b+'_m123']=M.mean(axis=1)
V['lat']=te[[f'login_lat_{i}' for i in (1,2,3)]].mean(axis=1)
for c in ['account_age_days','behavior_stability']: V[c]=te[c]
V=pd.DataFrame(V)
def gaps(x,name):
    x=x.dropna().sort_values().values; n=len(x)
    rng=x[-1]-x[0]
    dx=np.diff(x)
    typ=np.median(dx[dx>0])*50  # typical local spacing x50
    out=[]
    for i in np.where(dx>max(typ,rng*0.02))[0]:
        k=i+1
        if min(k,n-k)<n*0.1: out.append((name,round(x[i],3),round(x[i+1],3),'low' if k<n/2 else 'high', k if k<n/2 else n-k))
    return out
res=[]
for c in V: res+=gaps(V[c],c)
for r in res: print(r)
