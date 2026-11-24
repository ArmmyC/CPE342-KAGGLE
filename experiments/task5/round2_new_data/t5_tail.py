from common import *
import warnings; warnings.filterwarnings('ignore')
_,te=load(5); fl=np.load('out/t5_flag.npy').astype(bool); un=~fl
base=['login_count','location_changes','device_count','ip_hash_entropy','login_time_variance','session_duration_mean','session_duration_std','play_pattern_consistency','activity_level','skill_rating','skill_trajectory','kda_mean','kda_std','accuracy_mean','accuracy_std','reaction_time_mean','reaction_time_std','performance_stability','purchase_frequency','purchase_amount_mean','purchase_amount_std','refund_count','currency_transfers','friend_adds','friend_removes','communication_frequency','guild_changes']
rows=[]
for b in base:
    x=te[[f'{b}_{i}' for i in (1,2,3)]].mean(axis=1)[un].dropna()
    q=np.nanquantile(x,[0.0005,0.005,0.02,0.5,0.98,0.995,0.9995])
    spread=q[4]-q[2]
    rows.append((b,round((q[1]-q[0])/spread,3),round((q[2]-q[1])/spread,3),round((q[5]-q[4])/spread,3),round((q[6]-q[5])/spread,3)))
print(pd.DataFrame(rows,columns=['f','lowfar','lownear','highnear','highfar']).to_string())
x=te.account_age_days[un].dropna(); print('age',x.min(),x.max())
