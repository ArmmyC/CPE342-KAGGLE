from common import *
import warnings; warnings.filterwarnings('ignore')
_,te=load(5); fl=np.load('out/t5_flag.npy').astype(bool); un=~fl
base=['login_count','location_changes','device_count','ip_hash_entropy','login_time_variance','session_duration_mean','session_duration_std','play_pattern_consistency','activity_level','skill_rating','skill_trajectory','kda_mean','kda_std','accuracy_mean','accuracy_std','reaction_time_mean','reaction_time_std','performance_stability','purchase_frequency','purchase_amount_mean','purchase_amount_std','refund_count','currency_transfers','friend_adds','friend_removes','communication_frequency','guild_changes']
V={}
for b in base:
    M=te[[f'{b}_{i}' for i in range(1,5)]]
    V[b+'_L']=M.iloc[:,:3].mean(axis=1); V[b+'_D']=M.iloc[:,3]-V[b+'_L']; V[b+'_S']=M.iloc[:,:3].std(axis=1)
for c in ['account_age_days','behavior_stability','password_changed','vpn_usage','suspicious_login_time','mass_item_sale']: V[c]=te[c]
V=pd.DataFrame(V)
box=un&(V.kda_std_L>0.8).values&(V.accuracy_std_L>9.3).values
ref=un&~box
print('box rows',box.sum())
d=((V[box].mean()-V[ref].mean())/V[ref].std()).sort_values(key=abs,ascending=False)
print(d.head(20).round(2).to_string())
