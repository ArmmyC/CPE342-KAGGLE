from common import *
import warnings; warnings.filterwarnings('ignore')
_,te=load(5)
P=[1,2,3]
base=['login_count','location_changes','device_count','ip_hash_entropy','login_time_variance','session_duration_mean','session_duration_std','play_pattern_consistency','activity_level','skill_rating','skill_trajectory','kda_mean','kda_std','accuracy_mean','accuracy_std','reaction_time_mean','reaction_time_std','performance_stability','purchase_frequency','purchase_amount_mean','purchase_amount_std','refund_count','currency_transfers','friend_adds','friend_removes','communication_frequency','guild_changes']
# normal trajectory limits
for k in range(1,5):
    v=te[f'skill_trajectory_{k}'].dropna(); print(k,np.percentile(v,[0,0.1,1,99,99.9,100]).round(1), 'frac |v|>18',(v.abs()>18).sum())
v4=te.skill_trajectory_4.dropna(); print(np.histogram(v4,bins=[-300,-100,-50,-30,-20,-18,18,20,30,50,100,300,1000]))
# rating deviations
dev=te.skill_rating_4-te[['skill_rating_1','skill_rating_2','skill_rating_3']].mean(axis=1)
rel=dev/te[['skill_rating_1','skill_rating_2','skill_rating_3']].mean(axis=1)
print('rel dev pct',np.nanpercentile(rel,[0.1,1,5,50,95,99,99.9]).round(3))
tj=te.skill_trajectory_4
big_tj=(tj.abs()>18)
print('traj4 big',big_tj.sum(),'with rel dev>0.3',(big_tj&(rel.abs()>0.3)).sum())
# in traj4 big rows what are other indicators
P_=[f'{b}_4' for b in base]
for c in ['location_changes_4','purchase_frequency_4','currency_transfers_4','play_pattern_consistency_4','password_changed','mass_item_sale','vpn_usage']:
    print(c, te.loc[big_tj,c].mean().round(2), te.loc[~big_tj,c].mean().round(2))
