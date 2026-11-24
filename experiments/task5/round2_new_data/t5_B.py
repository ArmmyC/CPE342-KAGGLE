from common import *
import warnings; warnings.filterwarnings('ignore')
_,te=load(5)
P=range(1,5)
base=['login_count','location_changes','device_count','ip_hash_entropy','login_time_variance','session_duration_mean','session_duration_std','play_pattern_consistency','activity_level','skill_rating','skill_trajectory','kda_mean','kda_std','accuracy_mean','accuracy_std','reaction_time_mean','reaction_time_std','performance_stability','purchase_frequency','purchase_amount_mean','purchase_amount_std','refund_count','currency_transfers','friend_adds','friend_removes','communication_frequency','guild_changes']
lat=te[[f'login_lat_{i}' for i in P]].values; lon=te[[f'login_lon_{i}' for i in P]].values
d03=np.sqrt((lat[:,0]-lat[:,3])**2+(lon[:,0]-lon[:,3])**2)
A=np.nan_to_num(d03>5)|(te.location_changes_4.fillna(0)>8)
bot=(te.ip_hash_entropy_1.fillna(1)<0.2)
B=te.behavior_stability<0.5
N=~(A|bot|B)
print(A.sum(),bot.sum(),B.sum(),N.sum(), (A&B).sum())
feat={}
for b in base:
    M=te[[f'{b}_{i}' for i in P]]
    feat[b+'_mean']=M.mean(axis=1)
    feat[b+'_p4']=M.iloc[:,3]
    feat[b+'_p123']=M.iloc[:,:3].mean(axis=1)
F=pd.DataFrame(feat); 
for c in ['account_age_days','behavior_stability','password_changed','vpn_usage','suspicious_login_time','mass_item_sale']: F[c]=te[c]
rows=[]
for c in F:
    s=F.loc[N,c].std()
    rows.append((c,*[(F.loc[g,c].mean()-F.loc[N,c].mean())/s for g in (A,bot,B)]))
R=pd.DataFrame(rows,columns=['f','A','bot','B']).set_index('f')
print(R.round(2).to_string())
