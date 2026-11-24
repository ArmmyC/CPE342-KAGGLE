from common import *
import warnings; warnings.filterwarnings('ignore')
_,te=load(5)
base=['login_count','location_changes','device_count','ip_hash_entropy','login_time_variance','session_duration_mean','session_duration_std','play_pattern_consistency','activity_level','skill_rating','skill_trajectory','kda_mean','kda_std','accuracy_mean','accuracy_std','reaction_time_mean','reaction_time_std','performance_stability','purchase_frequency','purchase_amount_mean','purchase_amount_std','refund_count','currency_transfers','friend_adds','friend_removes','communication_frequency','guild_changes']
lat=te[[f'login_lat_{i}' for i in range(1,5)]].values; lon=te[[f'login_lon_{i}' for i in range(1,5)]].values
dl=np.sqrt((lat[:,3]-np.nanmean(lat[:,:3],1))**2+(lon[:,3]-np.nanmean(lon[:,:3],1))**2)
loc=(np.nan_to_num(dl)>5)|(te.location_changes_4.fillna(0)>8)
skill=te.skill_trajectory_4.fillna(0)>30
bot=te.ip_hash_entropy_1.fillna(1)<0.2
B=te.behavior_stability<0.5
known=loc|skill|bot|B
print('known',loc.sum(),skill.sum(),bot.sum(),B.sum(),known.sum())
out=[]
Zs={}
for b in base:
    M=te[[f'{b}_{i}' for i in range(1,5)]]
    ref=M.iloc[:,:3].mean(axis=1); sd=M.iloc[:,:3].std(axis=1)
    dev=M.iloc[:,3]-ref
    # robust scale from known-free rows: scale = global MAD of dev among ~known
    s=1.4826*np.nanmedian(np.abs(dev[~known]-np.nanmedian(dev[~known])))
    z=(dev-np.nanmedian(dev[~known]))/s
    Zs[b]=z
    n5=(z.abs()>5)
    out.append((b,round(s,3),int(n5.sum()),int((n5&~known).sum()),int((z>5).sum()),int((z<-5).sum())))
df=pd.DataFrame(out,columns=['f','scale','n|z|>5','n_new','pos','neg']); print(df.to_string())
Z=pd.DataFrame(Zs)
pd.to_pickle(Z,'t5_p4z.pkl')
