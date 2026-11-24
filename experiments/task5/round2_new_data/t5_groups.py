from common import *
import warnings; warnings.filterwarnings('ignore')
exec(open('t5_gap.py').read().split("def gaps")[0])
def m(b,n=4): return te[[f'{b}_{i}' for i in range(1,n+1)]].mean(axis=1)
G={}
lat=te[[f'login_lat_{i}' for i in range(1,5)]].values; lon=te[[f'login_lon_{i}' for i in range(1,5)]].values
dl=np.sqrt((lat[:,3]-np.nanmean(lat[:,:3],1))**2+(lon[:,3]-np.nanmean(lon[:,:3],1))**2)
G['loc']=(np.nan_to_num(dl)>5)|(te.location_changes_4.fillna(0)>8)
G['skill']=te.skill_trajectory_4.fillna(0)>30
G['bot']=(V.ip_hash_entropy_m123<0.2)|(V.login_time_variance_m123<1.0)
G['B']=te.behavior_stability<0.55
G['act']=V.activity_level_m123<0.33
G['comm']=V.communication_frequency_m123<0.16
G=pd.DataFrame(G)
print(G.sum().to_dict())
print('union',G.any(axis=1).sum(),'multi',(G.sum(axis=1)>1).sum())
print(G.T.dot(G.astype(int)))
G.to_pickle('t5_groups.pkl')
