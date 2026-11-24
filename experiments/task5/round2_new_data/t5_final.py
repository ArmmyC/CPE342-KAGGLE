"""Task 5: unsupervised anomaly detection. Each anomaly type is a separated sub-population:
  loc   : period-4 location/device/purchase/currency shift (location jump or location_changes_4)
  skill : period-4 skill spike (skill_trajectory_4)
  bot   : very low ip entropy / login time variance
  B     : behavior_stability far below the normal range
  act   : activity_level level far below the normal range
  comm  : communication_frequency level far below the normal range
plus rows with >=2 period-4 features beyond 5 robust local sd.
All thresholds sit inside empty gaps of the feature histograms (no labels used)."""
from common import *
import warnings; warnings.filterwarnings('ignore')
_,te=load(5)
def m(b,ps): return te[[f'{b}_{i}' for i in ps]].mean(axis=1)
lat=te[[f'login_lat_{i}' for i in range(1,5)]].values; lon=te[[f'login_lon_{i}' for i in range(1,5)]].values
dl=np.sqrt((lat[:,3]-np.nanmean(lat[:,:3],1))**2+(lon[:,3]-np.nanmean(lon[:,:3],1))**2)
G=pd.DataFrame({
 'loc':(np.nan_to_num(dl)>5)|(te.location_changes_4.fillna(0)>8),
 'skill':te.skill_trajectory_4.fillna(0)>30,
 'bot':(m('ip_hash_entropy',[1,2,3])<0.2)|(m('login_time_variance',[1,2,3])<1.0),
 'B':te.behavior_stability<0.55,
 'act':m('activity_level',[1,2,3])<0.33,
 'comm':m('communication_frequency',[1,2,3])<0.16})
Z=pd.read_pickle('t5_p4z.pkl').drop(columns=['device_count','refund_count','guild_changes','friend_removes'])
extra=((Z.abs()>5).sum(axis=1)>=2)
flag=G.any(axis=1)|extra
print(G.sum().to_dict(),'extra',int((extra&~G.any(axis=1)).sum()),'total',int(flag.sum()),flag.mean().round(4))
np.save('out/t5_flag.npy',flag.values.astype(int))
# 7th group (found later, see t5_g7.py): high-variance cluster in period means of
# kda_std (~0.86), accuracy_std (~10.4), performance_stability (~0.90); flagged where posterior > 0.1.
