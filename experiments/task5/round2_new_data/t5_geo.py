from common import *
import warnings; warnings.filterwarnings('ignore')
_,te=load(5)
for c in ['login_lat_1','login_lon_1','login_lat_4','login_lon_4']:
    v=te[c].dropna(); print(c,v.describe().round(2).to_dict())
lat=te[[f'login_lat_{i}' for i in range(1,5)]].values; lon=te[[f'login_lon_{i}' for i in range(1,5)]].values
print('corr lat periods',np.corrcoef(np.nan_to_num(lat[:,0]),np.nan_to_num(lat[:,1]))[0,1])
d=np.sqrt(np.diff(lat,axis=1)**2+np.diff(lon,axis=1)**2)
for i in range(3):
    x=d[:,i][~np.isnan(d[:,i])]; print('step',i,np.percentile(x,[1,10,50,90,95,99,99.9]).round(3), 'frac>1',(x>1).mean().round(4),'>5',(x>5).mean().round(4))
dm=np.nanmax(d,axis=1)
h,e=np.histogram(dm[~np.isnan(dm)],bins=40); print(h.tolist(), e[[0,-1]])
# location_changes vs distance
lc=te[[f'location_changes_{i}' for i in range(1,5)]].values
print(np.nanmean(lc,0), np.nanstd(lc,0))
# lat/lon relation to anything: is lat tied to account? print some rows
print(te[['login_lat_1','login_lon_1','login_lat_2','login_lon_2','login_lat_3','login_lon_3','login_lat_4','login_lon_4']].head(8).round(3).to_string())
