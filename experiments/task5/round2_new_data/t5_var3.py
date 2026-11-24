from common import *
import warnings; warnings.filterwarnings('ignore')
_,te=load(5); fl=np.load('out/t5_flag.npy').astype(bool); un=~fl
def L(b,ps=(1,2,3,4)): return te[[f'{b}_{i}' for i in ps]].mean(axis=1)
d=pd.DataFrame({'ks':L('kda_std'),'as':L('accuracy_std'),'ps':L('performance_stability')})
box=un&(d.ks>0.8).values&(d['as']>9.3).values
ctl=un&(d.ks.between(0.5,0.75)).values&(d['as'].between(5,8)).values
print('perf_stab in box',np.histogram(d.ps[box].dropna(),bins=np.linspace(0.55,1.1,12))[0].tolist())
print('perf_stab ctl   ',np.histogram(d.ps[ctl].dropna(),bins=np.linspace(0.55,1.1,12))[0].tolist())
# 2D 4-period
H,xe,ye=np.histogram2d(d.ks[un].fillna(0),d['as'][un].fillna(0),bins=[np.linspace(0.6,1.05,10),np.linspace(6,13,15)])
print(xe.round(2),ye.round(1)); print(H.astype(int))
# Restrict to perf_stab high
hp=un&(d.ps>0.85).values
H,xe,ye=np.histogram2d(d.ks[hp].fillna(0),d['as'][hp].fillna(0),bins=[np.linspace(0.6,1.05,10),np.linspace(6,13,15)])
print('perf_stab>0.85'); print(H.astype(int))
