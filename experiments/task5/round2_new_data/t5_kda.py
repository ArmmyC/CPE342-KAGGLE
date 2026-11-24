from common import *
import warnings; warnings.filterwarnings('ignore')
_,te=load(5); fl=np.load('out/t5_flag.npy').astype(bool); un=~fl
def m(b): return te[[f'{b}_{i}' for i in (1,2,3)]].mean(axis=1)
ks=m('kda_std'); km=m('kda_mean'); sr=m('skill_rating'); 
d=pd.DataFrame({'ks':ks,'km':km,'sr':sr,'acc':m('accuracy_mean'),'ksd':te[[f'kda_std_{i}' for i in (1,2,3)]].std(axis=1)})[un]
print(d.corr().round(3))
r=(d.ks/d.km); print('ks/km hist',np.histogram(r.dropna(),bins=30))
print('ks hist 2d vs km quantiles:')
q=pd.qcut(d.km,8)
print(d.groupby(q).ks.describe()[['mean','std','min','max']].round(3))
