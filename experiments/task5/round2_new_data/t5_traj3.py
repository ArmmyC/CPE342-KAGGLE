from common import *
import warnings; warnings.filterwarnings('ignore')
from scipy.stats import norm
_,te=load(5); fl=np.load('out/t5_flag.npy').astype(bool); un=~fl
T=te[[f'skill_trajectory_{i}' for i in (1,2,3)]][un]
m=T.mean(axis=1).dropna()
med=m.median(); s=1.4826*(m-med).abs().median()
print('robust sd',s.round(3),'n',len(m))
for k in (2,2.5,3,3.5,4,4.5,5):
    obs=((m-med).abs()>k*s).sum(); exp=2*norm.sf(k)*len(m); print(k,'obs',obs,'exp',round(exp,1))
# shape of body: kurtosis of central 95%
c=m[(m-med).abs()<2*s]; print('central uniformity? hist',np.histogram(c,bins=10)[0].tolist())
# within-row noise
sd=T.std(axis=1); print('row noise sd median',sd.median().round(3))
big=m[(m-med).abs()>3*s].index
print('tail rows: sign agreement across periods',(np.sign(T.loc[big]).nunique(axis=1)==1).mean().round(3) if False else (np.sign(T.loc[big]).abs().sum(axis=1)).value_counts().to_dict())
print(T.loc[big].head(8).round(1))
print('tail rows, p4 value:', te.skill_trajectory_4[un].loc[big].describe().round(2).to_dict())
