from common import *
import warnings; warnings.filterwarnings('ignore')
_,te=load(5); fl=np.load('out/t5_flag.npy').astype(bool); un=~fl
K=te[[f'kda_std_{i}' for i in (1,2,3)]][un]
ks=K.mean(axis=1); sd=K.std(axis=1)
print('noise ratio sd/level by level bins'); q=pd.qcut(ks,8); print(pd.DataFrame({'sd':sd,'q':q}).groupby('q').sd.agg(['mean','median']).round(4))
# single-period p1 distribution
p1=te.kda_std_1[un].dropna()
print(np.histogram(p1,bins=np.arange(0.1,1.3,0.05)))
print(np.histogram(ks.dropna(),bins=np.arange(0.1,1.2,0.05)))
# fit: body uniform [a,b] level, noise sd=c*level -> use sd to deconvolve: level_i ~ ks; edges via quantiles of rows with small noise
print('quantiles of ks',np.nanquantile(ks,[0.001,0.01,0.05,0.5,0.95,0.99,0.999]).round(3))
print('top sorted ks',np.sort(ks.dropna().values)[-15:].round(3))
