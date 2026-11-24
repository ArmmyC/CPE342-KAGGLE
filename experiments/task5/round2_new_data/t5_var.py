from common import *
import warnings; warnings.filterwarnings('ignore')
_,te=load(5); fl=np.load('out/t5_flag.npy').astype(bool); un=~fl
def L(b): return te[[f'{b}_{i}' for i in (1,2,3)]].mean(axis=1)
ks=L('kda_std'); as_=L('accuracy_std')
d=pd.DataFrame({'ks':ks,'as':as_})[un].dropna()
# 2D histogram in rank space of top corner
H,xe,ye=np.histogram2d(d.ks,d['as'],bins=[np.linspace(0.15,1.05,19),np.linspace(1.5,14,26)])
print('kda_std edges',xe.round(2)); print('acc_std edges',ye.round(1))
print(pd.DataFrame(H.astype(int),index=xe[:-1].round(2),columns=ye[:-1].round(1)).to_string())
