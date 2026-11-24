from common import *
import warnings; warnings.filterwarnings('ignore')
_,te=load(5); fl=np.load('out/t5_flag.npy').astype(bool); un=~fl
def L(b,ps=(1,2,3,4)): return te[[f'{b}_{i}' for i in ps]].mean(axis=1)
d=pd.DataFrame({'ks':L('kda_std'),'as':L('accuracy_std'),'ps':L('performance_stability')})[un]
n=len(d)
def box(k0,k1,a0,a1,p0,p1):
    A=d.ks.between(k0,k1); B=d['as'].between(a0,a1); C=d.ps.between(p0,p1)
    obs=(A&B&C).sum(); exp=A.mean()*B.mean()*C.mean()*n
    return int(obs),round(exp,1)
for spec in [(0.8,1.1,9.3,12,0.85,0.95),(0.78,1.1,9.0,12,0.84,0.96),(0.75,1.1,8.5,12,0.83,0.97),(0.7,1.1,8.0,12,0.82,0.98),(0.8,1.1,9.3,12,0.6,1.1),(0.6,1.1,6,12,0.85,0.95)]:
    print(spec, box(*spec))
# marginal views inside core box on each axis
core_ps=d.ps.between(0.85,0.95); core_as=d['as']>9.3; core_ks=d.ks>0.8
print('ks | as&ps core',np.histogram(d.ks[core_as&core_ps].dropna(),bins=np.linspace(0.5,1.05,12))[0].tolist())
print('ks | as&ps ctl ',np.histogram(d.ks[(d['as'].between(6,8))&core_ps].dropna(),bins=np.linspace(0.5,1.05,12))[0].tolist())
print('as | ks&ps core',np.histogram(d['as'][core_ks&core_ps].dropna(),bins=np.linspace(6,13,15))[0].tolist())
print('ps | ks&as core',np.histogram(d.ps[core_ks&core_as].dropna(),bins=np.linspace(0.6,1.05,19))[0].tolist())
