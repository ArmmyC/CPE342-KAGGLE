from common import *
import warnings; warnings.filterwarnings('ignore')
from scipy.stats import multivariate_normal, gaussian_kde
_,te=load(5); fl=np.load('out/t5_flag.npy').astype(bool); un=~fl
def L(b,ps=(1,2,3,4)): return te[[f'{b}_{i}' for i in ps]].mean(axis=1)
D=pd.DataFrame({'ks':L('kda_std'),'as':L('accuracy_std'),'ps':L('performance_stability')})
ok=D.notna().all(axis=1).values
U=D[un&ok]
# background marginals (kde on unflagged, contamination ~1%)
bk={c:gaussian_kde(U[c].values,bw_method=0.05) for c in D}
core=(U.ks.between(0.8,0.95))&(U['as'].between(9.5,11.2))&(U.ps.between(0.87,0.93))
g=U[core]; print('core rows',len(g))
mu=g.mean().values; cov=np.cov(g.values.T)
# widen covariance to undo truncation of the box (inflate sd by 1.3)
sd=np.sqrt(np.diag(cov))*1.3; cov=np.diag(sd**2)
print('group mean',mu.round(3),'sd',sd.round(3))
X=D[ok].values
pg=multivariate_normal(mu,cov).pdf(X)
pb=np.prod([bk[c](D.loc[ok,c].values) for c in D],axis=0)
for prior in (0.009,):
    post=prior*pg/(prior*pg+(1-prior)*pb)
    P=np.zeros(len(D)); P[ok]=post
    for t in (0.05,0.1,0.2,0.5,0.8,0.95):
        m=(P>t)&un
        print(f'thr {t}: flags {m.sum()}  expected TP {P[m].sum():.0f}')
    print('expected group size (sum post over unflagged)',P[un].sum().round(0))
    np.save('out/t5_g7_post.npy',P)
