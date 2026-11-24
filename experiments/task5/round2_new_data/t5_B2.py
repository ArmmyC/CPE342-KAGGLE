from common import *
import warnings; warnings.filterwarnings('ignore')
exec(open('t5_B.py').read().split("rows=[]")[0])
Nmask=N.values
cols=[c for c in F.columns if c.endswith('_mean') or c in('account_age_days','password_changed','vpn_usage','suspicious_login_time','mass_item_sale')]
# restrict to B-sensitive features first
cand=['login_count_mean','purchase_frequency_mean','account_age_days']
X=F[cand].copy()
# impute with N means
X=X.fillna(X[Nmask].mean())
mu=X[Nmask].mean().values; S=np.cov(X[Nmask].values.T)
dB=X[B.values].mean().values-mu
w=np.linalg.solve(S,dB)
s=((X.values-mu)@w)/np.sqrt(w@S@w)
print('B mean score',s[B.values].mean().round(2),'N sd',s[Nmask].std().round(2))
sn=s[Nmask]
for t in [1.5,2,2.5,3,3.5,4,5]:
    print(t,'N right',(sn>t).sum(),'N left',(sn<-t).sum())
# marginals of N
for c in cand: print(c,np.percentile(X.loc[Nmask,c],[0.1,1,5,25,50,75,95,99,99.9]).round(2))
import collections
print(np.histogram(X.loc[Nmask,'login_count_mean'],bins=25))
print(np.histogram(X.loc[Nmask,'purchase_frequency_mean'],bins=25))
