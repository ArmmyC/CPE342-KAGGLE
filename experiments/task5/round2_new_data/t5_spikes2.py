from common import *
import warnings; warnings.filterwarnings('ignore')
exec(open('t5_spikes.py').read().split("out=[]")[0])
def mad(x): x=x[~np.isnan(x)]; return 1.4826*np.median(np.abs(x-np.median(x)))
Zs={}; out=[]
for b in base:
    M=te[[f'{b}_{i}' for i in range(1,5)]].values
    mu=np.nanmean(M[:,:3],axis=1)
    # residuals p1..p3 vs mean of other two
    R=np.stack([M[:,t]-np.nanmean(np.delete(M[:,:3],t,axis=1),axis=1) for t in range(3)],1)
    kn=known.values
    qs=np.nanquantile(mu[~kn],np.linspace(0,1,21)); qs[0]=-np.inf; qs[-1]=np.inf
    bi=np.clip(np.searchsorted(qs,mu,side='right')-1,0,19)
    sig=np.zeros(20)
    for k in range(20):
        sel=(bi==k)&~kn
        sig[k]=mad(R[sel].ravel())/np.sqrt(1.5)
    cen=np.array([np.nanmedian(mu[(bi==k)&~kn]) for k in range(20)])
    s_row=np.interp(mu,cen,sig)
    dev=M[:,3]-mu
    z=dev/(s_row*np.sqrt(4/3)+1e-12)
    Zs[b]=z
    n5=np.abs(z)>5
    out.append((b,sig.min().round(3),sig.max().round(3),int(n5.sum()),int((n5&~kn).sum()),int((z>5).sum()),int((z<-5).sum())))
df=pd.DataFrame(out,columns=['f','smin','smax','n5','n_new','pos','neg']); print(df.to_string())
pd.to_pickle(pd.DataFrame(Zs),'t5_p4z.pkl')
