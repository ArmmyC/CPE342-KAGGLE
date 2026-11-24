import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d
class HistNB:
    """Class-conditional per-feature histogram likelihoods; NaN features skipped (exact under MCAR + cond. independence)."""
    def __init__(self,nbins=60,smooth=1.5,alpha=1.0,min_unique_cat=25):
        self.nbins=nbins;self.smooth=smooth;self.alpha=alpha;self.mu=min_unique_cat
    def fit(self,X,y,ncls=None):
        self.cols=list(X.columns); self.K=ncls or int(y.max())+1
        self.prior=np.bincount(y,minlength=self.K)/len(y); self.spec={}
        for c in self.cols:
            v=X[c]
            if v.dtype==object or str(v.dtype) in('str','category'):
                cats=pd.Series(v.dropna().unique()); mp={k:i for i,k in enumerate(cats)}
                idx=v.map(mp); m=idx.notna().values
                cnt=np.zeros((self.K,len(mp)))
                np.add.at(cnt,(y[m],idx[m].astype(int).values),1)
                lp=np.log((cnt+self.alpha)/(cnt.sum(1,keepdims=True)+self.alpha*len(mp)))
                self.spec[c]=('cat',mp,lp)
            else:
                x=v.values.astype(float); m=~np.isnan(x); u=np.unique(x[m])
                if len(u)<=self.mu: # discrete
                    mp={k:i for i,k in enumerate(u)}; idx=np.array([mp[a] for a in x[m]])
                    cnt=np.zeros((self.K,len(u))); np.add.at(cnt,(y[m],idx),1)
                    lp=np.log((cnt+self.alpha)/(cnt.sum(1,keepdims=True)+self.alpha*len(u)))
                    self.spec[c]=('disc',u,lp)
                else:
                    qs=np.unique(np.quantile(x[m],np.linspace(0,1,self.nbins+1)))
                    edges=qs.copy(); edges[0]=-np.inf; edges[-1]=np.inf
                    b=np.clip(np.searchsorted(edges,x[m],side='right')-1,0,len(edges)-2)
                    nb=len(edges)-1; cnt=np.zeros((self.K,nb)); np.add.at(cnt,(y[m],b),1)
                    # density per unit: divide by bin mass share (equal-mass bins => counts ~ density)
                    cnt=gaussian_filter1d(cnt,self.smooth,axis=1,mode='nearest')
                    lp=np.log((cnt+self.alpha)/(cnt.sum(1,keepdims=True)+self.alpha*nb))
                    self.spec[c]=('bin',edges,lp)
        return self
    def ll(self,X):
        n=len(X); out=np.tile(np.log(self.prior),(n,1))
        per={}
        for c in self.cols:
            kind,a,lp=self.spec[c]; v=X[c]
            if kind=='cat':
                idx=v.map(a); m=idx.notna().values; ii=idx[m].astype(int).values
            elif kind=='disc':
                x=v.values.astype(float); m=~np.isnan(x); ii=np.searchsorted(a,x[m]); ii=np.clip(ii,0,len(a)-1)
            else:
                x=v.values.astype(float); m=~np.isnan(x); ii=np.clip(np.searchsorted(a,x[m],side='right')-1,0,len(a)-2)
            add=np.zeros((n,self.K)); add[m]=lp[:,ii].T
            # unseen category -> 0 contribution already for NaN rows
            per[c]=add
        return out,per
    def predict_logit(self,X,drop=()):
        out,per=self.ll(X)
        for c,a in per.items():
            if c not in drop: out=out+a
        return out
    def predict_proba(self,X,drop=()):
        z=self.predict_logit(X,drop); z-=z.max(1,keepdims=True); p=np.exp(z); return p/p.sum(1,keepdims=True)
