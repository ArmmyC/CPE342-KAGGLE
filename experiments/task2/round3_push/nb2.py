import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d
DROP0=['random_metric_1','random_metric_2','random_metric_3']
NOINFO=['region','platform','device_type','payment_method','language','account_status','player_type_tag','skill_tier',
        'account_age_days','responds_to_discounts','preferred_game_mode','avg_match_length','item_type_preference_social']
class NB2:
    """Naive Bayes with per-feature binned densities; optional label-noise latent class model via EM.
    Continuous features: atoms for frequent exact values (clip bounds), equal-mass bins + gaussian smoothing."""
    def __init__(s,nbins=60,smooth=1.5,alpha=0.5,atom_frac=0.002,disc_max=25,em=0,Tfix=None,smooth_map=None):
        s.nbins=nbins;s.smooth=smooth;s.alpha=alpha;s.af=atom_frac;s.dm=disc_max;s.em=em;s.Tfix=Tfix;s.sm=smooth_map or {}
    def _codes(s,X,fit):
        # returns list of (code array with -1 for NaN, nlev, kind) per column; edges learned when fit
        if fit: s.enc={}
        out=[]
        for c in s.cols:
            v=X[c]
            if v.dtype==object or str(v.dtype) in('str','category'):
                if fit: s.enc[c]=('cat',{k:i for i,k in enumerate(pd.Series(v.dropna().unique()))})
                mp=s.enc[c][1]; code=v.map(mp).fillna(-1).astype(int).values; out.append((code,len(mp),'cat',0)); continue
            x=v.values.astype(float); m=~np.isnan(x)
            if fit:
                u,cnt=np.unique(x[m],return_counts=True)
                if len(u)<=s.dm: s.enc[c]=('disc',u)
                else:
                    atoms=u[cnt>=s.af*m.sum()]
                    xr=x[m][~np.isin(x[m],atoms)]
                    e=np.unique(np.quantile(xr,np.linspace(0,1,s.nbins+1))); e[0]=-np.inf; e[-1]=np.inf
                    s.enc[c]=('bin',atoms,e)
            sp=s.enc[c]; code=np.full(len(x),-1)
            if sp[0]=='disc':
                u=sp[1]; i=np.clip(np.searchsorted(u,x[m]),0,len(u)-1); code[m]=i; out.append((code,len(u),'disc',0))
            else:
                atoms,e=sp[1],sp[2]; nb=len(e)-1; xm=x[m]
                i=np.clip(np.searchsorted(e,xm,side='right')-1,0,nb-1)
                isat=np.isin(xm,atoms); ai=np.searchsorted(atoms,xm[isat])
                i[isat]=nb+ai; code[m]=i; out.append((code,nb+len(atoms),'bin',nb))
        return out
    def _fitdens(s,codes,R):
        K=R.shape[1]; s.lp=[]
        for j,(code,L,kind,nb) in enumerate(codes):
            m=code>=0; cnt=np.zeros((L,K)); np.add.at(cnt,code[m],R[m])
            cnt=cnt.T
            if kind=='bin':
                sm=s.sm.get(s.cols[j],s.smooth)
                if sm>0: cnt[:,:nb]=gaussian_filter1d(cnt[:,:nb],sm,axis=1,mode='nearest')
            lp=np.log((cnt+s.alpha)/(cnt.sum(1,keepdims=True)+s.alpha*L)); s.lp.append(lp)
    def _loglik(s,codes):
        n=len(codes[0][0]); z=np.tile(np.log(s.pi),(n,1))
        for (code,L,kind,nb),lp in zip(codes,s.lp):
            m=code>=0; z[m]+=lp[:,code[m]].T
        return z
    def fit(s,X,y,K=4):
        s.cols=list(X.columns); codes=s._codes(X,True); R=np.eye(K)[y]
        s.pi=R.mean(0); s.T=np.eye(K); s._fitdens(codes,R)
        for it in range(s.em):
            z=s._loglik(codes); T=s.Tfix if s.Tfix is not None else s.T
            if it==0 and s.Tfix is None: T=np.full((K,K),0.01)+np.eye(K)*0.96
            z=z+np.log(T[:,y].T); z-=z.max(1,keepdims=True); R=np.exp(z); R/=R.sum(1,keepdims=True)
            s.pi=R.mean(0); s._fitdens(codes,R)
            if s.Tfix is None:
                T=R.T@np.eye(K)[y]; s.T=T/T.sum(1,keepdims=True)
            else: s.T=s.Tfix
        return s
    def predict_proba(s,X,clean=False):
        z=s._loglik(s._codes(X,False)); z-=z.max(1,keepdims=True); p=np.exp(z); p/=p.sum(1,keepdims=True)
        return p if clean else p@s.T
