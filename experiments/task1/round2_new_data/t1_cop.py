from common import *
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import norm
tr,te=load(1)
tr=tr.dropna(subset=['is_cheater']).reset_index(drop=True)
y=tr.is_cheater.astype(int).values
p=np.load('t1_qda_oof.npy')[:,2]
a=tr.account_age_days.values
m=(a>200)&(a<900)
FS=['reports_received','crosshair_placement','game_sense_score','headshot_percentage','accuracy_score','level','spray_control_score','kill_death_ratio','aiming_smoothness','win_rate','account_age_days','kill_consistency','reaction_time_ms','input_consistency_score','friend_network_size']
S=tr.loc[m,FS].reset_index(drop=True); ys=y[m]; ps=p[m]
NB=30
def fit(X,yy):
    M={}
    for j,c in enumerate(FS):
        v=X[:,j]; ok=~np.isnan(v)
        u=np.unique(v[ok])
        if len(u)<=40: edges=np.r_[u-1e-6,u[-1]+1e-6]
        else: edges=np.unique(np.quantile(v[ok],np.linspace(0,1,NB+1))); edges[0]-=1e-9; edges[-1]+=1e-9
        M[c]=edges
    P={}
    for k in (0,1):
        Xk=X[yy==k]; dens=[]; Z=np.full(Xk.shape,np.nan)
        for j,c in enumerate(FS):
            e=M[c]; v=Xk[:,j]; ok=~np.isnan(v)
            h=np.histogram(v[ok],e)[0]+0.5; dens.append(np.log(h/h.sum()))
            # within-class rank -> normal (empirical cdf)
            sv=np.sort(v[ok]); Z[ok,j]=norm.ppf((np.searchsorted(sv,v[ok],'right')-0.5)/len(sv))
        R=pd.DataFrame(Z).corr().values
        P[k]=(dens,np.sort,[np.sort(Xk[~np.isnan(Xk[:,j]),j]) for j in range(len(FS))],R)
    return M,P
def score(M,P,X,cop=True):
    out=np.zeros(len(X))
    ll={}
    for k in (0,1):
        dens,_,sorts,R=P[k]; L=np.zeros(len(X)); Z=np.full(X.shape,np.nan)
        for j,c in enumerate(FS):
            v=X[:,j]; ok=~np.isnan(v)
            b=np.clip(np.searchsorted(M[c],v[ok],'right')-1,0,len(dens[j])-1)
            L[ok]+=dens[j][b]
            sv=sorts[j]; Z[ok,j]=norm.ppf(np.clip(np.searchsorted(sv,v[ok],'right')/(len(sv)+1),1e-4,1-1e-4))
        if cop:
            pat=np.isnan(Z); keys=np.packbits(pat,axis=1); kk=[bytes(r) for r in keys]
            dfk=pd.Series(range(len(X))).groupby(kk).apply(list)
            for idx in dfk:
                idx=np.array(idx); o=~pat[idx[0]]
                Rs=R[np.ix_(o,o)]; Ri=np.linalg.inv(Rs)-np.eye(o.sum()); z=Z[np.ix_(idx,np.where(o)[0])]
                L[idx]+=-0.5*np.linalg.slogdet(Rs)[1]-0.5*np.einsum('ij,jk,ik->i',z,Ri,z)
        ll[k]=L
    return ll[1]-ll[0]
oc=np.zeros(len(ys)); on=np.zeros(len(ys))
X=S.values
for a_,b_ in StratifiedKFold(5,shuffle=True,random_state=0).split(X,ys):
    M,P=fit(X[a_],ys[a_]); oc[b_]=score(M,P,X[b_]); on[b_]=score(M,P,X[b_],cop=False)
print('lgb',roc_auc_score(ys,ps),'nb',roc_auc_score(ys,on),'nb+copula',roc_auc_score(ys,oc))
lg=np.log(ps/(1-ps))
for w in (0.2,0.4,0.5,0.6):
    print(w,roc_auc_score(ys,(1-w)*lg+w*oc))
np.save('t1_cop_region.npy',oc)
