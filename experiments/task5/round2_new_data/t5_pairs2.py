from common import *
import warnings, itertools; warnings.filterwarnings('ignore')
_,te=load(5); fl=np.load('out/t5_flag_all.npy').astype(bool); un=~fl
base=['login_count','location_changes','device_count','ip_hash_entropy','login_time_variance','session_duration_mean','session_duration_std','play_pattern_consistency','activity_level','skill_rating','skill_trajectory','kda_mean','kda_std','accuracy_mean','accuracy_std','reaction_time_mean','reaction_time_std','performance_stability','purchase_frequency','purchase_amount_mean','purchase_amount_std','refund_count','currency_transfers','friend_adds','friend_removes','communication_frequency','guild_changes']
V={}
for b in base:
    M=te[[f'{b}_{i}' for i in range(1,5)]]
    V[b+'_L']=M.iloc[:,:3].mean(axis=1)
    V[b+'_D']=M.iloc[:,3]-M.iloc[:,:3].mean(axis=1)
for c in ['account_age_days','behavior_stability']: V[c]=te[c]
V=pd.DataFrame(V)[un].reset_index(drop=True)
R=V.rank(pct=True)
n=len(R); q=0.05
cols=list(R.columns)
hi=(R>1-q).values; lo=(R<q).values; nan=R.isna().values
res=[]
for i,j in itertools.combinations(range(len(cols)),2):
    for a,A in (('hi',hi),('lo',lo)):
        for b_,B in (('hi',hi),('lo',lo)):
            obs=(A[:,i]&B[:,j]).sum()
            exp=A[:,i].sum()*B[:,j].sum()/n
            z=(obs-exp)/np.sqrt(exp)
            res.append((cols[i],a,cols[j],b_,int(obs),round(exp,1),round(z,1)))
D0=pd.DataFrame(res,columns=["f1","s1","f2","s2","obs","exp","z"])
D=D0[D0.f1.str.rsplit("_",n=1).str[0]!=D0.f2.str.rsplit("_",n=1).str[0]].sort_values("z",ascending=False); print(D.head(25).to_string()); print(D.tail(5).to_string())
# flags vs features
F=te[['password_changed','vpn_usage','suspicious_login_time','mass_item_sale']][un].reset_index(drop=True)
res=[]
for f in F:
    on=(F[f]==1).values
    for c in cols:
        for a,A in (('hi',hi),('lo',lo)):
            k=cols.index(c); obs=(on&A[:,k]).sum(); exp=on.sum()*A[:,k].sum()/n
            res.append((f,c,a,int(obs),round(exp,1),round((obs-exp)/np.sqrt(exp),1)))
print(pd.DataFrame(res,columns=['flag','f','s','obs','exp','z']).sort_values('z',ascending=False).head(12).to_string())
