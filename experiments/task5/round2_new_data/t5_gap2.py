from common import *
import warnings; warnings.filterwarnings('ignore')
_,te=load(5)
G=pd.read_pickle('t5_groups.pkl'); un=~G.any(axis=1).values
base=['login_count','location_changes','device_count','ip_hash_entropy','login_time_variance','session_duration_mean','session_duration_std','play_pattern_consistency','activity_level','skill_rating','skill_trajectory','kda_mean','kda_std','accuracy_mean','accuracy_std','reaction_time_mean','reaction_time_std','performance_stability','purchase_frequency','purchase_amount_mean','purchase_amount_std','refund_count','currency_transfers','friend_adds','friend_removes','communication_frequency','guild_changes']
V={}
for b in base:
    M=te[[f'{b}_{i}' for i in range(1,5)]]
    V[b+'_m1234']=M.mean(axis=1); V[b+'_m123']=M.iloc[:,:3].mean(axis=1); V[b+'_sd123']=M.iloc[:,:3].std(axis=1)
    V[b+'_p1']=M.iloc[:,0]; V[b+'_p4']=M.iloc[:,3]; V[b+'_p4mp']=M.iloc[:,3]-M.iloc[:,:3].mean(axis=1)
for i in range(1,5): V[f'lat{i}']=te[f'login_lat_{i}']; V[f'lon{i}']=te[f'login_lon_{i}']
for c in ['account_age_days','behavior_stability']: V[c]=te[c]
V=pd.DataFrame(V)[un]
def gaps(x,name,fac=25,maxfrac=0.2):
    x=x.dropna().sort_values().values; n=len(x); rng=x[-1]-x[0]
    dx=np.diff(x); typ=np.median(dx[dx>0])*fac
    out=[]
    for i in np.where(dx>max(typ,rng*0.01))[0]:
        k=i+1
        if min(k,n-k)<n*maxfrac and min(k,n-k)>=15: out.append((name,round(x[i],3),round(x[i+1],3),'low' if k<n/2 else 'high', int(k if k<n/2 else n-k)))
    return out
for c in V:
    for r in gaps(V[c],c): print(r)
# binary flags
F=te[['password_changed','vpn_usage','suspicious_login_time','mass_item_sale']][un]
print(F.mean().round(3).to_dict(), F.dropna().astype(int).astype(str).agg(''.join,axis=1).value_counts().to_dict())
for g in G: print(g, te.loc[G[g],['password_changed','vpn_usage','suspicious_login_time','mass_item_sale']].mean().round(2).to_dict())
