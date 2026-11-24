from common import *
import warnings; warnings.filterwarnings('ignore')
_,te=load(5)
P=range(1,5)
lat=te[[f'login_lat_{i}' for i in P]].values; lon=te[[f'login_lon_{i}' for i in P]].values
def dist(i,j): return np.sqrt((lat[:,i]-lat[:,j])**2+(lon[:,i]-lon[:,j])**2)
# jump between each pair
J={(i,j):dist(i,j) for i in range(4) for j in range(i+1,4)}
big={k:(v>5) for k,v in J.items()}
for k,v in J.items(): print(k,'n>5',np.nansum(v>5))
# pattern: which period is the odd one out
def odd(r):
    return None
pat=[]
d01,d02,d03,d12,d13,d23=[J[k] for k in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]]
cnt={}
for a in range(len(te)):
    b=''.join('1' if (not np.isnan(x) and x>5) else ('?' if np.isnan(x) else '0') for x in [d01[a],d02[a],d03[a],d12[a],d13[a],d23[a]])
    cnt[b]=cnt.get(b,0)+1
for k,v in sorted(cnt.items(),key=lambda x:-x[1])[:15]: print(k,v)
# Anomaly rows with big move: examine how feature p3 / p4 deviate
anom=np.nan_to_num(d03>5)|np.nan_to_num(d23>5)|np.nan_to_num(d13>5)|np.nan_to_num(d12>5)
print('rows with any big jump',anom.sum())
base=['login_count','location_changes','device_count','ip_hash_entropy','login_time_variance','session_duration_mean','session_duration_std','play_pattern_consistency','activity_level','skill_rating','skill_trajectory','kda_mean','kda_std','accuracy_mean','accuracy_std','reaction_time_mean','reaction_time_std','performance_stability','purchase_frequency','purchase_amount_mean','purchase_amount_std','refund_count','currency_transfers','friend_adds','friend_removes','communication_frequency','guild_changes']
rows=[]
for b in base:
    M=te[[f'{b}_{i}' for i in P]].values
    sd=np.nanstd(M[~anom]); 
    rows.append((b,[np.nanmean(M[anom][:,i]-np.nanmean(M[~anom][:,i]))/np.nanstd(M[~anom][:,i]) for i in range(4)]))
for b,r in rows: print(f'{b:28s}',np.round(r,2))
for c in ['password_changed','vpn_usage','suspicious_login_time','mass_item_sale','behavior_stability','account_age_days']:
    print(c, te.loc[anom,c].mean().round(3), te.loc[~anom,c].mean().round(3))
