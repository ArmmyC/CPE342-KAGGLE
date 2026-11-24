from common import *
import warnings; warnings.filterwarnings('ignore')
from scipy.ndimage import gaussian_filter1d
_,te=load(5); fl=np.load('out/t5_flag.npy').astype(bool); un=~fl
base=['login_count','location_changes','ip_hash_entropy','login_time_variance','session_duration_mean','session_duration_std','play_pattern_consistency','activity_level','skill_rating','kda_mean','kda_std','accuracy_mean','accuracy_std','reaction_time_mean','reaction_time_std','performance_stability','purchase_frequency','purchase_amount_mean','purchase_amount_std','currency_transfers','friend_adds','friend_removes','communication_frequency']
out=[]
for b in base:
    K=te[[f'{b}_{i}' for i in (1,2,3)]][un]
    x=K.mean(axis=1).dropna(); sd=K.std(axis=1).loc[x.index]
    h,e=np.histogram(x,bins=80); hs=gaussian_filter1d(h.astype(float),1.0); c=(e[1:]+e[:-1])/2
    plateau=np.median(hs[20:60]); 
    lo_i=np.argmax(hs>plateau/2); hi_i=len(hs)-1-np.argmax(hs[::-1]>plateau/2)
    elo,ehi=c[lo_i],c[hi_i]
    # noise near edges
    nlo=sd[(x-elo).abs()<(e[-1]-e[0])*0.04].median()/np.sqrt(3); nhi=sd[(x-ehi).abs()<(e[-1]-e[0])*0.04].median()/np.sqrt(3)
    below=(x<elo-4*nlo).sum(); above=(x>ehi+4*nhi).sum()
    out.append((b,round(elo,3),round(ehi,3),round(nlo,3),round(nhi,3),int(below),int(above)))
print(pd.DataFrame(out,columns=['f','edge_lo','edge_hi','noise_lo','noise_hi','n_below','n_above']).to_string())
