from common import *
import warnings; warnings.filterwarnings('ignore')
_,te=load(5); fl=np.load('out/t5_flag.npy').astype(bool); un=~fl
def m(b,ps=(1,2,3,4)): return te[[f'{b}_{i}' for i in ps]].mean(axis=1)
feats=['accuracy_mean','kda_mean','skill_rating','reaction_time_mean','session_duration_mean','friend_adds','kda_std','purchase_amount_mean','currency_transfers']
for f in ['password_changed','mass_item_sale']:
    on=(te[f]==1).values&un; off=(te[f]==0).values&un
    print('==',f,on.sum())
    for b in feats:
        x=m(b); q=[0.01,0.05,0.25,0.5,0.75,0.95,0.99]
        a=np.nanquantile(x[on],q); c=np.nanquantile(x[off],q); sd=x[off].std()
        print(f'{b:22s}',((a-c)/sd).round(2))
