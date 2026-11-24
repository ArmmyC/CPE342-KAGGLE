from common import *
import warnings; warnings.filterwarnings('ignore')
_,te=load(5); fl=np.load('out/t5_flag.npy').astype(bool); un=~fl
X=te.drop(columns=['player_id'])
for f in ['suspicious_login_time','vpn_usage','password_changed','mass_item_sale']:
    on=(te[f]==1).values&un; off=(te[f]==0).values&un
    d=((X[on].mean()-X[off].mean())/X[off].std()).drop(f)
    print(f,'n_on',on.sum(),'top diffs',d.abs().sort_values(ascending=False).head(6).round(3).to_dict())
