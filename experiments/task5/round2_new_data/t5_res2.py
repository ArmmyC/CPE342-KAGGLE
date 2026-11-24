from common import *
import warnings; warnings.filterwarnings('ignore')
G=pd.read_pickle('t5_groups.pkl'); fl=np.load('out/t5_flag.npy').astype(bool); un=~fl
R=pd.read_parquet('t5_res.parquet')[un]
R=R.drop(columns=[c for c in R if c.endswith('_4') or c in('mass_item_sale','password_changed','suspicious_login_time','vpn_usage')])
Z=(R-R.median())/(1.4826*(R-R.median()).abs().median())
k=Z.apply(lambda s: s.kurt()).sort_values(ascending=False)
print(k.head(15).round(2).to_dict())
print((Z.abs()>5).sum().sort_values(ascending=False).head(12).to_dict())
