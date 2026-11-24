from common import *
import warnings; warnings.filterwarnings('ignore')
_,te=load(5)
for b in ['device_count','refund_count','guild_changes','friend_removes','location_changes','friend_adds','currency_transfers']:
    print('==',b)
    for t in range(1,5):
        v=te[f'{b}_{t}'].dropna()
        vc=v.round(0).value_counts().sort_index()
        print(t, 'nuniq',v.nunique(),'max',v.max().round(2),'mean',v.mean().round(3), (vc.head(12).to_dict() if v.nunique()<40 else ''))
