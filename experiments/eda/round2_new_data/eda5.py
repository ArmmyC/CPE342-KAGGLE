from common import *
tr,te=load(2)
f=[c for c in tr.columns if c not in('player_id','segment')]
core=[c for c in f if c not in('region','payment_method')]
n=tr[core].isna().sum(1); print(n.value_counts().head()); 
print('test',te[core].isna().sum(1).value_counts().head())
allna=n==len(core)
print(tr[allna].segment.value_counts(normalize=True).round(3).to_dict(), tr[~allna].segment.value_counts(normalize=True).round(3).to_dict())
print(tr[allna][['region','payment_method']].notna().mean())
print(tr[~allna][['region','payment_method']].notna().mean())
# also check duplicated rows w.r.t. features / near-dups between train and test
print(tr[~allna][core].describe().T[['mean','std','min','max']].round(2).to_string())
# non-all-NaN rows with strange values?
print(tr[['random_metric_1','random_metric_2','random_metric_3']].corrwith(tr.segment))
for c in ['vip_tier','preferred_game_mode','watches_esports','responds_to_discounts']: print(c, tr[c].nunique(), tr.groupby(c).segment.mean().round(2).head(8).to_dict())
