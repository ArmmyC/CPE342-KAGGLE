import numpy as np
from c1 import *
X,y,U,T=load()
d=X; ok=d[['level','level_progression_speed','account_age_days']].notna().all(1)
u=d.level/(d.level_progression_speed*d.account_age_days)
print('U range by class', u[ok&(y==1)].quantile([0,.01,.5,.99,1]).values, u[ok&(y==0)].quantile([0,.01,.5,.99,1]).values)
a=d.level/d.level_progression_speed  # = age*U
m=d.account_age_days.isna()&d.level.notna()&d.level_progression_speed.notna()
lo_age=a*0.16*0.999; hi_age=a*1.07*1.001  # age = a/U
print('rows age missing:',m.sum(),' with age certainly >1000:',(m&(a/6.67>1000)).sum(),' certainly <15:',(m&(a/0.925<15)).sum())
print(' cheat rate among those', y[(m&((a/6.67>1000)|(a/0.925<15))).values].mean() if (m&((a/6.67>1000)|(a/0.925<15))).sum() else None)
# existing model on those rows
base=np.load(P+'/t1_seedsbox_oof.npy'); sel=(m&((a/6.67>1000)|(a/0.925<15))).values
print(' prod OOF on those rows: mean',base[sel].mean(),'max',base[sel].max(),'n >0.37',(base[sel]>0.37).sum())
# level missing: level = lps*age*U in [0.16,1.07]*lps*age; box level 25-96
m2=d.level.isna()&d.level_progression_speed.notna()&d.account_age_days.notna(); b=d.level_progression_speed*d.account_age_days
sel2=(m2&((b*0.925>96)|(b*6.67<25))).values
print('level missing',m2.sum(),'certain outside',sel2.sum(),'cheat',y[sel2].mean() if sel2.sum() else None,'prod>0.37',(base[sel2]>0.37).sum())
m3=d.level_progression_speed.isna()&d.level.notna()&d.account_age_days.notna(); c=d.level/d.account_age_days
sel3=(m3&((c/0.925<0.0044)|(c/6.67>6.26))).values
print('lps missing',m3.sum(),'certain outside',sel3.sum(),'prod>0.37',(base[sel3]>0.37).sum())
