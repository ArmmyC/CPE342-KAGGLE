import numpy as np, pandas as pd
R='<project_root>/'
tr=pd.read_csv(R+'MLTest/cpe342-kaggle-main/02_multiclass_classification/train.csv')
pd.set_option('display.width',250)
for c in ['avg_monthly_spending','spending_frequency','rare_items_count','vip_tier','play_frequency','friend_count']:
    print(c); print(tr.groupby(['segment','loyalty_tier'])[c].agg(['mean','count']).unstack(0).round(1).to_string())
print(tr.groupby(['segment','engagement_level'])['play_frequency'].agg(['mean','count']).unstack(0).round(2).to_string())
print(tr.groupby(['segment','engagement_level'])['avg_monthly_spending'].agg(['mean']).unstack(0).round(0).to_string())
