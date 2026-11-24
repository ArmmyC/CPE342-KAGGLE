from common import *
tr,te=load(1)
tr=tr.dropna(subset=['is_cheater']).reset_index(drop=True)
y=tr.is_cheater.astype(int).values
for k in (1,0):
    s=tr[y==k]
    print('class',k)
    print(pd.crosstab(s.reports_received,pd.cut(s.game_sense_score,[0.4,.5,.6,.7,.75,.8,.85,.9,.95,.99,1.0001])).to_string())
