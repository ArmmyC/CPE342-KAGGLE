from common import *
tr,te=load(1)
print(tr.describe().T[['mean','std','min','max']].round(2).to_string())
print(tr.isna().mean().round(3)[tr.isna().mean()>0].to_dict())
y=tr.is_cheater
print(tr.drop(columns=['player_id']).corrwith(y).sort_values().round(3).to_string())
