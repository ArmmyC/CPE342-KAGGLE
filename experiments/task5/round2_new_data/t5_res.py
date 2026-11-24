from common import *
import warnings; warnings.filterwarnings('ignore')
_,te=load(5)
R=pd.read_parquet('t5_res.parquet')
for t in range(1,5):
    for u in range(1,5):
        print(t,u,'traj_t vs rating_u',round(te[f'skill_trajectory_{t}'].corr(te[f'skill_rating_{u}']),3), end=' | ')
    print()
dr=pd.DataFrame({f'd{t}':te[f'skill_rating_{t+1}']-te[f'skill_rating_{t}'] for t in range(1,4)})
for t in range(1,4): print('traj',t,'vs diff',[round(te[f'skill_trajectory_{k}'].corr(dr[f'd{t}']),3) for k in range(1,5)])
# robust z of residuals
Z=(R-R.median())/ (1.4826*(R-R.median()).abs().median())
kurt=Z.apply(lambda s: s.kurt()).sort_values(ascending=False)
print(kurt.head(25).round(1).to_dict())
big=(Z.abs()>5)
n=big.sum(axis=1); print(n.value_counts().sort_index().head(15).to_dict())
