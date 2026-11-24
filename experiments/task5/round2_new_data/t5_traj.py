from common import *
import warnings; warnings.filterwarnings('ignore')
exec(open('t5_B.py').read().split("feat={}")[0])
Nm=N.values
d3=te.skill_rating_4-te.skill_rating_3
t3=te.skill_trajectory_3
print('N: corr traj3~d3',t3[Nm].corr(d3[Nm]).round(3),' A:',t3[A.values].corr(d3[A.values]).round(3))
print('N: corr traj4~rating4',te.skill_trajectory_4[Nm].corr(te.skill_rating_4[Nm]).round(3))
# linear fit traj3 on d3 within N and residual dist
m=Nm&d3.notna().values&t3.notna().values
b=np.polyfit(d3[m],t3[m],1); print('slope',b)
res=t3[m]-np.polyval(b,d3[m]); print(res.describe().round(3).to_dict())
# is the correlation concentrated in a subgroup? bin by |d3|
q=pd.qcut(d3[m].abs(),10)
print(pd.DataFrame({'t3':t3[m],'d3':d3[m],'q':q}).groupby('q').apply(lambda g: g.t3.corr(g.d3)).round(3).to_dict())
print('skill_rating level vs sd of traj3', t3[m].std(), 'sd d3',d3[m].std())
print(np.percentile(t3[m],[0.1,1,50,99,99.9]), np.percentile(d3[m],[0.1,1,50,99,99.9]))
for k in range(1,5):
    tk=te[f'skill_trajectory_{k}']; print(k,'traj mean/std',tk[Nm].mean().round(3),tk[Nm].std().round(3),'corr w rating_k',tk[Nm].corr(te[f'skill_rating_{k}'][Nm]).round(3))
