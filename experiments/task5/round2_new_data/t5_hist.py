from common import *
import warnings; warnings.filterwarnings('ignore')
exec(open('t5_gap2.py').read().split("def gaps")[0])
for c in ['kda_std_m123','skill_rating_m123','skill_trajectory_m123','kda_mean_m123','accuracy_mean_m123','performance_stability_m123','activity_level_m123','reaction_time_mean_m123','kda_std_sd123']:
    x=V[c].dropna(); h,e=np.histogram(x,bins=30); print(c,e[[0,-1]].round(2),h.tolist())
