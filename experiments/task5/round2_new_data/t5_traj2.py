from common import *
import warnings; warnings.filterwarnings('ignore')
_,te=load(5); fl=np.load('out/t5_flag.npy').astype(bool)
T=te[[f'skill_trajectory_{i}' for i in (1,2,3)]]
for k in (1,2,3):
    v=T.iloc[:,k-1][~fl].dropna(); print(k,np.histogram(v,bins=[-30,-20,-16,-14,-13,-12,-10,-5,0,5,10,12,13,14,16,20,30])[0].tolist())
mx=T.abs().max(axis=1)
for th in (12.5,13,14,15,18): print(th,(mx[~fl]>th).sum())
# per-row: persistent? use sd across p1..3 vs mean
mean_abs=T.mean(axis=1)
print(np.histogram(mean_abs[~fl].dropna(),bins=24)[0].tolist(), np.histogram(mean_abs[~fl].dropna(),bins=24)[1][[0,-1]])
# correlation between periods among unflagged
print(T[~fl].corr().round(3))
