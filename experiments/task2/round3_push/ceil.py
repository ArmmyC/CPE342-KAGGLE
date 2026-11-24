from lib import *
tr,te=load(); y=tr.segment.values
o=np.load(OUT+'oof_em100.npy'); w=fit_w(y,o); lab=(o*w).argmax(1)
r=np.random.default_rng(0); s=[]
for _ in range(200):
    u=r.random(len(o))[:,None]; ys=(u>np.cumsum(o,1)).sum(1); s.append(mf1(ys,lab))
print('model-implied expected macroF1 of its own decision: %.4f sd %.4f ; observed %.4f'%(np.mean(s),np.std(s),mf1(y,lab)))
