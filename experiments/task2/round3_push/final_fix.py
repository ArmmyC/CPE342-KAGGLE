from lib import *
tr,te=load(); y=tr.segment.values.astype(int); fl=folds(y)
F='../t2fix/preds/'
bo,bt=np.load(F+'t2_blendnn_oof.npy'),np.load(F+'t2_blendnn_test.npy')
no,nt=np.load(OUT+'oof_em100.npy'),np.load(OUT+'test_em100.npy')
def mix(p,q,a):
    z=(1-a)*np.log(np.clip(p,1e-9,1))+a*np.log(np.clip(q,1e-9,1)); z-=z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)
blab=nested_pred(y,bo,fl); print('compliant base nested',round(mf1(y,blab),4))
r=np.random.default_rng(2); I=[r.integers(0,len(y),len(y)) for _ in range(500)]
for a in (0.3,0.4,0.5):
    p=mix(bo,no,a); lab=nested_pred(y,p,fl); d=np.array([mf1(y[i],lab[i])-mf1(y[i],blab[i]) for i in I])
    print(a,'nested',round(mf1(y,lab),4),'diff',round(mf1(y,lab)-mf1(y,blab),4),'sd',round(d.std(),4),'P>0',(d>0).mean())
a=0.4; p=mix(bo,no,a); w=fit_w(y,p); pt=mix(bt,nt,a); lab=(pt*w).argmax(1)
s=pd.read_csv('../../out/sub38_t4new.csv'); print('w',w,'changed vs sub38',(lab!=s.task2.values).sum())
np.save('t2_fix_labels.npy',lab)
