from common import *
from nb import HistNB
import lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, log_loss
tr,te=load(2)
feats=[c for c in tr.columns if c not in('player_id','segment')]
X=tr[feats].copy(); Xt=te[feats].copy(); y=tr.segment.astype(int).values
def fe(d):
    d=d.copy()
    d['months']=d.total_spending_thb/d.avg_monthly_spending
    d['spend_per_txn']=d.avg_monthly_spending/d.spending_frequency
    d['play_per_age']=d.total_playtime_hours/(d.account_age_days+1)
    return d
CATS={c:sorted(X[c].dropna().unique()) for c in feats if X[c].dtype==object or str(X[c].dtype)=='str'}
def catify(*ds):
    for d in ds:
        for c,cats in CATS.items(): d[c]=pd.Categorical(d[c],categories=cats)
    return ds
oof=np.zeros((len(X),4)); tp=np.zeros((len(Xt),4))
seeds=[0,1,2]
for sd in seeds:
  for f,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=sd).split(X,y)):
    nbm=HistNB().fit(X.iloc[a],y[a])
    def aug(d):
        z=nbm.predict_logit(d[feats]); d=fe(d)
        for k in range(4): d[f'nb{k}']=z[:,k]-z.max(1)
        return d
    Xa,Xb,Xc=aug(X.iloc[a]),aug(X.iloc[b]),aug(Xt)
    Xa,Xb,Xc=catify(Xa,Xb,Xc)
    m=lgb.LGBMClassifier(n_estimators=3000,learning_rate=0.02,num_leaves=15,min_child_samples=100,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=5,verbose=-1,n_jobs=16,random_state=sd)
    m.fit(Xa,y[a],eval_set=[(Xb,y[b])],callbacks=[lgb.early_stopping(150,verbose=False)])
    oof[b]+=m.predict_proba(Xb)/len(seeds); tp+=m.predict_proba(Xc[Xa.columns])/(5*len(seeds))
  print('seed',sd,'done',flush=True)
print('stack macroF1',f1_score(y,oof.argmax(1),average='macro'),'ll',log_loss(y,oof))
# class weight tuning (coordinate search)
w=np.ones(4)
best=f1_score(y,(oof*w).argmax(1),average='macro')
for it in range(3):
    for k in range(4):
        for v in np.linspace(0.7,1.5,33):
            w2=w.copy(); w2[k]=v; s=f1_score(y,(oof*w2).argmax(1),average='macro')
            if s>best: best,w=s,w2
print('weighted macroF1',best,w)
np.save('out/t2_oof.npy',oof); np.save('out/t2_test.npy',tp); np.save('out/t2_w.npy',w)
