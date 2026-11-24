from common import *
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import fbeta_score, roc_auc_score
tr,te=load(1)
print('label NaN',tr.is_cheater.isna().mean())
tr['nmiss']=tr.drop(columns=['player_id','is_cheater']).isna().sum(axis=1)
print(tr.groupby('nmiss').is_cheater.agg(['mean','count']).T.round(3).to_string())
lab=tr.dropna(subset=['is_cheater']).reset_index(drop=True)
feats=[c for c in lab.columns if c not in('player_id','is_cheater')]
X=lab[feats]; y=lab.is_cheater.astype(int).values
oof=np.zeros(len(X))
for f,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=0).split(X,y)):
    m=lgb.LGBMClassifier(n_estimators=2000,learning_rate=0.03,num_leaves=31,subsample=0.8,subsample_freq=1,colsample_bytree=0.7,verbose=-1,n_jobs=16)
    m.fit(X.iloc[a],y[a],eval_set=[(X.iloc[b],y[b])],callbacks=[lgb.early_stopping(100,verbose=False)])
    oof[b]=m.predict_proba(X.iloc[b])[:,1]; print(f,m.best_iteration_,roc_auc_score(y[b],oof[b]))
def mf(y,p,t,beta): return fbeta_score(y,(p>t).astype(int),beta=beta,average='macro')
ts=np.linspace(0.05,0.95,91)
for beta in (2,):
    s=[mf(y,oof,t,beta) for t in ts]; print('beta',beta,'best',max(s),ts[int(np.argmax(s))],'@0.5',mf(y,oof,0.5,beta))
imp=pd.Series(m.feature_importances_,feats).sort_values(ascending=False); print(imp.head(15).to_dict())
