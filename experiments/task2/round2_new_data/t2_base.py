from common import *
import lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, log_loss
tr,te=load(2)
print('label NaN',tr.segment.isna().mean())
lab=tr.dropna(subset=['segment']).reset_index(drop=True)
cats=[c for c in lab.columns if lab[c].dtype==object or str(lab[c].dtype)=='str']
for c in cats: lab[c]=lab[c].astype('category')
feats=[c for c in lab.columns if c not in('player_id','segment')]
print(lab[feats].isna().mean().round(2)[lambda s:s>0].to_dict())
X=lab[feats]; y=lab.segment.astype(int).values
oof=np.zeros((len(X),4))
for f,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=0).split(X,y)):
    m=lgb.LGBMClassifier(n_estimators=3000,learning_rate=0.03,num_leaves=31,subsample=0.8,subsample_freq=1,colsample_bytree=0.6,verbose=-1,n_jobs=16)
    m.fit(X.iloc[a],y[a],eval_set=[(X.iloc[b],y[b])],callbacks=[lgb.early_stopping(100,verbose=False)])
    oof[b]=m.predict_proba(X.iloc[b]); print(f,m.best_iteration_,f1_score(y[b],oof[b].argmax(1),average='macro'))
print('macroF1',f1_score(y,oof.argmax(1),average='macro'),'ll',log_loss(y,oof))
imp=pd.Series(m.feature_importances_,feats).sort_values(ascending=False); print(imp.head(20).to_dict()); print(imp.tail(10).to_dict())
from sklearn.metrics import confusion_matrix; print(confusion_matrix(y,oof.argmax(1)))
