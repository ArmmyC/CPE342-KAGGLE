from common import *
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import fbeta_score, roc_auc_score
from sklearn.preprocessing import QuantileTransformer
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA, LinearDiscriminantAnalysis as LDA
tr,te=load(1)
tr=tr.dropna(subset=['is_cheater']).reset_index(drop=True)
F=[c for c in tr.columns if c not in('player_id','is_cheater')]
X=tr[F]; y=tr.is_cheater.astype(int).values
def mf2(p):
    return max(fbeta_score(y,(p>t).astype(int),beta=2,average='macro') for t in np.arange(0.25,0.5,0.01))
oq=np.zeros(len(y)); ol=np.zeros(len(y)); ob=np.zeros(len(y)); o2=np.zeros(len(y))
for f,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=0).split(X,y)):
    qt=QuantileTransformer(output_distribution='normal',n_quantiles=2000,subsample=10**6).fit(X.iloc[a])
    Za=np.nan_to_num(qt.transform(X.iloc[a])); Zb=np.nan_to_num(qt.transform(X.iloc[b]))
    q=QDA(reg_param=0.01).fit(Za,y[a]); l=LDA().fit(Za,y[a])
    lo=lambda m,Z: np.log(np.clip(m.predict_proba(Z)[:,1],1e-9,1-1e-9))-np.log(np.clip(m.predict_proba(Z)[:,0],1e-9,1))
    oq[b]=q.predict_proba(Zb)[:,1]; ol[b]=l.predict_proba(Zb)[:,1]
    # in-fold stacking feature: inner OOF for train part
    inner=np.zeros(len(a))
    for a2,b2 in StratifiedKFold(5,shuffle=True,random_state=1).split(Za,y[a]):
        inner[b2]=lo(QDA(reg_param=0.01).fit(Za[a2],y[a][a2]),Za[b2])
    P=dict(n_estimators=3000,learning_rate=0.02,num_leaves=31,subsample=0.8,subsample_freq=1,colsample_bytree=0.7,verbose=-1,n_jobs=16)
    m=lgb.LGBMClassifier(**P).fit(X.iloc[a],y[a],eval_set=[(X.iloc[b],y[b])],callbacks=[lgb.early_stopping(150,verbose=False)])
    ob[b]=m.predict_proba(X.iloc[b])[:,1]
    Xa=X.iloc[a].assign(qda=inner); Xb=X.iloc[b].assign(qda=lo(q,Zb))
    m2=lgb.LGBMClassifier(**P).fit(Xa,y[a],eval_set=[(Xb,y[b])],callbacks=[lgb.early_stopping(150,verbose=False)])
    o2[b]=m2.predict_proba(Xb)[:,1]
    print(f,roc_auc_score(y[b],oq[b]),roc_auc_score(y[b],ob[b]),roc_auc_score(y[b],o2[b]),flush=True)
for n,o in [('qda',oq),('lda',ol),('lgb',ob),('lgb+qda',o2)]: print(n,round(roc_auc_score(y,o),4),round(mf2(o),4))
np.save('t1_qda_oof.npy',np.c_[oq,ol,ob,o2])
