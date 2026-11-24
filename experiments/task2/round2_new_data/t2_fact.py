from common import *
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
tr,te=load(2)
y=tr.segment.astype(int).values
F=[c for c in tr.columns if c not in('player_id','segment')]
X=tr[F].copy()
for c in X.columns:
    if not pd.api.types.is_numeric_dtype(X[c]): X[c]=X[c].astype('category')
A=(y>=2).astype(int); B=(y%2).astype(int)
print('P(A)',A.mean(),'P(B)',B.mean(),'joint vs product', pd.crosstab(A,B,normalize=True).values.round(4).tolist(), np.outer([1-A.mean(),A.mean()],[1-B.mean(),B.mean()]).round(4).tolist())
P=dict(n_estimators=4000,learning_rate=0.03,num_leaves=31,subsample=0.8,subsample_freq=1,colsample_bytree=0.6,min_child_samples=40,verbose=-1,n_jobs=16)
o4=np.zeros((len(y),4)); oa=np.zeros(len(y)); ob=np.zeros(len(y))
for f,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(X,y)):
    es=[lgb.early_stopping(200,verbose=False)]
    m=lgb.LGBMClassifier(**P).fit(X.iloc[a],y[a],eval_set=[(X.iloc[b],y[b])],callbacks=es); o4[b]=m.predict_proba(X.iloc[b])
    m=lgb.LGBMClassifier(**P).fit(X.iloc[a],A[a],eval_set=[(X.iloc[b],A[b])],callbacks=es); oa[b]=m.predict_proba(X.iloc[b])[:,1]
    m=lgb.LGBMClassifier(**P).fit(X.iloc[a],B[a],eval_set=[(X.iloc[b],B[b])],callbacks=es); ob[b]=m.predict_proba(X.iloc[b])[:,1]
    print(f,flush=True)
pf=np.c_[(1-oa)*(1-ob),(1-oa)*ob,oa*(1-ob),oa*ob]
from sklearn.metrics import roc_auc_score
print('A auc',roc_auc_score(A,oa),'from4',roc_auc_score(A,o4[:,2]+o4[:,3]),' B auc',roc_auc_score(B,ob),'from4',roc_auc_score(B,o4[:,1]+o4[:,3]))
def best(p):
    # simple class-weight search for macro F1
    bs=f1_score(y,p.argmax(1),average='macro'); bw=None
    for w in np.array(np.meshgrid(*[np.arange(0.8,1.25,0.05)]*3)).T.reshape(-1,3):
        s=f1_score(y,(p*np.r_[1,w]).argmax(1),average='macro')
        if s>bs: bs,bw=s,w
    return bs
print('4class',f1_score(y,o4.argmax(1),average='macro'),'factored',f1_score(y,pf.argmax(1),average='macro'),'avg',f1_score(y,(o4+pf).argmax(1),average='macro'))
np.save('t2_fact_oof.npy',np.c_[o4,pf])
# residual dependence: within true A, is B predictable differently? check corr of residuals
print('logloss 4',-np.log(o4[np.arange(len(y)),y]).mean(),'fact',-np.log(pf[np.arange(len(y)),y]).mean())
