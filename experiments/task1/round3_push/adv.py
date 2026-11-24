import numpy as np, lightgbm as lgb
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from c1 import *
X,y,U,T=load()
print('labeled',X.shape,'unlabeled',U.shape)
print('missing per row lab/unl/test', X.n_missing.mean(), U.n_missing.mean(), T.n_missing.mean())
A=pd.concat([X,U]); a=np.r_[np.zeros(len(X)),np.ones(len(U))]
m=lgb.LGBMClassifier(n_estimators=300,learning_rate=0.05,num_leaves=31,verbose=-1,n_jobs=6)
p=cross_val_predict(m,A,a,cv=StratifiedKFold(5,shuffle=True,random_state=0),method='predict_proba')[:,1]
print('adv AUC labeled vs unlabeled', roc_auc_score(a,p))
A2=A.drop(columns=['n_missing'])
p=cross_val_predict(m,A2.notna().astype(int),a,cv=StratifiedKFold(5,shuffle=True,random_state=0),method='predict_proba')[:,1]
print('adv AUC on missing mask only', roc_auc_score(a,p))
p=cross_val_predict(m,A2.fillna(-999),a,cv=StratifiedKFold(5,shuffle=True,random_state=0),method='predict_proba')[:,1]
print('adv AUC values', roc_auc_score(a,p))
for c in X.columns[:31]:
    print(f'{c:28s} miss lab {X[c].isna().mean():.3f} unl {U[c].isna().mean():.3f} test {T[c].isna().mean():.3f}')
# box fraction
o=box_outside(X,y,T)
print('outside box: lab',o(X).mean(),'unl',o(U).mean(),'test',o(T).mean())
