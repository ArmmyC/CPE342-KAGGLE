import sys, lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
from lib import *; from nb2 import DROP0,NOINFO
tr,te=load(); y=tr.segment.values; fl=folds(y)
blab=np.load(OUT+'base_nested_lab.npy')
feats=[c for c in tr.columns if c not in('player_id','segment')]
F2=[c for c in feats if c not in DROP0+NOINFO]
fs=F2 if sys.argv[2]=='F2' else [c for c in feats if c not in DROP0]
X=tr[fs].copy(); Xt=te[fs].copy()
for c in fs:
    if X[c].dtype==object or str(X[c].dtype)=='str':
        cats=sorted(X[c].dropna().unique()); X[c]=pd.Categorical(X[c],categories=cats); Xt[c]=pd.Categorical(Xt[c],categories=cats)
nl=int(sys.argv[1]); lr=float(sys.argv[3]) if len(sys.argv)>3 else 0.05
o=np.zeros((len(y),4)); tp=np.zeros((len(Xt),4)); its=[]
for a,b in fl:
    m=lgb.LGBMClassifier(n_estimators=20000,learning_rate=lr,num_leaves=nl,max_depth=-1 if nl>2 else 1,min_child_samples=200,
        subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=5,max_bin=255,cat_smooth=10,verbose=-1,n_jobs=6,random_state=0)
    m.fit(X.iloc[a],y[a],eval_set=[(X.iloc[b],y[b])],callbacks=[lgb.early_stopping(300,verbose=False)])
    o[b]=m.predict_proba(X.iloc[b]); tp+=m.predict_proba(Xt)/5; its.append(m.best_iteration_)
report(f'lgb leaves{nl} {sys.argv[2]} lr{lr} its{its}',y,o,fl,blab)
np.save(OUT+f'oof_lgb{nl}_{sys.argv[2]}.npy',o); np.save(OUT+f'test_lgb{nl}_{sys.argv[2]}.npy',tp)
