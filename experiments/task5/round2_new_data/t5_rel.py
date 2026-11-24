from common import *
import lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
_,te=load(5)
X=te.drop(columns=['player_id'])
cols=list(X.columns)
res={}
R=[]
for c in cols:
    y=X[c]; m=y.notna().values
    others=X.drop(columns=[c])
    oof=np.full(len(X),np.nan)
    for a,b in KFold(3,shuffle=True,random_state=0).split(X):
        a=a[m[a]]
        mdl=lgb.LGBMRegressor(n_estimators=150,learning_rate=0.1,num_leaves=15,subsample=0.8,subsample_freq=1,colsample_bytree=0.7,verbose=-1,n_jobs=16)
        mdl.fit(others.iloc[a],y.iloc[a]); oof[b]=mdl.predict(others.iloc[b])
    r2=1-np.nanmean((y.values-oof)[m]**2)/np.nanvar(y.values[m])
    R.append((c,r2)); res[c]=y.values-oof
R=pd.DataFrame(R,columns=['f','r2']).sort_values('r2',ascending=False)
print(R.head(40).to_string()); R.to_csv('t5_r2.csv',index=False)
pd.DataFrame(res).to_parquet('t5_res.parquet')
