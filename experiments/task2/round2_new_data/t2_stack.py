from common import *
from nb import HistNB
import lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, log_loss
tr,te=load(2)
feats=[c for c in tr.columns if c not in('player_id','segment')]
X=tr[feats].copy(); y=tr.segment.astype(int).values
def fe(d):
    d=d.copy()
    d['months']=d.total_spending_thb/d.avg_monthly_spending
    d['spend_per_txn']=d.avg_monthly_spending/d.spending_frequency
    d['tot_per_txn']=d.total_spending_thb/(d.spending_frequency*d.account_age_days/30+1)
    d['play_per_age']=d.total_playtime_hours/(d.account_age_days+1)
    d['sess_x_freq']=d.avg_session_duration*d.play_frequency
    return d
oof=np.zeros((len(X),4)); oofb=np.zeros((len(X),4))
for f,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=0).split(X,y)):
    nbm=HistNB().fit(X.iloc[a],y[a])
    def aug(d):
        d=fe(d); z=nbm.predict_logit(d[feats]); 
        for k in range(4): d[f'nb{k}']=z[:,k]-z.max(1)
        return d
    Xa,Xb=aug(X.iloc[a]),aug(X.iloc[b])
    for c in Xa.columns:
        if Xa[c].dtype==object or str(Xa[c].dtype)=='str': Xa[c]=Xa[c].astype('category'); Xb[c]=pd.Categorical(Xb[c],categories=Xa[c].cat.categories)
    m=lgb.LGBMClassifier(n_estimators=3000,learning_rate=0.02,num_leaves=15,min_child_samples=100,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=5,verbose=-1,n_jobs=16)
    m.fit(Xa,y[a],eval_set=[(Xb,y[b])],callbacks=[lgb.early_stopping(150,verbose=False)])
    oof[b]=m.predict_proba(Xb); print(f,m.best_iteration_,f1_score(y[b],oof[b].argmax(1),average='macro'),log_loss(y[b],oof[b]))
print('stack macroF1',f1_score(y,oof.argmax(1),average='macro'),'ll',log_loss(y,oof))
np.save('oof2_stack.npy',oof)
