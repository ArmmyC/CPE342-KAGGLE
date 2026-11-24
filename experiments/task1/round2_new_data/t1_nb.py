from common import *
from nb import HistNB
import lightgbm as lgb, warnings; warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import fbeta_score, roc_auc_score
tr,te=load(1)
lab=tr.dropna(subset=['is_cheater']).reset_index(drop=True)
feats=[c for c in lab.columns if c not in('player_id','is_cheater')]
X=lab[feats]; y=lab.is_cheater.astype(int).values
def fe(d):
    d=d.copy(); d['age_fr']=d.friend_network_size/(d.account_age_days**0.66); d['lps_exp']=d.level_progression_speed*d.account_age_days/d.level
    d['hs_acc']=d.headshot_percentage/d.accuracy_score; return d
def run(use_nb,use_fe,seed=0):
    oof=np.zeros(len(X))
    for a,b in StratifiedKFold(5,shuffle=True,random_state=seed).split(X,y):
        Xa,Xb=X.iloc[a].copy(),X.iloc[b].copy()
        if use_nb:
            nbm=HistNB(nbins=40).fit(Xa,y[a],2)
            for d in (Xa,Xb):
                z=nbm.predict_logit(d[feats]); d['nb']=z[:,1]-z[:,0]
        if use_fe: Xa,Xb=fe(Xa),fe(Xb)
        m=lgb.LGBMClassifier(n_estimators=3000,learning_rate=0.02,num_leaves=15,min_child_samples=60,subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=3,verbose=-1,n_jobs=16,random_state=seed)
        m.fit(Xa,y[a],eval_set=[(Xb,y[b])],callbacks=[lgb.early_stopping(150,verbose=False)])
        oof[b]=m.predict_proba(Xb)[:,1]
    ts=np.linspace(0.25,0.55,31); s=[fbeta_score(y,(oof>t).astype(int),beta=2,average='macro') for t in ts]
    print('nb',use_nb,'fe',use_fe,'auc',round(roc_auc_score(y,oof),4),'bestF2',round(max(s),4),'thr',round(ts[int(np.argmax(s))],3),flush=True); return oof
run(False,False); run(True,False); run(False,True); run(True,True)
