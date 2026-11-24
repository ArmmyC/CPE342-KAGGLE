import numpy as np, pandas as pd
from sklearn.metrics import fbeta_score
from sklearn.model_selection import StratifiedKFold
D='<project_root>/MLTest/cpe342-kaggle-main'
P='<project_root>/MLKaggleChallenge/preds'
OUT=D+'/work/push/t1'
NOISE=['device_changes_count','input_consistency_score','avg_session_length_min','sessions_per_day','night_play_ratio','weapon_switch_speed','movement_pattern_score','communication_rate','team_play_score','buy_decision_score','map_knowledge','clutch_success_rate','first_blood_rate','survival_time_avg','damage_per_round','utility_usage_rate']
BOX=['account_age_days','level','level_progression_speed','friend_network_size','reports_received','crosshair_placement']
def fe(d):
    d=d.copy()
    d['n_missing']=d.isna().sum(1)
    d['hs_x_acc']=d['headshot_percentage']*d['accuracy_score']
    d['kd_per_level']=d['kill_death_ratio']/(d['level']+1)
    d['reports_per_day']=d['reports_received']/(d['account_age_days']+1)
    d['react_x_acc']=d['reaction_time_ms']/(d['accuracy_score']+1e-3)
    return d
def load():
    tr=pd.read_csv(D+'/01_binary_classification/train.csv'); te=pd.read_csv(D+'/01_binary_classification/test.csv')
    feats=[c for c in tr.columns if c not in ('player_id','is_cheater')]
    lab=tr[tr.is_cheater.notna()].reset_index(drop=True); unl=tr[tr.is_cheater.isna()].reset_index(drop=True)
    y=lab.is_cheater.astype(int).values
    # n_missing computed on raw feats only (row-wise, as gbdt.py)
    X=fe(lab[feats]); U=fe(unl[feats]); T=fe(te[feats])
    return X,y,U,T
def folds(y,seed=42): return list(StratifiedKFold(5,shuffle=True,random_state=seed).split(np.zeros(len(y)),y))
TH=np.round(np.arange(0.20,0.60,0.01),2)
def mf2(y,p,t): return fbeta_score(y,p>t,beta=2,average='macro')
def curve(y,p): return np.array([mf2(y,p,t) for t in TH])
def best(y,p):
    c=curve(y,p); i=c.argmax(); return c[i],TH[i]
def box_outside(X,y,T):
    ch=X[y==1]; lo,hi=ch[BOX].min(),ch[BOX].max()
    o=lambda d: np.any([((d[c]<lo[c])|(d[c]>hi[c])).fillna(False).values for c in BOX],0)
    return o
