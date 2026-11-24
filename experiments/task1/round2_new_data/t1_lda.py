from common import *
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
tr,te=load(1)
tr=tr.dropna(subset=['is_cheater']).reset_index(drop=True)
y=tr.is_cheater.astype(int).values
a=tr.account_age_days.values; m=(a>200)&(a<900)
F=[c for c in tr.columns if c not in('player_id','is_cheater')]
SK=['kill_death_ratio','headshot_percentage','win_rate','accuracy_score','kill_consistency','reaction_time_ms','aiming_smoothness','spray_control_score','crosshair_placement','game_sense_score']
NZ=['input_consistency_score','movement_pattern_score','first_blood_rate','weapon_switch_speed','device_changes_count','avg_session_length_min','sessions_per_day','night_play_ratio','communication_rate','team_play_score','buy_decision_score','map_knowledge','clutch_success_rate','survival_time_avg','damage_per_round','utility_usage_rate']
S=tr[m]; ys=y[m]
cv=StratifiedKFold(5,shuffle=True,random_state=0)
for name,cols in [('skill',SK),('skill+noise',SK+NZ)]:
    pipe=make_pipeline(QuantileTransformer(output_distribution='normal'),SimpleImputer(strategy='median'),LDA())
    p=cross_val_predict(pipe,S[cols],ys,cv=cv,method='predict_proba')[:,1]
    print(name,roc_auc_score(ys,p))
# complete cases only (no NaN) to avoid imputation dilution
cc=S[SK+NZ].notna().all(1).values; print('complete',cc.sum())
for name,cols in [('skill',SK),('skill+noise',SK+NZ)]:
    pipe=make_pipeline(QuantileTransformer(output_distribution='normal'),LDA())
    p=cross_val_predict(pipe,S.loc[cc,cols],ys[cc],cv=cv,method='predict_proba')[:,1]
    print('cc',name,roc_auc_score(ys[cc],p))
