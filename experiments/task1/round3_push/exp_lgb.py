"""LightGBM variants on identical folds (StratifiedKFold(5, shuffle, rs=42)). usage: python exp_lgb.py NAME seed[,seed..]"""
import sys, time, numpy as np, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from c1 import *
NAME=sys.argv[1]; SEEDS=[int(s) for s in sys.argv[2].split(',')]
X,y,U,T=load()
BASE=dict(objective='binary',learning_rate=float(__import__('os').environ.get('LR','0.03')),num_leaves=63,min_child_samples=40,subsample=0.8,subsample_freq=1,
          colsample_bytree=0.6,reg_lambda=1.0,verbose=-1,n_jobs=int(__import__("os").environ.get("NJ","4")))
cols=list(X.columns); prm=dict(BASE); unl_mode=None; inbox=False
if NAME=='base': pass
elif NAME=='nonoise': cols=[c for c in cols if c not in NOISE]
elif NAME=='nonoise_raw': cols=[c for c in X.columns[:31] if c not in NOISE]
elif NAME=='raw': cols=list(X.columns[:31])
elif NAME=='nonoise_slow': cols=[c for c in cols if c not in NOISE]; prm.update(learning_rate=0.01)
elif NAME=='nonoise_reg': cols=[c for c in cols if c not in NOISE]; prm.update(num_leaves=31,min_child_samples=100,reg_lambda=5,colsample_bytree=0.8)
elif NAME=='nonoise_deep': cols=[c for c in cols if c not in NOISE]; prm.update(num_leaves=127,min_child_samples=20,colsample_bytree=0.8)
elif NAME=='nonoise_mono':
    cols=[c for c in cols if c not in NOISE]
    neg=['kill_death_ratio','headshot_percentage','win_rate','accuracy_score','kill_consistency','aiming_smoothness','spray_control_score']
    pos=['reaction_time_ms']
    prm.update(monotone_constraints=[-1 if c in neg else (1 if c in pos else 0) for c in cols],monotone_constraints_method='advanced')
elif NAME=='nonoise_proxy':
    for d in (X,U,T): d['age_proxy']=d.level/d.level_progression_speed; d['lvl_proxy']=d.level_progression_speed*d.account_age_days
    cols=[c for c in X.columns if c not in NOISE]
elif NAME=='nonoise_inbox': cols=[c for c in cols if c not in NOISE]; inbox=True
elif NAME.startswith('pseudo'):   # pseudo_hard / pseudo_soft (unlabeled TRAIN rows only, teacher trained in-fold)
    unl_mode=NAME.split('_')[1]
    if NAME.endswith('_nn'): cols=[c for c in cols if c not in NOISE]
else: raise SystemExit('unknown')
F=folds(y,42)
for seed in SEEDS:
    t0=time.time(); p=dict(prm,seed=seed); oof=np.zeros(len(y)); tp=np.zeros(len(T)); its=[]
    for k,(tr,va) in enumerate(F):
        Xtr,ytr,w=X.iloc[tr][cols],y[tr].astype(float),np.ones(len(tr))
        if inbox:
            ch=X.iloc[tr][y[tr]==1]; lo,hi=ch[BOX].min(),ch[BOX].max()
            outf=lambda d: np.any([((d[c]<lo[c])|(d[c]>hi[c])).fillna(False).values for c in BOX],0)
            keep=~outf(X.iloc[tr]); Xtr,ytr,w=Xtr[keep],ytr[keep],w[keep]
        if unl_mode:
            m0=lgb.train(p,lgb.Dataset(Xtr,y[tr]),10000,valid_sets=[lgb.Dataset(X.iloc[va][cols],y[va])],callbacks=[lgb.early_stopping(200,verbose=False)])
            pu=m0.predict(U[cols])
            if unl_mode=='hard':
                keep=(pu>0.9)|(pu<0.1); Xtr=pd.concat([Xtr,U[cols][keep]]); ytr=np.r_[ytr,(pu[keep]>0.5).astype(float)]; w=np.r_[w,np.ones(keep.sum())]
            else:  # soft: duplicate each unlabeled row as y=1 (weight p) and y=0 (weight 1-p)
                Xtr=pd.concat([Xtr,U[cols],U[cols]]); ytr=np.r_[ytr,np.ones(len(U)),np.zeros(len(U))]; w=np.r_[w,pu,1-pu]
        m=lgb.train(p,lgb.Dataset(Xtr,ytr,weight=w),10000,valid_sets=[lgb.Dataset(X.iloc[va][cols],y[va])],callbacks=[lgb.early_stopping(200,verbose=False)])
        pv,pt=m.predict(X.iloc[va][cols]),m.predict(T[cols])
        if inbox: pv=np.where(outf(X.iloc[va]),0,pv); pt=np.where(outf(T),0,pt)
        oof[va]=pv; tp+=pt/5; its.append(m.best_iteration); print(f"  {NAME} s{seed} fold {k} it {m.best_iteration} {time.time()-t0:.0f}s",flush=True)
    TAG='lgb' if prm['learning_rate']==0.03 else f"lgb{prm['learning_rate']}"
    np.save(f'{OUT}/oof_{TAG}_{NAME}_s{seed}.npy',oof); np.save(f'{OUT}/test_{TAG}_{NAME}_s{seed}.npy',tp)
    b,t=best(y,oof)
    print(f'{NAME} seed {seed}: AUC {roc_auc_score(y,oof):.5f} macroF2 best {b:.5f}@{t} @0.37 {mf2(y,oof,0.37):.5f} iters {its} {time.time()-t0:.0f}s',flush=True)
