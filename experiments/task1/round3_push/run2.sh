cd <project_root>/MLTest/cpe342-kaggle-main/work/push/t1
for n in base nonoise nonoise_proxy nonoise_inbox; do NJ=5 LR=0.06 python -u exp_lgb.py $n 1,2,3; done
echo ALLDONE
