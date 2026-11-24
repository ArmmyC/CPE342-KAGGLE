# usage: bash run3.sh "variant1 variant2 ..." -> each variant: seeds 1,2,3 in parallel, 2 threads each
cd <project_root>/MLTest/cpe342-kaggle-main/work/push/t1
for n in $1; do
  for s in 1 2 3; do NJ=${NJ:-1} LR=${LR:-0.06} python -u exp_lgb.py $n $s & done
  wait
done
echo ALLDONE
