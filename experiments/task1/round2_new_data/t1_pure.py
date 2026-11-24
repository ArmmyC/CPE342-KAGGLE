from common import *
from sklearn.metrics import fbeta_score
tr,te=load(1)
tr=tr.dropna(subset=['is_cheater']).reset_index(drop=True)
y=tr.is_cheater.astype(int).values
p=np.load('../../../MLKaggleChallenge/preds/t1_seeds_oof.npy')
a=tr.account_age_days; f=tr.friend_network_size; r=tr.reports_received; c=tr.crosshair_placement
rules={
 'age110-120 cheat':(a>110)&(a<120),
 'age180-200 legit':(a>180)&(a<200),
 'age>1000 legit':a>1000.0,
 'age<15 legit':a<15,
 'age15-110 & fr49-105 cheat':(a<110)&(f>49)&(f<=105),
 'age120-180 & fr<=27 cheat':(a>120)&(a<180)&(f<=27),
 'age120-180 & fr>105 legit':(a>120)&(a<180)&(f>105),
}
th=0.37
for n,m in rules.items():
    m=m.fillna(False).values
    pred=(p>th)
    print(n.ljust(30),'n',m.sum(),'rate',round(y[m].mean(),4),'wrong now',(pred[m]!=y[m]).sum())
base=fbeta_score(y,(p>th).astype(int),beta=2,average='macro')
q=p.copy()
for n,m in rules.items():
    m=m.fillna(False).values; q[m]=1.0 if 'cheat' in n else 0.0
print('macroF2',base,'->',fbeta_score(y,(q>th).astype(int),beta=2,average='macro'))
