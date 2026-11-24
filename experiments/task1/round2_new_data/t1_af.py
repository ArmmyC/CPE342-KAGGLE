from common import *
tr,te=load(1)
tr=tr.dropna(subset=['is_cheater']).reset_index(drop=True)
y=tr.is_cheater.astype(int).values
p=np.load('t1_qda_oof.npy')[:,2]
a=tr.account_age_days; f=tr.friend_network_size
ab=pd.cut(a,[0,15,30,60,110,120,180,200,300,500,700,900,1000,1200,1500]); fb=pd.qcut(f,10,duplicates='drop')
d=pd.DataFrame({'ab':ab,'fb':fb,'r':y-p,'v':p*(1-p),'y':y})
g=d.groupby(['ab','fb'],observed=True)
z=(g.r.sum()/np.sqrt(g.v.sum())).unstack().round(1); print(z.to_string())
print(g.y.mean().unstack().round(2).to_string())
# friends/age ratio per class within segments
for k in (0,1):
    s=tr[y==k]; m=(s.account_age_days>300)&(s.account_age_days<800)
    lr=np.log(s.friend_network_size[m]+1)-0.63*np.log(s.account_age_days[m])
    print(k,lr.quantile([0,.01,.05,.25,.5,.75,.95,.99,1]).round(3).values)
    print(k,'friends quantiles in 300-800',s.friend_network_size[m].quantile([0,.01,.05,.25,.5,.75,.95,.99,1]).values)
