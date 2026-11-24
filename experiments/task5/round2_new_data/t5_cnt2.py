from common import *
import warnings; warnings.filterwarnings('ignore')
exec(open('t5_spikes.py').read().split("out=[]")[0])
kn=known
cond={'dev4==4':te.device_count_4==4,'guild1==3':te.guild_changes_1==3,'guild4==4':te.guild_changes_4==4,
      'guild2==-1':te.guild_changes_2==-1,'refund2==-1':te.refund_count_2==-1,'refund4==2':te.refund_count_4==2}
for k,v in cond.items():
    print(k,'n',v.sum(),'known',(v&kn).sum(),'loc',(v&loc).sum(),'skill',(v&skill).sum(),'bot',(v&bot).sum(),'B',(v&B).sum())
g1=te.guild_changes_1==3
print('guild1==3 rows: other features mean vs rest')
print(pd.DataFrame({'g3':te[g1].mean(numeric_only=True),'rest':te[~g1&~kn].mean(numeric_only=True)}).assign(d=lambda x:(x.g3-x.rest)/te[~kn].std(numeric_only=True)).sort_values('d',key=abs,ascending=False).head(12).round(3))
print(pd.crosstab(te.guild_changes_1,te.guild_changes_2))
print(pd.crosstab(te.guild_changes_1,te.guild_changes_4))
print(pd.crosstab(te.device_count_1,te.device_count_4))
