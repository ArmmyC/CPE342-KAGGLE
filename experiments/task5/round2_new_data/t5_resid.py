from common import *
import warnings; warnings.filterwarnings('ignore')
Z=pd.read_pickle('t5_p4z.pkl'); G=pd.read_pickle('t5_groups.pkl'); un=~G.any(axis=1).values
Zc=Z.drop(columns=['device_count','refund_count','guild_changes','friend_removes'])  # discrete/unstable scale
n5=(Zc.abs()>5).sum(axis=1); n4=(Zc.abs()>4).sum(axis=1); n3=(Zc.abs()>3).sum(axis=1)
print('unassigned rows with >=1/2/3 feats |z|>5:',[(int(((n5>=k)&un).sum())) for k in (1,2,3)])
print('all rows with >=2 feats |z|>5',int((n5>=2).sum()),'in loc|skill',int(((n5>=2)&(G.loc|G.skill).values).sum()))
print('unassigned n4>=2',int(((n4>=2)&un).sum()),'n3>=3',int(((n3>=3)&un).sum()),'n3>=4',int(((n3>=4)&un).sum()))
print(pd.Series(n3[un]).value_counts().sort_index().head(10).to_dict())
r=(n5>=1)&un
print(Zc[r].abs().gt(5).sum().sort_values(ascending=False).head(10).to_dict())
