from common import *
import warnings; warnings.filterwarnings('ignore')
exec(open('t5_spikes.py').read().split("out=[]")[0])
cm=te[[f'communication_frequency_{i}' for i in range(1,5)]].mean(axis=1)
print(np.histogram(cm.dropna(),bins=30))
low=cm<0.35
print('low comm rows',low.sum(),'known overlap',(low&known).sum(),'guild1==3',(low&(te.guild_changes_1==3)).sum())
g=low&~known
print(te.loc[g,[c for c in te.columns if c.startswith(('friend_adds','guild_changes','friend_removes'))]].describe().T[['mean','std','min','max']].round(2).head(12))
fa=te[[f'friend_adds_{i}' for i in range(1,5)]].mean(axis=1)
print('friend_adds mean hist',np.histogram(fa.dropna(),bins=30))
