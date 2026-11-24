from common import *
import warnings; warnings.filterwarnings('ignore')
_,te=load(5); fl=np.load('out/t5_flag.npy').astype(bool); P=np.load('out/t5_g7_post.npy')
cols=[c for c in te.columns if c!='player_id']
base=sorted(set(c.rsplit('_',1)[0] for c in cols if c[-2]=='_' and c[-1] in '1234'))
other=[c for c in cols if not(c[-2]=='_' and c[-1] in '1234')]
print('non-period cols',other)
core=(P>0.9)&~fl; bg=(P<0.01)&~fl
print('core',core.sum(),'bg',bg.sum())
res=[]
for b in base:
    for i in (1,2,3,4):
        c=f'{b}_{i}'; a=te.loc[core,c]; z=te.loc[bg,c]
        res.append((c,round((a.median()-z.median())/(z.quantile(.75)-z.quantile(.25)+1e-9),2)))
    v=te[[f'{b}_{i}' for i in (1,2,3,4)]].mean(1)
    res.append((b+'_mean',round((v[core].median()-v[bg].median())/(v[bg].quantile(.75)-v[bg].quantile(.25)+1e-9),2)))
for c in other:
    a=te.loc[core,c]; z=te.loc[bg,c]
    if pd.api.types.is_numeric_dtype(a): res.append((c,round((a.median()-z.median())/(z.quantile(.75)-z.quantile(.25)+1e-9),2)))
print(sorted(res,key=lambda t:-abs(t[1]))[:30])
