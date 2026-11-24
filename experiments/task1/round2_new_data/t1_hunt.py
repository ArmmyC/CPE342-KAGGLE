from common import *
tr,te=load(1)
tr=tr.dropna(subset=['is_cheater'])
F=[c for c in tr.columns if c not in('player_id','is_cheater')]
y=tr.is_cheater.values
print(len(tr),y.mean())
# per-class rank correlations
R0=tr.loc[y==0,F].rank(pct=True).corr(); R1=tr.loc[y==1,F].rank(pct=True).corr()
D=(R1-R0)
n0,n1=(y==0).sum(),(y==1).sum()
se=np.sqrt(1/n0+1/n1)
rows=[]
for i,a in enumerate(F):
    for b in F[i+1:]:
        rows.append((a,b,R0.loc[a,b],R1.loc[a,b],D.loc[a,b]/se))
df=pd.DataFrame(rows,columns=['a','b','r0','r1','z']).sort_values('z',key=abs,ascending=False)
print(df.head(30).round(3).to_string())
print('n |z|>4:',(df.z.abs()>4).sum(),'of',len(df))
