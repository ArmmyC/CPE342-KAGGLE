from common import *
import warnings; warnings.filterwarnings('ignore')
_,te=load(5); s=pd.read_csv('out/sub43_t4newblend.csv'); fl=s.task5.values.astype(bool); u=~fl
cols=[c for c in te.columns if c!='player_id']
base=sorted(set(c.rsplit('_',1)[0] for c in cols if c[-2]=='_' and c[-1] in '1234'))
other=[c for c in cols if not(c[-2]=='_' and c[-1] in '1234')]
D={}
for b in base:
    for nm,ps in [('m123',(1,2,3)),('p4',(4,))]:
        D[f'{b}_{nm}']=te[[f'{b}_{i}' for i in ps]].mean(1)
    D[f'{b}_d4']=te[f'{b}_4']-te[[f'{b}_{i}' for i in (1,2,3)]].mean(1)
for c in other: D[c]=te[c]
D=pd.DataFrame(D)[u].reset_index(drop=True)
R=D.rank(pct=True)
q=0.03; n=len(D)
lo=(R<q).values; hi=(R>1-q).values; ok=D.notna().values
F=list(D.columns); out=[]
for i in range(len(F)):
    for j in range(i+1,len(F)):
        m=ok[:,i]&ok[:,j]; nn=m.sum()
        for ai,A in (('lo',lo),('hi',hi)):
            for bj,B in (('lo',lo),('hi',hi)):
                c=(A[:,i]&B[:,j]&m).sum(); e=nn*(A[m,i].mean())*(B[m,j].mean())
                out.append((F[i],ai,F[j],bj,c,round(e,1),(c-e)/np.sqrt(e+1)))
d=pd.DataFrame(out,columns=['a','sa','b','sb','n','exp','z']).sort_values('z',ascending=False)
# drop trivially related pairs (same base)
d=d[d.a.str.rsplit('_',n=1).str[0]!=d.b.str.rsplit('_',n=1).str[0]]
print(d.head(25).to_string())
