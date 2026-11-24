from common import *
tr,te=load(1)
tr=tr.dropna(subset=['is_cheater']).reset_index(drop=True)
y=tr.is_cheater.astype(int).values
a=tr.account_age_days.values; m=(a>200)&(a<900)
F=[c for c in tr.columns if c not in('player_id','is_cheater')]
S=tr[m]; ys=y[m]
Q={c:pd.qcut(S[c].rank(method='first'),8,labels=False).values for c in F}
rows=[]
for i,c1 in enumerate(F):
    for c2 in F[i+1:]:
        r=[c1,c2]
        for k in (0,1):
            ok=(ys==k)&~np.isnan(Q[c1])&~np.isnan(Q[c2])
            t=pd.crosstab(Q[c1][ok],Q[c2][ok]).values.astype(float)
            e=t.sum(1,keepdims=True)*t.sum(0,keepdims=True)/t.sum()
            r.append(((t-e)**2/e).sum()/49)  # ~1 under independence
        rows.append(r)
d=pd.DataFrame(rows,columns=['a','b','chi0','chi1']); d['diff']=d.chi1-d.chi0
print(d.sort_values('diff',key=abs,ascending=False).head(25).round(2).to_string())
print(d.sort_values('chi1',ascending=False).head(15).round(2).to_string())
