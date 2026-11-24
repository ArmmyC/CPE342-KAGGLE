import sys
from lib import *; from nb2 import *
tr,te=load(); y=tr.segment.values; fl=folds(y)
bo,_,_=baseline(); blab=np.load(OUT+'base_nested_lab.npy')
feats_all=[c for c in tr.columns if c not in('player_id','segment')]
def cv(name,fs,**kw):
    o=np.zeros((len(y),4))
    for a,b in fl:
        m=NB2(**kw).fit(tr[fs].iloc[a],y[a]); o[b]=m.predict_proba(tr[fs].iloc[b])
    report(name,y,o,fl,blab); return o,m
F1=[c for c in feats_all if c not in DROP0]; F2=[c for c in F1 if c not in NOINFO]
cv('nb all feats bins60 s1.5',feats_all)
cv('nb noRandom',F1)
cv('nb informative only',F2)
for nb,sm in [(30,1),(60,0),(100,2.5),(150,3),(200,4),(300,6)]:
    cv(f'nb F2 bins{nb} s{sm}',F2,nbins=nb,smooth=sm)
o,m=cv('nb F2 EM10',F2,em=10); print('T',m.T.round(4),'pi',m.pi.round(3))
