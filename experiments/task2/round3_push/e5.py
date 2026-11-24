import sys
from lib import *; from nb2 import *
tr,te=load(); y=tr.segment.values
blab=np.load(OUT+'base_nested_lab.npy')
feats=[c for c in tr.columns if c not in('player_id','segment')]
F2=[c for c in feats if c not in DROP0+NOINFO]
nb_,sm_,tag=int(sys.argv[1]),float(sys.argv[2]),sys.argv[3]; seed=int(sys.argv[4]) if len(sys.argv)>4 else 42
fl=folds(y,seed)
o=np.zeros((len(y),4)); tp=np.zeros((len(te),4))
for a,b in fl:
    m=NB2(nbins=nb_,smooth=sm_,em=8).fit(tr[F2].iloc[a],y[a]); o[b]=m.predict_proba(tr[F2].iloc[b]); tp+=np.log(m.predict_proba(te[F2]))/5
report(f'EM-NB bins{nb_} s{sm_} foldseed{seed}',y,o,fl,blab if seed==42 else None)
np.save(OUT+f'oof_{tag}.npy',o); tp=np.exp(tp); tp/=tp.sum(1,keepdims=True); np.save(OUT+f'test_{tag}.npy',tp)
