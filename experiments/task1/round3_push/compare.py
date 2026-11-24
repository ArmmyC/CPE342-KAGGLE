"""Paired comparison of variants vs 'base' on identical folds. usage: python compare.py prefix var1 var2 ..."""
import sys, glob, numpy as np
from sklearn.metrics import roc_auc_score
from c1 import *
X,y,U,T=load(); outb=box_outside(X,y,T)(X)
pre=sys.argv[1]; names=sys.argv[2:]
def get(n):
    fs=sorted(glob.glob(f'{OUT}/oof_{pre}_{n}_s*.npy')); return {f.split('_s')[-1][:-4]:np.load(f) for f in fs}
B=get('base')
for n in names:
    V=get(n); S=[s for s in V if s in B]
    if not S: print(n,'no seeds'); continue
    d_auc=[roc_auc_score(y,V[s])-roc_auc_score(y,B[s]) for s in S]
    bb=[best(y,np.where(outb,0,B[s]))[0] for s in S]; vb=[best(y,np.where(outb,0,V[s]))[0] for s in S]
    d=np.array(vb)-np.array(bb)
    avV=np.where(outb,0,np.mean([V[s] for s in S],0)); avB=np.where(outb,0,np.mean([B[s] for s in S],0))
    (bv,tv),(bB,tB)=best(y,avV),best(y,avB)
    print(f'{n:16s} seeds {len(S)} | per-seed macroF2 {np.mean(vb):.5f} vs base {np.mean(bb):.5f}: diff {d.mean():+.5f} (sd {d.std(ddof=1) if len(d)>1 else 0:.5f}, se {d.std(ddof=1)/np.sqrt(len(d)) if len(d)>1 else 0:.5f}) | dAUC {np.mean(d_auc):+.5f} | seed-avg {bv:.5f}@{tv} vs {bB:.5f}@{tB} ({bv-bB:+.5f})')
