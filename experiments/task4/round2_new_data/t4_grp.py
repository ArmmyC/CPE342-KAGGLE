import torch, numpy as np, pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
E=torch.load('t4/emb_cnl_f0.pt'); tr=E['tr'].float(); te=E['te'].float(); n=len(tr)
y=pd.read_csv('t4_train.csv').label.values
V,I=[],[]
for i in range(0,n,2048):
    v,ix=(tr[i:i+2048]@tr.T).topk(21,dim=1); V.append(v[:,1:]); I.append(ix[:,1:])
V=torch.cat(V).numpy(); I=torch.cat(I).numpy(); np.savez('t4/trknn.npz',V=V,I=I)
for th in (0.8,0.85,0.9,0.95):
    r=np.repeat(np.arange(n),20); m=V.ravel()>th
    A=coo_matrix((np.ones(m.sum()),(r[m],I.ravel()[m])),shape=(n,n))
    k,lab=connected_components(A,directed=False); sz=np.bincount(lab)
    big=np.where(sz>=5)[0]
    pur=np.average([np.bincount(y[lab==c]).max()/sz[c] for c in big],weights=sz[big])
    print(th,'components',k,'singletons',(sz==1).sum(),'largest',np.sort(sz)[-6:],'n>=5',len(big),'weighted purity',round(pur,4),flush=True)
    np.save(f't4/grp_{th}.npy',lab)
