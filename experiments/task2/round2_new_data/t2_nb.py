from common import *
from nb import HistNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, log_loss
tr,te=load(2)
feats=[c for c in tr.columns if c not in('player_id','segment')]
X=tr[feats]; y=tr.segment.astype(int).values
oof=np.zeros((len(X),4))
for a,b in StratifiedKFold(5,shuffle=True,random_state=0).split(X,y):
    m=HistNB().fit(X.iloc[a],y[a]); oof[b]=m.predict_proba(X.iloc[b])
print('NB macroF1',f1_score(y,oof.argmax(1),average='macro'),'ll',log_loss(y,oof))
np.save('oof2_nb.npy',oof)
