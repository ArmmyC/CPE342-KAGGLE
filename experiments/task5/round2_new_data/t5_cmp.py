from common import *
G=pd.read_pickle('t5_groups.pkl')
S='<project_root>/MLKaggleChallenge/subs/'
for f in ['sub25_newdata','sub26_t5tight','sub28_t4full','sub29_t5r13']:
    s=pd.read_csv(S+f+'.csv'); t5=s.task5.values.astype(int)==1
    print(f,'flagged',t5.sum(),{g:int((t5&G[g].values).sum()) for g in G},'of',G.sum().to_dict(),'union covered',int((t5&G.any(axis=1).values).sum()))
