from common import *
tr,te=load(2)
y=tr.segment.astype(int).values
for nm in ['t2_blendnn','t2_lgb_s42','t2_nn','t2_blend']:
    P=np.load(f'push/t2fix/preds/{nm}_oof.npy'); Y=np.eye(4)[y]
    print('==',nm)
    for c in ['loyalty_tier','vip_tier','engagement_level']:
        g=tr[c].astype(str)
        obs=pd.DataFrame(Y).groupby(g.values).mean(); pr=pd.DataFrame(P).groupby(g.values).mean()
        print(c, 'n',g.value_counts().to_dict()); print(pd.concat([obs.round(3),pr.round(3)],axis=1,keys=['obs','pred']).to_string())
    if nm!='t2_blendnn': break
