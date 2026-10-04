"""Train-only correction fitting, committed validation predictions and scoring."""
from pathlib import Path
import numpy as np
from . import io,features,correction as fc,data as dataset
from .training import ensemble,fold_roles


def numeric_head(model):
    return {k:np.asarray(v) if k in ('mean','scale','theta') else v for k,v in model.items()}


def threshold(y,p,records,target=.95):
    positive=y==1;w=features.weights(records)[positive];scores=p[positive]
    if not positive.any():raise ValueError('No development positives')
    order=np.argsort(-scores,kind='stable');j=int(np.searchsorted(np.cumsum(w[order]),target*w.sum(),side='left'))
    return float(scores[order[min(j,len(order)-1)]])


def operating(y,p,records,t):
    w=features.weights(records);pos=y==1;neg=~pos;decision=p>=t
    return {'threshold':t,'sensitivity':float(w[pos]@decision[pos]/w[pos].sum()),'specificity':float(w[neg]@(~decision[neg])/w[neg].sum()),
            'false_negative_events':int((pos&~decision).sum()),'false_positive_events':int((neg&decision).sum())}


def corrections(data,y,config,out,family):
    folder=Path(out)/'corrections'/family
    if (folder/'COMPLETE.json').exists():io.verify(folder);return
    folder.mkdir(parents=True,exist_ok=True)
    anchor,calibration,components=ensemble(data,out,family,config['seeds'])
    local={**data,'anchor':anchor};train=np.flatnonzero(data['split']=='train');val=np.flatnonzero(data['split']=='val')
    arms=fc.ARMS if family=='QMF' else fc.ARMS[:4]
    predictions={a:anchor.copy() if a=='qmf_anchor' else np.full(len(anchor),np.nan) for a in arms};heads={a:{} for a in arms[1:]};validation={a:[] for a in arms[:4]}
    for fold in range(5):
        _,cal,test=fold_roles(data['fold'],fold)
        for arm in arms[1:]:
            q=calibration[fold];x=fc.matrix(local,arm,cal,q)
            model=fc.fit_head(x,y[cal],data['record'][cal],q,data['eligible'][cal],arm!='graft_unbounded')
            model.update(calibration_fold=(fold+1)%5,predictor_names=fc.feature_specs()[arm]['predictor_names'])
            heads[arm][str(fold)]=model
            q=anchor[test];predictions[arm][test]=fc.predict(model,fc.matrix(local,arm,test,q),q,data['eligible'][test])
            if family=='QMF' and arm in validation:
                q=components[fold];validation[arm].append(fc.predict(model,fc.matrix(local,arm,val,q),q,data['eligible'][val]))
        if family=='QMF':validation['qmf_anchor'].append(components[fold])
    io.write(folder/'HEADS.json',heads)
    train_predictions={a:p[train] for a,p in predictions.items()}
    report={'scope':'Development evaluation with fixed cohort and record-group folds','metrics':{a:fc.metrics(y[train],p,data['record'][train],anchor[train]) for a,p in train_predictions.items()},
        'paired_record_comparisons':fc.comparisons(y[train],train_predictions,data['record'][train],config['bootstrap_repetitions'],20260927 if family=='QMF' else 20261002),
        'folds':{str(f):{a:features.logloss(y[data['fold']==f],p[data['fold'][train]==f],data['record'][data['fold']==f]) for a,p in train_predictions.items()} for f in range(5)}}
    io.write(folder/'DEVELOPMENT.json',report)
    io.save_npz(folder/'development_predictions.npz',event=data['event'][train],record=data['record'][train],fold=data['fold'][train],eligible=data['eligible'][train],**train_predictions)
    if family=='QMF':
        thresholds={a:threshold(y[train],train_predictions[a],data['record'][train]) for a in validation}
        io.write(folder/'THRESHOLDS.json',thresholds)
        vp={a:np.mean(p,axis=0) for a,p in validation.items()}
        if any(not np.array_equal(p[~data['eligible'][val]],vp['qmf_anchor'][~data['eligible'][val]]) for p in vp.values()):raise ValueError('Validation fallback changed')
        io.save_npz(folder/'validation_predictions.npz',event=data['event'][val],record=data['record'][val],eligible=data['eligible'][val],**vp)
        io.write(folder/'VALIDATION_COMMITMENT.json',{'prediction_sha256':io.digest(folder/'validation_predictions.npz'),'threshold_sha256':io.digest(folder/'THRESHOLDS.json'),'validation_labels_used':False})
    io.finish(folder)


def score_validation(raw,rows,config,out):
    folder=Path(out)/'corrections/QMF';io.verify(folder)
    destination=Path(out)/'validation'
    if (destination/'COMPLETE.json').exists():io.verify(destination);return
    commitment=io.read(folder/'VALIDATION_COMMITMENT.json')
    if io.digest(folder/'validation_predictions.npz')!=commitment['prediction_sha256'] or commitment['validation_labels_used']:raise ValueError('Prediction commitment differs')
    p=io.load_npz(folder/'validation_predictions.npz');valrows=[r for r in rows if r['split']=='val']
    if p['event'].tolist()!=[r['event'] for r in valrows]:raise ValueError('Validation binding differs')
    y=dataset.labels(raw,rows,'val');thresholds=io.read(folder/'THRESHOLDS.json')
    metrics={a:{**fc.metrics(y,p[a],p['record'],p['qmf_anchor']),
        'at_development_95pct_sensitivity_threshold':operating(y,p[a],p['record'],thresholds[a]),
        'at_0_5':operating(y,p[a],p['record'],.5)} for a in thresholds}
    comparisons=fc.comparisons(y,{a:p[a] for a in thresholds},p['record'],config['bootstrap_repetitions'],20260930)
    report={'scope':'Internal validation evaluation using committed predictions','events':len(y),'record_groups':len(set(p['record'])),
        'metrics':metrics,'paired_record_comparisons':comparisons,'prediction_sha256':commitment['prediction_sha256']}
    io.write(destination/'RESULTS.json',report);io.finish(destination)
    print('Validation scoring complete; predictions were committed before labels were joined.',flush=True)
