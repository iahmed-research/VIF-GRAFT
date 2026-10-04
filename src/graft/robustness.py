"""Reproduce the fixed chest-only perturbation experiment from raw-derived arrays."""
from pathlib import Path
import numpy as np
from . import io,features,correction as fc,perturbations as rb
from .evaluation import numeric_head

ARMS=('qmf_anchor','graft_ecg2','graft','graft_unbounded')


def record_mean(x,records):
    _,inv,counts=np.unique(records,return_inverse=True,return_counts=True)
    return np.bincount(inv,weights=x)/counts


def run(data,y,config,out):
    folder=Path(out)/'robustness'
    if (folder/'COMPLETE.json').exists():io.verify(folder);return
    source=Path(out)/'corrections/QMF';io.verify(source)
    heads=io.read(source/'HEADS.json');saved=io.load_npz(source/'development_predictions.npz')
    ix=np.flatnonzero(data['split']=='train');local={k:v[ix] for k,v in data.items()};local['anchor']=saved['qmf_anchor'];y=y[ix]
    n=len(ix);pred=np.empty((len(rb.CONDITIONS),3,n,len(ARMS)));eligibility=np.empty((len(rb.CONDITIONS),3,n),bool)
    for k,condition in enumerate(rb.CONDITIONS):
        for seed_index,seed in enumerate(rb.SEEDS):
            stressed=local['B'].copy();mask=local['eligible'].copy()
            for i,event in enumerate(local['event']):
                if not mask[i]:continue
                if condition=='missing':mask[i]=False;continue
                if condition=='clean':continue
                x=io.load_npz(Path(out)/'data/waveforms'/f'{event}.npz')
                stressed[i],mask[i]=rb.stressed_features(local['B'][i],x['signals'][0],x['model_valid_mask'][0],x['chest_physical'],x['chest_valid'],condition,str(event),seed,config['preprocessing'])
            d={**local,'B':stressed};eligibility[k,seed_index]=mask;pred[k,seed_index,:,0]=local['anchor']
            for fold in range(5):
                rows=np.flatnonzero(local['fold']==fold);q=local['anchor'][rows]
                for j,arm in enumerate(ARMS[1:],1):
                    m=numeric_head(heads[arm][str(fold)])
                    pred[k,seed_index,rows,j]=fc.predict(m,fc.matrix(d,arm,rows,q),q,mask[rows])
        print(f'Chest robustness: {condition}',flush=True)
    for j,arm in enumerate(ARMS):
        if not np.allclose(pred[0,0,:,j],saved[arm],rtol=0,atol=1e-8):raise ValueError('Clean robustness reproduction differs')
    if not np.array_equal(pred[1,:,:,2],np.broadcast_to(local['anchor'],(3,n))):raise ValueError('Missing fallback differs')
    results=score(y,local['record'],pred,eligibility,local['eligible'])
    io.write(folder/'RESULTS.json',{'conditions':results,'scope':'Fixed development perturbations; average losses across seeds, not probabilities'})
    io.save_npz(folder/'PREDICTIONS.npz',event=local['event'],record=local['record'],prediction=pred,eligibility=eligibility,conditions=np.array(rb.CONDITIONS),arms=np.array(ARMS),seeds=np.array(rb.SEEDS))
    plot(folder,results)
    io.finish(folder)


def plot(folder,results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=list(rb.CONDITIONS);mean=np.array([results[k]['graft_minus_qmf_anchor']['loss_difference'] for k in names]);ci=np.array([results[k]['graft_minus_qmf_anchor']['ci95'] for k in names])
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    ax.errorbar(mean,np.arange(len(names)),xerr=np.maximum(0,np.array([mean-ci[:,0],ci[:,1]-mean])),fmt='o',capsize=3)
    ax.axvline(0,color='gray',linestyle='--');ax.set_yticks(np.arange(len(names)),names);ax.invert_yaxis();ax.set_xlabel('GRAFT minus QMF record-balanced log loss (paired 95% interval)')
    for suffix in ('png','pdf','svg'):fig.savefig(Path(folder)/f'ROBUSTNESS.{suffix}',dpi=180)
    plt.close(fig)


def score(y,records,pred,eligibility,original_eligible):
    # pred: condition, seed, event, arm. Repeated seeds are not independent records.
    w=features.weights(records); loss=fc.losses(y[None,None,:,None],pred); meanloss=loss.mean(axis=1)
    rng=np.random.Generator(np.random.PCG64(20260930))
    contrasts=[]; results={}; sizes=len(np.unique(records))
    for k,name in enumerate(rb.CONDITIONS):
        entry={'metrics':{},'eligible_events_per_seed':eligibility[k].sum(axis=1).tolist(),
               'rejected_originally_eligible_per_seed':((~eligibility[k]) & original_eligible).sum(axis=1).tolist()}
        for j,arm in enumerate(ARMS):
            entry['metrics'][arm]={'record_balanced_logloss':float(w @ meanloss[k,:,j]),
                                  'Brier':float(np.mean((pred[k,:,:,j]-y)**2,axis=0) @ w),
                                  'weighted_mean_absolute_change_from_clean':float(np.mean(abs(pred[k,:,:,j]-pred[0,:,:,j]),axis=0) @ w)}
        for group,mask in [('accepted',eligibility[k]),('rejected',~eligibility[k])]:
            mass=float(np.sum(mask*w)/len(rb.SEEDS))
            entry[group]={'event_seed_observations':int(mask.sum()),'global_record_weight_mass':mass,
                          'GRAFT_logloss':float(np.sum(loss[k,:,:,2]*mask*w)/len(rb.SEEDS)/mass) if mass else None,
                          'QMF_logloss':float(np.sum(loss[k,:,:,0]*mask*w)/len(rb.SEEDS)/mass) if mass else None}
        for label,difference in [('graft_minus_qmf_anchor',meanloss[k,:,2]-meanloss[k,:,0]),
                                 ('graft_minus_unbounded',meanloss[k,:,2]-meanloss[k,:,3]),
                                 ('graft_stressed_minus_clean',meanloss[k,:,2]-meanloss[0,:,2])]:
            contrasts.append((name,label,record_mean(difference,records)))
        results[name]=entry
    matrix=np.column_stack([x[2] for x in contrasts]); samples=[]
    for _ in range(100):
        ix=rng.integers(0,sizes,size=(100,sizes)); samples.append(matrix[ix].mean(axis=1))
    samples=np.concatenate(samples)
    for j,(name,label,diff) in enumerate(contrasts):
        results[name][label]={'loss_difference':float(diff.mean()),'ci95':np.quantile(samples[:,j],[.025,.975]).tolist()}
    return results
