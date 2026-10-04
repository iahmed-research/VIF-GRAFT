"""Exact correction accounting, interventions, fixed cases and reference sensitivity."""
from pathlib import Path
import numpy as np
from . import io,features,correction as fc,shapley as sh
from .evaluation import numeric_head

GROUPS=('Intercept','QMF confidence','Original ECG','Chest ECG','PPG','ABP','ECG agreement')
INDICES=([0],[1],list(range(2,14)),list(range(14,26)),list(range(26,38)),list(range(38,50)),list(range(50,53)))


def representative(mask,magnitude,event):
    ix=np.flatnonzero(mask)
    if not len(ix):return None
    median=np.median(magnitude[ix])
    return int(min(ix,key=lambda i:(abs(magnitude[i]-median),str(event[i]))))


def run(data,y,out):
    folder=Path(out)/'explanations'
    if (folder/'COMPLETE.json').exists():io.verify(folder);return
    source=Path(out)/'corrections/QMF';io.verify(source)
    ix=np.flatnonzero(data['split']=='train');d={k:v[ix] for k,v in data.items()};y=y[ix]
    saved=io.load_npz(source/'development_predictions.npz');d['anchor']=saved['qmf_anchor'];p=saved['graft'];q=d['anchor']
    heads=io.read(source/'HEADS.json')['graft'];n=len(ix)
    terms=np.zeros((n,53));neutral=np.zeros((n,7));permuted=np.zeros(n);error=0.
    for fold in range(5):
        rows=np.flatnonzero(d['fold']==fold);m=numeric_head(heads[str(fold)])
        x=fc.matrix(d,'graft',rows,q[rows]);z=features.design(x,m['mean'],m['scale']);t=z*m['theta'];terms[rows]=t
        raw=t.sum(axis=1);baseline=q[rows];eligible=d['eligible'][rows]
        reproduced=baseline+sh.delta(baseline,raw,eligible)
        error=max(error,float(np.max(abs(reproduced-p[rows]))))
        for j,indices in enumerate(INDICES):
            altered=z.copy();altered[:,indices]=0
            direct=baseline+sh.delta(baseline,altered@m['theta'],eligible)
            accounted=baseline+sh.delta(baseline,raw-t[:,indices].sum(axis=1),eligible)
            error=max(error,float(np.max(abs(direct-accounted))))
            neutral[rows,j]=direct-p[rows]
        theta=m['theta'].copy();theta[1:]=np.random.default_rng(20261001+fold).permutation(theta[1:])
        permuted[rows]=baseline+sh.delta(baseline,z@theta,eligible)
    if error>1e-8:raise ValueError('Exact explanation accounting differs')
    groups=np.column_stack([terms[:,indices].sum(axis=1) for indices in INDICES])*d['eligible'][:,None]
    w=features.weights(d['record']);ew=w*d['eligible'];ew/=ew.sum()
    change=fc.losses(y,p)-fc.losses(y,q);records,inv,counts=np.unique(d['record'],return_inverse=True,return_counts=True)
    record_change=np.bincount(inv,weights=change)/counts
    report={'events':n,'reconstruction_and_intervention_max_error':error,
        'group_contributions':[{'group':g,'mean_absolute_score':float(ew@abs(groups[:,j])),'mean_signed_score':float(ew@groups[:,j]),'mean_absolute_neutralization_probability_change':float(ew@abs(neutral[:,j]))} for j,g in enumerate(GROUPS)],
        'coefficient_permutation_mean_absolute_probability_change':float(ew@abs(permuted-p)),
        'improved_record_groups':int((record_change < -1e-12).sum()),'worsened_record_groups':int((record_change>1e-12).sum()),'unchanged_record_groups':int((abs(record_change)<=1e-12).sum()),
        'interpretation':'Exact pre-tanh score decomposition and computational interventions, not causal physiology.'}
    theta=np.array([heads[str(f)]['theta'] for f in range(5)])
    names=['Intercept']+heads['0']['predictor_names']
    report['feature_terms_and_fold_signs']=[{'feature':name,'mean_absolute_score':float(ew@abs(terms[:,j]*d['eligible'])),
        'negative_heads':int((theta[:,j]<-1e-10).sum()),'positive_heads':int((theta[:,j]>1e-10).sum()),'near_zero_heads':int((abs(theta[:,j])<=1e-10).sum())} for j,name in enumerate(names)]
    report['class_logloss']={str(label):{'QMF':float(w[y==label]@fc.losses(y[y==label],q[y==label])/w[y==label].sum()),
        'GRAFT':float(w[y==label]@fc.losses(y[y==label],p[y==label])/w[y==label].sum())} for label in (0,1)}
    cases=[]
    for label in (0,1):
        for sign,name in [(-1,'help'),(1,'harm')]:
            i=representative(d['eligible']&(y==label)&(change*sign>1e-12),abs(change),d['event'])
            if i is not None:cases.append({'index':i,'category':f'class_{label}_{name}'})
    missing=np.flatnonzero(~d['eligible'])
    if len(missing):cases.append({'index':int(min(missing,key=lambda i:str(d['event'][i]))),'category':'missing_chest'})
    stress_path=Path(out)/'robustness/PREDICTIONS.npz';noise=None
    if stress_path.exists():
        io.verify(stress_path.parent);stress=io.load_npz(stress_path)
        if not np.array_equal(stress['event'],d['event']):raise ValueError('Stress explanation event mismatch')
        k=list(stress['conditions']).index('noise_0dB');j=list(stress['arms']).index('graft');noise=stress['prediction'][k,:,:,j]
        harm=fc.losses(y[None,:],noise).mean(axis=0)-fc.losses(y,p)
        i=representative(d['eligible']&stress['eligibility'][k].all(axis=0)&(harm>1e-12),harm,d['event'])
        if i is not None:cases.append({'index':i,'category':'accepted_severe_noise_harm','noise_probability_first_seed':float(noise[0,i])})
    for number,case in enumerate(cases,1):
        i=case['index'];case.update(case_id=f'C{number}',event=str(d['event'][i]),outcome=int(y[i]),qmf_probability=float(q[i]),graft_probability=float(p[i]),group_terms=groups[i].tolist())
    io.write(folder/'RESULTS.json',report);io.write(folder/'CASES.json',cases)
    io.save_npz(folder/'EXPLANATIONS.npz',event=d['event'],record=d['record'],fold=d['fold'],raw_terms=terms,effective_group_terms=groups,neutralization_change=neutral,permuted_probability=permuted)
    shap=[]
    for i in sh.selected(d,20):
        model=numeric_head(heads[str(d['fold'][i])]);target,context=sh.components(model,d['B'][[i]],q[[i]])
        row={'event':str(d['event'][i]),'record':str(d['record'][i]),'fold':int(d['fold'][i]),'correction':float(p[i]-q[i])}
        if not d['eligible'][i]:row.update(status='exact_missing_fallback',phi=[0.,0.,0.],reference_correction=0.)
        else:
            pools=sh.reference_sets(d,i,model['calibration_fold'])
            if pools is None:row['status']='insufficient_matched_reference_records'
            else:
                row.update(status='explained',references=[])
                for name,pool in zip(sh.REFERENCES,pools):
                    donors,_=sh.components(model,d['B'][pool],np.full(len(pool),q[i]))
                    phi,values=sh.explain(target[0],donors,sh.weights(d['record'][pool]),q[i],context[0])
                    if abs(values[-1]-row['correction'])>1e-8:raise ValueError('Shapley endpoint differs')
                    row['references'].append({'name':name,'phi':phi.tolist(),'reference_correction':float(values[0]),'events':d['event'][pool].tolist(),'records':len(set(d['record'][pool]))})
                phis=np.array([r['phi'] for r in row['references']]);row['spread']=float(np.ptp(phis,axis=0).max())
                row['sign_change']=bool(np.any((phis.min(axis=0)<-1e-6)&(phis.max(axis=0)>1e-6)))
        shap.append(row)
    explained=[r for r in shap if r['status']=='explained']
    io.write(folder/'SHAPLEY.json',{'events':shap,'summary':{'explained':len(explained),'sign_change_events':sum(r['sign_change'] for r in explained),'maximum_reference_spread':max((r['spread'] for r in explained),default=None)}})
    plot(folder,out,cases,report)
    io.finish(folder)


def plot(folder,out,cases,report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .perturbations import perturb
    from .preprocessing import preprocess_physical_channel
    config=io.read(Path(out)/'BINDING.json')['config']['study']
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    ax.barh(GROUPS[::-1],[r['mean_absolute_score'] for r in report['group_contributions']][::-1]);ax.set_xlabel('Mean absolute contribution to pre-tanh correction score')
    for suffix in ('png','pdf','svg'):fig.savefig(Path(folder)/f'EXPLANATION.{suffix}',dpi=180)
    plt.close(fig)
    if not cases:return
    fig,axes=plt.subplots(len(cases),2,figsize=(13,3*len(cases)),squeeze=False,layout='constrained')
    for index,case in enumerate(cases):
        x=io.load_npz(Path(out)/'data/waveforms'/f"{case['event']}.npz");t=np.arange(2500)/250-10
        traces=[x['signals'][0],None,x['signals'][2],x['signals'][3]];valid=[x['model_valid_mask'][0],None,x['model_valid_mask'][2],x['model_valid_mask'][3]]
        if x['eligible']:
            pp=preprocess_physical_channel(x['chest_physical'],x['chest_valid'],role='ecg',sampling_frequency_hz=250,config=config['preprocessing']);traces[1]=pp.tensor;valid[1]=pp.model_valid_mask
        offsets=[7.5,5,2.5,0];axis=axes[index,0]
        for j,(trace,mask) in enumerate(zip(traces,valid)):
            if trace is None or not mask.any():axis.text(-9.8,offsets[j],'unavailable');continue
            axis.plot(t,np.where(mask,trace/12+offsets[j],np.nan),linewidth=.65)
        if case['category']=='accepted_severe_noise_harm':
            physical,mask=perturb(x['chest_physical'],x['chest_valid'],'noise_0dB',case['event'],20260927)
            pp=preprocess_physical_channel(physical,mask,role='ecg',sampling_frequency_hz=250,config=config['preprocessing'])
            axis.plot(t,np.where(pp.model_valid_mask,pp.tensor/12+5,np.nan),color='orange',linewidth=.55)
        axis.set_yticks(offsets,['Original ECG','Chest ECG','PPG','ABP']);axis.set_xlabel('Seconds before alarm')
        axis.set_title(f"{case['case_id']}: {case['category']} | QMF {case['qmf_probability']:.3f} -> GRAFT {case['graft_probability']:.3f}")
        axes[index,1].barh(GROUPS[::-1],case['group_terms'][::-1]);axes[index,1].axvline(0,color='gray');axes[index,1].set_xlabel('Signed clean score contribution')
    for suffix in ('png','pdf','svg'):fig.savefig(Path(folder)/f'WAVEFORM_CASES.{suffix}',dpi=180)
    plt.close(fig)
