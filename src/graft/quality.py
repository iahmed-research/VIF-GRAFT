"""Rebuild the original fold-local quality inputs for early concatenation."""
from pathlib import Path
import numpy as np
from . import io
from .quality_core import serialize_representations,deserialize_representations,pair_feature_blocks
from .quality_pairs import extract_event_representations,feature_sets
from .quality_materialization import materialize_fold_role_pairs,m2_fold_roles
from .quality_fusion import fit_fold_local_credibility,supported_credibility_tasks


def build(raw,rows,y,config,out):
    folder=Path(out)/'quality'
    if (folder/'COMPLETE.json').exists():io.verify(folder);return
    folder.mkdir(parents=True,exist_ok=True)
    channels=io.read(Path(out)/'data/channels.json');by_event={}
    for c in channels:by_event.setdefault(c['event'],[]).append(c)
    extracted={};availability={}
    for i,row in enumerate(rows):
        if row['split']!='train':continue
        cache=folder/'representations'/f"{row['event']}.json"
        if cache.exists():representation=deserialize_representations(io.read(cache)['representation'])
        else:
            representation=extract_event_representations(Path(raw),row,by_event[row['event']],config)
            io.write(cache,{'representation':serialize_representations(representation)})
        extracted[row['event']]={**row,'label':int(y[i]),'representations':representation}
        a=io.load_npz(Path(out)/'data/waveforms'/f"{row['event']}.npz")['available']
        availability[row['event']]='both' if a[1] and a[2] else 'ppg_only' if a[1] else 'abp_only' if a[2] else 'no_usable_pulse'
        if len(extracted)%250==0:print(f'Quality representations: {len(extracted)}',flush=True)
    specs=feature_sets(config);names={t:specs['full' if t=='PA' else 'marginal'] for t in ('EP','EA','PA')}
    # Uncorrupted application features are computed without outcomes or donor selection.
    applications=[];supported={}
    for event,row in extracted.items():
        tasks=supported_credibility_tasks(availability[event],row['representations'],config);supported[event]=tasks
        for task in tasks:
            spec=config['pair_tasks'][task];reps=row['representations']
            a,b=pair_feature_blocks(reps[spec['anchor']],reps[spec['target']],spec,config);values={**a,**b}
            applications.append({'task':task,'record':row['record'],'event':event,'partition_role':'application',**{k:float(values[k]) for k in names[task]}})
    for fold in range(5):
        dest=folder/f'fold_{fold}'
        if (dest/'COMPLETE.json').exists():io.verify(dest);continue
        roles=m2_fold_roles(fold);pairs={}
        for role in ('fit','calibration'):
            local={e:{**r,'partition':role} for e,r in extracted.items() if r['fold'] in roles[role]}
            pairs[role]=materialize_fold_role_pairs(local,config,partition_role=role,allowed_folds=roles[role])
        bundle=fit_fold_local_credibility(pairs['fit'].rows,pairs['calibration'].rows,features_by_task=names,immutable_context={'outer_fold':fold,'quality_config':io.canonical(config)})
        dest.mkdir(parents=True,exist_ok=True);bundle.save(dest/'bundle.json')
        result=bundle.transform(applications,expected_event_records={e:r['record'] for e,r in extracted.items()},availability_by_event=availability,supported_tasks_by_event=supported)
        io.write(dest/'coordinates.json',{r['event']:{k:r[k] for k in ('r_ppg','r_abp','eta')} for r in result})
        io.write(dest/'PAIR_AUDIT.json',{r:dict(p.audit) for r,p in pairs.items()});io.finish(dest)
        print(f'Fold-local quality complete: {fold}',flush=True)
    io.finish(folder)
