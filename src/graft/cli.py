"""One portable command tree for the complete raw-data reproduction."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(key,'1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import argparse
import json
import time
import numpy as np
import torch
from . import io,data as dataset,training,evaluation,robustness,explanations,quality


def benchmark(data,config,out,device):
    from .models import QMFModel
    from .losses import QMFHistory
    from .training_math import training_loss,augment_batch
    fit,_,_=training.fold_roles(data['fold'],0);events=data['event'][fit[:64]]
    x=[io.load_npz(Path(out)/'data/waveforms'/f'{e}.npz') for e in events]
    signal=torch.from_numpy(np.stack([r['signals'] for r in x]));available=torch.from_numpy(np.stack([r['available'] for r in x]));valid=torch.from_numpy(np.stack([r['model_valid_mask'] for r in x]))
    model=QMFModel().to(device);torch.use_deterministic_algorithms(True);elapsed=[]
    for repetition in range(3):
        model.zero_grad(set_to_none=True)
        if device=='cuda':torch.cuda.synchronize()
        start=time.perf_counter();s=augment_batch(signal,available,valid.clone(),20260911+repetition).to(device)
        with torch.autocast(device_type=device,dtype=torch.float16,enabled=device=='cuda'):
            p=model(s,available.to(device))
        p={k:v.float() if v.is_floating_point() else v for k,v in p.items()}
        loss,_=training_loss('QMF',p,torch.zeros(len(events),device=device),torch.ones(len(events),device=device),epoch=1,event_indices=torch.arange(len(events),device=device),qmf_history=QMFHistory.create(len(events)))
        loss.backward()
        if device=='cuda':torch.cuda.synchronize()
        elapsed.append(time.perf_counter()-start)
    seconds=float(np.median(elapsed[1:]));batches=int(np.ceil(len(fit)/64))
    result={'status':'benchmark_pass','device':device,'batch_size':len(events),'forward_backward_batch_seconds':seconds,'optimizer_steps':0,'outcomes_loaded':False,
        'QMF_25_tracks_100_epochs_forward_backward_hours':seconds*batches*25*100/3600,
        'note':'Rough training-only ceiling at 100 epochs. Excludes calibration, optimizer, disk, quality and early-concatenation training. Early stopping may reduce work; this is not a cost or runtime guarantee.'}
    io.write(Path(out)/'BENCHMARK.json',result);print(json.dumps(result,indent=2))


def report(out):
    out = Path(out)
    method_labels = {
        'qmf_anchor': 'QMF anchor',
        'qmf_recalibration': 'QMF recalibration',
        'graft_ecg2': 'GRAFT-ECG2',
        'graft': 'GRAFT',
        'graft_without_agreement': 'GRAFT without ECG agreement',
        'graft_without_supplementary_ecg': 'GRAFT without supplementary ECG',
        'graft_unbounded': 'GRAFT unbounded residual',
    }
    sections = []
    for name, path in [
        ('QMF anchor extension', out/'corrections/QMF/DEVELOPMENT.json'),
        ('Early-concatenation anchor extension', out/'corrections/early_concatenation/DEVELOPMENT.json'),
        ('Internal validation', out/'validation/RESULTS.json'),
    ]:
        if path.exists():
            current = io.read(path)
            rows = []
            for method, value in current['metrics'].items():
                rows.append({
                    'method': method_labels.get(method, method),
                    'record_balanced_logloss': value['record_balanced_logloss'],
                    'relative_gain_vs_qmf': value.get('relative_gain_vs_qmf'),
                })
            sections.append({'evaluation': name, 'metrics': rows})

    io.write(out/'RESULTS_SUMMARY.json', {'sections': sections})
    lines = ['# GRAFT reproduction results', '']
    for section in sections:
        lines += [
            f"## {section['evaluation']}",
            '',
            '| Method | Record-balanced log loss | Relative gain vs QMF anchor |',
            '|---|---:|---:|',
        ]
        for row in section['metrics']:
            gain = row['relative_gain_vs_qmf']
            gain_text = '' if gain is None else f"{100*gain:.2f}%"
            lines.append(f"| {row['method']} | {row['record_balanced_logloss']:.8f} | {gain_text} |")
        lines.append('')
    lines += [
        'Detailed metrics and paired record intervals are written under `corrections/*/DEVELOPMENT.json` and `validation/RESULTS.json`.',
        'Robustness outputs are written under `robustness/`. Exact score decomposition, waveform cases and grouped Shapley outputs are written under `explanations/`.',
        'The pipeline starts from raw VTaC 1.0 waveforms and regenerates preprocessing, training, calibration, GRAFT residual heads, validation scoring, robustness analysis and explanations.',
    ]
    (out/'ALL_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['doctor','download','prepare','benchmark','quality','train','fit','score','robustness','explain','report','all'])
    p.add_argument('--raw',type=Path,default=Path('data/vtac-1.0'))
    p.add_argument('--out',type=Path,default=Path('outputs/reproduction'))
    p.add_argument('--config',type=Path,default=Path('configs/study.json'))
    p.add_argument('--quality-config',type=Path,default=Path('configs/quality.json'))
    p.add_argument('--cohort',type=Path,default=Path('configs/cohort.csv'))
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--family',choices=['QMF','early_concatenation','both'],default='both')
    p.add_argument('--fold',type=int,choices=range(5))
    p.add_argument('--seed',type=int,choices=range(20260911,20260916))
    p.add_argument('--epoch-limit',type=int,help='Pause training after this absolute epoch; cannot mark an incomplete track complete')
    args=p.parse_args(argv);torch.set_num_threads(4)
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable; CPU fallback is not automatic')
    if args.command=='doctor':print(json.dumps(io.environment(),indent=2));return
    if args.command=='report':report(args.out);return
    if args.command=='all' and (args.fold is not None or args.seed is not None or args.epoch_limit or args.family!='both'):p.error('Use train for partial jobs; all requires the complete prespecified experiment')
    rows=io.cohort(args.cohort);config=io.read(args.config);qconfig=io.read(args.quality_config)
    if args.command=='download':dataset.download(args.raw,rows);return
    io.bind_run(args.out,{'study':config,'quality':qconfig},args.cohort,args.device)
    if args.command in ('prepare','all'):
        data=dataset.prepare(args.raw,rows,config,args.out)
        if args.command=='prepare':return
    else:
        io.verify(args.out/'data');data=io.load_npz(args.out/'data/features.npz')
    if data['event'].tolist()!=[r['event'] for r in rows]:raise ValueError('Cached cohort differs')
    raw_manifest=io.read(args.out/'data/RAW_MANIFEST.json')
    for name in ('event_labels.csv','benchmark_data_split.csv'):
        if io.digest(args.raw/name)!=raw_manifest[name]:raise ValueError('Raw metadata differs from prepared data: '+name)
    if args.command=='benchmark':benchmark(data,config,args.out,args.device);return
    y=np.full(len(rows),-1,dtype=int);train=data['split']=='train';y[train]=dataset.labels(args.raw,rows,'train')
    families=['QMF','early_concatenation'] if args.family=='both' else [args.family]
    if args.command in ('quality','all') or args.command=='train' and 'early_concatenation' in families:
        quality.build(args.raw,rows,y,qconfig,args.out)
        if args.command=='quality':return
    if args.command in ('train','all'):
        for family in families:
            for fold in ([args.fold] if args.fold is not None else range(5)):
                coordinates=io.read(args.out/'quality'/f'fold_{fold}'/'coordinates.json') if family=='early_concatenation' else None
                if family=='early_concatenation':io.verify(args.out/'quality'/f'fold_{fold}')
                for seed in ([args.seed] if args.seed is not None else config['seeds']):
                    training.train_track(data,y,config,args.out,family,fold,seed,args.device,coordinates,args.epoch_limit)
        if args.command=='train':return
    if args.command in ('fit','all'):
        for family in families:evaluation.corrections(data,y,config,args.out,family)
    if args.command in ('score','all'):evaluation.score_validation(args.raw,rows,config,args.out)
    if args.command in ('robustness','all'):robustness.run(data,y,config,args.out)
    if args.command in ('explain','all'):explanations.run(data,y,args.out)
    if args.command=='all':report(args.out);io.write(args.out/'ALL_COMPLETE.json',{'status':'complete','scope':'Full fresh-training pipeline','environment':io.environment()})


if __name__=='__main__':main()
