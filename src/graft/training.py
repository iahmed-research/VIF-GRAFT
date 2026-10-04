"""From-initialization training, inner calibration selection and resumable seeds."""
from pathlib import Path
from dataclasses import asdict
import hashlib
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
from . import io,features
from .models import QMFModel,EarlyConcatenationModel
from .losses import QMFHistory,record_balanced_weights
from .training_math import augment_batch,learning_rate_multiplier,training_loss
from .calibration import fit_nonnegative_logistic_calibrator,NonnegativeLogisticCalibrator


def fold_roles(folds,outer):
    cal=(outer+1)%5
    fit=np.flatnonzero((folds>=0)&(folds!=outer)&(folds!=cal))
    return fit,np.flatnonzero(folds==cal),np.flatnonzero(folds==outer)


def cross_calibrate(raw,y,records):
    partitions=np.array([int.from_bytes(hashlib.sha256(f'20260925\x1f{r}'.encode()).digest()[:8],'big')%5 for r in records])
    result=np.empty(len(raw))
    if set(partitions)!=set(range(5)):raise ValueError('Missing inner calibration partition')
    for k in range(5):
        fit=partitions!=k;test=~fit
        cal=fit_nonnegative_logistic_calibrator(raw[fit],y[fit],records[fit],input_kind='probability')
        result[test]=cal.predict_from_probabilities(raw[test])
    p=np.clip(result,1e-6,1-1e-6)
    return float(features.weights(records)@(-y*np.log(p)-(1-y)*np.log1p(-p)))


class Waveforms(Dataset):
    def __init__(self,folder,events,labels,weights):
        self.folder=Path(folder);self.events=events;self.labels=labels;self.weights=weights
    def __len__(self):return len(self.events)
    def __getitem__(self,i):
        x=io.load_npz(self.folder/(str(self.events[i])+'.npz'))
        return {'signals':torch.from_numpy(x['signals']),'availability':torch.from_numpy(x['available']),
            'valid':torch.from_numpy(x['model_valid_mask']),'target':torch.tensor(self.labels[i],dtype=torch.float32),
            'weight':torch.tensor(self.weights[i],dtype=torch.float32),'index':torch.tensor(i),'event':str(self.events[i])}


def forward(model,family,signals,available,event,quality,device):
    if family=='QMF':return model(signals,available)
    if family!='early_concatenation' or quality is None:raise ValueError('Early concatenation requires fitted fold-local quality')
    rows=[]
    for e,both in zip(event,(available[:,1]&available[:,2]).detach().cpu().tolist()):
        row=quality[e];eta=row['eta']
        if eta is None:
            if both:raise ValueError('Missing eta for both-pulse event: '+e)
            eta=0.0  # Training adapter: unavailable pair has no dependence coordinate.
        rows.append([row['r_ppg'],row['r_abp'],eta])
    q=torch.tensor(rows,dtype=torch.float32,device=device)
    return model(signals,available,q[:,0],q[:,1],q[:,2])


def infer(model,family,waveforms,events,device,quality=None,batch_size=64):
    loader=DataLoader(Waveforms(waveforms,events,np.zeros(len(events)),np.ones(len(events))),batch_size=batch_size,shuffle=False,num_workers=0)
    result=[];model.eval()
    with torch.inference_mode():
        for batch in loader:
            p=forward(model,family,batch['signals'].to(device),batch['availability'].to(device),batch['event'],quality,device)['probability']
            result.extend(p.cpu().numpy().astype(float).tolist())
    p=np.array(result)
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Invalid base probabilities')
    return p


def atomic_checkpoint(path,state):
    temp=Path(path).with_suffix('.tmp');torch.save(state,temp);temp.replace(path)


def train_track(data,y,config,out,family,fold,seed,device,quality=None,epoch_limit=None):
    folder=Path(out)/'models'/family/f'fold_{fold}'/f'seed_{seed}'
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'COMPLETE.json').exists():io.verify(folder);return
    fit,cal,test=fold_roles(data['fold'],fold)
    if set(data['record'][fit])&set(data['record'][cal]) or set(data['record'][test])&set(data['record'][cal]):raise ValueError('Fold leakage')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    model=(QMFModel() if family=='QMF' else EarlyConcatenationModel()).to(device)
    cfg=config['training'];amp=bool(cfg['mixed_precision'] and device=='cuda')
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    history=QMFHistory.create(len(fit)) if family=='QMF' else None
    metrics=[];best_loss=float('inf');best_epoch=0;completed=0
    latest=folder/'latest.pt'
    if latest.exists():
        state=torch.load(latest,map_location='cpu',weights_only=False)
        if (state['family'],state['fold'],state['seed'])!=(family,fold,seed):raise ValueError('Checkpoint identity differs')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        completed=state['epoch'];metrics=state['metrics'];best_loss=state['best_loss'];best_epoch=state['best_epoch']
        if history is not None:history=QMFHistory(**state['qmf_history'])
        random.setstate(state['rng_python']);np.random.set_state(state['rng_numpy']);torch.set_rng_state(state['rng_torch'])
        if device=='cuda':torch.cuda.set_rng_state_all(state['rng_cuda'])
    raw_weights=record_balanced_weights(data['record'][fit].tolist())
    w=(raw_weights*(len(fit)/float(raw_weights.sum()))).numpy()
    waveforms=Path(out)/'data/waveforms';dataset=Waveforms(waveforms,data['event'][fit],y[fit],w)
    limit=min(cfg['maximum_epochs'],epoch_limit or cfg['maximum_epochs'])
    stopped=completed>=cfg['minimum_epochs'] and completed-best_epoch>=cfg['early_stopping_patience']
    for epoch in range(completed+1,limit+1):
        if stopped:break
        loader=DataLoader(dataset,batch_size=cfg['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(seed+100003*epoch),num_workers=0,drop_last=False)
        for group in optimizer.param_groups:group['lr']=cfg['learning_rate']*learning_rate_multiplier(epoch,cfg['maximum_epochs'])
        model.train();started=time.perf_counter()
        for batch_number,batch in enumerate(loader):
            signals=augment_batch(batch['signals'],batch['availability'],batch['valid'].clone(),seed+epoch*1000003+batch_number).to(device)
            available=batch['availability'].to(device);target=batch['target'].to(device);weight=batch['weight'].to(device);indices=batch['index'].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device,dtype=torch.float16,enabled=amp):
                output=forward(model,family,signals,available,batch['event'],quality,device)
            output={k:v.float() if v.is_floating_point() else v for k,v in output.items()}
            loss,update=training_loss(family,output,target,weight,epoch=epoch,event_indices=indices,qmf_history=history)
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_norm_clip'])
            scaler.step(optimizer);scaler.update()
            if update is not None:history.update(indices,*update)
        raw=infer(model,family,waveforms,data['event'][cal],device,quality)
        score=cross_calibrate(raw,y[cal],data['record'][cal]);seconds=time.perf_counter()-started
        metrics.append({'epoch':epoch,'cross_calibrated_record_logloss':score,'seconds':seconds})
        if score<best_loss:
            best_loss=score;best_epoch=epoch
            atomic_checkpoint(folder/'best.pt',{'model':model.state_dict(),'epoch':epoch,'family':family,'fold':fold,'seed':seed})
            io.save_npz(folder/'best_calibration.npz',event=data['event'][cal],raw_probability=raw)
        state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,
            'family':family,'fold':fold,'seed':seed,'metrics':metrics,'best_loss':best_loss,'best_epoch':best_epoch,
            'qmf_history':asdict(history) if history is not None else None,'rng_python':random.getstate(),
            'rng_numpy':np.random.get_state(),'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all() if device=='cuda' else []}
        atomic_checkpoint(latest,state)
        io.write(folder/'SELECTION.json',{'history':metrics,'selected_epoch':best_epoch,'outer_evaluation_used':False})
        print(f'{family} fold={fold} seed={seed} epoch={epoch} calibration_loss={score:.8f}',flush=True)
        completed=epoch
        stopped=epoch>=cfg['minimum_epochs'] and epoch-best_epoch>=cfg['early_stopping_patience']
    if epoch_limit and not stopped and completed<cfg['maximum_epochs']:
        print('Track paused at requested epoch limit; resume with the same output directory.',flush=True);return
    if not stopped and completed<cfg['maximum_epochs']:raise ValueError('Incomplete training track')
    best=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(best['model'])
    raw=io.load_npz(folder/'best_calibration.npz')['raw_probability']
    calibrator=fit_nonnegative_logistic_calibrator(raw,y[cal],data['record'][cal],input_kind='probability')
    io.write(folder/'CALIBRATOR.json',asdict(calibrator))
    evaluation=np.r_[test,np.flatnonzero(data['split']=='val')] if family=='QMF' else test
    p=infer(model,family,waveforms,data['event'][evaluation],device,quality)
    io.save_npz(folder/'predictions.npz',event=data['event'][evaluation],probability=calibrator.predict_from_probabilities(p),raw_probability=p)
    io.save_npz(folder/'calibration_predictions.npz',event=data['event'][cal],probability=calibrator.predict_from_probabilities(raw))
    io.finish(folder,{'selected_epoch':best_epoch,'training_complete':True,'validation_labels_used':False})


def ensemble(data,out,family,seeds):
    anchor=np.full(len(data['event']),np.nan);val=np.flatnonzero(data['split']=='val');components={};calibration={}
    for fold in range(5):
        _,cal,test=fold_roles(data['fold'],fold);pp=[];cc=[]
        expected=np.r_[test,val] if family=='QMF' else test
        for seed in seeds:
            folder=Path(out)/'models'/family/f'fold_{fold}'/f'seed_{seed}';io.verify(folder)
            p=io.load_npz(folder/'predictions.npz');c=io.load_npz(folder/'calibration_predictions.npz')
            if not np.array_equal(p['event'],data['event'][expected]) or not np.array_equal(c['event'],data['event'][cal]):raise ValueError('Ensemble event alignment differs')
            pp.append(p['probability']);cc.append(c['probability'])
        averaged=np.mean(pp,axis=0);anchor[test]=averaged[:len(test)];calibration[fold]=np.mean(cc,axis=0)
        if family=='QMF':components[fold]=averaged[len(test):]
    if family=='QMF':anchor[val]=np.mean(list(components.values()),axis=0)
    return anchor,calibration,components
