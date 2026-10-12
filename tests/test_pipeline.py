"""Exercise orchestration with a tiny trainable model, not a substitute scientific model."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from graft import io,training,evaluation,correction as fc,explanations


class TinyQMF(torch.nn.Module):
    def __init__(self):
        super().__init__();self.linear=torch.nn.Linear(4,6)
    def forward(self,x,a):
        views=self.linear(x.mean(-1)).reshape(-1,3,2)
        confidence=.1*torch.logsumexp(views,dim=-1)*a
        fused=(views*confidence[:,:,None]).sum(1)
        return {'probability':fused.softmax(1)[:,1],'fused_logits':fused,'view_logits':views,'view_mask':a,'energy_confidence':confidence}


def fixture(tmp_path):
    rows=[]
    for fold in range(5):
        groups={k:[] for k in range(5)};j=0
        while any(len(v)<2 for v in groups.values()):
            record=f'f{fold}r{j}';j+=1
            k=int.from_bytes(hashlib.sha256(f'20260925\x1f{record}'.encode()).digest()[:8],'big')%5
            if len(groups[k])<2:groups[k].append(record)
        for record in sorted(sum(groups.values(),[])):
            for label in (0,1):rows.append({'event':f'{record}_{label}','record':record,'split':'train','fold':fold,'label':label})
    rows+= [{'event':f'v{i}','record':f'v{i}','split':'val','fold':-1,'label':i%2} for i in range(4)]
    rng=np.random.default_rng(3)
    for r in rows:
        io.save_npz(tmp_path/'data/waveforms'/f"{r['event']}.npz",signals=rng.normal(size=(4,128)).astype('float32'),available=np.ones(3,bool),model_valid_mask=np.ones((4,128),bool))
    d={k:np.array([r[k] for r in rows]) for k in ('event','record','split','fold')}
    d.update(ecg2_features=rng.normal(size=(len(rows),51)),graft_features=rng.normal(size=(len(rows),51)),eligible=np.ones(len(rows),bool))
    d['graft_features'][:,[0,12,24,36]]=1
    y=np.array([r['label'] if r['split']=='train' else -1 for r in rows])
    config=io.read(Path(__file__).resolve().parents[1]/'configs/study.json')
    config['training'].update(maximum_epochs=1,minimum_epochs=1,batch_size=32,mixed_precision=False)
    config.update(seeds=[20260911],bootstrap_repetitions=100)
    return rows,d,y,config


def test_training_ensemble_heads_and_separate_scoring(tmp_path,monkeypatch):
    monkeypatch.setattr(training,'QMFModel',TinyQMF)
    rows,d,y,config=fixture(tmp_path)
    for fold in range(5):training.train_track(d,y,config,tmp_path,'QMF',fold,20260911,'cpu')
    original=io.digest(tmp_path/'models/QMF/fold_0/seed_20260911/best.pt')
    training.train_track(d,y,config,tmp_path,'QMF',0,20260911,'cpu')
    assert io.digest(tmp_path/'models/QMF/fold_0/seed_20260911/best.pt')==original
    evaluation.corrections(d,y,config,tmp_path,'QMF')
    committed=io.read(tmp_path/'corrections/QMF/VALIDATION_COMMITMENT.json')
    assert committed['validation_labels_used'] is False
    raw=tmp_path/'raw';raw.mkdir()
    (raw/'event_labels.csv').write_text('record,event,decision\n'+'\n'.join(f"{r['record']},{r['event']},{bool(r['label'])}" for r in rows)+'\n')
    evaluation.score_validation(raw,rows,config,tmp_path)
    report=io.read(tmp_path/'validation/RESULTS.json')
    assert report['events']==4 and len(report['metrics'])==4
    assert report['prediction_sha256']==committed['prediction_sha256']
    io.verify(tmp_path/'validation')
    monkeypatch.setattr(explanations,'plot',lambda *args:None)
    explanations.run(d,y,tmp_path)
    io.verify(tmp_path/'explanations')
    assert io.read(tmp_path/'explanations/RESULTS.json')['reconstruction_and_intervention_max_error']<1e-8
    assert io.read(tmp_path/'explanations/SHAPLEY.json')['summary']['explained']>0


def test_resume_training_matches_uninterrupted(tmp_path,monkeypatch):
    monkeypatch.setattr(training,'QMFModel',TinyQMF)
    _,d,y,config=fixture(tmp_path)
    config['training'].update(maximum_epochs=2,minimum_epochs=2)
    training.train_track(d,y,config,tmp_path,'QMF',0,20260911,'cpu',epoch_limit=1)
    folder=tmp_path/'models/QMF/fold_0/seed_20260911'
    assert not (folder/'COMPLETE.json').exists()
    training.train_track(d,y,config,tmp_path,'QMF',0,20260911,'cpu')
    resumed=torch.load(folder/'latest.pt',weights_only=False)['model']
    # Independent folder, identical generated data and configuration.
    other=tmp_path/'other';_,d2,y2,c2=fixture(other);c2['training'].update(maximum_epochs=2,minimum_epochs=2)
    training.train_track(d2,y2,c2,other,'QMF',0,20260911,'cpu')
    uninterrupted=torch.load(other/'models/QMF/fold_0/seed_20260911/latest.pt',weights_only=False)['model']
    for key in resumed:assert torch.equal(resumed[key],uninterrupted[key])
