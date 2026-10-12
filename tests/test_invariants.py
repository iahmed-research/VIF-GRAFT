import itertools
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from graft import io,features,correction as fc,data,training,shapley
from graft.models import QMFModel
from graft.quality_pairs import extract_event_representations
from graft.training_math import training_loss
from graft.losses import QMFHistory

ROOT=Path(__file__).resolve().parents[1]


def test_fixed_cohort_and_record_separation():
    rows=io.cohort(ROOT/'configs/cohort.csv')
    assert sum(r['split']=='train' for r in rows)==4060
    assert sum(r['split']=='val' for r in rows)==495
    folds=np.array([r['fold'] for r in rows]);records=np.array([r['record'] for r in rows])
    for outer in range(5):
        fit,cal,test=training.fold_roles(folds,outer)
        assert not (set(records[fit])&set(records[cal]) or set(records[fit])&set(records[test]) or set(records[cal])&set(records[test]))
        assert np.all(folds[fit]>=0)


def test_qmf_real_forward_backward_missing_views():
    torch.set_num_threads(2);torch.manual_seed(2)
    model=QMFModel();signals=torch.randn(2,4,128);a=torch.tensor([[1,0,0],[1,1,1]],dtype=torch.bool)
    out=model(signals,a)
    loss,update=training_loss('QMF',out,torch.tensor([0.,1.]),torch.ones(2),epoch=1,event_indices=torch.arange(2),qmf_history=QMFHistory.create(2))
    loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
    assert out['energy_confidence'][0,1:].tolist()==[0.,0.]
    assert update[0].shape==(2,3)


def test_correction_gradient_against_finite_difference():
    rng=np.random.default_rng(4);z=np.c_[np.ones(20),rng.normal(size=(20,4))];theta=rng.normal(size=5)*.1
    y=np.arange(20)%2;w=np.ones(20)/20;offset=rng.normal(size=20);eligible=np.arange(20)%3!=0
    _,g=features.objective(theta,z,y,w,offset,eligible,.1,True)
    numeric=[]
    for j in range(5):
        eps=np.eye(5)[j]*1e-6
        numeric.append((features.objective(theta+eps,z,y,w,offset,eligible,.1,True)[0]-features.objective(theta-eps,z,y,w,offset,eligible,.1,True)[0])/2e-6)
    np.testing.assert_allclose(g,numeric,atol=1e-8)


@pytest.mark.parametrize('q',[0.,1e-10,.3,.8,1.])
def test_exact_anchor_and_missing(q):
    x=np.ones((3,2));m={'mean':np.zeros(2),'scale':np.ones(2),'theta':np.zeros(3),'bounded':True}
    p=np.full(3,q)
    assert np.array_equal(fc.predict(m,x,p,np.ones(3,bool)),p)
    m['theta'][:]=100
    assert np.array_equal(fc.predict(m,x,p,np.zeros(3,bool)),p)


def test_shapley_permutation_reference():
    v=np.random.default_rng(3).normal(size=8);expected=np.zeros(3)
    for permutation in itertools.permutations(range(3)):
        mask=0
        for j in permutation:
            nxt=mask|1<<j;expected[j]+=(v[nxt]-v[mask])/6;mask=nxt
    np.testing.assert_allclose(shapley.shapley(v,3),expected,atol=1e-14)


def test_shapley_reference_plus_terms_and_dummy():
    phi,v=shapley.explain(np.array([.3,-.2,0]),np.array([[.1,.2,0],[-.4,.1,0]]),np.array([.5,.5]),.7,.1)
    assert abs(v[0]+phi.sum()-v[-1])<1e-14 and phi[2]==0


def test_receipt_detects_mutation(tmp_path):
    io.write(tmp_path/'x.json',{'x':1});io.finish(tmp_path);io.verify(tmp_path)
    io.write(tmp_path/'x.json',{'x':2})
    with pytest.raises(ValueError):io.verify(tmp_path)


def test_cohort_rejects_record_leak(tmp_path):
    p=tmp_path/'cohort.csv';p.write_text('event,record,split,fold\na,r,train,0\nb,r,val,-1\n')
    with pytest.raises(ValueError):io.cohort(p)


def test_raw_waveform_and_quality_extraction(tmp_path):
    row={'record':'r','event':'e'};folder=tmp_path/'waveforms/r';folder.mkdir(parents=True)
    t=np.arange(75000)/250
    digital=np.column_stack([1000*np.sin(2*np.pi*f*t) for f in [2,2.1,1.9,1.3,1.4]]).astype('<i2')
    (folder/'e.dat').write_bytes(digital.tobytes())
    header=['e 5 250 75000']+[f'e.dat 16 1000/mV 16 0 0 0 0 {name}' for name in ['II','I','V','PLETH','ABP']]
    (folder/'e.hea').write_text('\n'.join(header)+'\n')
    config=io.read(ROOT/'configs/study.json');qconfig=io.read(ROOT/'configs/quality.json')
    x,channels=data.extract(tmp_path,row,config)
    assert x['signals'].shape==(4,2500) and x['graft_features'].shape==(51,) and x['eligible']
    assert x['selected_channel_indices'].tolist()==[0,1,3,4] and x['chest_index']==2
    reps=extract_event_representations(tmp_path,row,channels,qconfig)
    assert set(reps)=={'ecg','ppg','abp'}


def test_validation_label_alignment_filters_other_splits(tmp_path):
    (tmp_path/'event_labels.csv').write_text('record,event,decision\nr,a,True\ns,b,False\nt,c,SECRET_TEST_VALUE\n')
    rows=[{'event':'a','record':'r','split':'train'},{'event':'b','record':'s','split':'val'}]
    assert data.labels(tmp_path,rows,'train').tolist()==[1]
    assert data.labels(tmp_path,rows,'val').tolist()==[0]


def test_reference_presence_isolation():
    d={'graft_features':np.ones((6,51)),'fold':np.array([0,1,1,1,1,1]),'record':np.array(['a','b','c','d','e','f']),'eligible':np.ones(6,bool)}
    d['graft_features'][-1,24]=0
    pools=shapley.reference_sets(d,0,1)
    assert set(pools[0])=={1,2,3,4} and not set(pools[1])&set(pools[2])
