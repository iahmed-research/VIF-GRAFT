import pytest
import torch
from graft import training
from graft.models import EarlyConcatenationModel


def test_missing_eta_matches_original_adapter_and_backpropagates():
    torch.set_num_threads(2);torch.manual_seed(17)
    model=EarlyConcatenationModel().eval()
    signal=torch.randn(4,4,128)
    available=torch.tensor([[1,1,1],[1,1,0],[1,0,1],[1,0,0]],dtype=torch.bool)
    events=['both','ppg','abp','neither']
    quality={e:dict(r_ppg=.6 if a[1] else 0.,r_abp=.7 if a[2] else 0.,eta=.2 if i==0 else None) for i,(e,a) in enumerate(zip(events,available))}
    actual=training.forward(model,'early_concatenation',signal,available,events,quality,'cpu')
    expected=model(signal,available,torch.tensor([.6,.6,0.,0.]),torch.tensor([.7,0.,.7,0.]),torch.tensor([.2,0.,0.,0.]))
    assert torch.equal(actual['probability'],expected['probability'])
    actual['probability'].sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())


def test_missing_eta_for_both_pulses_is_not_silently_imputed():
    with pytest.raises(ValueError,match='Missing eta for both-pulse'):
        training.forward(None,'early_concatenation',torch.zeros(1,4,128),torch.ones(1,3,dtype=torch.bool),['e'],{'e':{'r_ppg':.5,'r_abp':.5,'eta':None}},'cpu')


def test_qmf_ignores_quality_coordinates():
    class Probe:
        def __call__(self,signal,available):return signal
    signal=torch.randn(1,4,128)
    assert training.forward(Probe(),'QMF',signal,None,['e'],None,'cpu') is signal
