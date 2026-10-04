"""Fixed raw-chest perturbations; preprocessing and head remain frozen."""
import hashlib
import numpy as np
from . import features as cp
from .preprocessing import preprocess_physical_channel
SEEDS=(20260927,20260928,20260929)
CONDITIONS=('clean','missing','noise_20dB','noise_10dB','noise_0dB','gap_0.1s','gap_0.5s','gap_2s')

def rng_for(event,seed):
    digest=hashlib.sha256((str(seed)+'|'+str(event)).encode()).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(digest[:8],'little')))


def perturb(physical,valid,condition,event,seed):
    x=np.asarray(physical,float).copy(); valid=np.asarray(valid,bool).copy()
    rng=rng_for(event,seed)
    if condition=='clean': return x,valid
    if condition=='missing': return x,np.zeros_like(valid)
    if condition.startswith('noise_'):
        db=float(condition[6:-2]); ix=np.flatnonzero(valid)
        if len(ix):
            rms=np.sqrt(np.mean((x[ix]-x[ix].mean())**2))
            noise=rng.normal(size=len(ix)); noise-=noise.mean()
            noise/=max(np.sqrt(np.mean(noise**2)),1e-15)
            x[ix]+=noise*rms*10**(-db/20)
    elif condition.startswith('gap_'):
        width=round(float(condition[4:-1])*250)
        start=int(rng.integers(0,len(x)-500+1)); valid[start:start+width]=False
    else: raise ValueError('Unknown condition')
    return x,valid


def usable(x,valid):
    good=valid & np.isfinite(x); pairs=good[1:] & good[:-1]
    return bool(good.mean()>=.9 and good.any() and np.ptp(x[good])>0 and pairs.any()
                and np.mean(x[1:][pairs]==x[:-1][pairs])<.999)


def stressed_features(base,ecg,ecg_valid,physical,valid,condition,event,seed,preprocessing):
    x,mask=perturb(physical,valid,condition,event,seed)
    if not usable(x,mask): return base.copy(),False
    processed=preprocess_physical_channel(x,mask,role='ecg',sampling_frequency_hz=250,config=preprocessing)
    if not processed.model_valid_mask.any(): return base.copy(),False
    b=base.copy()
    b[12:24]=cp.channel_features(processed.tensor,processed.model_valid_mask,True)
    b[48:51]=cp.pair_features(ecg,processed.tensor,ecg_valid & processed.model_valid_mask)
    return b,True
