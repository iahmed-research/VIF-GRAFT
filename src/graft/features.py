"""Waveform descriptors and fitted residual-head primitives used by B."""
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
CHEST={'V','V1','V2','V3','V4','V5','V6','MCL'}
CHANNEL_NAMES=['present','valid_fraction','clipped_fraction','log_rms','low_power_fraction','high_power_fraction','spectral_entropy','dominant_frequency','autocorrelation_peak','autocorrelation_lag','kurtosis','late_early_log_energy_ratio']
PAIR_NAMES=['absolute_correlation','absolute_derivative_correlation','difference_energy_ratio']

def gross_usable(x):
    valid=np.isfinite(x)&(x!=-32768)
    pairs=valid[1:]&valid[:-1]
    return bool(valid.mean()>=.9 and valid.any() and np.ptp(x[valid])>0 and
        pairs.any() and np.mean(x[1:][pairs]==x[:-1][pairs])<.999)


def choose_chest(names,raw,selected):
    # Fixed header order among unused grossly usable chest-region channels.
    return next((i for i,n in enumerate(names) if n.upper() in CHEST and i not in selected and gross_usable(raw[:,i])),None)


def read_raw_window(hp,start=72500,stop=75000):
    lines=[l.strip() for l in hp.read_text().splitlines() if l.strip() and not l.startswith('#')]
    h=lines[0].split(); count=int(h[1]); fs=float(h[2]); length=int(h[3])
    parts=[l.split() for l in lines[1:1+count]]
    if fs!=250 or stop>75000 or start<0 or stop-start!=2500 or length<stop:raise ValueError('Invalid causal window/header')
    if len(parts)!=count or len({p[0] for p in parts})!=1 or any(p[1]!='16' for p in parts):raise ValueError('Unsupported WFDB layout')
    dp=(hp.parent/parts[0][0]).resolve()
    if dp.parent!=hp.parent.resolve() or dp.stat().st_size!=length*count*2:raise ValueError('Raw data layout differs')
    mm=np.memmap(dp,dtype='<i2',mode='r',shape=(length,count))
    raw=np.array(mm[start:stop],dtype=np.float64); del mm
    return raw,parts


def channel_features(x,valid,present):
    if not present:return np.zeros(len(CHANNEL_NAMES))
    good=np.asarray(valid,bool)&np.isfinite(x); v=np.where(good,x,0.).astype(float)
    v=v-(v[good].mean() if good.any() else 0.); v[~good]=0
    energy=float(np.mean(v*v)); spectrum=abs(np.fft.rfft(v*np.hanning(len(v))))**2
    hz=np.fft.rfftfreq(len(v),1/250); band=(hz>=1)&(hz<=30)
    power=spectrum[band]; total=max(float(power.sum()),1e-12); distribution=power/total
    ac=np.fft.irfft(abs(np.fft.rfft(v,n=8192))**2,n=8192)[:len(v)]
    ac=ac/max(float(ac[0]),1e-12); lo,hi=50,376
    peak=int(np.argmax(ac[lo:hi]))+lo
    peak_band=(hz>=1)&(hz<=8)
    result=np.array([1.,good.mean(),float(np.mean(np.abs(x[good])>=7.999)) if good.any() else 0.,
        np.log1p(np.sqrt(energy)),spectrum[(hz>=1)&(hz<5)].sum()/total,
        spectrum[(hz>=15)&(hz<=30)].sum()/total,
        -np.sum(distribution*np.log(np.maximum(distribution,1e-15)))/np.log(max(len(power),2)),
        hz[peak_band][np.argmax(spectrum[peak_band])] if energy>1e-12 else 0.,
        ac[peak] if energy>1e-12 else 0.,peak/250 if energy>1e-12 else 0.,
        min(float(np.mean(v**4)/max(energy**2,1e-12)),100.),
        np.clip(np.log((np.mean(v[1250:]**2)+1e-8)/(np.mean(v[:1250]**2)+1e-8)),-8,8)])
    if not np.isfinite(result).all():raise ValueError('Nonfinite channel features')
    return result


def pair_features(x,y,valid):
    mask=np.asarray(valid,bool)
    if mask.sum()<100:return np.zeros(3)
    def corr(a,b):
        a=a-a.mean();b=b-b.mean();return float(abs(a@b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-12))
    a=x[mask].astype(float); b=y[mask].astype(float)
    pairs=mask[1:]&mask[:-1]
    dc=corr(np.diff(x)[pairs],np.diff(y)[pairs]) if pairs.sum()>=100 else 0.
    a=(a-a.mean())/max(a.std(),1e-8);b=(b-b.mean())/max(b.std(),1e-8)
    # Polarity-invariant difference; do not treat different lead polarity as conflict.
    return np.array([corr(a,b),dc,min(np.mean((a-b)**2),np.mean((a+b)**2))/2])


def feature_arms(signals,valid,chest,chest_valid,available):
    features=[channel_features(signals[i],valid[i],bool(valid[i].any()) if i<2 else bool(available[i-1])) for i in range(4)]
    pair=pair_features(signals[0],signals[1],valid[0]&valid[1])
    a=np.concatenate(features+[pair])
    if chest is None:
        return a,a.copy(),np.r_[a,np.zeros(len(CHANNEL_NAMES)+6)]
    cf=channel_features(chest,chest_valid,True)
    p0=pair_features(signals[0],chest,valid[0]&chest_valid)
    p1=pair_features(signals[1],chest,valid[1]&chest_valid)
    return a,np.concatenate([features[0],cf,features[2],features[3],p0]),np.r_[a,cf,p0,p1]


def weights(records):
    _,inv,counts=np.unique(records,return_inverse=True,return_counts=True)
    w=1/counts[inv];return w/w.sum()


def logloss(y,p,records):
    p=np.clip(p,1e-15,1-1e-15)
    return float(weights(records)@(-(y*np.log(p)+(1-y)*np.log1p(-p))))


def design_fit(x,records):
    w=weights(records);mean=w@x;scale=np.sqrt(w@((x-mean)**2));scale=np.maximum(scale,.001)
    return mean,scale


def design(x,mean,scale):return np.c_[np.ones(len(x)),np.clip((x-mean)/scale,-8,8)]


def objective(theta,z,y,w,offset,eligible,alpha,bounded):
    raw=z@theta
    correction=np.tanh(raw) if bounded else raw
    derivative=1-correction**2 if bounded else np.ones_like(raw)
    linear=offset+eligible*correction
    penalty=np.r_[.01,np.ones(len(theta)-1)]
    value=float(w@(np.logaddexp(0,linear)-y*linear)+alpha/2*np.sum(penalty*theta**2))
    gradient=z.T@(w*(expit(linear)-y)*eligible*derivative)+alpha*penalty*theta
    return value,gradient


def fit(x,y,records,anchor,eligible,alpha,bounded,maxiter):
    mean,scale=design_fit(x,records);z=design(x,mean,scale)
    offset=np.log(np.clip(anchor,1e-6,1-1e-6))-np.log1p(-np.clip(anchor,1e-6,1-1e-6)) if bounded else np.zeros(len(y))
    result=minimize(objective,np.zeros(z.shape[1]),args=(z,y,weights(records),offset,eligible,alpha,bounded),
        method='L-BFGS-B',jac=True,options={'maxiter':maxiter,'ftol':1e-10,'gtol':1e-6})
    if not result.success or not np.isfinite(result.fun):raise RuntimeError('Compact fit did not converge: '+str(result.message))
    return {'mean':mean,'scale':scale,'theta':result.x,'iterations':int(result.nit)}


def predict(model,x,anchor,eligible,bounded):
    raw=design(x,model['mean'],model['scale'])@model['theta']
    if not bounded:return expit(raw)
    pc=np.clip(anchor,1e-6,1-1e-6); offset=np.log(pc)-np.log1p(-pc)
    p=np.clip(anchor+(expit(offset+np.tanh(raw))-expit(offset)),0,1)
    return np.where(eligible,p,anchor)
