"""Build every waveform tensor and descriptor directly from VTaC 1.0."""
from pathlib import Path
import csv
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import numpy as np
from . import io,features
from .preprocessing import preprocess_channel,select_channels

ROLES={'ecg':{'I','II','III','AVR','AVL','AVF'}|features.CHEST,'ppg':{'PLETH'},'abp':{'ABP'}}
URL='https://physionet.org/files/vtac/1.0/'


def published_hashes(raw):
    result={}
    for line in (Path(raw)/'SHA256SUMS.txt').read_text().splitlines():
        if not line.strip():continue
        h,name=line.split(maxsplit=1);result[name.lstrip('*').removeprefix('./')]=h
    return result


def download(raw,rows,workers=4):
    if not isinstance(workers,int) or not 1<=workers<=8:raise ValueError('Download workers must be between 1 and 8')
    raw=Path(raw);raw.mkdir(parents=True,exist_ok=True)
    locks={};guard=Lock()
    def fetch(relative,expected=None):
        # A shared waveform referenced by multiple headers must have one writer.
        with guard:lock=locks.setdefault(relative,Lock())
        with lock:fetch_locked(relative,expected)
    def fetch_locked(relative,expected=None):
        p=raw/relative
        if p.exists():
            if expected and io.digest(p)!=expected:raise ValueError('Existing data checksum differs: '+relative)
            return
        p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.download')
        with urllib.request.urlopen(URL+relative,timeout=90) as source,temp.open('wb') as target:
            while block:=source.read(1024*1024):target.write(block)
        if expected and io.digest(temp)!=expected:raise ValueError('Downloaded checksum differs: '+relative)
        temp.replace(p)
    fetch('SHA256SUMS.txt');hashes=published_hashes(raw)
    for name in ('LICENSE.txt','benchmark_data_split.csv','event_labels.csv'):fetch(name,hashes.get(name))
    def event(r):
        stem=f"waveforms/{r['record']}/{r['event']}"
        fetch(stem+'.hea',hashes[stem+'.hea'])
        parts=(raw/(stem+'.hea')).read_text().splitlines()
        filename=parts[1].split()[0]
        if Path(filename).name!=filename:raise ValueError('Unexpected waveform path')
        relative=f"waveforms/{r['record']}/{filename}"
        fetch(relative,hashes[relative])
    print(f'Checking/downloading {len(rows)} selected events with {workers} workers',flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending=[pool.submit(event,r) for r in rows]
        try:
            for number,future in enumerate(as_completed(pending),1):
                future.result()
                if number%25==0 or number==len(rows):print(f'Verified {number}/{len(rows)} selected events',flush=True)
        except BaseException:
            for future in pending:future.cancel()
            raise


def labels(raw,rows,split):
    allowed={r['event']:r for r in rows if r['split']==split}
    found={}
    with (Path(raw)/'event_labels.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if r['event'] not in allowed:continue
            if r['record']!=allowed[r['event']]['record'] or r['event'] in found:raise ValueError('Label identity differs')
            value=r['decision'].lower()
            if value not in ('true','false','1','0'):raise ValueError('Non-binary label')
            found[r['event']]=int(value in ('true','1'))
    if set(found)!=set(allowed):raise ValueError('Missing labels')
    return np.array([found[r['event']] for r in rows if r['split']==split])


def extract(raw_root,row,config):
    hp=Path(raw_root)/'waveforms'/row['record']/(row['event']+'.hea')
    raw,parts=features.read_raw_window(hp)
    channel_rows=[];scales=[]
    for i,p in enumerate(parts):
        name=p[-1].upper();role=next((role for role,names in ROLES.items() if name in names),'other')
        m=re.fullmatch(r'([0-9.eE+-]+)(?:\(([0-9+-]+)\))?/(.+)',p[2])
        if not m:raise ValueError('Unsupported WFDB gain')
        gain=float(m[1]);baseline=float(m[2]) if m[2] else float(p[4])
        if not np.isfinite(gain) or gain<=0:raise ValueError('Invalid WFDB gain')
        scales.append((baseline,gain))
        valid=raw[:,i]!=-32768;pairs=valid[:-1]&valid[1:]
        invalid_fraction=float(1-valid.mean())
        usable=invalid_fraction<=.10 and valid.sum()>=2 and np.ptp(raw[valid,i])>0 and pairs.any() and np.mean(raw[1:,i][pairs]==raw[:-1,i][pairs])<.999
        channel_rows.append({'channel_index':str(i),'signal_role':role,'provisional_usable':str(bool(usable)),'signal_name':name,'gain':str(gain),'baseline':str(baseline)})
    selected=select_channels(channel_rows,2)
    signals=np.zeros((4,2500),np.float32);valid=np.zeros_like(signals,bool);indices=np.full(4,-1,np.int16)
    available=np.array([True,bool(selected['ppg']),bool(selected['abp'])])
    def process(i,role):
        baseline,gain=scales[i]
        return preprocess_channel(raw[:,i],baseline=baseline,gain=gain,role=role,sampling_frequency_hz=250,config=config['preprocessing'])
    for role,slots in [('ecg',(0,1)),('ppg',(2,)),('abp',(3,))]:
        for slot,i in zip(slots,selected[role]):
            processed=process(i,role);signals[slot]=processed.tensor;valid[slot]=processed.model_valid_mask;indices[slot]=i
    chest=features.choose_chest([p[-1] for p in parts],raw,selected['ecg'])
    p=process(chest,'ecg') if chest is not None else None
    eligible=p is not None and bool(p.model_valid_mask.any())
    a,b,_=features.feature_arms(signals,valid,p.tensor if eligible else None,p.model_valid_mask if eligible else None,available)
    physical=np.zeros(2500);chest_valid=np.zeros(2500,bool)
    if chest is not None:
        baseline,gain=scales[chest];physical=(raw[:,chest]-baseline)/gain;chest_valid=np.isfinite(raw[:,chest])&(raw[:,chest]!=-32768)
    return {'signals':signals,'available':available,'model_valid_mask':valid,'selected_channel_indices':indices,
        'A':a,'B':b,'eligible':np.array(eligible),'chest_index':np.array(-1 if chest is None else chest),
        'chest_physical':physical,'chest_valid':chest_valid},channel_rows


def prepare(raw,rows,config,out):
    raw=Path(raw).resolve();folder=Path(out)/'data';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'COMPLETE.json').exists():
        io.verify(folder)
        prior=io.read(folder/'RAW_MANIFEST.json')
        for relative,h in prior.items():
            if io.digest(raw/relative)!=h:raise ValueError('Raw source changed: '+relative)
        return io.load_npz(folder/'features.npz')
    published=published_hashes(raw)
    official={r['event']:r for r in io.csv_rows(raw/'benchmark_data_split.csv')}
    manifest={};combined={k:[] for k in ('A','B','eligible','chest_index')};channels=[]
    for i,row in enumerate(rows):
        if official[row['event']]['record']!=row['record'] or official[row['event']]['split']!=row['split']:raise ValueError('Official split differs')
        hp=Path('waveforms')/row['record']/(row['event']+'.hea')
        filename=(raw/hp).read_text().splitlines()[1].split()[0]
        if Path(filename).name!=filename:raise ValueError('Unsafe waveform filename')
        for p in (hp,hp.parent/filename):
            relative=p.as_posix();h=io.digest(raw/p)
            if published.get(relative)!=h:raise ValueError('Dataset source checksum differs: '+relative)
            manifest[relative]=h
        x,channel=extract(raw,row,config)
        io.save_npz(folder/'waveforms'/f"{row['event']}.npz",**x)
        for key in combined:combined[key].append(x[key])
        channels.extend([{**c,'event':row['event'],'record':row['record']} for c in channel])
        if (i+1)%250==0:print(f'Prepared {i+1}/{len(rows)} raw events',flush=True)
    data={k:np.asarray(v) for k,v in combined.items()}
    data.update(event=np.array([r['event'] for r in rows]),record=np.array([r['record'] for r in rows]),fold=np.array([r['fold'] for r in rows]),split=np.array([r['split'] for r in rows]))
    for name in ('SHA256SUMS.txt','benchmark_data_split.csv','event_labels.csv'):
        manifest[name]=io.digest(raw/name)
        if name in published and manifest[name]!=published[name]:raise ValueError('Metadata checksum differs')
    io.save_npz(folder/'features.npz',**data);io.write(folder/'RAW_MANIFEST.json',manifest);io.write(folder/'channels.json',channels)
    io.finish(folder,{'events':len(rows),'outcome_arrays_created':False})
    return data
