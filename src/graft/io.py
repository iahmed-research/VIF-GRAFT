"""Portable paths and content-bound, resumable artifact storage."""
from pathlib import Path
import csv
import hashlib
import json
import platform
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False,default=serial)+'\n',encoding='utf-8')
    temp.replace(path)


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def cohort(path):
    rows=csv_rows(path)
    if not rows or len({r['event'] for r in rows})!=len(rows):raise ValueError('Duplicate/empty cohort')
    groups={}
    for r in rows:
        r['fold']=int(r['fold'])
        if r['split'] not in ('train','val') or (r['split']=='train' and r['fold'] not in range(5)) or (r['split']=='val' and r['fold']!=-1):raise ValueError('Invalid cohort role')
        if any(not x or Path(x).name!=x or '/' in x or '\\' in x or x in ('.','..') for x in (r['event'],r['record'])):raise ValueError('Unsafe identifier')
        role=(r['split'],r['fold'])
        if r['record'] in groups and groups[r['record']]!=role:raise ValueError('Record crosses roles')
        groups[r['record']]=role
    return sorted(rows,key=lambda r:(r['split']!='train',r['event']))


def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def save_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,**arrays)
    temp.replace(path)


def finish(folder,metadata=None):
    folder=Path(folder)
    files={p.relative_to(folder).as_posix():digest(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name not in ('COMPLETE.json','RUN.log') and not p.name.endswith('.tmp')}
    write(folder/'COMPLETE.json',{'status':'complete','files':files,**(metadata or {})})


def verify(folder):
    folder=Path(folder);r=read(folder/'COMPLETE.json')
    for name,h in r['files'].items():
        p=(folder/name).resolve()
        if not p.is_relative_to(folder.resolve()) or digest(p)!=h:raise ValueError('Artifact changed: '+name)
    return r


def environment():
    import scipy, sklearn, torch
    return {'python':platform.python_version(),'platform':platform.platform(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def bind_run(out,config,cohort_path,device):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    source={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}
    context={'config':config,'cohort_sha256':digest(cohort_path),'source':source,'environment':environment(),'device':device}
    path=out/'BINDING.json'
    if path.exists() and read(path)!=context:raise ValueError('Run settings/source/environment changed; use a new output directory')
    write(path,context)
    return canonical(context)
