"""Create a source-only transfer archive; never include data, weights or credentials."""
from pathlib import Path
import hashlib
import json
import re
import zipfile

root=Path(__file__).resolve().parents[1]
directories=('src','tests','configs','docs','reference','scripts','.github')
files=[root/name for name in ('README.md','pyproject.toml','Dockerfile','.dockerignore','.gitignore','.gitattributes','LICENSE','LICENSE-DATA.txt')]
for name in directories:
    files.extend(p for p in (root/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts and not any(part.endswith('.egg-info') for part in p.parts))
files=sorted(set(files))
for path in files:
    if path.suffix in ('.pt','.npz','.dat','.hea','.log') or path.name=='.env':raise ValueError('Forbidden release file: '+str(path))
    if path.suffix in ('.py','.json','.md','.toml','.yml','.ps1','.sh'):
        text=path.read_text(encoding='utf-8')
        if re.search(r'(?i)(?:[a-z]:[\\/](?:users|vif_local_\d+)[\\/]|/home/[a-z0-9_.-]+/)',text):raise ValueError('Workstation path in release: '+str(path))
destination=root/'dist';destination.mkdir(exist_ok=True)
archive=destination/'graft-reproducibility-source.zip'
manifest={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in files:z.write(p,'graft-reproducibility/'+p.relative_to(root).as_posix())
    z.writestr('graft-reproducibility/SOURCE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
h=hashlib.sha256(archive.read_bytes()).hexdigest()
(destination/'SHA256SUMS.txt').write_text(f'{h}  {archive.name}\n')
print(json.dumps({'archive':archive.name,'files':len(files),'bytes':archive.stat().st_size,'sha256':h},indent=2))
