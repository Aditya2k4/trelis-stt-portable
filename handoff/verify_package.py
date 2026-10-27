"""Verify every included file, including all model weights, against packaged SHA256."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'SHA256SUMS.json').read_text(encoding='utf-8'))
problems=[]
for name,item in manifest.items():
    path=ROOT/name
    if not path.is_file():problems.append('MISSING '+name);continue
    if path.stat().st_size!=item['bytes']:problems.append('SIZE '+name);continue
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(4*1024*1024),b''):digest.update(data)
    if digest.hexdigest()!=item['sha256']:problems.append('HASH '+name)
print(json.dumps({'ok':not problems,'checked_files':len(manifest),'problems':problems},indent=2))
sys.exit(bool(problems))
