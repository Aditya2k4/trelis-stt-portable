"""Combine included public pilot data with separately supplied reviewed test clips."""
import argparse,json,os
from pathlib import Path
from common import ROOT,manifest,validate_splits
p=argparse.ArgumentParser();p.add_argument('--test',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
a=p.parse_args();rows=manifest(ROOT/'data/public-manifest.jsonl');test=manifest(a.test)
if not test or any(r.get('split')!='test' for r in test):raise SystemExit('Provide only reviewed test rows in --test; never move them into training.')
rows+=test;validate_splits(rows,allow_publisher_labels=True)
if a.output.exists():raise SystemExit('Output already exists; choose another path.')
a.output.parent.mkdir(parents=True,exist_ok=True)
for row in rows:row['audio']=os.path.relpath(row['audio'],a.output.parent.resolve())
a.output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
print(f'Saved {len(rows)} rows with independent train, validation, and reviewed test splits.')
