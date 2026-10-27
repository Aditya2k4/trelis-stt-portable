import ctypes, hashlib, json, re, unicodedata
from pathlib import Path

ROOT=Path(__file__).resolve().parent
CONFIG=json.loads((ROOT/'experiment.json').read_text(encoding='utf-8'))
CONFIG['model_path']=str((ROOT/CONFIG['model_path']).resolve())

def free_ram_gib():
    import psutil
    return psutil.virtual_memory().available/2**30


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def manifest(path):
    path=Path(path).resolve()
    rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    for row in rows:
        audio=Path(row['audio'])
        row['audio']=str(audio if audio.is_absolute() else path.parent/audio)
    if len({r['id'] for r in rows})!=len(rows): raise ValueError('Duplicate clip IDs')
    return rows

def validate_splits(rows, allow_publisher_labels=False):
    groups={};hashes={}
    for row in rows:
        split=row['split']
        if split not in ('train','validation','test'): raise ValueError('Assign train/validation/test explicitly')
        publisher_ok=(allow_publisher_labels and split in ('train','validation') and
                      row.get('label_origin')=='publisher' and row.get('annotation_status')=='publisher_unverified' and
                      bool(row.get('source_license',{}).get('source_url')))
        if (not row.get('reviewed') and not publisher_ok) or not row.get('text','').strip(): raise ValueError('Human-reviewed transcripts required unless an explicitly documented publisher-label pilot is requested')
        if not row.get('license_verified'): raise ValueError('Verify source permission/license before admitting data')
        group=row.get('group_id')
        if not group: raise ValueError('Speaker/recording group is required')
        if group in groups and groups[group]!=split: raise ValueError('Group leakage across splits')
        groups[group]=split
        digest=hashlib.sha256(Path(row['audio']).read_bytes()).hexdigest()
        if row.get('audio_sha256') and row['audio_sha256']!=digest: raise ValueError('Audio changed after review')
        if digest in hashes and hashes[digest]!=split: raise ValueError('Duplicate audio across splits')
        hashes[digest]=split
        audio_array(row['audio'])
    if {r['split'] for r in rows}!={'train','validation','test'}: raise ValueError('All three independent splits are required')

def normalize(text):
    text=unicodedata.normalize('NFC',text).casefold()
    # Preserve Hindi matras/nuktas and all combining marks. Do not transliterate.
    return ' '.join(''.join(' ' if unicodedata.category(c)[0] in ('P','S') else c for c in text).split())

def errors(reference,hypothesis):
    ref=normalize(reference).split();hyp=normalize(hypothesis).split()
    previous=list(range(len(hyp)+1))
    for i,a in enumerate(ref,1):
        current=[i]
        for j,b in enumerate(hyp,1):current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(a!=b)))
        previous=current
    return previous[-1],len(ref)

def prefix(tokenizer):
    vocab=tokenizer.get_vocab()
    parts=['<|startoftranscript|>','<|hi|>','<|transcribe|>','<|notimestamps|>']
    if not all(p in vocab for p in parts):raise ValueError('Unexpected tokenizer')
    mixed=tokenizer('<|mixedcode|>',add_special_tokens=False).input_ids
    if not mixed:raise ValueError('Missing mixed-code marker')
    return [vocab[parts[0]],vocab[parts[1]],*mixed,vocab[parts[2]],vocab[parts[3]]]

def audio_array(path):
    import numpy as np, soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd
    a,sr=sf.read(path,dtype='float32',always_2d=True);a=a.mean(axis=1)
    if not len(a) or len(a)/sr>CONFIG['max_audio_seconds']:raise ValueError('Clip must be nonempty and at most 30 seconds; segment with aligned text')
    if sr!=16000:
        g=gcd(sr,16000);a=resample_poly(a,16000//g,sr//g).astype(np.float32)
    return a
