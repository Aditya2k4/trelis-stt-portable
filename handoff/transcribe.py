"""Portable offline Trelis transcription on Intel GPU; no machine-specific paths."""
import argparse,collections,copy,hashlib,json,os,shutil,subprocess,sys,time,wave
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REVISION='eab1188fd2d0e91f2584229b32b3bfe1901c896c'
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',
                  OPENVINO_TELEMETRY_DISABLED='1',NNCF_TELEMETRY_DISABLED='1',TOKENIZERS_PARALLELISM='false')
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(4*1024*1024),b''):h.update(part)
    return h.hexdigest()
def write(path,data):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)
def stamp(seconds):
    n=int(seconds);return f'{n//3600:02}:{n//60%60:02}:{n%60:02}'
def repetitive(text):
    w=text.casefold().split();counts=collections.Counter(tuple(w[i:i+3]) for i in range(len(w)-2))
    return '\ufffd' in text or max(counts.values(),default=0)>=5

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('files',type=Path,nargs='+')
    p.add_argument('--output',type=Path,default=ROOT/'transcripts')
    p.add_argument('--language',choices=['en','hi'],default='en')
    p.add_argument('--mixed-code',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--batch-size',type=int,choices=range(1,9),default=1)
    p.add_argument('--ffmpeg',default='ffmpeg')
    p.add_argument('--model-dir',type=Path,default=ROOT/'models/trelis')
    p.add_argument('--ov-dir',type=Path,default=ROOT/'models/trelis-openvino-fp16')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    ffmpeg=shutil.which(a.ffmpeg)
    if not ffmpeg:raise SystemExit('FFmpeg not found. Install FFmpeg or pass --ffmpeg path/to/ffmpeg.exe.')
    if not (a.ov_dir/'openvino_decoder_model.bin').exists():raise SystemExit('Model weights missing. Extract the FULL package, not only the KIT.')
    import numpy as np
    settings={'model_revision':REVISION,'backend':'OpenVINO GPU fp16','language':a.language,'mixed_code':a.mixed_code,
              'decoding':'greedy; max_new_tokens=440; explicit prefix; no aligned timestamps'}
    work=a.output/'audio-cache';work.mkdir(exist_ok=True)
    jobs=[];seen_source={};seen_pcm={};duplicates=[]
    for source in a.files:
        source=source.resolve()
        if not source.is_file():raise FileNotFoundError(source)
        digest=sha(source)
        if digest in seen_source:
            duplicates.append({'source':str(source),'duplicate_of':seen_source[digest],'evidence':'identical source SHA256'});continue
        seen_source[digest]=str(source)
        wav=work/(digest+'.wav')
        if not wav.exists():
            temp=work/(digest+'.tmp.wav')
            subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-y','-i',str(source),'-vn','-ac','1','-ar','16000','-c:a','pcm_s16le',str(temp)],check=True)
            temp.replace(wav)
        with wave.open(str(wav),'rb') as f:
            assert (f.getnchannels(),f.getsampwidth(),f.getframerate())==(1,2,16000)
            duration=f.getnframes()/16000;pcm=hashlib.sha256()
            while part:=f.readframes(16000*30):pcm.update(part)
        if duration<=0:raise ValueError('Empty recording: '+str(source))
        if pcm.hexdigest() in seen_pcm:
            duplicates.append({'source':str(source),'duplicate_of':seen_pcm[pcm.hexdigest()],'evidence':'identical normalized PCM SHA256'});continue
        seen_pcm[pcm.hexdigest()]=str(source)
        base=a.output/(source.stem+' - '+digest[:8])
        report_path=Path(str(base)+'.json')
        report=json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {
            'source':str(source),'source_sha256':digest,'duration_seconds':duration,'settings':settings,
            'timestamp_type':'approximate original audio-window boundaries; not word-aligned',
            'segments':[],'complete':False}
        if report['source_sha256']!=digest or report['settings']!=settings:raise ValueError('Existing report uses different settings. Choose another output folder.')
        jobs.append({'source':source,'wav':wav,'duration':duration,'base':base,'report_path':report_path,'report':report})
    write(a.output/'duplicate-check.json',duplicates)
    def export(j):
        r=j['report'];write(j['report_path'],r)
        lines=[j['source'].name,'Automatic Trelis transcript; approximate original-recording timestamps.',
               'Names and unclear speech may be misrecognized.','']
        for s in r['segments']:lines += [f"[{stamp(s['start'])}–{stamp(s['end'])}]",s['text'].strip() or '[No intelligible speech detected]','']
        Path(str(j['base'])+'.txt').write_text('\n'.join(lines),encoding='utf-8')
    def audio(j,start,end):
        with wave.open(str(j['wav']),'rb') as f:
            f.setpos(round(start*16000));raw=f.readframes(round(end*16000)-round(start*16000))
        return np.frombuffer(raw,dtype='<i2').astype(np.float32)/32768
    def boundary(j,start):
        if j['duration']-start<=23:return j['duration']
        samples=audio(j,start+18,start+22)
        energy=np.mean(samples[:64000].reshape(40,1600)**2,axis=1)
        return start+18+int(np.argmin(energy))*.1+.05
    if all(j['report']['complete'] for j in jobs):
        print('All inputs already complete, or duplicates.');return
    import torch,openvino as ov
    from transformers import AutoConfig,WhisperProcessor
    from optimum.intel.openvino import OVModelForSpeechSeq2Seq
    torch.set_num_threads(4)
    if not any(str(d).startswith('GPU') for d in ov.Core().available_devices):raise SystemExit('Intel OpenVINO GPU unavailable. Check the Intel graphics driver. No CPU fallback was used.')
    cache=ROOT/'.cache/openvino';cache.mkdir(parents=True,exist_ok=True)
    print('Loading Trelis on Intel GPU...',flush=True)
    proc=WhisperProcessor.from_pretrained(str(a.model_dir),local_files_only=True)
    model=OVModelForSpeechSeq2Seq._from_pretrained(str(a.ov_dir),
        config=AutoConfig.from_pretrained(str(a.ov_dir),local_files_only=True),device='GPU',compile=False,
        local_files_only=True,trust_remote_code=False,load_in_8bit=False,
        ov_config={'PERFORMANCE_HINT':'LATENCY','INFERENCE_PRECISION_HINT':'f16','CACHE_DIR':str(cache)})
    model.compile()
    actual={}
    for name,component in model.components.items():
        request=component.request;compiled=request.get_compiled_model() if hasattr(request,'get_compiled_model') else request
        devices=[str(d) for d in compiled.get_property('EXECUTION_DEVICES')]
        if not devices or any(not d.startswith('GPU') for d in devices):raise RuntimeError('GPU execution not confirmed: '+repr(devices))
        actual[name]=devices
    print('Execution devices: '+json.dumps(actual),flush=True)
    vocab=proc.tokenizer.get_vocab();prefix=[model.config.decoder_start_token_id,vocab[f'<|{a.language}|>']]
    if a.mixed_code:prefix+=proc.tokenizer('<|mixedcode|>',add_special_tokens=False).input_ids
    prefix += [vocab['<|transcribe|>'],vocab['<|notimestamps|>']]
    generation=copy.deepcopy(model.generation_config);generation.forced_decoder_ids=None;generation.language=None
    def generate(arrays):
        features=proc(arrays,sampling_rate=16000,return_tensors='pt',return_attention_mask=True)
        with torch.inference_mode():
            output=model.generate(input_features=features.input_features,attention_mask=features.attention_mask,
                generation_config=generation,decoder_input_ids=torch.tensor([prefix]*len(arrays)),language=a.language,
                task='transcribe',return_timestamps=False,do_sample=False,num_beams=1,max_new_tokens=min(440,448-len(prefix)))
        texts=[]
        for item in output:
            ids=item.tolist()
            if ids[:len(prefix)]==prefix:ids=ids[len(prefix):]
            texts.append(proc.tokenizer.decode(ids,skip_special_tokens=True).strip())
        assert len(texts)==len(arrays)
        return texts
    while pending:=[j for j in jobs if not j['report']['complete']]:
        planned={str(j['source']):(j['report']['segments'][-1]['end'] if j['report']['segments'] else 0) for j in pending}
        tasks=[]
        for _ in range(a.batch_size):
            available=[j for j in pending if planned[str(j['source'])]<j['duration']-.0001]
            if not available:break
            j=max(available,key=lambda x:x['duration']-planned[str(x['source'])])
            start=planned[str(j['source'])];end=boundary(j,start);planned[str(j['source'])]=end
            tasks.append((j,start,end))
        if not tasks:raise RuntimeError('Unexpected empty pending batch; preserve reports for inspection')
        texts=generate([audio(j,s,e) for j,s,e in tasks])
        for (j,start,end),text in zip(tasks,texts):
            r=j['report'];rows=[]
            if repetitive(text):
                r.setdefault('rechecks',[]).append({'start':start,'end':end,'raw_text':text})
                cursor=start
                while cursor<end:
                    stop=min(cursor+10,end);retry=generate([audio(j,cursor,stop)])[0];bad=repetitive(retry)
                    rows.append({'start':cursor,'end':stop,'text':'[Unclear speech; repeated model output could not be resolved.]' if bad else retry,'uncertain':bad})
                    cursor=stop
            else:rows=[{'start':start,'end':end,'text':text}]
            r['segments'].extend(rows);r['complete']=abs(end-j['duration'])<.001;export(j)
            print(f"{j['source'].name}: {stamp(end)}/{stamp(j['duration'])}"+(' COMPLETE' if r['complete'] else ''),flush=True)
    for j in jobs:
        rows=j['report']['segments']
        assert rows and rows[0]['start']==0 and abs(rows[-1]['end']-j['duration'])<.001
        assert all(abs(x['end']-y['start'])<.001 for x,y in zip(rows,rows[1:]))
    print('All unique recordings complete; full audio-window coverage checked.',flush=True)

if __name__=='__main__':main()
