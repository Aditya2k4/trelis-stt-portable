"""Same-process, same-precision baseline/LoRA comparison with fixed decoding."""
import argparse,contextlib,copy,hashlib,json,os,time
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration,WhisperProcessor
from transformers.modeling_outputs import BaseModelOutput
from common import CONFIG,manifest,prefix,audio_array,errors,free_ram_gib,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--adapter',required=True)
    p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
    if out.exists():raise SystemExit('Use a new paired evaluation output file')
    if free_ram_gib()<CONFIG['minimum_free_ram_gib']:raise SystemExit('Free at least 10 GiB RAM before evaluation')
    if not torch.xpu.is_available():raise SystemExit('Intel GPU unavailable')
    rows=manifest(a.manifest);test=[r for r in rows if r['split']=='test'];validation=[r for r in rows if r['split']=='validation']
    if not test or any(not r.get('reviewed') for r in test):raise SystemExit('Human-reviewed test references required')
    for r in rows:
        if hashlib.sha256(Path(r['audio']).read_bytes()).hexdigest()!=r['audio_sha256']:raise ValueError('Audio changed')
    torch.set_num_threads(4)
    proc=WhisperProcessor.from_pretrained(CONFIG['model_path'],local_files_only=True)
    base=WhisperForConditionalGeneration.from_pretrained(CONFIG['model_path'],local_files_only=True,
        torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,attn_implementation='sdpa').to('xpu')
    model=PeftModel.from_pretrained(base,a.adapter,local_files_only=True).eval()
    model.config.use_cache=True;pref=prefix(proc.tokenizer)
    gen=copy.deepcopy(model.generation_config);gen.forced_decoder_ids=None;gen.language=None;gen.use_cache=True
    report={'model_revision':CONFIG['model_revision'],'adapter':str(Path(a.adapter).resolve()),
            'manifest_sha256':hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),
            'precision':'bfloat16','runtime':'PyTorch 2.8 XPU; baseline uses disable_adapter on the same frozen base',
            'prefix_ids':pref,'num_beams':1,'max_new_tokens':min(440,448-len(pref)),
            'normalizer':'NFC, casefold, punctuation/symbol-to-space, preserves Hindi marks, no transliteration',
            'validation':[],'test':[],'complete':False,'installed_model_replaced':False}
    write_json(out,report)
    def inputs(row):
        encoded=proc(audio_array(row['audio']),sampling_rate=16000,return_tensors='pt',return_attention_mask=True)
        features=encoded.input_features.to('xpu',torch.bfloat16)
        with torch.no_grad():hidden=model.get_encoder()(features).last_hidden_state
        return encoded,features,hidden
    # Public validation measures token-weighted teacher-forced loss, not WER.
    # No checkpoint tuning or selection is performed against the meeting test.
    for row in validation:
        encoded,features,hidden=inputs(row)
        ids=proc.tokenizer(row['text'],add_special_tokens=False).input_ids
        decoder=torch.tensor([pref+ids],device='xpu')
        labels=torch.tensor([[-100]*(len(pref)-1)+ids+[proc.tokenizer.eos_token_id]],device='xpu')
        if decoder.shape[1]>448:raise ValueError('Validation transcript exceeds context')
        losses={}
        for name in ('baseline','adapter'):
            with (model.disable_adapter() if name=='baseline' else contextlib.nullcontext()),torch.no_grad():
                loss=model(encoder_outputs=BaseModelOutput(last_hidden_state=hidden),decoder_input_ids=decoder,labels=labels).loss
            if not torch.isfinite(loss):raise ValueError('Non-finite validation loss')
            losses[name]=loss.item()
        report['validation'].append({'id':row['id'],'tokens':len(ids)+1,**losses})
        write_json(out,report);print('Validation',row['id'],flush=True)
    for row in test:
        encoded,features,hidden=inputs(row)
        item={'id':row['id'],'group_id':row['group_id'],'audio_sha256':row['audio_sha256'],'reference':row['text']}
        for name in ('baseline','adapter'):
            started=time.monotonic()
            with (model.disable_adapter() if name=='baseline' else contextlib.nullcontext()),torch.no_grad():
                tokens=model.generate(input_features=features,encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
                    attention_mask=encoded.attention_mask.to('xpu'),generation_config=gen,
                    decoder_input_ids=torch.tensor([pref],device='xpu'),language='hi',task='transcribe',
                    return_timestamps=False,do_sample=False,num_beams=1,max_new_tokens=min(440,448-len(pref)),use_cache=True)
            ids=tokens[0].cpu().tolist()
            if ids[:len(pref)]==pref:ids=ids[len(pref):]
            text=proc.tokenizer.decode(ids,skip_special_tokens=True).strip();e,n=errors(row['text'],text)
            item[name]={'text':text,'word_errors':e,'reference_words':n,'seconds':time.monotonic()-started}
            print('Test',row['id'],name,e,n,flush=True)
        report['test'].append(item);write_json(out,report)
    total_tokens=sum(r['tokens'] for r in report['validation'])
    report['summary']={}
    for name in ('baseline','adapter'):
        e=sum(r[name]['word_errors'] for r in report['test']);n=sum(r[name]['reference_words'] for r in report['test'])
        report['summary'][name]={'word_errors':e,'reference_words':n,'wer':e/n if n else None,
            'validation_token_loss':sum(r[name]*r['tokens'] for r in report['validation'])/total_tokens if total_tokens else None}
    b=report['summary']['baseline']['word_errors'];c=report['summary']['adapter']['word_errors']
    report['summary']['relative_error_reduction']=(b-c)/b if b else None
    report['complete']=True;write_json(out,report);print(json.dumps(report['summary'],indent=2),flush=True)

if __name__=='__main__':main()
