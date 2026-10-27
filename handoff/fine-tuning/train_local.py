"""Local decoder-only LoRA pilot. Explicit memory guard; no network or app changes."""
import argparse, hashlib, json, os, time
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
import torch
from peft import LoraConfig,get_peft_model
from transformers import WhisperForConditionalGeneration,WhisperProcessor
from transformers.modeling_outputs import BaseModelOutput
from common import ROOT,CONFIG,free_ram_gib,manifest,validate_splits,prefix,audio_array,write_json

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True)
    p.add_argument('--engineering-smoke',action='store_true',help='One unreviewed public example; measure resources; discard adapter')
    p.add_argument('--steps',type=int,default=20)
    p.add_argument('--publisher-label-pilot',action='store_true',help='Admit documented publisher labels for public train/validation only; test still requires human review')
    p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise SystemExit('Use a new output directory; existing results are preserved')
    if not 1<=a.steps<=CONFIG['pilot_max_optimizer_steps']:raise SystemExit('Pilot step count must be 1..100')
    available=free_ram_gib()
    if available<CONFIG['minimum_free_ram_gib']:
        raise SystemExit(f'Only {available:.1f} GiB RAM free; need {CONFIG["minimum_free_ram_gib"]} GiB. Close unused apps and retry.')
    if not torch.xpu.is_available():raise SystemExit('Intel GPU unavailable. No silent CPU fallback.')
    rows=manifest(a.manifest)
    if a.engineering_smoke:
        train=[r for r in rows if r.get('text') and r.get('audio')][:1]
        steps=1;accum=1
    else:
        validate_splits(rows,allow_publisher_labels=a.publisher_label_pilot)
        train=[r for r in rows if r['split']=='train'];steps=a.steps;accum=CONFIG['gradient_accumulation']
    if not train:raise SystemExit('No training examples')
    torch.set_num_threads(4);torch.manual_seed(CONFIG['seed'])
    processor=WhisperProcessor.from_pretrained(CONFIG['model_path'],local_files_only=True)
    pref=prefix(processor.tokenizer)
    out.mkdir(parents=True)
    report={'config':CONFIG,'mode':'engineering_smoke_discarded' if a.engineering_smoke else 'pilot',
            'manifest_sha256':hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),
            'initial_free_ram_gib':available,'prefix_ids':pref,'steps':[],'accuracy_improvement_established':False,
            'publisher_label_pilot':a.publisher_label_pilot,'public_labels_human_reviewed':False if a.publisher_label_pilot else True}
    write_json(out/'run.json',report)
    print('Loading original Trelis checkpoint...',flush=True)
    model=WhisperForConditionalGeneration.from_pretrained(CONFIG['model_path'],local_files_only=True,
                torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,attn_implementation='sdpa').to('xpu')
    model=get_peft_model(model,LoraConfig(r=CONFIG['lora_rank'],lora_alpha=CONFIG['lora_alpha'],
                lora_dropout=CONFIG['lora_dropout'],target_modules=CONFIG['target_modules'],bias='none'))
    model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    assert named and all('decoder.layers.' in n and 'lora_' in n for n,p in named)
    report['trainable_parameters']=sum(p.numel() for n,p in named)
    report['trainable_names']=[n for n,p in named]
    optimizer=torch.optim.AdamW([p for n,p in named],lr=CONFIG['learning_rate'],foreach=False)
    # Cache frozen encoder outputs for this small local pilot, avoiding repeated encoder work.
    cache=out/'encoder-cache';cache.mkdir()
    prepared=[]
    model.eval()
    for i,row in enumerate(train):
        audio=audio_array(row['audio'])
        tokens=processor.tokenizer(row['text'],add_special_tokens=False).input_ids
        decoder=pref+tokens
        labels=[-100]*(len(pref)-1)+tokens+[processor.tokenizer.eos_token_id]
        if len(decoder)>model.config.max_target_positions:raise ValueError('Transcript exceeds decoder context; do not truncate labels')
        features=processor.feature_extractor(audio,sampling_rate=16000,return_tensors='pt').input_features.to('xpu',torch.bfloat16)
        with torch.no_grad(): hidden=model.get_encoder()(features).last_hidden_state.detach().cpu()
        file=cache/f'{i:06}.pt';torch.save(hidden,file)
        prepared.append((file,decoder,labels))
        print(f'Encoded {i+1}/{len(train)}',flush=True)
        del features,hidden
    model.train();optimizer.zero_grad(set_to_none=True)
    initial=next(p for n,p in named if 'lora_B' in n).detach().cpu().clone()
    for step in range(steps):
        started=time.monotonic();loss_sum=0.0
        for micro in range(accum):
            file,decoder,labels=prepared[(step*accum+micro)%len(prepared)]
            hidden=torch.load(file,weights_only=True).to('xpu')
            loss=model(encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
                       decoder_input_ids=torch.tensor([decoder],device='xpu'),
                       labels=torch.tensor([labels],device='xpu')).loss
            if not torch.isfinite(loss):raise RuntimeError('Non-finite loss; stopping')
            (loss/accum).backward();loss_sum+=loss.item()/accum
        grads=[p.grad for n,p in named if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all().item() for g in grads):raise RuntimeError('Missing or non-finite LoRA gradients')
        torch.nn.utils.clip_grad_norm_([p for n,p in named],1.0,foreach=False)
        optimizer.step();optimizer.zero_grad(set_to_none=True);torch.xpu.synchronize()
        row={'step':step+1,'training_loss':loss_sum,'seconds':time.monotonic()-started}
        report['steps'].append(row);write_json(out/'run.json',report);print(json.dumps(row),flush=True)
    report['adapter_changed']=not torch.equal(initial,next(p for n,p in named if 'lora_B' in n).detach().cpu())
    if not report['adapter_changed']:raise RuntimeError('LoRA weights did not update')
    if not a.engineering_smoke:
        model.save_pretrained(out/'adapter',safe_serialization=True)
        processor.save_pretrained(out/'adapter')
    report['completed']=True;write_json(out/'run.json',report)
    print('Completed. Evaluate on held-out references before considering deployment.',flush=True)

if __name__=='__main__':main()
