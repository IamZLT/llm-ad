"""Fixed-sample, training-free comparison of full/uniform/H-guided visual tokens.

Run from project root. Results include raw outputs, fixed paths and all failures.
Development and heldout manifests are generated before inspecting model predictions.
"""
import argparse,json,sys,time,hashlib,random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from data.scan import load_prior_split
from data.prior_dataset import PriorCoTDataset,PriorCollator,apply_chat_template_safe
from models.qwen35 import setup_model_and_processor
from models.anomaly_prior import AnomalyPrior
from models.h_token_budget import pack_sparse_pair,greedy_decode_sparse
from outcome.inputs import encode_pair_canonical,encode_visual_merged,pixel_budget
from outcome.metrics import component_metrics,mask_iou
from utils.config import load_yaml_config

PROMPT='''Image 1 is a defect-free reference. Image 2 is the inspection image of {class_name}.
Compare the inspection image with the reference. Distinguish actual defects from normal texture, illumination and pose differences. Inspect the full object.
Return only one JSON object: {{"is_anomaly": true or false, "bboxes_2d": [[x1,y1,x2,y2], ...]}}.
Use one tight box per distinct defect, at most 16 boxes. Coordinates refer to the FULL second image, normalized to integers from 0 to 1000. For a normal image return false and an empty list. Do not box the entire object unless the entire object is defective.'''


def parse(text):
    try:
        start=text.index('{'); obj,end=json.JSONDecoder().raw_decode(text[start:])
        if text[start+end:].strip().strip('`').strip():return None
        pred=obj['is_anomaly']; boxes=obj['bboxes_2d']
        if type(pred)!=bool or not isinstance(boxes,list) or len(boxes)>16:return None
        if bool(boxes)!=pred:return None
        for b in boxes:
            if not isinstance(b,list) or len(b)!=4 or any(type(x) not in (int,float) for x in b):return None
            if not (0<=b[0]<b[2]<=1000 and 0<=b[1]<b[3]<=1000):return None
        return obj
    except (ValueError,KeyError,TypeError):return None


def freeze_manifest(cfg,path,per_split):
    if path.exists():return json.loads(path.read_text())
    _,samples=load_prior_split(cfg)
    rng=random.Random(2173)
    abnormal=[s for s in samples if s.get('metadata',{}).get('anomaly')]
    normal=[s for s in samples if not s.get('metadata',{}).get('anomaly')]
    # Round robin categories after per-category shuffling: no H/GT geometry filtering.
    def ordered(items):
        groups={}
        for s in items:groups.setdefault(s['metadata']['class'],[]).append(s)
        for g in groups.values():rng.shuffle(g)
        names=sorted(groups);rng.shuffle(names);out=[]
        while any(groups.values()):
            for name in names:
                if groups[name]:out.append(groups[name].pop())
        return out
    abnormal,normal=ordered(abnormal),ordered(normal)
    manifest={'seed':2173,'selection':'category-round-robin; label stratified; no H/prediction filtering','splits':{}}
    for j,name in enumerate(['dev','heldout']):
        na=max(1,per_split*3//4);nn=per_split-na
        manifest['splits'][name]=abnormal[j*na:(j+1)*na]+normal[j*nn:(j+1)*nn]
        if len(manifest['splits'][name]) != per_split:
            raise ValueError('not enough samples for the requested disjoint manifests')
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(manifest,indent=2))
    return manifest


def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--config',default='configs/qwen35_2b_outcome_multibox.yaml')
    ap.add_argument('--output',required=True);ap.add_argument('--manifest',required=True)
    ap.add_argument('--split',choices=['dev','heldout'],default='dev')
    ap.add_argument('--samples',type=int,default=8);ap.add_argument('--ratio',type=float,default=.5)
    ap.add_argument('--global-fraction',type=float,default=.5);ap.add_argument('--halo',type=float,default=.25)
    ap.add_argument('--modes',default='full,uniform,h');ap.add_argument('--max-new-tokens',type=int,default=192)
    ap.add_argument('--image-size',type=int,default=448)
    ap.add_argument('--h-image-size',type=int,default=768)
    args=ap.parse_args();torch.manual_seed(2173);torch.set_num_threads(2)
    cfg=load_yaml_config(args.config);cfg['data']['max_image_size']=args.image_size
    manifest=freeze_manifest(cfg,Path(args.manifest),args.samples)
    outdir=Path(args.output);outdir.mkdir(parents=True,exist_ok=True)
    (outdir/'experiment.json').write_text(json.dumps(vars(args),indent=2))
    model,processor=setup_model_and_processor(cfg,for_inference=True,freeze_vision=True)
    model.to('cuda' if torch.cuda.is_available() else 'cpu').eval()
    prior=AnomalyPrior.from_qwen(model,cfg);device=next(model.parameters()).device
    collator=PriorCollator(processor,prior,cfg)
    ds=PriorCoTDataset(manifest['splits'][args.split],cfg,processor,'eval')
    tokenizer=processor.tokenizer
    rows=[]
    output=outdir/'results.jsonl'
    if output.exists():raise FileExistsError(f'refusing to overwrite {output}')
    with output.open('w') as f,torch.inference_mode():
        for i in range(len(ds)):
            item=ds[i];ref,test=collator._align_pair(item['ref'],item['test'])
            messages=[{'role':'user','content':[{'type':'image','image':ref},{'type':'image','image':test},
                {'type':'text','text':PROMPT.format(class_name=item['class_name'])}]}]
            text=apply_chat_template_safe(processor,messages,True,False)
            batch=processor(text=[text],images=[ref,test],return_tensors='pt')
            batch={k:v.to(device) for k,v in batch.items()}
            start=time.perf_counter()
            if args.h_image_size == args.image_size:
                vis=encode_pair_canonical(prior,batch['pixel_values'],batch['image_grid_thw'])
            else:
                with pixel_budget(processor,args.h_image_size):
                    high=processor.image_processor(images=[item['ref'],item['test']],return_tensors='pt')
                vis=encode_pair_canonical(prior,high['pixel_values'].to(device),high['image_grid_thw'].to(device))
                vis['merged_embeddings']=encode_visual_merged(prior,batch['pixel_values'],batch['image_grid_thw'])
                del high
            torch.cuda.synchronize() if device.type=='cuda' else None
            encode_seconds=time.perf_counter()-start
            for mode in args.modes.split(','):
                start=time.perf_counter()
                packed=pack_sparse_pair(model,batch,vis['merged_embeddings'],vis['patch_map'],mode,args.ratio,args.global_fraction,args.halo)
                # Full-token prefill must match the original official cached-image path.
                if i==0 and mode=='full':
                    from models.vision_cache import bind_cached_image_features
                    with bind_cached_image_features(model,vis['merged_embeddings']):
                        official=model(**batch,use_cache=False,logits_to_keep=1).logits
                    explicit=model(inputs_embeds=packed['inputs_embeds'],position_ids=packed['position_ids'],
                        attention_mask=batch['attention_mask'],use_cache=False,logits_to_keep=1).logits
                    diff=float((official-explicit).abs().max());print('FULL_PREFILL_MAX_ERROR',diff,flush=True)
                    if diff>.05:raise RuntimeError(f'full token prefill mismatch {diff}')
                    del official,explicit
                if i==0 and mode=='full':
                    torch.cuda.synchronize() if device.type=='cuda' else None
                    start=time.perf_counter()  # exclude the extra correctness check
                result=greedy_decode_sparse(model,packed,tokenizer,args.max_new_tokens)
                torch.cuda.synchronize() if device.type=='cuda' else None
                seconds=time.perf_counter()-start
                parsed=parse(result['text']);gt=item.get('component_bboxes') or ([item['gt_box_px']] if item['gt_box_px'] else [])
                pred=[]
                if parsed:
                    w,h=item['orig_size'];pred=[[b[0]*w/1000,b[1]*h/1000,b[2]*w/1000,b[3]*h/1000] for b in parsed['bboxes_2d']]
                correct=bool(parsed is not None and parsed['is_anomaly']==item['is_anomaly'])
                cm=component_metrics(pred,gt)
                row=dict(index=i,mode=mode,image=item['image_path'],reference=item['ref_path'],class_name=item['class_name'],
                    anomaly=item['is_anomaly'],gt=gt,parsed=parsed,correct=correct,valid=parsed is not None,
                    set_iou=cm['set_iou'] if correct else 0,mask_iou=mask_iou(pred,gt,item['orig_size']) if correct else 0,
                    text=result['text'],tokens=len(result['ids']),truncated=result['truncated'],seconds=seconds,
                    encode_seconds=encode_seconds,selection=packed['selection'])
                f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();rows.append(row)
                print(json.dumps({k:row[k] for k in ['index','mode','class_name','anomaly','correct','set_iou','tokens','seconds']}),flush=True)
                del packed
            del vis,batch
    summary={}
    for mode in args.modes.split(','):
        rr=[r for r in rows if r['mode']==mode];aa=[r for r in rr if r['anomaly']];nn=[r for r in rr if not r['anomaly']]
        mean=lambda a:sum(a)/len(a) if a else None
        summary[mode]=dict(n=len(rr),accuracy=mean([r['correct'] for r in rr]),valid=mean([r['valid'] for r in rr]),
            anomaly_set_iou=mean([r['set_iou'] for r in aa]),anomaly_mask_iou=mean([r['mask_iou'] for r in aa]),
            normal_fpr=mean([r['parsed'] is not None and r['parsed']['is_anomaly'] for r in nn]),
            truncation=mean([r['truncated'] for r in rr]),mean_tokens=mean([r['tokens'] for r in rr]))
    (outdir/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
