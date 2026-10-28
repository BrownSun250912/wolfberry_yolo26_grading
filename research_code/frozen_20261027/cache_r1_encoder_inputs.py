"""Cache exact frozen-runtime uint8 RGBA crops, normalized geometry and lineage."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from run_r1_ap_followup import load_cache,Grader,sha256,save,STEM
from cache_r1_quality_features import EXQ,load_quality
from cache_r1_quality_color import DETECTOR,polygon_to_mask

ENC=EXQ/'encoder_inputs'
MEAN=torch.tensor([.485,.456,.406]).reshape(1,3,1,1)
STD=torch.tensor([.229,.224,.225]).reshape(1,3,1,1)


def encode(x):
    rgb=((x[:,:3].cpu()*STD+MEAN)*255).round().clamp(0,255).to(torch.uint8)
    mask=(x[:,3:4].cpu()*255).round().clamp(0,255).to(torch.uint8)
    return torch.cat([rgb,mask],1)


def decode(x):
    raw=x.float()/255;return torch.cat([(raw[:,:3]-MEAN.to(raw.device))/STD.to(raw.device),raw[:,3:4]],1)


def main(split):
    if split not in ['train','val']:raise ValueError('No test access')
    ENC.mkdir(parents=True,exist_ok=True);directory=ENC/split;directory.mkdir(exist_ok=True)
    plan=ENC/'plan.json'
    if not plan.exists():save(plan,{'date':'2026-10-09','task':'encoder micro-finetuning exact runtime input cache',
        'train':'matched high predictions only, no test; train labels join by immutable instance_id','val':'all valid detections conf.001, exploratory evaluation only',
        'representation':'uint8 letterboxed RGB+mask, frozen geometry normalization','augmentation_at_cache':False,
        'script_sha256':sha256(Path(__file__)),'R1_sha256':sha256(STEM)})
    LOGGER.setLevel(40);torch.set_num_threads(4);cache=load_cache(split);q=load_quality(split);grader=Grader();detector=YOLO(str(DETECTOR));captured=[]
    def hook(module,args):captured.append((args[0].detach().cpu(),args[1].detach().cpu()))
    handle=grader.stem.register_forward_pre_hook(hook);entries=[];started=time.perf_counter()
    try:
        for n,(e,qe) in enumerate(zip(cache,q)):
            path=directory/(e['image_id']+'.pt')
            if path.exists():
                item=torch.load(path,map_location='cpu',weights_only=False)
                if item['image_sha256']!=e['image_sha256'] or item['R1_sha256']!=sha256(STEM):raise ValueError('Input cache resume lineage differs')
            else:
                result=detector.predict(e['image_path'],imgsz=640,conf=.001,iou=.7,max_det=300,device=0,retina_masks=True,half=False,verbose=False)[0]
                base=result.boxes.cls.cpu().numpy().astype(np.int64)
                if not np.array_equal(base,e['baseline']) or not np.array_equal(result.boxes.conf.cpu().numpy(),e['conf']):raise ValueError('Detector drift')
                h,w=result.orig_img.shape[:2];masks=[] if result.masks is None else [polygon_to_mask(p,h,w) for p in result.masks.xy]
                captured.clear();_,score,stem=grader.classify(result.orig_img,masks,base)
                if not np.allclose(stem,e['stem'],atol=1e-5,rtol=0,equal_nan=True):raise ValueError('Runtime crop drift')
                image=torch.cat([v[0] for v in captured]);geometry=torch.cat([v[1] for v in captured]);valid=np.flatnonzero(np.isfinite(stem))
                if len(valid)!=len(image) or not np.array_equal(valid,qe['valid_indices']):raise ValueError('Input row alignment differs')
                encoded=encode(image)
                if float((decode(encoded)-image).abs().max())>1e-6:raise ValueError('Uint8 crop encoding is not reversible')
                if split=='train':
                    matched={m['prediction_index']:m for m in qe['matched_high_predictions']};keep=np.array([i for i,j in enumerate(valid) if j in matched],int)
                    image,geometry,valid=encoded[keep],geometry[keep],valid[keep];records=[matched[j] for j in valid]
                else:image=encoded;records=[]
                item={'image_id':e['image_id'],'image_sha256':e['image_sha256'],'R1_sha256':sha256(STEM),'image_uint8':image,
                    'geometry':geometry,'prediction_indices':valid,'matched':records,'split':split}
                torch.save(item,path)
            entries.append({'image_id':e['image_id'],'path':str(path),'sha256':sha256(path),'n':len(item['prediction_indices'])})
            if (n+1)%50==0 or n+1==len(cache):print(f'ENCODER INPUT {split} {n+1}/{len(cache)} elapsed={time.perf_counter()-started:.1f}s',flush=True)
    finally:handle.remove()
    save(ENC/f'{split}_manifest.json',{'split':split,'entries':entries,'images':len(entries),'n':sum(e['n'] for e in entries),'test_accessed':False})


def load_inputs(split):
    if split not in ['train','val']:raise ValueError('No test access')
    data=json.loads((ENC/f'{split}_manifest.json').read_text(encoding='utf-8'));result=[]
    for e in data['entries']:
        p=Path(e['path'])
        if sha256(p)!=e['sha256']:raise ValueError('Input cache hash differs')
        result.append(torch.load(p,map_location='cpu',weights_only=False))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('split',choices=['train','val']);main(p.parse_args().split)
