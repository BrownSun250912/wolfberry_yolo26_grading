"""New bounded boundary/robustness experiments; no test or production mutation.

Commands and stage definitions are committed before cache construction/training.
Existing high-confidence prediction crops are paired with manual polygons for
boundary supervision. GT is never used to choose a crop or runtime application.
"""
from __future__ import annotations
import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from ultralytics import YOLO
from ultralytics.utils import LOGGER

from run_r1_ap_followup import ROOT, EX, DETECTOR, STEM, TARGETS, load_cache, sha256
from analyze_val_instance_errors import polygon_to_mask, optimal_match
from build_r1_multitask_targets import parse_polygon

STAGE = ROOT / 'models/experiments/direction_experiments_20261023'
OUT = ROOT / 'outputs/reports/direction_experiments_20261023'
SIZE = 160
SEEDS = [20261023, 20261024, 20261025]
RESET_AT = 1792768574


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def stop_requested():
    return (STAGE / 'STOP_AFTER_CURRENT_EPOCH').exists() or time.time() >= RESET_AT-180


def assert_split(split):
    if split not in ('train', 'val'):
        raise ValueError('Only original train/development val; test access prohibited')


def plan():
    STAGE.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    value = {
        'date': '2026-10-23', 'authorization': 'User requests experiments based on the analysis until current5h quota is exhausted',
        'mode': 'run', 'sources': {p.name: {'path':str(p),'sha256':sha256(p)} for p in [DETECTOR, STEM, TARGETS, Path(__file__)]},
        'budget': {'starting_used_percent':16, 'starting_paid_balance':556.4829495, 'window_reset_unix':RESET_AT,
                   'begin_wrapup_percent':94, 'stop_new_fits_percent':97, 'absolute_stop_percent':99,
                   'resets_forbidden':True,'paid_spillover_authorized':False,'next_window_not_automatically_authorized':True},
        'no_test':True,'labels_thresholds_original_checkpoints_preserved':True,
        'diagnostic': 'Predicted high-conf mask versus corresponding manual polygon; whole-image optimal IoU>=.5 matching; matched and unmatched retained',
        'boundary_crop': {'size':SIZE,'padding':'max(12px,.25longest predicted-mask side)','no_GT_crop':True,'all_runtime_predictions_conf_ge':.25},
        'training': {'variants':['mask_only','rgb_mask','rgb_mask_geometry'], 'seeds':SEEDS,
                     'split':'Existing head_image_split.csv, 773fit/193held original train images; all predictions of same image together',
                     'targets':'Manual polygon binary foreground; no claim of separate fruit-body/pedicel semantic ground truth',
                     'selection':'highest held mean full-crop IoU; lower held relative-area error tie-break; epoch-zero eligible',
                     'family_gate':'mean held IoU gain>=.001 versus identity, all3seeds completed; no worse relative area error by>.005',
                     'max_epochs':40,'patience':10,'optimizer':'AdamW lr.001 wd.0001 batch32 strictFP32',
                     'architecture':'same24channel residual boundary CNN; zero final layer gives identity; only4pixel crop boundary band modifiable',
                     'paired_control':'mask_only uses zeros for RGB channels, same allocated architecture/init/order; effective capacity caveat retained',
                     'geometry_weight':.2,'geometry_target':'relative soft area and normalized second moments, proxies not OpenCV fitted axes'},
        'later': ['diagnose geometric errors','evaluate non-destructive image degradation','review risk-coverage from existing labels, split internal calibration'],
        'limits':['Upstream detector has seen original train, internal split only isolates new refiner','Development repeatedly reused; exploratory only',
                  'No manual defect/keypoint labels are fabricated; spatially supervised directions require such labels',
                  'No automatic primary replacement and no tuning existing grade thresholds'],
    }
    path = STAGE / 'plan.json'
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'))
        if old != value:
            if list(STAGE.glob('cache_*/*.pt')) or list(STAGE.glob('boundary_*/context.json')):
                raise ValueError('Plan/source drift after experimental data processing')
            stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S')
            save(STAGE / ('pre_execution_amendment_'+stamp+'.json'),{'reason':'Correct documented channel width24 before any cache/fit; preserve smoke-tested initial plan',
                 'old_plan':old,'new_plan':value,'no_cache_or_fit_exists':True})
            save(path,value)
    else: save(path,value)
    save(OUT / 'state.json', {'status':'planned','updated_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'test_accessed':False})
    return value


def crop_bounds(mask):
    ys,xs=np.nonzero(mask)
    if not len(xs): return None
    x0,y0,x1,y1=int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1
    p=max(12,math.ceil(max(x1-x0,y1-y0)*.25))
    return [max(0,x0-p),max(0,y0-p),min(mask.shape[1],x1+p),min(mask.shape[0],y1+p)]


def boundary_band(mask, radius=4):
    kernel=np.ones((radius*2+1,radius*2+1),np.uint8)
    m=(mask>0).astype(np.uint8)
    return cv2.dilate(m,kernel)-cv2.erode(m,kernel)


def gt_polygons(entry, target_index):
    rows=target_index.loc[entry['gt_ids']]
    label=ROOT / str(rows.iloc[0].label_path)
    if sha256(label)!=entry['label_sha256']: raise ValueError('GT label hash differs')
    lines=label.read_text(encoding='utf-8').splitlines(); h,w=entry['image_height'],entry['image_width']
    return [parse_polygon(lines[int(row.source_polygon_row)-1],w,h)[1] for row in rows.itertuples()]


def cache(split):
    assert_split(split); plan(); LOGGER.setLevel(40);torch.set_num_threads(4);cv2.setNumThreads(1)
    entries=load_cache(split);targets=pd.read_csv(TARGETS).set_index('instance_id')
    model=YOLO(str(DETECTOR)); directory=STAGE / ('cache_'+split);directory.mkdir(exist_ok=True)
    manifest=[]; started=time.monotonic()
    for k,e in enumerate(entries):
        if stop_requested(): break
        path=directory / (e['image_id']+'.pt')
        if path.exists():
            item=torch.load(path,map_location='cpu',weights_only=False)
            if item['image_sha256']!=e['image_sha256'] or item['label_sha256']!=e['label_sha256']:raise ValueError('Cache lineage drift')
        else:
            r=model.predict(e['image_path'],imgsz=640,conf=.001,iou=.7,max_det=300,device=0,retina_masks=True,half=False,verbose=False)[0]
            if not np.array_equal(r.boxes.cls.cpu().numpy().astype(np.int64),e['baseline']) or not np.array_equal(r.boxes.conf.cpu().numpy(),e['conf']):
                raise ValueError('Fresh detector does not reproduce frozen source')
            h,w=r.orig_img.shape[:2];polys=[] if r.masks is None else [p.copy() for p in r.masks.xy]
            masks=[polygon_to_mask(p,h,w) for p in polys]
            if not np.array_equal(np.array([np.count_nonzero(m) for m in masks],dtype=np.float32),e['mask_area']):raise ValueError('Fresh mask geometry drift')
            gps=gt_polygons(e,targets);high=np.flatnonzero(e['conf']>=.25)
            matched={int(high[pj]):int(gi) for gi,pj,_ in optimal_match(e['mask_iou'][:,high],.5)}
            records=[]
            for j in high:
                bb=crop_bounds(masks[j])
                if bb is None:continue
                x0,y0,x1,y1=bb;rgb=cv2.cvtColor(r.orig_img[y0:y1,x0:x1],cv2.COLOR_BGR2RGB)
                cm=cv2.resize(masks[j][y0:y1,x0:x1].astype(np.uint8),(SIZE,SIZE),interpolation=cv2.INTER_NEAREST)
                rec={'prediction_index':int(j),'bounds':bb,'rgb':torch.from_numpy(cv2.resize(rgb,(SIZE,SIZE)).transpose(2,0,1).copy()),
                     'coarse':torch.from_numpy(cm),'band':torch.from_numpy(boundary_band(cm)),'gt_index':matched.get(int(j),-1)}
                if int(j) in matched:
                    gi=matched[int(j)];gm=polygon_to_mask(gps[gi],h,w)
                    rec['gt']=torch.from_numpy(cv2.resize(gm[y0:y1,x0:x1].astype(np.uint8),(SIZE,SIZE),interpolation=cv2.INTER_NEAREST))
                    rec['instance_id']=e['gt_ids'][gi];rec['gt_class']=int(e['gt_cls'][gi]);rec['original_IoU']=float(e['mask_iou'][gi,j])
                    rec['gt_outside_crop_fraction']=float(1-np.count_nonzero(gm[y0:y1,x0:x1])/max(np.count_nonzero(gm),1))
                records.append(rec)
            item={'image_id':e['image_id'],'split':split,'source_group':e['source_group'],'image_path':e['image_path'],
                  'image_sha256':e['image_sha256'],'label_sha256':e['label_sha256'],'shape':[h,w],'records':records,
                  'polygons':polys,'gt_polygons':gps,'original':e}
            torch.save(item,path)
        manifest.append({'image_id':e['image_id'],'path':str(path),'sha256':sha256(path),
                         'high_predictions':len(item['records']),'matched':sum(x['gt_index']>=0 for x in item['records'])})
        if (k+1)%25==0 or k+1==len(entries): print(f'BOUNDARY CACHE {split} {k+1}/{len(entries)} {time.monotonic()-started:.1f}s',flush=True)
        save(OUT/('cache_'+split+'_status.json'),{'status':'running','images_done':len(manifest),'images_total':len(entries),'test_accessed':False})
    complete=len(manifest)==len(entries)
    save(STAGE / ('cache_'+split+'_manifest.json'),{'split':split,'complete':complete,'entries':manifest,'test_accessed':False,'script_sha256':sha256(Path(__file__))})
    save(OUT/('cache_'+split+'_status.json'),{'status':'completed' if complete else 'budget_stopped','images_done':len(manifest),
         'images_total':len(entries),'high_predictions':sum(x['high_predictions'] for x in manifest),'matched':sum(x['matched'] for x in manifest),'test_accessed':False})


def load_records(split):
    assert_split(split); m=json.loads((STAGE / ('cache_'+split+'_manifest.json')).read_text(encoding='utf-8'))
    if not m['complete']:raise ValueError('Incomplete boundary cache')
    items=[]
    for row in m['entries']:
        path=Path(row['path'])
        if sha256(path)!=row['sha256']: raise ValueError('Boundary cache checksum differs')
        items.append(torch.load(path,map_location='cpu',weights_only=False))
    return items


def arrays(items):
    records=[(e,r) for e in items for r in e['records'] if r['gt_index']>=0]
    x=torch.stack([torch.cat((r['rgb'],r['coarse'][None]*255)) for _,r in records])
    y=torch.stack([r['gt'] for _,r in records])[:,None]
    band=torch.stack([r['band'] for _,r in records])[:,None]
    meta=pd.DataFrame([{'image_id':e['image_id'],'source_group':e['source_group'],'prediction_index':r['prediction_index'],
                        'instance_id':r['instance_id'],'gt_class':r['gt_class'],'original_IoU':r['original_IoU'],
                        'gt_outside_crop_fraction':r['gt_outside_crop_fraction']} for e,r in records])
    return x,y,band,meta


class BoundaryRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(4,24,3,padding=1),nn.SiLU(),nn.Conv2d(24,24,3,padding=1),nn.SiLU(),
                               nn.Conv2d(24,16,3,padding=1),nn.SiLU(),nn.Conv2d(16,1,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,x,band,kind):
        if kind=='mask_only':x=torch.cat((torch.zeros_like(x[:,:3]),x[:,3:4]),1)
        coarse=x[:,3:4]
        return (coarse*2-1)*2 + 6*torch.tanh(self.net(x))*band


def moment_features(p):
    p=p[:,0];h,w=p.shape[-2:];xx=torch.linspace(-1,1,w,device=p.device)[None,None,:]; yy=torch.linspace(-1,1,h,device=p.device)[None,:,None]
    area=p.sum((1,2)).clamp_min(1);mx=(p*xx).sum((1,2))/area;my=(p*yy).sum((1,2))/area
    vx=(p*(xx-mx[:,None,None])**2).sum((1,2))/area;vy=(p*(yy-my[:,None,None])**2).sum((1,2))/area
    return area,torch.stack((vx,vy),1)


def objective(logits,y,band,geometry):
    bce=(F.binary_cross_entropy_with_logits(logits,y,reduction='none')*(1+4*band)).mean()
    p=logits.sigmoid();dice=1-(2*(p*y).sum((1,2,3))+1)/(p.sum((1,2,3))+y.sum((1,2,3))+1)
    loss=bce+dice.mean()
    if geometry:
        a,m=moment_features(p);ga,gm=moment_features(y)
        loss=loss+.2*((a-ga).abs()/ga.clamp_min(1)).mean()+.2*F.l1_loss(m,gm)
    return loss


@torch.inference_mode()
def assess(model,x,y,band,kind,indices,device):
    model.eval();iou=[];area=[];bce=[]
    for ids in indices.split(32):
        bx=x[ids].to(device).float()/255;by=y[ids].to(device).float();bb=band[ids].to(device).float()
        logits=model(bx,bb,kind);pred=logits>0;truth=by>0
        inter=(pred&truth).sum((1,2,3));union=(pred|truth).sum((1,2,3)).clamp_min(1)
        iou.extend((inter/union).cpu().tolist());area.extend(((pred.sum((1,2,3))-truth.sum((1,2,3))).abs()/truth.sum((1,2,3)).clamp_min(1)).cpu().tolist())
        bce.append(float(objective(logits,by,bb,kind=='rgb_mask_geometry')))
    return {'mean_crop_IoU':float(np.mean(iou)),'mean_relative_area_error':float(np.mean(area)),'loss':float(np.mean(bce))}


def fit(kind,seed):
    if kind not in ('mask_only','rgb_mask','rgb_mask_geometry'):raise ValueError(kind)
    cfg=plan();torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(seed);np.random.seed(seed)
    x,y,band,meta=arrays(load_records('train'))
    split=pd.read_csv(ROOT / 'models/experiments/r1_quality_followup_20261009/head_image_split.csv').set_index('image_id')
    held=torch.tensor([i for i,row in meta.iterrows() if split.loc[row.image_id,'role']=='head_holdout'])
    train=torch.tensor([i for i,row in meta.iterrows() if split.loc[row.image_id,'role']=='head_fit'])
    if set(meta.iloc[held.numpy()].image_id)&set(meta.iloc[train.numpy()].image_id):raise ValueError('Image leakage')
    directory=STAGE / f'boundary_{kind}_seed{seed}';directory.mkdir(exist_ok=False)
    device=torch.device('cuda:0');model=BoundaryRefiner().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    generator=torch.Generator().manual_seed(seed);history=[];started=time.monotonic();best=None;bestkey=None;bestepoch=0
    meta.to_csv(directory/'population.csv',index=False)
    save(directory/'context.json',{'plan_sha256':sha256(STAGE/'plan.json'),'kind':kind,'seed':seed,'fit':len(train),'held':len(held),
         'fit_images':meta.iloc[train.numpy()].image_id.nunique(),'held_images':meta.iloc[held.numpy()].image_id.nunique(),'parameters':sum(p.numel() for p in model.parameters()),'test_accessed':False})
    first_order=None;reason='max_epochs'
    for epoch in range(41):
        if epoch:
            model.train();order=train[torch.randperm(len(train),generator=generator)]
            if first_order is None:first_order=meta.iloc[order[:32].numpy()].instance_id.tolist()
            for ids in order.split(32):
                bx=x[ids].to(device).float()/255;by=y[ids].to(device).float();bb=band[ids].to(device).float()
                optimizer.zero_grad(set_to_none=True);loss=objective(model(bx,bb,kind),by,bb,kind=='rgb_mask_geometry')
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        metrics=assess(model,x,y,band,kind,held,device);history.append({'epoch':epoch,**metrics})
        key=(metrics['mean_crop_IoU'],-metrics['mean_relative_area_error'])
        if bestkey is None or key>bestkey:
            bestkey=key;bestepoch=epoch;best=copy.deepcopy(model.state_dict())
            torch.save({'state':{k:v.cpu() for k,v in best.items()},'kind':kind,'seed':seed,'epoch':epoch,'metrics':metrics,'script_sha256':sha256(Path(__file__))},directory/'best.pt')
        torch.save({'state':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'bestepoch':bestepoch,
                    'generator_state':generator.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},directory/'last.pt')
        pd.DataFrame(history).to_csv(directory/'history.csv',index=False)
        save(OUT/'boundary_training_status.json',{'status':'running','kind':kind,'seed':seed,'epoch':epoch,'bestepoch':bestepoch,'held':metrics})
        print(f'BOUNDARY FIT {kind} seed{seed} epoch{epoch} {metrics} elapsed{time.monotonic()-started:.1f}s',flush=True)
        if stop_requested():reason='budget_stop';break
        if epoch-bestepoch>=10:reason='held_early_stop';break
    clone=BoundaryRefiner();cp=torch.load(directory/'best.pt',map_location='cpu',weights_only=False);clone.load_state_dict(cp['state'])
    summary={'status':'completed' if reason!='budget_stop' else 'budget_stopped','kind':kind,'seed':seed,'epochs':epoch,'bestepoch':bestepoch,
             'best':cp['metrics'],'identity':history[0],'reason':reason,'seconds':time.monotonic()-started,'first32_instance_ids':first_order,
             'checkpoint_sha256':sha256(directory/'best.pt'),'cpu_reload':'PASS','test_accessed':False}
    save(directory/'completion.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)


def smoke():
    torch.set_num_threads(2);torch.manual_seed(23)
    x=torch.rand(2,4,32,32);x[:,3]=(x[:,3]>.5).float();band=torch.ones(2,1,32,32);model=BoundaryRefiner()
    for kind in ('mask_only','rgb_mask','rgb_mask_geometry'):
        logits=model(x,band,kind)
        assert torch.equal(logits>0,x[:,3:4]>0)
        y=x[:,3:4].clone();y[:,:,5:9,5:9]=1-y[:,:,5:9,5:9]
        loss=objective(logits,y,band,kind=='rgb_mask_geometry');loss.backward();assert model.net[-1].weight.grad.abs().sum()>0
        model.zero_grad()
    for bad in ('test','TEST','../test'):
        try:assert_split(bad)
        except ValueError:pass
        else:raise AssertionError('Test split permitted')
    m=np.zeros((30,40),np.uint8);m[10:20,10:30]=1;bb=crop_bounds(m);assert bb[0]<=10 and bb[2]>=30
    logits=model(x,torch.zeros_like(band),'rgb_mask');assert torch.equal(logits>0,x[:,3:4]>0)
    save(OUT/'smoke_tests.json',{'identity_all3_kinds':'PASS','gradient_nonzero':'PASS','forbid_test':'PASS','no_band_no_change':'PASS','crop_only_prediction':'PASS'})
    print('5 meaningful boundary preflight checks PASS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','smoke','cache','fit']);p.add_argument('--split',choices=['train','val']);p.add_argument('--kind');p.add_argument('--seed',type=int,default=20261023)
    a=p.parse_args()
    if a.action=='plan':plan()
    elif a.action=='smoke':plan();smoke()
    elif a.action=='cache':cache(a.split)
    else:fit(a.kind,a.seed)
