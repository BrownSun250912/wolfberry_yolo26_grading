"""Frozen R1 fused features on raw detector crops; no training or test access."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from run_r1_ap_followup import ROOT,load_cache,save,sha256
from validate_r1_closure_runtime import Grader,DETECTOR,STEM,TARGETS
from analyze_val_instance_errors import polygon_to_mask,optimal_match

EXQ=ROOT/'models/experiments/r1_quality_followup_20261009'
OUTQ=ROOT/'outputs/reports/r1_quality_followup_20261009'
GTCACHE=ROOT/'models/experiments/r1_score_specialist_20261007/feature_cache.pt'


def plan():
    path=EXQ/'quality_plan.json'
    if path.exists():return
    save(path,{'date':'2026-10-09','question':'Does prediction-crop supervision improve frozen R1 quality gating over original GT-crop supervision?',
        'budget':'account five-hour window; no total experiment-count cap; never consume a reset credit or authorize paid spillover',
        'labels_scores_thresholds_splits_unchanged':True,'test_accessed':False,
        'stage_design':{'training_domain':['GT crop','matched prediction crop'],'head':['linear','MLP'],'BCE':['unweighted','positive_class_weighted'],
            'paired_initialization_seeds':[20261009,20261010,20261011],
            'note':'24 fits are this factorial stage schedule, not an upper limit on subsequent experiments'},
        'checkpoint_selection':'original-train image-holdout prediction-crop balanced accuracy at0.5; tie-break unweighted BCE; no development-val fitting',
        'quality_policy':'fixed0.5 UQ gate in both directions only original detector conf>=.25; grade from unchanged score/stem, class0 outside scope unchanged',
        'confidence_policy':'original detector confidence for head-only comparison; old class-conditioned calibrator not transplanted after changing gate',
        'deployment_seed':20261009,'later_stages':'based on mechanism and outcomes, persist all including negatives; any stage plan saved before running',
        'artifacts':{k:{'path':str(p),'sha256':sha256(p)} for k,p in {'detector':DETECTOR,'R1':STEM,'targets':TARGETS,'GT_feature_cache':GTCACHE,'script':Path(__file__)}.items()},
        'limitations':'Repeated development validation remains exploratory; original R1 already learned original train, inner holdout is only new head holdout, not an unseen full-system test'})


def verify_gt():
    data=torch.load(GTCACHE,map_location='cpu',weights_only=False)
    for name in ['checkpoint','targets']:
        item=data['inputs'][name]
        if sha256(Path(item['path']))!=item['sha256']:raise ValueError('GT feature lineage drift')
    if tuple(data['train'].shape)!=(4277,256):raise ValueError('GT feature population differs')
    return data


def cache(split):
    if split not in ['train','val']:raise ValueError('No test extraction allowed')
    plan();verify_gt();LOGGER.setLevel(40);torch.set_num_threads(4)
    data=load_cache(split);grader=Grader();detector=YOLO(str(DETECTOR));directory=EXQ/f'features_{split}';directory.mkdir(parents=True,exist_ok=True)
    captured=[]
    def hook(module,args,value):captured.append(value.detach().cpu())
    handle=grader.stem.fusion.register_forward_hook(hook);manifest=[];started=time.perf_counter()
    try:
        for k,e in enumerate(data):
            path=directory/(e['image_id']+'.pt')
            if path.exists():
                previous=torch.load(path,map_location='cpu',weights_only=False)
                if previous['image_sha256']!=e['image_sha256'] or previous['R1_sha256']!=sha256(STEM):raise ValueError('Feature resume lineage differs')
            else:
                r=detector.predict(e['image_path'],imgsz=640,conf=.001,iou=.7,max_det=300,device=0,retina_masks=True,half=False,verbose=False)[0]
                image=r.orig_img;h,w=image.shape[:2];masks=[] if r.masks is None else [polygon_to_mask(p,h,w) for p in r.masks.xy]
                base=r.boxes.cls.cpu().numpy().astype(np.int64)
                if not np.array_equal(base,e['baseline']) or not np.array_equal(r.boxes.conf.cpu().numpy(),e['conf']):raise ValueError('Prediction drift')
                captured.clear();_,scores,stems=grader.classify(image,masks,base)
                if not np.allclose(scores,e['score'],atol=1e-4,rtol=0,equal_nan=True) or not np.allclose(stems,e['stem'],atol=1e-5,rtol=0,equal_nan=True):raise ValueError('Grader crop protocol drift')
                valid=np.flatnonzero(np.isfinite(stems));values=torch.cat(captured) if captured else torch.empty((0,256))
                if len(values)!=len(valid):raise ValueError('Feature alignment differs')
                features=torch.zeros((len(base),256));features[valid]=values
                use=np.flatnonzero(e['conf']>=.25);matched=[]
                for gi,pj,iou in optimal_match(e['mask_iou'][:,use],.5):
                    j=int(use[pj])
                    if j in valid:matched.append({'prediction_index':j,'gt_index':gi,'instance_id':e['gt_ids'][gi],'quality_target':int(e['gt_cls'][gi]==0),'gt_class':int(e['gt_cls'][gi]),'mask_iou':iou})
                previous={'image_id':e['image_id'],'source_group':e['source_group'],'image_sha256':e['image_sha256'],'R1_sha256':sha256(STEM),'features':features,'valid_indices':valid,'matched_high_predictions':matched,'split':split}
                torch.save(previous,path)
            manifest.append({'image_id':e['image_id'],'path':str(path),'sha256':sha256(path),'predictions':len(previous['features']),'matched_high':len(previous['matched_high_predictions'])})
            if (k+1)%25==0 or k+1==len(data):print(f'QUALITY FEATURES {split} {k+1}/{len(data)} elapsed={time.perf_counter()-started:.1f}s',flush=True)
    finally:handle.remove()
    save(EXQ/f'features_{split}_manifest.json',{'status':'COMPLETED','split':split,'images':len(manifest),'predictions':sum(e['predictions'] for e in manifest),'matched_high':sum(e['matched_high'] for e in manifest),'entries':manifest,'new_fits':0,'test_accessed':False})


def load_quality(split):
    if split not in ['train','val']:raise ValueError('No test access')
    manifest=json.loads((EXQ/f'features_{split}_manifest.json').read_text(encoding='utf-8'));data=[]
    for item in manifest['entries']:
        path=Path(item['path'])
        if sha256(path)!=item['sha256']:raise ValueError('Quality feature cache changed')
        data.append(torch.load(path,map_location='cpu',weights_only=False))
    return data


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','train','val']);args=p.parse_args()
    plan() if args.action=='plan' else cache(args.action)
