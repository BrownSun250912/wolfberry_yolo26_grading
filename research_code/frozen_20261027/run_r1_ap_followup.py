"""Controlled follow-up: cache original-resolution IoUs once, vary class policy.

No test access, no old report overwrite, no label/threshold/model replacement.
Full factorial contrasts are deterministic paired diagnostics, not replications.
"""
from __future__ import annotations
import argparse
import itertools
import json
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils.metrics import SegmentMetrics,box_iou,mask_iou
from validate_r1_closure_runtime import ROOT,Grader,DETECTOR,STEM,TARGETS,LOW,HIGH
from evaluate_r1a_frozen_predictions import DEFAULT_R0,sha256,load_fixed_mask
from analyze_val_instance_errors import polygon_to_mask
from build_r1_multitask_targets import parse_polygon

EX=ROOT/'models/experiments/r1_ap_followup_20261008'
OUT=ROOT/'outputs/reports/r1_ap_followup_20261008'
NAMES={0:'unqualified',1:'grade_1',2:'grade_2',3:'grade_3'}

def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def plan():
    if (EX/'plan.json').exists():return
    EX.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    selected=Path(json.loads((ROOT/'models/experiments/r1_score_closure_20261008/selection.json').read_text(encoding='utf-8'))['selected_checkpoint'])
    artifacts={k:{'path':str(p),'sha256':sha256(p)} for k,p in {'detector':DETECTOR,'stem':STEM,'score':selected,'targets':TARGETS,'script':Path(__file__)}.items()}
    save(EX/'plan.json',{'date':'2026-10-08','artifacts':artifacts,'question':'Separate score relabeling, stem override and low-confidence scope; then train-only confidence calibration if warranted',
        'factorial':{'score_relabel':[False,True],'stem_override':[False,True],'only_conf_ge_025':[False,True]},
        'scope_threshold':.25,'scope_reason':'Previously frozen deployment threshold, not a newly optimized threshold',
        'prediction_protocol':{'imgsz':640,'conf':.001,'iou':.7,'max_det':300,'retina_masks':True,'half':False,'batch':1},
        'primary_outcomes':['mask_mAP50','mask_mAP50_95','fixed_500_grading_accuracy'],
        'maximum_new_calibration_trainings':6,'allowed_calibration_families':['linear_confidence','mlp_confidence'],
        'calibration_seed_list':[20261008,20261009,20261010],
        'calibration_data':'train predictions only; split train image IDs 80/20 for calibration fitting/checkpoint selection; no validation labels in fit',
        'candidate_acceptance':'mean deployment mask AP50 >= baseline+0.005, mask AP50:95 >= baseline-0.002, localized correct >=439/500; no automatic primary model replacement',
        'no_new_score_training':True,'test_accessed':False,'old_outputs_preserved':True,
        'budget':{'starting_balance':556.4829495,'starting_5h_used_percent':0,'stop_at_used_percent':80,'begin_wrapup_at_used_percent':75,'paid_credits_authorized_this_round':False}})

def match_flags(classes,gt,ious):
    validator=BaseValidator(args={'plots':False})
    validator.iouv=torch.linspace(.5,.95,10)
    return validator.match_predictions(torch.as_tensor(classes),torch.as_tensor(gt),torch.as_tensor(ious)).numpy()

def cache_split(split):
    if split not in ['train','val']:raise ValueError('Only train and val permitted')
    torch.set_num_threads(4)
    plan();cfg=json.loads((EX/'plan.json').read_text(encoding='utf-8'))
    directory=EX/f'cache_{split}';directory.mkdir(exist_ok=True)
    for key in ['detector','stem','score','targets']:
        artifact=cfg['artifacts'][key]
        if sha256(Path(artifact['path']))!=artifact['sha256']:raise ValueError(f'Changed input {key}')
    targets=pd.read_csv(TARGETS);targets=targets[targets.split==split]
    if len(targets)!={'train':4277,'val':518}[split]:raise ValueError('GT population changed')
    grader=Grader();detector=YOLO(str(DETECTOR));started=time.perf_counter()
    fixed=pd.read_csv(DEFAULT_R0/'fixed_predictions.csv') if split=='val' else None
    entries=[]
    for n,(iid,group) in enumerate(targets.groupby('image_id',sort=True)):
        dest=directory/f'{iid}.pt'
        if dest.exists():
            entry=torch.load(dest,map_location='cpu',weights_only=False)
            if entry['input_hashes']!={k:v['sha256'] for k,v in cfg['artifacts'].items() if k!='script'}:raise ValueError('Stale cache')
            entries.append({'image_id':iid,'path':str(dest),'sha256':sha256(dest),'predictions':len(entry['conf'])})
            continue
        path=ROOT/group.iloc[0].image_path
        result=detector.predict(str(path),imgsz=640,conf=.001,iou=.7,max_det=300,agnostic_nms=False,retina_masks=True,half=False,device=0,verbose=False)[0]
        image=result.orig_img;h,w=image.shape[:2]
        masks=[] if result.masks is None else [polygon_to_mask(p,h,w) for p in result.masks.xy]
        baseline=result.boxes.cls.cpu().numpy().astype(np.int64)
        classes,scores,stems=grader.classify(image,masks,baseline)
        conf=result.boxes.conf.cpu().numpy();boxes=result.boxes.xyxy.cpu().numpy()
        gt_masks,gt_boxes,gt_cls,gt_ids=[],[],[],[]
        lines=(ROOT/group.iloc[0].label_path).read_text(encoding='utf-8').splitlines()
        for row in group.sort_values('source_polygon_row').itertuples():
            _,poly=parse_polygon(lines[int(row.source_polygon_row)-1],w,h)
            gt_masks.append(polygon_to_mask(poly,h,w));gt_boxes.append([poly[:,0].min(),poly[:,1].min(),poly[:,0].max(),poly[:,1].max()]);gt_cls.append(row.final_class_id);gt_ids.append(row.instance_id)
        with torch.inference_mode():
            mt=torch.as_tensor(np.stack(masks) if masks else np.empty((0,h,w)),dtype=torch.float32,device='cuda:0')
            gmt=torch.as_tensor(np.stack(gt_masks),dtype=torch.float32,device='cuda:0')
            miou=mask_iou(gmt.flatten(1),mt.flatten(1)).cpu().numpy()
            biou=box_iou(torch.as_tensor(gt_boxes,dtype=torch.float32),torch.as_tensor(boxes,dtype=torch.float32)).numpy()
        old_ids=['']*len(conf)
        if split=='val':
            high=np.flatnonzero(conf>=.25);old=fixed[fixed.image_id==iid].sort_values('pred_index')
            if len(high)!=len(old):raise ValueError(f'Frozen count mismatch {iid}')
            for j,row in zip(high,old.itertuples()):
                if baseline[j]!=row.baseline_class_id or not np.array_equal(masks[j],load_fixed_mask(row,DEFAULT_R0)>0):raise ValueError('Frozen prediction drift')
                old_ids[j]=row.prediction_id
        entry={'image_id':iid,'source_group':str(group.iloc[0].source_group),'image_path':str(path),'image_sha256':sha256(path),'label_sha256':sha256(ROOT/group.iloc[0].label_path),
            'input_hashes':{k:v['sha256'] for k,v in cfg['artifacts'].items() if k!='script'},
            'conf':conf,'baseline':baseline,'candidate':classes,'score':scores,'stem':stems,'boxes':boxes,'mask_iou':miou,'box_iou':biou,
            'mask_area':np.asarray([np.count_nonzero(m) for m in masks],np.float32),'image_height':h,'image_width':w,'gt_cls':np.asarray(gt_cls,np.int64),'gt_ids':gt_ids,'old_prediction_ids':old_ids}
        torch.save(entry,dest)
        entries.append({'image_id':iid,'path':str(dest),'sha256':sha256(dest),'predictions':len(conf)})
        if (n+1)%25==0 or n+1==targets.image_id.nunique():print(f'CACHE {split} {n+1}/{targets.image_id.nunique()} elapsed={time.perf_counter()-started:.1f}s',flush=True)
    save(EX/f'cache_{split}_manifest.json',{'split':split,'images':len(entries),'gt_instances':len(targets),'predictions':sum(x['predictions'] for x in entries),'entries':entries,'fallbacks_this_run':grader.fallbacks,'elapsed_seconds':time.perf_counter()-started})

def load_cache(split):
    manifest=json.loads((EX/f'cache_{split}_manifest.json').read_text(encoding='utf-8'))
    result=[]
    for item in manifest['entries']:
        path=Path(item['path'])
        if sha256(path)!=item['sha256']:raise ValueError('Cache changed')
        result.append(torch.load(path,map_location='cpu',weights_only=False))
    return result

def policy(entry,score,stem,high):
    cls=entry['baseline'].copy()
    active=(cls!=0)&np.isfinite(entry['score'])&np.isfinite(entry['stem'])
    if high:active&=entry['conf']>=.25
    if score:cls[active]=np.where(entry['score'][active]>=HIGH,1,np.where(entry['score'][active]>=LOW,2,3))
    if stem:cls[active&(entry['stem']>=.5)]=3
    return cls

def evaluate(cache,name,labels,confidences=None):
    metrics=SegmentMetrics(names=NAMES)
    fixed=pd.read_csv(ROOT/'outputs/reports/h1_valfix_replay_20261006/variant_instance_outcomes.csv')
    fixed=fixed[(fixed.variant=='baseline')&(fixed.localization_matched==True)].set_index('prediction_id')
    correct,seen,transitions,rows=0,0,[],[]
    for k,entry in enumerate(cache):
        cls=np.asarray(labels[k],np.int64);conf=entry['conf'] if confidences is None else np.asarray(confidences[k])
        if len(cls)!=len(entry['conf']) or not np.isfinite(conf).all() or (conf<0).any() or (conf>1).any():raise ValueError('Invalid policy predictions')
        tp=match_flags(cls,entry['gt_cls'],entry['box_iou']);tpm=match_flags(cls,entry['gt_cls'],entry['mask_iou'])
        metrics.update_stats({'tp':tp,'tp_m':tpm,'target_cls':entry['gt_cls'],'target_img':np.unique(entry['gt_cls']),'conf':conf,'pred_cls':cls,'im_name':entry['image_id']})
        for j,pid in enumerate(entry['old_prediction_ids']):
            if pid in fixed.index:
                correct+=int(cls[j]==int(fixed.loc[pid,'gt_class_id']));seen+=1
            transitions.append({'variant':name,'image_id':entry['image_id'],'prediction_index':j,'old_prediction_id':pid,'source_conf':float(entry['conf'][j]),'new_conf':float(conf[j]),'baseline_class':int(entry['baseline'][j]),'pred_class':int(cls[j]),'tp_mask50':bool(tpm[j,0]),'score':float(entry['score'][j]),'stem_probability':float(entry['stem'][j])})
    if seen!=500:raise ValueError('Fixed matched cohort changed')
    metrics.process(save_dir=OUT,plot=False)
    report={'variant':name,'correct_500':correct,'accuracy_500':correct/500,**metrics.results_dict}
    for j,clsid in enumerate(metrics.seg.ap_class_index):rows.append({'variant':name,'class_id':int(clsid),'mask_ap50':float(metrics.seg.ap50[j]),'mask_ap50_95':float(metrics.seg.ap[j])})
    return report,rows,transitions

def factorial():
    cache=load_cache('val');reports,classes,transitions=[],[],[]
    for score,stem,high in itertools.product([False,True],repeat=3):
        name=f'score{int(score)}_stem{int(stem)}_high{int(high)}'
        labels=[policy(e,score,stem,high) for e in cache]
        report,c,t=evaluate(cache,name,labels)
        report.update({'score_relabel':score,'stem_override':stem,'only_conf_ge_025':high})
        reports.append(report);classes.extend(c);transitions.extend(t)
        print('FACTORIAL '+str(report),flush=True)
    pd.DataFrame(reports).to_csv(OUT/'factorial_metrics.csv',index=False)
    pd.DataFrame(classes).to_csv(OUT/'factorial_class_ap.csv',index=False)
    frame=pd.DataFrame(transitions);frame.to_csv(OUT/'factorial_prediction_outcomes.csv',index=False)
    frame['confidence_bin']=pd.cut(frame.source_conf,[0,.01,.05,.1,.25,.5,.75,1.],right=False)
    frame['changed']=frame.baseline_class!=frame.pred_class
    frame['false_positive']=~frame.tp_mask50
    frame.groupby(['variant','confidence_bin'],observed=True).agg(predictions=('tp_mask50','size'),true_positive=('tp_mask50','sum'),false_positive=('false_positive','sum'),relabelled=('changed','sum')).reset_index().to_csv(OUT/'confidence_bin_counts.csv',index=False)
    baseline=reports[0];full=next(r for r in reports if r['variant']=='score1_stem1_high0')
    old=json.loads((ROOT/'outputs/reports/r1_score_closure_20261008/deployment_ap.json').read_text(encoding='utf-8'))
    for key in ['metrics/mAP50(M)','metrics/mAP50-95(M)','metrics/mAP50(B)','metrics/mAP50-95(B)']:
        if abs(baseline[key]-old['baseline'][key])>1e-8 or abs(full[key]-old['candidate'][key])>1e-8:raise ValueError('Cached official AP does not reproduce previous AP')
    save(OUT/'factorial_summary.json',{'status':'COMPLETED','cache_reproduces_previous_AP':True,'protocol':'same original single-image detections, full-resolution IoUs, same official AP aggregation','runs':8,'unique_conditions':7,'not_independent_replications':True,'test_accessed':False,'metrics':reports})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['plan','cache-val','cache-train','factorial'])
    args=parser.parse_args()
    if args.action=='plan':plan()
    elif args.action=='cache-val':cache_split('val')
    elif args.action=='cache-train':cache_split('train')
    else:factorial()
