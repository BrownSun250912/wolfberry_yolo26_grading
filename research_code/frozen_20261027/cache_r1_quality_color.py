"""GT-free masked colour summaries for conservative quality correction."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from run_r1_ap_followup import ROOT,load_cache,save,sha256
from validate_r1_closure_runtime import DETECTOR
from analyze_val_instance_errors import polygon_to_mask
from cache_r1_quality_features import EXQ,OUTQ

COLOR_NAMES=['R_mean','G_mean','B_mean','R_std','G_std','B_std','L_mean','a_mean','b_mean','L_std','a_std','b_std',
    'S_mean','V_mean','S_std','V_std','V_q10','V_q50','V_q90','low_V_fraction','low_S_fraction','hue_cos_mean','hue_sin_mean','R_minus_G_mean']


def color_features(image,mask):
    mask=np.asarray(mask,dtype=bool)
    if mask.shape!=image.shape[:2]:raise ValueError('Image/mask shape differs')
    yy,xx=np.nonzero(mask)
    if not len(xx):return np.zeros(len(COLOR_NAMES),np.float32),False
    x0,x1,y0,y1=xx.min(),xx.max()+1,yy.min(),yy.max()+1
    crop=image[y0:y1,x0:x1];use=mask[y0:y1,x0:x1]
    rgb=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)[use].astype(np.float32)/255
    lab=cv2.cvtColor(crop,cv2.COLOR_BGR2LAB)[use].astype(np.float32)/255
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)[use].astype(np.float32)
    sat,value=hsv[:,1]/255,hsv[:,2]/255;hue=hsv[:,0]*(2*np.pi/180)
    vector=np.r_[rgb.mean(0),rgb.std(0),lab.mean(0),lab.std(0),sat.mean(),value.mean(),sat.std(),value.std(),np.quantile(value,[.1,.5,.9]),(value<.25).mean(),(sat<.2).mean(),np.cos(hue).mean(),np.sin(hue).mean(),(rgb[:,0]-rgb[:,1]).mean()].astype(np.float32)
    if vector.shape!=(len(COLOR_NAMES),) or not np.isfinite(vector).all():raise ValueError('Invalid color vector')
    return vector,True


def main(split):
    if split not in ['train','val']:raise ValueError('No test access')
    LOGGER.setLevel(40);torch.set_num_threads(4)
    plan=EXQ/'color_feature_plan.json'
    if not plan.exists():save(plan,{'feature_names':COLOR_NAMES,'script_sha256':sha256(Path(__file__)),'source':'actual original RGB pixels inside frozen detector masks, no GT input to feature construction',
        'pixel_thresholds':'V<.25 and S<.2 are fixed feature definitions, not quality decisions; classifier learns mapping',
        'labels_or_detector_masks_changed':False,'test_accessed':False})
    cache=load_cache(split);directory=EXQ/f'color_{split}';directory.mkdir(parents=True,exist_ok=True);detector=YOLO(str(DETECTOR));manifest=[]
    for i,e in enumerate(cache):
        path=directory/(e['image_id']+'.pt')
        if path.exists():
            d=torch.load(path,map_location='cpu',weights_only=False)
            if d['image_sha256']!=e['image_sha256']:raise ValueError('Image lineage differs')
        else:
            r=detector.predict(e['image_path'],imgsz=640,conf=.001,iou=.7,max_det=300,device=0,retina_masks=True,half=False,verbose=False)[0]
            if not np.array_equal(r.boxes.cls.cpu().numpy().astype(np.int64),e['baseline']) or not np.array_equal(r.boxes.conf.cpu().numpy(),e['conf']):raise ValueError('Frozen detection drift')
            image=r.orig_img;h,w=image.shape[:2];masks=[] if r.masks is None else [polygon_to_mask(p,h,w) for p in r.masks.xy]
            results=[color_features(image,m) for m in masks]
            x=np.stack([row[0] for row in results]) if results else np.empty((0,len(COLOR_NAMES)),np.float32)
            valid=np.asarray([row[1] for row in results],bool)
            d={'image_id':e['image_id'],'image_sha256':e['image_sha256'],'features':x,'valid':valid,'feature_names':COLOR_NAMES}
            torch.save(d,path)
        manifest.append({'image_id':e['image_id'],'path':str(path),'sha256':sha256(path),'predictions':len(d['features'])})
        if (i+1)%50==0 or i+1==len(cache):print(f'COLOR CACHE {split} {i+1}/{len(cache)}',flush=True)
    save(EXQ/f'color_{split}_manifest.json',{'split':split,'images':len(manifest),'entries':manifest,'test_accessed':False})


def load_color(split):
    if split not in ['train','val']:raise ValueError('No test access')
    manifest=json.loads((EXQ/f'color_{split}_manifest.json').read_text(encoding='utf-8'));result=[]
    for e in manifest['entries']:
        path=Path(e['path'])
        if sha256(path)!=e['sha256']:raise ValueError('Color cache changed')
        result.append(torch.load(path,map_location='cpu',weights_only=False))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('split',choices=['train','val']);args=p.parse_args();main(args.split)
