"""Frozen score candidate: raw-image replay, paired AP and paired latency audit.

Only class IDs change; detector confidence, masks, boxes and NMS are retained.
Tiny invalid masks retain the detector class, with explicit fallback counts.
"""
from pathlib import Path
import json
import math
import time
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from ultralytics.models.yolo.segment.val import SegmentationValidator
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import SegmentMetrics
from ultralytics.utils import ops
from train_r1a_hierarchical_grader import R1HierarchicalGrader, compute_geometry, letterbox
from train_r1_score_geometry_s2 import GeometryScoreHead, mask_geometry
from run_r1_refined_geometry_batch import refine
from evaluate_r1a_frozen_predictions import DEFAULT_R0, load_fixed_mask, sha256
from analyze_val_instance_errors import polygon_to_mask

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/reports/r1_score_closure_20261008'
EX=ROOT/'models/experiments/r1_score_closure_20261008'
LOW=37.40298225745565
HIGH=65.62684031710077
DETECTOR=ROOT/'models/experiments/yolo26n_seg_offlineaug_e200_mask50_seed20260915/weights/best.pt'
STEM=ROOT/'models/experiments/r1a_hierarchical_grader_e200_resume_epoch150_seed20260929/best.pt'
TARGETS=ROOT/'data/grading_four_class_v3_valfix_20261006/audit/r1_multitask_targets.csv'

def save(name,data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

class Grader:
    def __init__(self):
        self.device=torch.device('cuda:0')
        self.path=Path(json.loads((EX/'selection.json').read_text(encoding='utf-8'))['selected_checkpoint'])
        cp=torch.load(self.path,map_location='cpu',weights_only=False)
        self.score=GeometryScoreHead(True).to(self.device).eval()
        self.score.load_state_dict(cp['head_state'])
        self.mean=np.asarray(cp['mean'],np.float32)
        self.std=np.asarray(cp['std'],np.float32)
        cache=torch.load(ROOT/'models/experiments/r1_score_refined_geometry_20261008_artifactfix/refined_geometry_cache.pt',weights_only=False)
        self.config=cache['config']
        cp=torch.load(STEM,map_location='cpu',weights_only=False)
        self.stem=R1HierarchicalGrader().to(self.device).eval()
        self.stem.load_state_dict(cp['model_state'])
        self.size=cp['config']['image_size']
        self.sm=np.asarray(cp['geometry_mean'],np.float32)
        self.ss=np.asarray(cp['geometry_std'],np.float32)
        self.fallbacks=0

    @torch.inference_mode()
    def classify(self,image,masks,baseline):
        classes=np.asarray(baseline,dtype=int).copy()
        scores=np.full(len(masks),np.nan)
        probs=np.full(len(masks),np.nan)
        tensors,oldgeom,newgeom,indices=[],[],[],[]
        for i,mask in enumerate(masks):
            mask=(np.asarray(mask)>0).astype(np.uint8)*255
            try:
                ys,xs=np.nonzero(mask)
                if not len(xs): raise ValueError('empty mask')
                x,y=int(xs.min()),int(ys.min())
                w,h=int(xs.max()-x+1),int(ys.max()-y+1)
                pad=max(12,math.ceil(max(w,h)*.25))
                x1,y1=max(0,x-pad),max(0,y-pad)
                x2,y2=min(image.shape[1],x+w+pad),min(image.shape[0],y+h+pad)
                bgr=image[y1:y2,x1:x2]
                mc=mask[y1:y2,x1:x2]
                refined,_,_=refine(bgr,mc,self.config)
                geom=mask_geometry(refined)
                rawgeom=compute_geometry(mc,bgr.shape[:2])
                rgb,mm=letterbox(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),mc,self.size)
                rgb=(rgb.astype(np.float32)/255-np.asarray([.485,.456,.406],np.float32))/np.asarray([.229,.224,.225],np.float32)
                tensor=np.concatenate([rgb,(mm.astype(np.float32)/255)[...,None]],axis=2).transpose(2,0,1)
                tensors.append(tensor);oldgeom.append(rawgeom);newgeom.append(geom);indices.append(i)
            except ValueError:
                self.fallbacks+=1
        if indices:
            g=torch.as_tensor((np.stack(newgeom)-self.mean)/self.std,device=self.device)
            scores[indices]=(self.score(torch.zeros((len(indices),256),device=self.device),g)*100).cpu().numpy()
            # Bounded memory even for conf=.001 validation batches.
            for start in range(0,len(indices),64):
                ii=indices[start:start+64]
                im=torch.as_tensor(np.stack(tensors[start:start+64]),device=self.device)
                gg=torch.as_tensor((np.stack(oldgeom[start:start+64])-self.sm)/self.ss,device=self.device)
                output=self.stem(im,gg)
                probs[ii]=torch.sigmoid(output['stem_logit']).cpu().numpy()
            for i in indices:
                if classes[i]==0: continue
                classes[i]=3 if probs[i]>=.5 else (1 if scores[i]>=HIGH else 2 if scores[i]>=LOW else 3)
        return classes,scores,probs

def replay(grader):
    fixed=pd.read_csv(DEFAULT_R0/'fixed_predictions.csv')
    expected=pd.read_csv(grader.path.parent/(grader.path.stem.replace('_best','')+'_prediction_scores.csv')).set_index('prediction_id')
    h1=pd.read_csv(ROOT/'outputs/reports/h1_frozen_replay_20261006/h1_predictions.csv').set_index('prediction_id')
    rows=[]
    for _,group in fixed.groupby('image_id',sort=False):
        image=cv2.imread(str(group.iloc[0].image_path))
        masks=[load_fixed_mask(row,DEFAULT_R0) for row in group.itertuples()]
        classes,scores,probs=grader.classify(image,masks,group.baseline_class_id.to_numpy())
        for i,row in enumerate(group.itertuples()):
            rows.append({'prediction_id':row.prediction_id,'class_id':int(classes[i]),'score':float(scores[i]),'stem_probability':float(probs[i]),'score_delta':abs(float(scores[i])-float(expected.loc[row.prediction_id,'score_prediction'])),'stem_decision_equal':bool((probs[i]>=.5)==(h1.loc[row.prediction_id,'stem_missing_probability']>=.5))})
    frame=pd.DataFrame(rows)
    frame.to_csv(OUT/'runtime_replay.csv',index=False)
    result={'n':len(frame),'max_score_delta':float(frame.score_delta.max()),'all_stem_decisions_equal':bool(frame.stem_decision_equal.all()),'fallbacks':grader.fallbacks}
    save('runtime_verification.json',result)
    if result['max_score_delta']>.002 or not result['all_stem_decisions_equal'] or result['fallbacks']:
        raise ValueError(f'Runtime replay verification failed: {result}')
    print('RUNTIME VERIFIED '+str(result),flush=True)

def ap(grader):
    class PairedValidator(SegmentationValidator):
        def postprocess(self,preds):
            proto=preds[0][1] if isinstance(preds[0],tuple) else preds[1]
            result=DetectionValidator.postprocess(self,preds[0])
            for i,pred in enumerate(result):
                coefficient=pred.pop('extra')
                pred['masks']=self.process(proto[i],coefficient,pred['bboxes'],shape=[4*x for x in proto.shape[2:]])
                pred['runtime_proto']=proto[i]
                pred['runtime_coefficients']=coefficient
            return result

        def build_dataset(self,img_path,mode='val',batch=None):
            dataset=super().build_dataset(img_path,mode,batch)
            targets=pd.read_csv(TARGETS)
            targets=targets[targets.split=='val']
            total,changed=0,0
            for label in dataset.labels:
                image_id=Path(label['im_file']).stem
                target=targets[targets.image_id==image_id].sort_values('source_polygon_row')
                cls=target.final_class_id.to_numpy(dtype=np.float32).reshape(-1,1)
                if cls.shape!=label['cls'].shape: raise ValueError('GT label overlay shape mismatch')
                changed+=int(np.count_nonzero(cls!=label['cls']))
                total+=len(cls);label['cls']=cls
            if total!=518: raise ValueError('GT population changed')
            self.overlay={'instances':total,'changed_classes':changed}
            return dataset

        def init_metrics(self,model):
            super().init_metrics(model)
            self.baseline_metrics=SegmentMetrics(names=self.names)
            self.predictions=0
            self.grading_seconds=0.
            self.runtime_rows=[]

        def update_metrics(self,preds,batch):
            original,seen=self.metrics,self.seen
            self.metrics=self.baseline_metrics
            super().update_metrics(preds,batch)
            self.metrics,self.seen=original,seen
            adjusted=[]
            for si,pred in enumerate(preds):
                started=time.perf_counter()
                pb=self._prepare_batch(si,batch)
                image=cv2.imread(str(pb['im_file']))
                # Match actual retina_masks=True deployment: scale boxes then
                # decode prototypes directly at original resolution. AP masks
                # remain unchanged and are evaluated by the official validator.
                boxes=ops.scale_boxes(pb['imgsz'],pred['bboxes'].clone(),pb['ori_shape'],ratio_pad=pb['ratio_pad'])
                masks=ops.process_mask_native(pred['runtime_proto'],pred['runtime_coefficients'],boxes,pb['ori_shape'])
                polygons=ops.masks2segments(masks)
                masks=[polygon_to_mask(p,image.shape[0],image.shape[1]) for p in polygons]
                cls,scores,stems=grader.classify(image,masks,pred['cls'].cpu().numpy())
                for j,box in enumerate(boxes.cpu().numpy()):
                    self.runtime_rows.append({'image_id':Path(pb['im_file']).stem,'confidence':float(pred['conf'][j]),'baseline_class':int(pred['cls'][j]),'candidate_class':int(cls[j]),'score':float(scores[j]),'stem_probability':float(stems[j]),'x1':float(box[0]),'y1':float(box[1]),'x2':float(box[2]),'y2':float(box[3])})
                copy=dict(pred);copy['cls']=torch.as_tensor(cls,device=pred['cls'].device,dtype=pred['cls'].dtype)
                adjusted.append(copy)
                self.predictions+=len(cls)
                self.grading_seconds+=time.perf_counter()-started
            super().update_metrics(adjusted,batch)

    instances=[]
    def factory(**kwargs):
        value=PairedValidator(**kwargs);instances.append(value);return value
    detector=YOLO(str(DETECTOR))
    start=time.perf_counter()
    detector.val(validator=factory,data=str(ROOT/'data/grading_four_class_v3/data.yaml'),split='val',imgsz=640,batch=4,workers=0,device=0,conf=.001,iou=.7,max_det=300,plots=False,save_json=False,save_txt=False,project=str(OUT),name='paired_ap',exist_ok=True,verbose=False)
    elapsed=time.perf_counter()-start
    validator=instances[0]
    pd.DataFrame(validator.runtime_rows).to_csv(OUT/'ap_runtime_predictions.csv',index=False)
    validator.baseline_metrics.process(save_dir=OUT,plot=False)
    result={'baseline':validator.baseline_metrics.results_dict,'candidate':validator.metrics.results_dict,'label_overlay':validator.overlay,'predictions':validator.predictions,'fallbacks_cumulative':grader.fallbacks,'grading_seconds':validator.grading_seconds,'paired_validation_wall_seconds':elapsed,'conf':.001,'iou':.7,'max_det':300,'confidence_policy':'unchanged detector confidence','mask_policy':'unchanged native validation masks for AP; scaled masks only for class refinement','test_accessed':False}
    result['runtime_mask_policy']='original-resolution retina decode from prototypes, scaled boxes, polygon rasterization; identical policy to frozen replay'
    save('paired_ap.json',result)
    print('PAIRED AP '+json.dumps(result),flush=True)
    return detector

def latency(grader,detector):
    images=sorted((ROOT/'data/grading_four_class_v3/images/val').glob('*'))[:24]
    detector.predict(str(images[0]),imgsz=640,conf=.25,iou=.7,device=0,retina_masks=True,verbose=False)
    times=[]
    for repeat in range(3):
        for path in images:
            torch.cuda.synchronize();start=time.perf_counter()
            image=cv2.imread(str(path))
            result=detector.predict(image,imgsz=640,conf=.25,iou=.7,device=0,retina_masks=True,verbose=False)[0]
            torch.cuda.synchronize();mid=time.perf_counter()
            masks=[] if result.masks is None else [polygon_to_mask(p,image.shape[0],image.shape[1]) for p in result.masks.xy]
            grader.classify(image,masks,result.boxes.cls.cpu().numpy())
            torch.cuda.synchronize();end=time.perf_counter()
            times.append({'image':path.name,'repeat':repeat,'detector_ms':(mid-start)*1000,'candidate_ms':(end-start)*1000,'added_ms':(end-mid)*1000,'instances':len(masks)})
    frame=pd.DataFrame(times);frame.to_csv(OUT/'latency_samples.csv',index=False)
    means={k:float(frame[k].mean()) for k in ['detector_ms','candidate_ms','added_ms']}
    means.update({'n_images':len(images),'repeats':3,'device':torch.cuda.get_device_name(),'disk_decode_included':True,'warmup_excluded':True,'comparison':'paired same detector execution; candidate adds mask polygon reconstruction, refinement, stem inference and score inference','detector_parameters':sum(p.numel() for p in detector.model.parameters()),'stem_parameters':sum(p.numel() for p in grader.stem.parameters()),'score_parameters':sum(p.numel() for p in grader.score.parameters())})
    save('latency_summary.json',means)
    print('LATENCY '+str(means),flush=True)

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    previous=OUT/'paired_ap.json'
    if previous.exists() and not (OUT/'diagnostic_lowresolution_ap.json').exists():
        save('diagnostic_lowresolution_ap.json',json.loads(previous.read_text(encoding='utf-8')))
    torch.set_num_threads(4)
    grader=Grader()
    save('runtime_protocol.json',{'script_sha256':sha256(Path(__file__)),'detector_sha256':sha256(DETECTOR),'stem_sha256':sha256(STEM),'score_sha256':sha256(grader.path),'targets_sha256':sha256(TARGETS),'fallback':'invalid masks retain baseline class; count and report','selection':'fixed seed 20261007; previous selection.json','test_accessed':False})
    replay(grader)
    detector=ap(grader)
    latency(grader,detector)
