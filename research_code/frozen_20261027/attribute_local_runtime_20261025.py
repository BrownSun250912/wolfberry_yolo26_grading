"""Opt-in prediction-only runtime: shared frozen prefix, original grading masks, optional refined output."""
from attribute_local_common_20261025 import *
setup()
import cv2,numpy as np,torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.utils import ops
from validate_r1_closure_runtime import DETECTOR
from joint_pipeline_capture_20261024 import CandidateCapture
from joint_pipeline_runtime_20261024 import SharedGrader
from joint_pipeline_policy_20261024 import decide,keep_indices
from attribute_local_fit_20261025 import original,AttributeModel,predict
from attribute_local_proto_fit_20261025 import PointHead,apply_native
from attribute_local_proto_evaluate_20261025 import deltas
from attribute_local_prior_20261025 import correct
from attribute_local_cache_20261025 import crop_tensor
from attribute_local_evaluate_20261025 import labels_for
from attribute_local_target_evaluate_20261025 import gate_labels
from direction_native_inference_20261023 import prediction_records
from analyze_val_instance_errors import polygon_to_mask
setup()

class DeferredSoftCapture:
    """Keep same-forward coefficients; compute crops after public polygon rasterization."""
    def __init__(self,candidate):self.candidate=candidate;self.original=ops.process_mask_native;self.raw=None
    def attach(self):ops.process_mask_native=self.call
    def restore(self):ops.process_mask_native=self.original;self.raw=None
    def call(self,protos,masks_in,boxes,shape):
        binary=self.original(protos,masks_in,boxes,shape)
        allowed=np.flatnonzero(self.candidate.last['top_conf'].detach().cpu().numpy()>.001)[:300]
        if len(allowed)!=len(binary):raise ValueError('Native/topk lineage changed')
        valid=torch.nonzero(binary.flatten(1).any(1),as_tuple=False).squeeze(1).cpu().numpy()
        self.raw=(protos.detach(),masks_in.detach(),boxes.detach(),shape,valid);return binary
    def extract(self,bounds):
        if not bounds:return {}
        if self.raw is None:raise ValueError('No same-forward prototype capture')
        protos,masks_in,boxes,shape,valid=self.raw;native=ops.scale_masks(protos[None].float(),shape)[0];c,mh,mw=protos.shape;coeff=masks_in@protos.float().view(c,-1);values={}
        for public,source in enumerate(valid):
            if public not in bounds:continue
            x0,y0,x1,y1=bounds[public];logits=ops.scale_masks(coeff[source:source+1].view(1,1,mh,mw),shape)[0,0];support=ops.crop_mask(torch.ones_like(logits)[None],boxes[source:source+1])[0]>0;logits=torch.where(support,logits,torch.full_like(logits,-8))
            soft=F.interpolate((logits[y0:y1,x0:x1].clamp(-8,8)/8)[None,None],(160,160),mode='bilinear',align_corners=False)[0].cpu().half()
            proto=F.interpolate(native[:,y0:y1,x0:x1][None],(40,40),mode='bilinear',align_corners=False)[0].div(4).tanh().cpu().half();values[public]=dict(soft=soft,prototype=proto)
        self.raw=None;return values

class AttributePipeline:
    def __init__(self,attribute='none',output_mask='original',seed=20261025,reuse_prefix=False,recompute_global=False):
        if attribute not in ['none','primary_stem','target_selected','creation99_raw','creation99_corrected','defect_evidence_only']:raise ValueError(attribute)
        if output_mask not in ['original','prototype']:raise ValueError(output_mask)
        if reuse_prefix and recompute_global:raise ValueError('Choose one feature graph')
        self.attribute=attribute;self.output_mask=output_mask;self.seed=seed;self.reuse_prefix=reuse_prefix;self.recompute_global=recompute_global;self.detector=YOLO(str(DETECTOR));self.grader=SharedGrader(rule_sources=True);self.candidate=None;self.soft=None;self.handles=[];self.prefix=[];self.geometry=[];self.attribute_model=None;self.attribute_cp=None;self.point=None
        if attribute!='none':
            name={'primary_stem':f'fits/global_global_finetune_seed{seed}/best.pt','target_selected':f'target_fits/defect_unweighted_clean_duplicate_seed{seed}/best.pt','creation99_raw':f'target_fits/defect_balanced_neutral_pair_seed{seed}/best.pt','creation99_corrected':f'target_fits/defect_balanced_neutral_pair_seed{seed}/best.pt','defect_evidence_only':f'target_fits/defect_balanced_neutral_pair_seed{seed}/best.pt'}[attribute]
            self.attribute_cp=torch.load(BASE/name,map_location='cpu',weights_only=False);self.attribute_model=AttributeModel(original(),'global_global',True).cuda().eval();self.attribute_model.load_state_dict(self.attribute_cp['state'])
            if reuse_prefix:
                self.handles.append(self.grader.stem.visual[3].register_forward_hook(lambda m,a,v:self.prefix.append(v.detach().clone())))
                self.handles.append(self.grader.stem.register_forward_pre_hook(lambda m,a:self.geometry.append(a[1].detach().clone())))
        if output_mask=='prototype':
            cp=torch.load(BASE/f'prototype_fits/soft_prototype_seed{seed}/best.pt',map_location='cpu',weights_only=False);self.point=PointHead().cuda().eval();self.point.load_state_dict(cp['state'])
        self.options=dict(imgsz=640,conf=.001,iou=.7,max_det=300,agnostic_nms=False,retina_masks=True,half=False,device=0,verbose=False,save=False)
    @torch.inference_mode()
    def infer(self,image):
        self.prefix.clear();self.geometry.clear()
        if self.point is not None and self.candidate is None:
            self.detector.predict(image,**self.options);self.candidate=CandidateCapture(self.detector);self.candidate.attach();self.soft=DeferredSoftCapture(self.candidate)
        if self.soft is not None:self.soft.attach()
        try:result=self.detector.predict(image,**self.options)[0]
        finally:
            if self.soft is not None:ops.process_mask_native=self.soft.original
        base=result.boxes.cls.cpu().numpy().astype(int);conf=result.boxes.conf.cpu().numpy();boxes=result.boxes.xyxy.cpu().numpy();polygons=[] if result.masks is None else result.masks.xy
        masks=[polygon_to_mask(p,*image.shape[:2]) for p in polygons];entry=dict(baseline=base,conf=conf,boxes=boxes,mask_area=np.array([np.count_nonzero(m) for m in masks],np.float32),image_height=image.shape[0],image_width=image.shape[1])
        if self.candidate is not None:entry.update(self.candidate.align(result))
        outputs=self.grader.infer_sources(image,masks,base,conf);entry.update(score=outputs['score_sources']['rule_anchored22'],stem=outputs['stem'],defect_logit=outputs['defect_logit']);labels=decide(entry,'neighbour',True,True,False);attribute_probabilities=None;attribute_ids=np.array([],int)
        if self.attribute_model is not None:
            if self.reuse_prefix:
                valid=np.flatnonzero(np.isfinite(outputs['stem']));active=np.flatnonzero(conf[valid]>=.25);attribute_ids=valid[active]
                features=torch.cat(self.prefix) if self.prefix else torch.empty((0,192,10,10),device='cuda');geometry=torch.cat(self.geometry) if self.geometry else torch.empty((0,6),device='cuda')
                if len(features)!=len(valid) or len(geometry)!=len(valid):raise ValueError('Frozen prefix/grader row lineage')
                paired_features=features[active,None].expand(-1,2,-1,-1,-1);active_geometry=geometry[active]
            else:
                tensors=[];geoms=[];ids=[]
                for j in np.flatnonzero(conf>=.25):
                    try:
                        g,geom=crop_tensor(image,masks[j],self.grader.size,.25,12)
                        if not self.recompute_global:local,_=crop_tensor(image,masks[j],self.grader.size,.08,4)
                    except ValueError:continue
                    tensors.extend([g] if self.recompute_global else [g,local]);geoms.append(geom);ids.append(int(j))
                attribute_ids=np.array(ids,int);prefix=[]
                for batch in np.array_split(np.stack(tensors),max(1,int(np.ceil(len(tensors)/128)))) if tensors else []:prefix.append(self.grader.stem.visual[:4](torch.as_tensor(batch,device='cuda')))
                paired_features=(torch.cat(prefix)[:,None].expand(-1,2,-1,-1,-1) if self.recompute_global else torch.cat(prefix).reshape(len(ids),2,192,10,10)) if prefix else torch.empty((0,2,192,10,10),device='cuda');active_geometry=torch.tensor((np.stack(geoms)-self.grader.sm)/self.grader.ss,device='cuda') if geoms else torch.empty((0,6),device='cuda')
            pp=predict(self.attribute_model,paired_features,active_geometry).sigmoid().cpu().numpy();attribute_probabilities=correct(pp,self.attribute_cp,True,False) if self.attribute in ['creation99_corrected','defect_evidence_only'] else pp
            rs=[dict(prediction_indices=attribute_ids,image_id='online')];entry['image_id']='online'
            if self.attribute=='defect_evidence_only':labels=decide(entry,'neighbour',True,True,False)
            elif self.attribute=='primary_stem':labels=labels_for([entry],rs,[attribute_probabilities],'replace_stem')[0]
            else:labels=gate_labels([entry],rs,[attribute_probabilities],'defect','hysteresis.05_.95' if self.attribute=='target_selected' else 'creation_only.99')[0]
        refined=masks;delta=[];records=[]
        if self.point is not None:
            records=prediction_records(image,masks,conf);values=self.soft.extract({r['prediction_index']:r['bounds'] for r in records});kernel=np.ones((9,9),np.uint8)
            for r in records:
                j=r['prediction_index'];x0,y0,x1,y1=r['bounds'];band=cv2.dilate(masks[j].astype(np.uint8),kernel)-cv2.erode(masks[j].astype(np.uint8),kernel)
                r.update(values[j],native_coarse=torch.from_numpy(masks[j][y0:y1,x0:x1].astype(np.uint8).copy()),native_band=torch.from_numpy(band[y0:y1,x0:x1].copy()))
            delta=deltas(self.point,records,True,True);refined=[m.copy() for m in masks]
            for r,d in zip(records,delta):
                x0,y0,x1,y1=r['bounds'];refined[r['prediction_index']][y0:y1,x0:x1]=apply_native(r,d)
        keep=keep_indices(boxes,labels,conf,.7,conf>=.25)
        return dict(entry=entry,labels_all=labels,keep_indices=keep,grading_masks=masks,output_masks=refined,attribute_ids=attribute_ids,attribute_probabilities=attribute_probabilities,refiner_records=records,deltas=delta,dual_representation=self.point is not None)
    def close(self):
        if self.soft is not None:self.soft.restore()
        if self.candidate is not None:self.candidate.restore()
        for handle in self.handles:handle.remove()
        self.grader.close();self.prefix.clear();self.geometry.clear()

class RankedPipeline:
    """Use the internally frozen rank choice; optional explicit diagnostic control checkpoint."""
    def __init__(self,policy='RR22',output_mask='original',seed=20261025,reuse_prefix=False,recompute_global=False,rank_checkpoint=None):
        from attribute_local_rank_20261025 import Rank
        if rank_checkpoint is None:
            choices=json.loads((OUT/'attribute_rank_selections.json').read_text(encoding='utf-8'));choice=[c['selected'] for c in choices if c['policy']==policy][0]
            if choice is None:raise ValueError('No internally eligible rank family')
            kind='attribute4' if choice['extra'] else 'zero4';rank_checkpoint=BASE/'attribute_rank_fits'/f"{policy}_{kind}_{choice['target']}_seed{seed}/best.pt"
        self.cp=torch.load(rank_checkpoint,map_location='cpu',weights_only=False);self.rank=Rank().eval();self.rank.load_state_dict(self.cp['state']);self.policy=self.cp['policy']
        needs_new=self.cp['extra'] is True or isinstance(self.cp['extra'],np.bool_) and bool(self.cp['extra']);attribute='creation99_corrected' if self.policy=='correctedcreation99' else 'defect_evidence_only' if needs_new else 'none'
        self.base=AttributePipeline(attribute,output_mask,seed,reuse_prefix=reuse_prefix,recompute_global=recompute_global)
    @torch.inference_mode()
    def infer(self,image):
        from attribute_local_rank_20261025 import confidence
        result=self.base.infer(image);entry=result['entry'];defect=np.full(len(entry['conf']),np.nan,np.float32)
        if result['attribute_probabilities'] is not None:defect[result['attribute_ids']]=result['attribute_probabilities'][:,0]
        else:defect[(entry['conf']>=.25)&np.isfinite(entry['stem'])]=.5
        rank=confidence([entry],[result['labels_all']],[defect],self.cp,self.rank)[0];result['ranking_all']=rank;result['keep_indices']=keep_indices(entry['boxes'],result['labels_all'],rank,.7,entry['conf']>=.25);return result
    def close(self):self.base.close()
