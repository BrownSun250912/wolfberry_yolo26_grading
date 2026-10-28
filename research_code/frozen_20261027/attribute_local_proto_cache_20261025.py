"""Capture frozen YOLO soft masks/prototypes for a new boundary2x2 study."""
from attribute_local_common_20261025 import *
setup()
import argparse
import cv2,numpy as np,torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.utils import ops,LOGGER
from validate_r1_closure_runtime import DETECTOR
from run_r1_ap_followup import load_cache
from direction_experiments_20261023 import load_records
from joint_pipeline_capture_20261024 import CandidateCapture
from analyze_val_instance_errors import polygon_to_mask

def register():
    p=BASE/"prototype_boundary_plan.json"
    if not p.exists():save(p,dict(registered_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
       trigger="Historical binaryRGB boundaryCNN improved shape but degraded finalgrading. Attribute12matrix has no accepted valgain; targeted24evaluation underway. New experiment adds frozen native soft/prototypeinformation rather than repeating binaryCNN.",
       factors=dict(coarse_input=["hard","soft"],prototype_input=["zero","actual"]),fits=12,seeds=SEEDS,
       architecture="Pointwise1x1CNN36->32->16->1,RGB3+coarse1+prototype32. Sameallocatedarchitecture; zero-prototype controls have fewer effective degrees of freedom, explicitly disclosed. Not an implementation/equivalence claim of fullPointRend.",
       features="Same-forward native prethreshold masklogit clipped/8 and prototypechannels tanh(/4), crop selected only by frozen predictedmask;prototype40x40storedFP16;RGB/coarse/GT160x160. Sourceprecision and maxerrors checked.",
       identity="Zero finalweights => no nativepixelchange. Upsample learned residual, add to original polygon-raster binarylogit +/-2 in fixed native4px boundaryband, threshold0. Never replace originalmask with resampled160grid.",
       training="Only train matched maskIoU>=.5 instances, original773/193roles; BCEweighted1+4band + Dice; AdamWlr.001wd.0001,batch32,max30epochs,patience8,FP32;pairedseed/init/sampleorder",
       checkpoint="Native held mean IoU minus original baseline, allGT area denominator includes outsidecrop. Select maxIoU, lowerrelativearea tie. Full3seed family mean gain>=.001 and areaerror no worse by>.005; fixedseed25;val onlyafterfreeze.",
       pipeline="Mask-only with frozenoriginalgrades first; also truefullgrading with refinedmasks and freshlegacyattributes/scores, same-confidence actualNMS. Shapeimprovement doesnot certify gradingimprovement.",
       note="GT trainingonly and evaluationonly; no separate stem/body GT invented. Existing YOLOweights and legacygrader remain unchanged; allval exploratory.",test_accessed=False,primary_replaced=False))

class SoftCapture:
    def __init__(self,candidate):self.candidate=candidate;self.original=ops.process_mask_native;self.bounds={};self.values={}
    def attach(self):ops.process_mask_native=self.call
    def restore(self):ops.process_mask_native=self.original
    def call(self,protos,masks_in,boxes,shape):
        binary=self.original(protos,masks_in,boxes,shape);self.values={}
        conf=self.candidate.last["top_conf"].detach().cpu().numpy();allowed=np.flatnonzero(conf>.001)[:300]
        if len(allowed)!=len(binary):raise ValueError("Native/topk lineage changed")
        valid=torch.nonzero(binary.flatten(1).any(1),as_tuple=False).squeeze(1).cpu().numpy();positions={int(v):j for j,v in enumerate(valid)}
        if not len(self.bounds):return binary
        native_proto=ops.scale_masks(protos[None].float(),shape)[0]
        c,mh,mw=protos.shape;coeff=masks_in@protos.float().view(c,-1)
        for i,j in positions.items():
            if j not in self.bounds:continue
            x0,y0,x1,y1=self.bounds[j]
            logits=ops.scale_masks(coeff[i:i+1].view(1,1,mh,mw),shape)[0,0]
            support=ops.crop_mask(torch.ones_like(logits)[None],boxes[i:i+1])[0]>0
            logits=torch.where(support,logits,torch.full_like(logits,-8))
            soft_float=F.interpolate((logits[y0:y1,x0:x1].clamp(-8,8)/8)[None,None],(160,160),mode="bilinear",align_corners=False)[0].cpu()
            proto_float=F.interpolate(native_proto[:,y0:y1,x0:x1][None],(40,40),mode="bilinear",align_corners=False)[0].div(4).tanh().cpu()
            soft,proto=soft_float.half(),proto_float.half()
            self.values[j]=dict(soft=soft,prototype=proto,soft_cast_abs=float((soft.float()-soft_float).abs().max()),prototype_cast_abs=float((proto.float()-proto_float).abs().max()))
        return binary

def main(split,limit=None):
    plan();register()
    if split=="val" and not (OUT/"prototype_internal_selection.json").exists():raise ValueError("Freeze boundaryfamily before valcapture")
    LOGGER.setLevel(40);entries=load_cache(split);old=load_records(split);model=YOLO(str(DETECTOR))
    opts=dict(imgsz=640,conf=.001,iou=.7,max_det=300,device=0,retina_masks=True,half=False,verbose=False)
    model.predict(entries[0]["image_path"],**opts);candidate=CandidateCapture(model);candidate.attach();capture=SoftCapture(candidate);capture.attach()
    directory=BASE/("prototype_"+split);directory.mkdir(exist_ok=True);manifest=[];max_cast=0
    try:
        for n,(e,r) in enumerate(zip(entries,old)):
            if stop_requested() or (limit is not None and n>=limit):break
            path=directory/(e["image_id"]+".pt")
            if not path.exists() or torch.load(path,map_location="cpu",weights_only=False).get("format")!=2:
                capture.bounds={rec["prediction_index"]:rec["bounds"] for rec in r["records"]};result=model.predict(e["image_path"],**opts)[0]
                if not np.array_equal(result.boxes.conf.cpu().numpy(),e["conf"]) or not np.array_equal(result.boxes.cls.cpu().numpy().astype(int),e["baseline"]) or not np.array_equal(result.boxes.xyxy.cpu().numpy(),e["boxes"]):raise ValueError("Detector output drift")
                polygons=[] if result.masks is None else result.masks.xy
                if len(polygons)!=len(r["polygons"]) or not all(np.array_equal(a,b) for a,b in zip(polygons,r["polygons"])):raise ValueError("Polygon output drift")
                area=np.array([np.count_nonzero(polygon_to_mask(p,*r["shape"])) for p in polygons],np.float32)
                if not np.array_equal(area,e["mask_area"]):raise ValueError("Softcapture changes originalmasks")
                records=[]
                for rec in r["records"]:
                    j=rec["prediction_index"]
                    if j not in capture.values:raise ValueError("Missing high nativefeatures")
                    value=capture.values[j];x0,y0,x1,y1=rec["bounds"];coarse=polygon_to_mask(polygons[j],*r["shape"]).astype(np.uint8)
                    kernel=np.ones((9,9),np.uint8);band=cv2.dilate(coarse,kernel)-cv2.erode(coarse,kernel)
                    extra=dict(native_coarse=torch.from_numpy(coarse[y0:y1,x0:x1].copy()),native_band=torch.from_numpy(band[y0:y1,x0:x1].copy()),band=torch.from_numpy(cv2.resize(band[y0:y1,x0:x1],(160,160),interpolation=cv2.INTER_NEAREST)))
                    if rec["gt_index"]>=0:
                        gt=polygon_to_mask(r["gt_polygons"][rec["gt_index"]],*r["shape"]);extra.update(native_gt=torch.from_numpy(gt[y0:y1,x0:x1].copy()),full_gt_area=int(np.count_nonzero(gt)))
                    records.append(dict(rec,**value,**extra))
                torch.save(dict(image_id=e["image_id"],shape=r["shape"],records=records,source_group=e["source_group"],detector_parity=True,format=2),path)
            manifest.append(dict(image_id=e["image_id"],path=str(path),sha256=sha256(path)))
            if (n+1)%50==0 or n+1==len(entries):print(f"PROTOTYPE CAPTURE {split} {n+1}/{len(entries)}",flush=True)
    finally:capture.restore();candidate.restore()
    save(BASE/f"prototype_{split}_manifest.json",dict(status="completed" if len(manifest)==len(entries) else "partial",entries=manifest,test_accessed=False,detector_predictions_unchanged=True))
    save(OUT/f"prototype_{split}_cache_status.json",dict(status="completed" if len(manifest)==len(entries) else "partial",images=len(manifest),test_accessed=False))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("split",choices=["train","val"]);p.add_argument("--limit",type=int);a=p.parse_args();main(a.split,a.limit)
