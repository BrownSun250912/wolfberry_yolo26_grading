"""Same-forward raw four-class scores with candidate-index/public-row lineage."""
from joint_pipeline_common_20261024 import *
setup()
import argparse
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER,ops
from run_r1_ap_followup import load_cache
from validate_r1_closure_runtime import DETECTOR


class CandidateCapture:
    def __init__(self,detector):
        self.detector=detector;self.last={};self.handles=[];self.head=None

    def attach(self):
        network=self.detector.predictor.model.model
        self.head=network.model[-1]
        if not self.head.end2end or self.head.nc!=4:raise ValueError("Expected current four-class end2end detector")
        self.old_topk=self.head.get_topk_index;self.old_post=self.head.postprocess
        def before(module,args):self.last["input_hw"]=tuple(args[0].shape[-2:])
        self.handles.append(network.register_forward_pre_hook(before))
        def post(preds):
            self.last["raw_boxes"]=preds[0,:,:4].detach().clone()
            return self.old_post(preds)
        def topk(scores,max_det):
            result=self.old_topk(scores,max_det)
            if scores.shape[0]!=1:raise ValueError("Single-image lineage only")
            ids=result[2][0,:,0]
            self.last["candidate_indices"]=ids.detach().clone()
            self.last["class_scores"]=scores[0,ids,:].detach().clone()
            self.last["top_conf"]=result[0][0,:,0].detach().clone()
            self.last["top_cls"]=result[1][0,:,0].detach().clone()
            return result
        self.head.postprocess=post;self.head.get_topk_index=topk

    def restore(self):
        if self.head is not None:self.head.get_topk_index=self.old_topk;self.head.postprocess=self.old_post
        for h in self.handles:h.remove()

    def align(self,result,threshold=.001):
        x=self.last;idx=x["candidate_indices"]
        boxes=x["raw_boxes"][idx].clone()
        boxes=ops.scale_boxes(x["input_hw"],boxes,result.orig_shape).cpu().numpy()
        conf=x["top_conf"].cpu().numpy();cls=x["top_cls"].cpu().numpy().astype(int)
        allowed=np.flatnonzero(conf>threshold)[:300]
        actual=result.boxes.data.cpu().numpy()
        mapping=[];cursor=0
        for row in actual:
            matches=[]
            for pos in range(cursor,len(allowed)):
                j=allowed[pos]
                if cls[j]==int(row[5]) and conf[j]==row[4] and np.max(np.abs(boxes[j]-row[:4]))<=1e-3:
                    matches.append((pos,j))
            if not matches:raise ValueError("No exact class/conf and geometrically aligned candidate row")
            pos,j=matches[0];mapping.append(j);cursor=pos+1
        mapping=np.asarray(mapping,int)
        values=x["class_scores"].cpu().numpy()[mapping]
        source_ids=idx.cpu().numpy()[mapping]
        if len(actual) and not np.array_equal(values[np.arange(len(values)),actual[:,5].astype(int)],actual[:,4]):
            raise ValueError("Captured top class does not equal public confidence")
        return {"class_scores":values,"candidate_indices":source_ids,"topk_row_indices":mapping,"input_hw":x["input_hw"],
                "public_result_lineage_exact":True,"filtered_empty_mask_rows":len(allowed)-len(actual)}


def main(split):
    if split not in ["train","val"]:raise ValueError("No test capture")
    plan();LOGGER.setLevel(40);entries=load_cache(split);detector=YOLO(str(DETECTOR))
    options=dict(imgsz=640,conf=.001,iou=.7,max_det=300,agnostic_nms=False,retina_masks=True,half=False,device=0,verbose=False,save=False)
    detector.predict(entries[0]["image_path"],**options)
    capture=CandidateCapture(detector);capture.attach()
    directory=BASE/("raw_classes_"+split);directory.mkdir(exist_ok=True)
    manifest=[];started=time.perf_counter();drifts=[]
    try:
        for n,e in enumerate(entries):
            if stop_requested():break
            path=directory/(e["image_id"]+".pt")
            if path.exists():
                record=torch.load(path,map_location="cpu",weights_only=False)
                if record["image_sha256"]!=e["image_sha256"] or record["detector_sha256"]!=sha256(DETECTOR):raise ValueError("Raw class cache lineage drift")
            else:
                r=detector.predict(e["image_path"],**options)[0]
                record=capture.align(r)
                fresh_cls=r.boxes.cls.cpu().numpy().astype(int);fresh_conf=r.boxes.conf.cpu().numpy();fresh_boxes=r.boxes.xyxy.cpu().numpy()
                exact=np.array_equal(fresh_cls,e["baseline"]) and np.array_equal(fresh_conf,e["conf"]) and np.array_equal(fresh_boxes,e["boxes"])
                record.update(image_id=e["image_id"],image_sha256=e["image_sha256"],detector_sha256=sha256(DETECTOR),split=split,
                    baseline=fresh_cls,conf=fresh_conf,boxes=fresh_boxes,old_prediction_parity_exact=bool(exact),
                    old_n=len(e["conf"]),fresh_n=len(fresh_conf),device="cuda:0",precision="FP32,matmulTF32False,cudnnTF32True")
                if not exact:
                    drifts.append(e["image_id"])
                    # Preserve evidence and fail before misjoining new evidence to old scores/masks.
                    torch.save(record,directory/(e["image_id"]+"_parity_failure.pt"))
                    save(OUT/("capture_"+split+"_failure.json"),{"image_id":e["image_id"],"old_n":len(e["conf"]),"fresh_n":len(fresh_conf),"reason":"Original cached detector predictions differ; no evidence joined."})
                    raise ValueError("Original detector parity failed at "+e["image_id"])
                torch.save(record,path)
            manifest.append({"image_id":e["image_id"],"path":str(path),"sha256":sha256(path),"n":len(record["conf"])})
            if (n+1)%50==0 or n+1==len(entries):
                print(f"CAPTURE {split} {n+1}/{len(entries)} elapsed={time.perf_counter()-started:.1f}s",flush=True)
                save(OUT/"execution_status.json",{"status":"running","stage":"P1_capture","split":split,"images":n+1,"total":len(entries),"new_fits":0,"test_accessed":False})
    finally:capture.restore()
    save(BASE/("raw_classes_"+split+"_manifest.json"),{"status":"completed" if len(manifest)==len(entries) else "partial","split":split,"images":len(manifest),"entries":manifest,
        "all_detector_predictions_exact":not drifts,"same_forward_raw_scores":True,"script_sha256":sha256(Path(__file__)),"test_accessed":False})


def load_raw(split):
    if split not in ["train","val"]:raise ValueError("No test")
    m=json.loads((BASE/("raw_classes_"+split+"_manifest.json")).read_text(encoding="utf-8"))
    if m["status"]!="completed":raise ValueError("Incomplete class capture")
    data=[]
    for r in m["entries"]:
        if sha256(r["path"])!=r["sha256"]:raise ValueError("Class cache changed")
        data.append(torch.load(r["path"],map_location="cpu",weights_only=False))
    return data


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("split",choices=["train","val"]);args=parser.parse_args();main(args.split)
