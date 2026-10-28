"""Full-mask AP first, then actual legacy scoring replay on refined masks."""
from attribute_local_common_20261025 import *
setup()
import cv2,numpy as np,pandas as pd,torch
import torch.nn.functional as F
from ultralytics.utils.metrics import mask_iou
from attribute_local_proto_fit_20261025 import PointHead,apply_native
from joint_pipeline_decision_matrix_20261024 import joined_sources
from joint_pipeline_policy_20261024 import decide
from joint_pipeline_evaluate_20261024 import evaluate
from joint_pipeline_runtime_20261024 import SharedGrader
from direction_experiments_20261023 import load_records
from analyze_val_instance_errors import polygon_to_mask
setup()

def deltas(model,records,soft,proto):
    if not records:return []
    with torch.no_grad():
        rgb=torch.stack([r["rgb"] for r in records]).cuda().float()/255;coarse=torch.stack([r["coarse"] for r in records])[:,None].cuda().float()
        sm=torch.stack([r["soft"] for r in records]).cuda().float();pp=torch.stack([r["prototype"] for r in records]).cuda().float()
        return model(rgb,coarse,sm,pp,soft,proto).cpu().numpy()[:,0]

def full_masks(record,coarse,delta):
    masks=[polygon_to_mask(p,*coarse["shape"]) for p in coarse["polygons"]]
    for r,d in zip(record["records"],delta):
        j=r["prediction_index"];x0,y0,x1,y1=r["bounds"];mask=masks[j].copy();mask[y0:y1,x0:x1]=apply_native(r,d);masks[j]=mask
    return masks

def recompute_iou(gt,masks):
    if not masks:return np.zeros((len(gt),0),np.float32)
    return mask_iou(torch.tensor(np.stack(gt),device="cuda",dtype=torch.float32).flatten(1),torch.tensor(np.stack(masks),device="cuda",dtype=torch.float32).flatten(1)).cpu().numpy()

def main():
    plan();selection=json.loads((OUT/"prototype_internal_selection.json").read_text(encoding="utf-8"))
    if selection["selected"] is None:raise ValueError("No internally eligible boundaryfamily; do not pick valwinner")
    sources,_=joined_sources("val");original=sources["rule_anchored22"];coarse=load_records("val")
    manifest=json.loads((BASE/"prototype_val_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"]!="completed":raise ValueError("Incomplete valprototype cache")
    data=[torch.load(r["path"],map_location="cpu",weights_only=False) for r in manifest["entries"]]
    reference_labels=[decide(e,"neighbour",True,True,False) for e in original];rows=[];pointrows=[];selected=selection["selected"];grader=None
    try:
        for r in pd.read_csv(OUT/"prototype_fit_ledger.csv").to_dict("records"):
            cp=torch.load(r["path"],map_location="cpu",weights_only=False);model=PointHead().cuda().eval();model.load_state_dict(cp["state"])
            chosen=r["soft"]==selected["soft"] and r["prototype"]==selected["prototype"]
            if chosen and grader is None:grader=SharedGrader(rule_sources=True)
            entries=[];rescored=[];new_labels=[];directory=BASE/"refined_val"/str(r["seed"])
            if chosen:directory.mkdir(parents=True,exist_ok=True)
            for e,old,record in zip(original,coarse,data):
                delta=deltas(model,record["records"],cp["soft"],cp["prototype"]);masks=full_masks(record,old,delta);gt=[polygon_to_mask(p,*old["shape"]) for p in old["gt_polygons"]]
                iou=recompute_iou(gt,masks);entry=dict(e,mask_iou=iou,mask_area=np.array([np.count_nonzero(m) for m in masks],np.float32));entries.append(entry)
                for rec,d in zip(record["records"],delta):
                    if rec["gt_index"]<0:continue
                    j=rec["prediction_index"];gi=rec["gt_index"]
                    pointrows.append(dict(soft=r["soft"],prototype=r["prototype"],seed=r["seed"],image_id=e["image_id"],prediction_index=j,instance_id=rec["instance_id"],old_IoU=float(e["mask_iou"][gi,j]),new_IoU=float(iou[gi,j])))
                if chosen:
                    outputs=grader.infer_sources(cv2.imread(e["image_path"]),masks,e["baseline"],e["conf"])
                    value=dict(entry,score=outputs["score_sources"]["rule_anchored22"],stem=outputs["stem"],defect_logit=outputs["defect_logit"])
                    rescored.append(value);new_labels.append(decide(value,"neighbour",True,True,False))
                    torch.save(dict(entry=value,mask_only_labels=reference_labels[len(entries)-1],full_labels=new_labels[-1],deltas=delta,refiner_checkpoint=r["path"],empty_masks=int(sum(not np.any(m) for m in masks))),directory/(e["image_id"]+".pt"))
            m,_,_,_=evaluate(entries,reference_labels,dedup="scope_priority");rows.append(dict(soft=r["soft"],prototype=r["prototype"],seed=r["seed"],arm="mask_only_fixed_grades",**m))
            if chosen:
                m,_,cases,_=evaluate(rescored,new_labels,dedup="scope_priority",case_rows=True);rows.append(dict(soft=r["soft"],prototype=r["prototype"],seed=r["seed"],arm="full_legacy_regrading",**m))
                pd.DataFrame(cases).to_csv(OUT/f"prototype_full_seed{r['seed']}_cases.csv",index=False)
            pd.DataFrame(rows).to_csv(OUT/"prototype_val_metrics.csv",index=False);pd.DataFrame(pointrows).to_csv(OUT/"prototype_val_instance_metrics.csv",index=False)
            print(f"PROTOTYPE VAL soft={r['soft']} proto={r['prototype']} seed={r['seed']} rows={len(rows)}",flush=True)
    finally:
        if grader is not None:grader.close()
    save(OUT/"prototype_val_status.json",dict(status="completed",rows=len(rows),selected=selected,test_accessed=False,primary_replaced=False))

if __name__=="__main__":main()
