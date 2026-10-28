"""Full retained-prediction metrics and separate fixed matched-cohort diagnosis."""
from joint_pipeline_policy_20261024 import keep_indices
import numpy as np
import pandas as pd


def evaluate(entries,labels,ranking=None,dedup="none",cohort=None,case_rows=False,keeps=None,allow_missing_classes=False):
    from analyze_r1_ap_bootstrap import pure_match
    from analyze_val_instance_errors import optimal_match
    from ultralytics.utils.metrics import ap_per_class
    rank=[e["conf"] for e in entries] if ranking is None else ranking
    cohorts={}
    if cohort is not None:
        for row in cohort:cohorts.setdefault(row["image_id"],[]).append(row)
    blocks=[];bb=[];cases=[]
    counters={"predictions_before":0,"predictions_after":0,"high_before":0,"high_after":0,
              "GT":0,"localized":0,"correct_grade":0,"class_aware_TP50":0,
              "fixed_cohort_n":0,"fixed_correct_labels":0,"fixed_original_row_kept":0,"fixed_correct_row_kept":0,
              "UQ_GT":0,"UQ_localized":0,"UQ_correct":0,"UQ_escape":0,"qualified_rejected":0}
    cm=np.zeros((4,4),int)
    for ei,(e,cls,conf) in enumerate(zip(entries,labels,rank)):
        cls=np.asarray(cls,int);conf=np.asarray(conf,float)
        if len(cls)!=len(e["conf"]) or len(conf)!=len(cls) or not np.isfinite(conf).all():raise ValueError("Evaluation alignment")
        if keeps is not None:keep=np.asarray(keeps[ei],int)
        elif dedup=="none":keep=np.arange(len(cls))
        elif dedup in ["same_class","scope_priority"]:
            scope=e["conf"]>=.25 if dedup=="scope_priority" else None
            keep=keep_indices(e["boxes"],cls,conf,.7,scope)
        else:raise ValueError(dedup)
        kept=np.zeros(len(cls),bool);kept[keep]=True
        tp=pure_match(cls[keep],e["gt_cls"],e["mask_iou"][:,keep])
        blocks.append((tp,conf[keep],cls[keep],e["gt_cls"]))
        if "box_iou" in e:
            bb.append((pure_match(cls[keep],e["gt_cls"],e["box_iou"][:,keep]),conf[keep],cls[keep],e["gt_cls"]))
        high=keep[np.asarray(e["conf"])[keep]>=.25]
        matches=optimal_match(e["mask_iou"][:,high],.5)
        counters["predictions_before"]+=len(cls);counters["predictions_after"]+=len(keep)
        counters["high_before"]+=int((e["conf"]>=.25).sum());counters["high_after"]+=len(high)
        counters["GT"]+=len(e["gt_cls"]);counters["localized"]+=len(matches)
        counters["class_aware_TP50"]+=int(pure_match(cls[high],e["gt_cls"],e["mask_iou"][:,high])[:,0].sum())
        counters["UQ_GT"]+=int((e["gt_cls"]==0).sum())
        for gi,pj,iou in matches:
            j=high[pj];gt=int(e["gt_cls"][gi]);pred=int(cls[j]);cm[gt,pred]+=1
            counters["correct_grade"]+=int(gt==pred)
            if gt==0:
                counters["UQ_localized"]+=1;counters["UQ_correct"]+=int(pred==0);counters["UQ_escape"]+=int(pred!=0)
            else:counters["qualified_rejected"]+=int(pred==0)
            if case_rows:cases.append({"image_id":e["image_id"],"gt_index":gi,"instance_id":e["gt_ids"][gi],"prediction_index":int(j),"gt_class":gt,"pred_class":pred,"correct":gt==pred,"mask_iou":float(iou),"raw_conf":float(e["conf"][j]),"rank_conf":float(conf[j])})
        for r in cohorts.get(e["image_id"],[]):
            j=int(r["prediction_index"]);correct=int(cls[j]==int(r["gt_class"]))
            counters["fixed_cohort_n"]+=1;counters["fixed_correct_labels"]+=correct
            counters["fixed_original_row_kept"]+=int(kept[j]);counters["fixed_correct_row_kept"]+=int(kept[j])*correct
    def aps(bs):
        tp,c,cl,gt=[np.concatenate([b[j] for b in bs]) for j in range(4)]
        ap=ap_per_class(tp,c,cl,gt,plot=False)[5]
        if ap.shape!=(4,10) and not allow_missing_classes:raise ValueError("Missing target class changes macro AP denominator")
        return float(ap[:,0].mean()),float(ap.mean()),ap
    a50,a95,ap=aps(blocks)
    metrics={**counters,"Mask_AP50":a50,"Mask_AP50_95":a95,
             "conditional_grade_accuracy":counters["correct_grade"]/max(counters["localized"],1),
             "correct_per_all_GT":counters["correct_grade"]/max(counters["GT"],1),
             "high_class_precision50":counters["class_aware_TP50"]/max(counters["high_after"],1),
             "high_class_recall50":counters["class_aware_TP50"]/max(counters["GT"],1),
             "UQ_recall_among_localized":counters["UQ_correct"]/max(counters["UQ_localized"],1)}
    if bb:
        b50,b95,_=aps(bb);metrics.update(Box_AP50=b50,Box_AP50_95=b95)
    present=np.unique(np.concatenate([e["gt_cls"] for e in entries])).astype(int)
    metrics["present_GT_class_count"]=len(present)
    for c in range(4):
        if c in present:
            j=int(np.flatnonzero(present==c)[0]);metrics[f"class{c}_Mask_AP50"]=float(ap[j,0]);metrics[f"class{c}_Mask_AP95"]=float(ap[j].mean())
        else:metrics[f"class{c}_Mask_AP50"]=None;metrics[f"class{c}_Mask_AP95"]=None
    return metrics,cm,cases,blocks
