"""Load immutable original caches and scoring sources, never test."""
from joint_pipeline_common_20261024 import ROOT,BASE,OUT,sha256,save
import numpy as np
import pandas as pd
import torch


def load_sources(split,include_all_seeds=True):
    if split not in ["train","val"]:raise ValueError("No test access")
    import refine_rule_anchored_residual as anchor
    import refine_score_ranking as ranking
    from refine_score_direction import STAGE as SCORE
    from cache_r1_quality_features import load_quality
    from validate_r1_closure_runtime import STEM
    original,rule=anchor.load_rule_entries(split);quality=load_quality(split)
    cp=torch.load(STEM,map_location="cpu",weights_only=False)
    weight=cp["model_state"]["defect_head.weight"].numpy()[0];bias=float(cp["model_state"]["defect_head.bias"][0])
    for i,(e,q) in enumerate(zip(original,quality)):
        if e["image_id"]!=q["image_id"] or len(e["conf"])!=len(q["features"]):raise ValueError("Quality/cache misalignment")
        logits=q["features"].numpy()@weight+bias
        valid=np.zeros(len(logits),bool);valid[np.asarray(q["valid_indices"],int)]=True
        logits[~valid]=np.nan
        original[i]=dict(e,defect_logit=logits)
        rule[i]=dict(rule[i],defect_logit=logits)
    score_path=SCORE/"control_seed20261022/best.pt"
    cp_score=torch.load(score_path,map_location="cpu",weights_only=False)
    sources={"original":original,"selected22":ranking.revised(original,quality,cp_score),"direct_rule":rule}
    for seed in ([20261022,20261023,20261024] if include_all_seeds else [20261022]):
        cp=torch.load(anchor.BASE/f"rule_base_residual_seed{seed}/best.pt",map_location="cpu",weights_only=False)
        sources[f"rule_anchored{str(seed)[-2:]}"]=anchor.revise(rule,quality,cp)
    return sources,quality


def ranking_for(source,entries,seed=20261008):
    import refine_score_ranking as r
    import refine_rule_anchored_residual as a
    import refine_rule_reliability as d
    from train_r1_confidence_followup import ReliabilityHead,confidences
    if source=="original":path=r.BASE/"original_score"/f"rank_seed{seed}/best.pt"
    elif source=="selected22":path=r.BASE/"control_score20261022"/f"rank_seed{seed}/best.pt"
    elif source=="direct_rule":path=d.BASE/"reliability/direct_rule"/f"rank_seed{seed}/best.pt"
    elif source.startswith("rule_anchored"):
        path=a.BASE/"reliability"/f"rule_base_score202610{source[-2:]}"/f"rank_seed{seed}/best.pt"
    else:raise ValueError(source)
    cp=torch.load(path,map_location="cpu",weights_only=False)
    model=ReliabilityHead(cp["family"]);model.load_state_dict(cp["head_state"]);model.eval()
    return confidences(model,cp,entries,"product"),{"path":str(path),"sha256":sha256(path)}


def cohort_manifest(split,entries,quality):
    roles=pd.read_csv(ROOT/"models/experiments/r1_quality_followup_20261009/head_image_split.csv").set_index("image_id") if split=="train" else None
    rows=[]
    for e,q in zip(entries,quality):
        for m in q["matched_high_predictions"]:
            j=m["prediction_index"]
            rows.append({"split":split,"image_id":e["image_id"],"source_group":e["source_group"],
                         "role":"val" if roles is None else roles.loc[e["image_id"],"role"],
                         "instance_id":m["instance_id"],"prediction_index":j,"gt_index":m["gt_index"],"gt_class":m["gt_class"],
                         "original_class":int(e["baseline"][j]),"raw_conf":float(e["conf"][j]),"mask_iou":m["mask_iou"]})
    frame=pd.DataFrame(rows);frame.to_csv(OUT/f"cohort_{split}.csv",index=False)
    return rows
