"""Task-specific heads with explicit conservative interventions and no val choice."""
from attribute_local_common_20261025 import *
setup()
import argparse
import numpy as np,pandas as pd,torch
from attribute_local_evaluate_20261025 import load_records,outputs,labels_for
from attribute_local_prior_20261025 import correct
from joint_pipeline_decision_matrix_20261024 import joined_sources
from joint_pipeline_policy_20261024 import decide,evidence,logit
from joint_pipeline_evaluate_20261024 import evaluate
setup()

def register():
    p=BASE/"target_intervention_plan.json"
    if not p.exists():save(p,dict(registered_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
       question="Avoid whole-population replacement when new binary heads have asymmetric errors; compare fixed conservative updates to direct replacement.",
       registered_after="Primary12val and priorcorrection results observed; target24fits started, no targetheld/val pipeline outputs observed.",
       defect_variants=["direct.5","creation_only.9","creation_only.95","creation_only.99","hysteresis.05_.95","detector_consensus.05_.95"],
       stem_variants="Direct stem replace with/without analytic fitweight odds correction; originaldefect gate retained",
       corrections="balanced heads raw andfitweight-corrected,unweighted raw only; fixedgrid, no threshold optimized onval",
       selection="All3seed clean held mean correct>=reference,UQescape<=reference,qualifiedreject<=reference,newlymisaccepted_GT<=0. PrioritizeAP95 thencorrect, fewer interventions ties. Reference wins noimprovement. Separate taskgate selection; fixedseed25.",
       interpretation="Thresholds are engineering prespecified candidates,not calibratedrisk guarantees. Newescape usesGT only for evaluation,not runtimeintervention. Reusedval/adaptive development disclosed.",test_accessed=False))

def gate_labels(entries,records,probs,target,variant):
    if target=="stem":return labels_for(entries,records,probs,"replace_stem")
    labels=[]
    for e,r,p in zip(entries,records,probs):
        ids=r["prediction_indices"];state=(e["baseline"]==0).copy();d=p[:,0]
        if variant=="direct.5":state[ids]=d>=.5
        elif variant.startswith("creation_only."):
            threshold=float("0."+variant.split(".",1)[1]);state[ids[d>=threshold]]=True
        elif variant=="hysteresis.05_.95":
            state[ids[d>=.95]]=True;state[ids[d<=.05]]=False
        elif variant=="detector_consensus.05_.95":
            c=logit(evidence(e,"neighbour"));margin=c[:,0]-c[:,1:].max(1)
            state[ids[(d>=.95)&(margin[ids]>0)]]=True;state[ids[(d<=.05)&(margin[ids]<0)]]=False
        else:raise ValueError(variant)
        labels.append(decide(e,"neighbour",True,True,False,uq_state=state))
    return labels

def main(split):
    plan();register();ledger=pd.read_csv(OUT/"target_fit_ledger.csv")
    if len(ledger)!=24 or ledger.partial.any():raise ValueError("Complete24targetfits needed")
    if split=="val" and not (OUT/"target_internal_selection.json").exists():raise ValueError("Freeze targetfamily first")
    cache="train" if split=="held" else "val";sources,_=joined_sources(cache);entries=sources["rule_anchored22"];records=load_records(cache)
    if split=="held":
        roles=pd.read_csv(ROOT/"models/experiments/r1_quality_followup_20261009/head_image_split.csv").set_index("image_id")
        ids=[i for i,e in enumerate(entries) if roles.loc[e["image_id"],"role"]=="head_holdout"];entries=[entries[i] for i in ids];records=[records[i] for i in ids]
    reference,_,refcases,_=evaluate(entries,[decide(e,"neighbour",True,True,False) for e in entries],dedup="scope_priority",case_rows=True)
    safe_ids={r["instance_id"] for r in refcases if r["gt_class"]==0 and r["pred_class"]==0};rows=[];directory=BASE/("target_outputs_"+split);directory.mkdir(exist_ok=True)
    for r in ledger.to_dict("records"):
        cp=torch.load(r["path"],map_location="cpu",weights_only=False);path=directory/(Path(r["path"]).parent.name+".pt")
        if path.exists():probs=torch.load(path,map_location="cpu",weights_only=False)["probabilities"]
        else:
            probs=outputs(cp,records);torch.save(dict(image_ids=[e["image_id"] for e in entries],probabilities=probs,checkpoint_sha256=sha256(r["path"])),path)
        for correction in ([False,True] if r["balanced"] else [False]):
            pp=[correct(p,cp,correction and r["target"]=="defect",correction and r["target"]=="stem") for p in probs]
            variants=["direct.5"] if r["target"]=="stem" else ["direct.5","creation_only.9","creation_only.95","creation_only.99","hysteresis.05_.95","detector_consensus.05_.95"]
            for variant in variants:
                labels=gate_labels(entries,records,pp,r["target"],variant);m,cm,cases,_=evaluate(entries,labels,dedup="scope_priority",case_rows=True)
                newly=sum(c["instance_id"] in safe_ids and c["pred_class"]!=0 for c in cases if c["gt_class"]==0)
                rows.append(dict(target=r["target"],balanced=r["balanced"],paired=r["paired"],seed=r["seed"],corrected=correction,variant=variant,new_UQ_escape=newly,**m))
        pd.DataFrame(rows).to_csv(OUT/f"{split}_target_metrics.csv",index=False);print(f"TARGET {split} {Path(r['path']).parent.name} {len(rows)}",flush=True)
    if split=="held":
        f=pd.DataFrame(rows);family=f.groupby(["target","balanced","paired","corrected","variant"]).agg(seeds=("seed","nunique"),correct=("correct_per_all_GT","mean"),AP95=("Mask_AP50_95","mean"),AP50=("Mask_AP50","mean"),UQ=("UQ_escape","mean"),reject=("qualified_rejected","mean"),new_UQ=("new_UQ_escape","mean")).reset_index()
        family["eligible"]=(family.seeds==3)&(family.correct>=reference["correct_per_all_GT"])&(family.UQ<=reference["UQ_escape"])&(family.reject<=reference["qualified_rejected"])&(family.new_UQ==0)
        family.to_csv(OUT/"held_target_families.csv",index=False);good=family[family.eligible]
        selected=None if not len(good) else good.sort_values(["AP95","correct","paired","balanced"],ascending=[False,False,True,True],kind="stable").iloc[0].to_dict()
        if selected and selected["AP95"]<=reference["Mask_AP50_95"] and selected["correct"]<=reference["correct_per_all_GT"]:selected=None
        save(OUT/"target_internal_selection.json",dict(selected=selected,reference=reference,fixed_seed=20261025,before_target_val=True,test_accessed=False))
    save(OUT/f"{split}_target_status.json",dict(status="completed",rows=len(rows),test_accessed=False))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("split",choices=["held","val"]);main(p.parse_args().split)
