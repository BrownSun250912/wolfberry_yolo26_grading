"""Analytic fit-weight odds correction; not a claim of empirical calibration."""
from attribute_local_common_20261025 import *
setup()
import argparse
import numpy as np,pandas as pd,torch
from attribute_local_evaluate_20261025 import load_records,labels_for
from joint_pipeline_decision_matrix_20261024 import joined_sources
from joint_pipeline_policy_20261024 import decide
from joint_pipeline_evaluate_20261024 import evaluate
setup()

def correct(prob,cp,defect,stem):
    p=np.clip(prob,1e-6,1-1e-6);z=np.log(p/(1-p));ratio=(cp["positive_weights"]/cp["negative_weights"]).numpy()
    z=z-np.log(ratio)*np.array([defect,stem],np.float32)
    return 1/(1+np.exp(-z))

def main(split):
    path=BASE/"prior_correction_plan.json"
    if not path.exists():save(path,dict(registered_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
      trigger="Primary held balancedBCE reduces UQescapes but increases qualified rejects; weightedBCE output .5 is not an unweighted population posterior threshold. Primary val evaluation not yet inspected.",
      rule="Subtract log(wpositive/wnegative) from corresponding output logit; weights computed only from head_fit targets. Under ideal weightedBCE optimum restores fit population odds, not empirically verified probability calibration.",
      corrections="Stemonly raw/corrected; defectonly raw/corrected; both fourfactorcombinations; all8effectivevariants/model",
      selection="All3seed mean held correct/allGT>=oldrrreference,UQescape<=reference,qualifiedreject<=reference; sortAP95,correct,lessadaptation thenfixedname; oldreference wins ties/noimprovement. fixedseed25. Separate selection does not overwriteprimary.",
      risks="Representativeness/optimization assumptions may fail; prior correction notECE validation. Reused internalheld andval remain exploratory.",test_accessed=False,new_fits=0))
    if split=="val" and not (OUT/"prior_internal_selection.json").exists():raise ValueError("Freeze correction family before val")
    sources,_=joined_sources("train" if split=="held" else "val");entries=sources["rule_anchored22"];records=load_records("train" if split=="held" else "val")
    if split=="held":
        roles=pd.read_csv(ROOT/"models/experiments/r1_quality_followup_20261009/head_image_split.csv").set_index("image_id")
        ids=[i for i,e in enumerate(entries) if roles.loc[e["image_id"],"role"]=="head_holdout"];entries=[entries[i] for i in ids];records=[records[i] for i in ids]
    reference,_,_,_=evaluate(entries,[decide(e,"neighbour",True,True,False) for e in entries],dedup="scope_priority")
    rows=[]
    for r in pd.read_csv(OUT/"fit_ledger.csv").to_dict("records"):
        cp=torch.load(r["path"],map_location="cpu",weights_only=False);cached=torch.load(BASE/("attribute_outputs_"+split)/(Path(r["path"]).parent.name+".pt"),map_location="cpu",weights_only=False)
        if cached["image_ids"]!=[e["image_id"] for e in entries]:raise ValueError("Output lineage mismatch")
        for policy,dc,sc in [("replace_stem",False,False),("replace_stem",False,True),("replace_defect",False,False),("replace_defect",True,False)]+[("replace_both",d,s) for d in [False,True] for s in [False,True]]:
            probabilities=[correct(p,cp,dc,sc) for p in cached["probabilities"]]
            labels=labels_for(entries,records,probabilities,policy);m,_,_,_=evaluate(entries,labels,dedup="scope_priority")
            rows.append(dict(view=r["view"],adapt=r["adapt"],seed=r["seed"],policy=policy,correct_defect=dc,correct_stem=sc,**m))
        print(f"PRIOR {split} {Path(r['path']).parent.name} {len(rows)}",flush=True)
    pd.DataFrame(rows).to_csv(OUT/f"{split}_prior_metrics.csv",index=False)
    if split=="held":
        f=pd.DataFrame(rows);family=f.groupby(["view","adapt","policy","correct_defect","correct_stem"]).agg(seeds=("seed","nunique"),correct=("correct_per_all_GT","mean"),AP95=("Mask_AP50_95","mean"),AP50=("Mask_AP50","mean"),UQ=("UQ_escape","mean"),reject=("qualified_rejected","mean")).reset_index()
        family["eligible"]=(family.seeds==3)&(family.correct>=reference["correct_per_all_GT"])&(family.UQ<=reference["UQ_escape"])&(family.reject<=reference["qualified_rejected"])
        family.to_csv(OUT/"held_prior_families.csv",index=False);good=family[family.eligible]
        best=None if not len(good) else good.sort_values(["AP95","correct","adapt","view"],ascending=[False,False,True,True],kind="stable").iloc[0].to_dict()
        if best and best["AP95"]<=reference["Mask_AP50_95"] and best["correct"]<=reference["correct_per_all_GT"]:best=None
        save(OUT/"prior_internal_selection.json",dict(selected=best,reference=reference,fixed_seed=20261025,test_accessed=False,before_correction_val=True))
    save(OUT/f"{split}_prior_status.json",dict(status="completed",rows=len(rows),test_accessed=False,new_fits=0))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("split",choices=["held","val"]);main(p.parse_args().split)
