"""Held-only factorial family selection; later val evaluation of fixed candidates."""
from attribute_local_common_20261025 import *
setup()
import argparse
import numpy as np,pandas as pd,torch
from attribute_local_fit_20261025 import AttributeModel,original,predict
from joint_pipeline_decision_matrix_20261024 import joined_sources
from joint_pipeline_policy_20261024 import decide
from joint_pipeline_evaluate_20261024 import evaluate
setup()
POLICIES=["replace_stem","replace_defect","replace_both"]

def load_records(split):
    m=json.loads((BASE/f"prefix_{split}_manifest.json").read_text(encoding="utf-8"))
    if m["status"]!="completed":raise ValueError("Incomplete prefix")
    return [torch.load(r["path"],map_location="cpu",weights_only=False) for r in m["entries"]]

def outputs(cp,records):
    model=AttributeModel(original(),cp["view"],cp["adapt"]).cuda().eval();model.load_state_dict(cp["state"]);values=[]
    for e in records:
        with torch.no_grad():p=predict(model,e["prefix"].cuda(),torch.tensor(e["geometry"],device="cuda")).sigmoid().cpu().numpy()
        values.append(p)
    return values

def labels_for(entries,records,probabilities,policy,defect_threshold=.5,stem_threshold=.5):
    labels=[]
    for e,r,p in zip(entries,records,probabilities):
        assert e["image_id"]==r["image_id"];ids=r["prediction_indices"];entry=dict(e);uq=e["baseline"]==0
        if policy in ["replace_defect","replace_both"]:
            uq=uq.copy();uq[ids]=p[:,0]>=defect_threshold
        if policy in ["replace_stem","replace_both"]:
            stem=e["stem"].copy()
            # Map calibrated threshold to the policy's fixed .5 logit origin; identity at.5.
            pp=np.clip(p[:,1],1e-6,1-1e-6);shift=np.log(stem_threshold/(1-stem_threshold))
            stem[ids]=1/(1+np.exp(-(np.log(pp/(1-pp))-shift)));entry["stem"]=stem
        labels.append(decide(entry,"neighbour",True,True,False,uq_state=uq))
    return labels

def main(split):
    plan();setup();ledger=pd.read_csv(OUT/"fit_ledger.csv")
    if len(ledger)!=12 or ledger.partial.any():raise ValueError("All12pairedfits must finish before family selection")
    if split=="val" and not (OUT/"internal_selection.json").exists():raise ValueError("No frozen selection")
    cache_split="train" if split=="held" else "val";sources,_=joined_sources(cache_split);entries=sources["rule_anchored22"];records=load_records(cache_split)
    if split=="held":
        roles=pd.read_csv(ROOT/"models/experiments/r1_quality_followup_20261009/head_image_split.csv").set_index("image_id")
        ids=[i for i,e in enumerate(entries) if roles.loc[e["image_id"],"role"]=="head_holdout"];entries=[entries[i] for i in ids];records=[records[i] for i in ids]
    reference,_,_,_=evaluate(entries,[decide(e,"neighbour",True,True,False) for e in entries],dedup="scope_priority")
    rows=[];prediction_dir=BASE/("attribute_outputs_"+split);prediction_dir.mkdir(exist_ok=True)
    for r in ledger.to_dict("records"):
        cp=torch.load(r["path"],map_location="cpu",weights_only=False);probs=outputs(cp,records)
        torch.save(dict(image_ids=[e["image_id"] for e in entries],prediction_indices=[r["prediction_indices"] for r in records],probabilities=probs,checkpoint_sha256=sha256(r["path"])),prediction_dir/(Path(r["path"]).parent.name+".pt"))
        for policy in POLICIES:
            m,cm,cases,_=evaluate(entries,labels_for(entries,records,probs,policy),dedup="scope_priority",case_rows=split=="val" and r["seed"]==20261025)
            rows.append(dict(view=r["view"],adapt=r["adapt"],seed=r["seed"],policy=policy,checkpoint=r["path"],**m))
            if cases:pd.DataFrame(cases).to_csv(OUT/f"{split}_{Path(r['path']).parent.name}_{policy}_cases.csv",index=False)
        pd.DataFrame(rows).to_csv(OUT/f"{split}_pipeline_metrics.csv",index=False)
        print(f"ATTRIBUTE {split} {Path(r['path']).parent.name} {len(rows)}rows",flush=True)
    if split=="held":
        f=pd.DataFrame(rows);family=f.groupby(["view","adapt","policy"]).agg(seeds=("seed","nunique"),correct=("correct_per_all_GT","mean"),AP95=("Mask_AP50_95","mean"),AP50=("Mask_AP50","mean"),UQ=("UQ_escape","mean"),reject=("qualified_rejected","mean")).reset_index()
        family["eligible"]=(family.seeds==3)&(family.correct>=reference["correct_per_all_GT"])&(family.UQ<=reference["UQ_escape"])&(family.reject<=reference["qualified_rejected"])
        family.to_csv(OUT/"held_family_metrics.csv",index=False);good=family[family.eligible]
        if len(good):
            best=good.sort_values(["AP95","correct","adapt","view"],ascending=[False,False,True,True],kind="stable").iloc[0].to_dict()
            if best["AP95"]<=reference["Mask_AP50_95"] and best["correct"]<=reference["correct_per_all_GT"]:best=None
        else:best=None
        save(OUT/"internal_selection.json",dict(selected=best,reference=reference,fixed_seed=20261025,before_val=True,test_accessed=False))
    save(OUT/f"{split}_evaluation_status.json",dict(status="completed",rows=len(rows),reference=reference,test_accessed=False))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("split",choices=["held","val"]);main(p.parse_args().split)
