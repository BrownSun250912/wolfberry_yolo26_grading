"""Registered fixed-parameter factor matrix; selection uses internal held only."""
from joint_pipeline_common_20261024 import *
setup()
import itertools
import numpy as np
import pandas as pd
from joint_pipeline_data_20261024 import load_sources,cohort_manifest
from joint_pipeline_capture_20261024 import load_raw
from joint_pipeline_policy_20261024 import decide
from joint_pipeline_evaluate_20261024 import evaluate

RECIPES={"standard":(False,False,False),"boundary":(True,False,False),"stem":(False,True,False),
         "boundary_stem":(True,True,False),"recovery":(False,False,True),
         "boundary_recovery":(True,False,True),"stem_recovery":(False,True,True),"full":(True,True,True)}
EVIDENCE=["top1","raw_same_candidate","neighbour"]


def register():
    p=BASE/"decision_matrix_plan.json"
    value={"registered_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"change_from_overall_plan":"Complete both remaining interaction cells of three binary decisions before evaluation; standard evidence repeated only once.",
      "sources":["original","selected22","direct_rule","rule_anchored22","rule_anchored23","rule_anchored24"],
      "recipes":{k:list(v) for k,v in RECIPES.items()},"evidence":EVIDENCE,"dedup":["none","same_class","scope_priority"],
      "confidence":"Raw detector confidence, no old ranker transplanted after new class decisions",
      "partitions":["train_head_held","val"],
      "selection":"Within fixed source rule_anchored22, raw confidence, scope_priority; internal held max correct_per_all_GT with UQ_escape<=source standard and qualified_rejected<=standard; ties fewer enabled decisions then fixed evidence order top1/raw_same_candidate/neighbour. No val endpoint for selection.",
      "fixed_reference":"neighbour + full same fixed alpha.5/.5/lambda1.5 is reported regardless of gate; unsafe recovery not promoted",
      "parameter_variants":"Fixed main parameters only in this matrix; historical held-selected alpha.35/.5/lambda2 separately retained, not reoptimized here.",
      "no_new_fits":True,"test_accessed":False,"script_sha256":sha256(Path(__file__))}
    if p.exists():return json.loads(p.read_text(encoding="utf-8"))
    save(p,value);return value


def joined_sources(split):
    sources,quality=load_sources(split);raw=load_raw(split)
    for name,entries in sources.items():
        for i,(e,r) in enumerate(zip(entries,raw)):
            if e["image_id"]!=r["image_id"] or not np.array_equal(e["conf"],r["conf"]) or not np.array_equal(e["boxes"],r["boxes"]):raise ValueError("Raw score evidence alignment failure")
            entries[i]=dict(e,class_scores=r["class_scores"],candidate_indices=r["candidate_indices"])
    return sources,quality


def main():
    plan();registration=register();rows=[];started=time.perf_counter();selection=None
    roles=pd.read_csv(ROOT/"models/experiments/r1_quality_followup_20261009/head_image_split.csv").set_index("image_id")
    for split in ["train","val"]:
        if stop_requested():break
        sources,quality=joined_sources(split)
        cohort=cohort_manifest(split,sources["original"],quality)
        if split=="train":
            ids=[i for i,e in enumerate(sources["original"]) if roles.loc[e["image_id"],"role"]=="head_holdout"]
            sources={k:[v[i] for i in ids] for k,v in sources.items()}
            images={e["image_id"] for e in sources["original"]};cohort=[r for r in cohort if r["image_id"] in images]
        partition="train_head_held" if split=="train" else "val"
        for source,entries in sources.items():
            for recipe,flags in RECIPES.items():
                evidences=["top1"] if recipe=="standard" else EVIDENCE
                for ev in evidences:
                    labels=[decide(e,ev,*flags) for e in entries]
                    for dedup in ["none","same_class","scope_priority"]:
                        metrics,cm,cases,_=evaluate(entries,labels,None,dedup,cohort,split=="val" and source in ["selected22","rule_anchored22"] and dedup=="scope_priority")
                        row=dict(partition=partition,source=source,recipe=recipe,evidence=ev,dedup=dedup,enabled=sum(flags),**metrics)
                        rows.append(row)
                        if cases:pd.DataFrame(cases).to_csv(OUT/f"p2_{source}_{recipe}_{ev}_{dedup}_cases.csv",index=False)
            pd.DataFrame(rows).to_csv(OUT/"p2_decision_matrix.csv",index=False)
            print(f"P2 {partition} {source} {len(rows)} conditions",flush=True)
            if stop_requested():break
        if split=="train":
            frame=pd.DataFrame(rows);focus=frame[(frame.source=="rule_anchored22")&(frame.dedup=="scope_priority")].copy()
            reference=focus[focus.recipe=="standard"].iloc[0]
            eligible=focus[(focus.UQ_escape<=reference.UQ_escape)&(focus.qualified_rejected<=reference.qualified_rejected)].copy()
            eligible["evidence_order"]=eligible.evidence.map({e:i for i,e in enumerate(EVIDENCE)})
            selected=eligible.sort_values(["correct_per_all_GT","enabled","evidence_order"],ascending=[False,True,True],kind="stable").iloc[0]
            selection={"source":selected.source,"recipe":selected.recipe,"evidence":selected.evidence,"dedup":selected.dedup,
                       "held_correct":int(selected.correct_grade),"held_localized":int(selected.localized),"held_GT":int(selected.GT),
                       "held_UQ_escape":int(selected.UQ_escape),"held_qualified_rejected":int(selected.qualified_rejected),
                       "standard_UQ_escape":int(reference.UQ_escape),"standard_qualified_rejected":int(reference.qualified_rejected),
                       "selected_before_val_matrix":True,"val_not_independent":True}
            save(OUT/"p2_internal_selection.json",selection)
    pd.DataFrame(rows).to_csv(OUT/"p2_decision_matrix.csv",index=False)
    save(OUT/"p2_status.json",{"status":"completed" if len(rows)==792 else "partial","rows":len(rows),"elapsed_s":time.perf_counter()-started,"new_fits":0,"selection":selection,"test_accessed":False})
    print(f"P2 DONE rows={len(rows)} elapsed={time.perf_counter()-started:.1f}s",flush=True)


if __name__=="__main__":main()
