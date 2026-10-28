"""Paired train-only ranking fits after new decisions and true duplicate removal."""
from joint_pipeline_common_20261024 import *
setup()
import itertools
import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from joint_pipeline_decision_matrix_20261024 import joined_sources
from joint_pipeline_policy_20261024 import decide,evidence,logit,LOW,HIGH
from joint_pipeline_evaluate_20261024 import evaluate
from analyze_r1_ap_bootstrap import pure_match

SEEDS=[20261024,20261025,20261026]
POLICIES=[("selected22","standard"),("selected22","boundary_stem"),("rule_anchored22","boundary_stem")]
TARGETS=["one_to_one_tp50","independent_tp50","independent_multi_iou"]
FEATURES=["base14_zero4","base14_evidence4"]
FITS=BASE/"rank_fits"


def label_entries(entries,recipe):
    return [decide(e,"neighbour",recipe!="standard",recipe!="standard",False) for e in entries]


def inputs(e,cls,extra):
    conf=np.clip(e["conf"],1e-5,1-1e-5);s=np.nan_to_num(e["score"],nan=50.)
    stem=np.clip(np.nan_to_num(e["stem"],nan=.5),1e-6,1-1e-6)
    b=e["boxes"];w=np.maximum(b[:,2]-b[:,0],1.);h=np.maximum(b[:,3]-b[:,1],1.)
    area=e["image_height"]*e["image_width"]
    cols=[np.log(conf/(1-conf)),s/100,np.minimum(np.abs(s-LOW),np.abs(s-HIGH))/100,
          np.log(stem/(1-stem))/10,np.log(np.maximum(e["mask_area"]/area,1e-7)),np.log(np.maximum(w*h/area,1e-7)),np.log(w/h),(cls==e["baseline"]).astype(float)]
    cols.extend([(cls==c).astype(float) for c in [1,2,3]])
    cols.extend([(e["baseline"]==c).astype(float) for c in [1,2,3]])
    x=np.stack(cols,1).astype(np.float32)
    add=logit(evidence(e,"neighbour"))/10 if extra else np.zeros((len(cls),4),np.float32)
    out=np.concatenate([x,add],1)
    if out.shape!=(len(cls),18) or not np.isfinite(out).all():raise ValueError("Ranking features invalid")
    return out


def targets(e,cls,kind):
    if kind=="one_to_one_tp50":return pure_match(cls,e["gt_cls"],e["mask_iou"])[:,0].astype(np.float32)
    compatible=np.where(e["gt_cls"][:,None]==cls[None,:],e["mask_iou"],0.)
    best=compatible.max(0) if len(compatible) else np.zeros(len(cls))
    if kind=="independent_tp50":return (best>=.5).astype(np.float32)
    if kind=="independent_multi_iou":return (best[:,None]>=np.linspace(.5,.95,10)[None,:]).mean(1).astype(np.float32)
    raise ValueError(kind)


class RankHead(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(18,16),nn.SiLU(),nn.Linear(16,1))
    def forward(self,x):return self.net(x).squeeze(1)


def register():
    FITS.mkdir(exist_ok=True);path=BASE/"rank_fit_plan.json"
    value={"registered_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"reason":"Distinguish pointwise label noise from duplicate competition, after true-NMS baseline and class-evidence matrix.",
       "policies":POLICIES,"targets":TARGETS,"features":FEATURES,"seeds":SEEDS,"fits":54,
       "architecture":"18->16->1 SiLU,321 allocated parameters; base14 control zero-fills extra4; equal effective capacity is NOT claimed",
       "population":"All original train predictions with baseline!=0, including low confidence and unmatched; labels fixed by candidate inference, GT only defines offline rank targets.",
       "split":"Existing calibration_image_split.csv image roles; no val labels in fit/normalization/checkpoint selection; original encoder saw original train.",
       "dose":"AdamW lr.003 wd.0001 batch256, max300, patience30, best held BCE; epoch0 eligible; same seed init and full shuffle order across target/features within each source/recipe",
       "normalization":"fit rows only, same core14 normalizer across families; extra4 zero columns std1",
       "confidence":"sigmoid(rank)*raw detector confidence, original baseline0 confidence preserved; same-class NMS with raw-scope priority; rank score is not calibrated probability",
       "family_selection":"For each fixed policy, complete3seed families only; mean held AP50>=raw-NMS -.002, mean held AP95>=raw-NMS+.002, mean held UQ escapes<=raw-NMS. Among eligible maximize mean held AP95 then AP50; fixed seed20261024, report all3 seeds; if no eligible use raw confidence.",
       "development":"Only after all families for a policy are selected from held, evaluate all completed seeds on reused val, retain all negatives; do not choose best val seed.",
       "budget":"Stop after epoch/checkpoint if new-fit cutoff or original-window deadline; partials excluded from complete-family selection",
       "test_accessed":False,"primary_replaced":False,"script_sha256":sha256(Path(__file__))}
    if path.exists():return json.loads(path.read_text(encoding="utf-8"))
    save(path,value);return value


def rank_confidence(entries,labels,cp,head):
    out=[]
    with torch.inference_mode():
        for e,cls in zip(entries,labels):
            x=inputs(e,cls,cp["feature_kind"]=="base14_evidence4")
            p=torch.sigmoid(head(torch.tensor((x-cp["mean"])/cp["std"]))).numpy()
            c=p*e["conf"];c[e["baseline"]==0]=e["conf"][e["baseline"]==0];out.append(c.astype(float))
    return out


def prepare(entries,labels,role,feature_kind,target_kind):
    data={"fit":[[],[],[]],"held":[[],[],[]]}
    for e,cls in zip(entries,labels):
        x=inputs(e,cls,feature_kind=="base14_evidence4");y=targets(e,cls,target_kind);use=np.flatnonzero(e["baseline"]!=0)
        key="held" if role.loc[e["image_id"],"role"]=="calibration_holdout" else "fit"
        data[key][0].append(x[use]);data[key][1].append(y[use]);data[key][2].extend([(e["image_id"],int(j)) for j in use])
    fx,fy=[np.concatenate(data["fit"][i]) for i in [0,1]];hx,hy=[np.concatenate(data["held"][i]) for i in [0,1]]
    mean=fx.mean(0);std=fx.std(0);std[std<1e-6]=1
    return tuple(torch.tensor(v) for v in [(fx-mean)/std,fy,(hx-mean)/std,hy]),mean,std,data["fit"][2]


def fit(source,recipe,feature_kind,target_kind,seed,data,mean,std,ids):
    directory=FITS/f"{source}_{recipe}_{feature_kind}_{target_kind}_seed{seed}"
    if (directory/"completion.json").exists():return json.loads((directory/"completion.json").read_text(encoding="utf-8"))
    if directory.exists():raise FileExistsError("Inspect partial rank fit rather than silently restart: "+str(directory))
    directory.mkdir();torch.manual_seed(seed);model=RankHead();x,y,hx,hy=data
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    torch.save(initial,directory/"initial_state.pt")
    generator=torch.Generator().manual_seed(seed);first=torch.randperm(len(x),generator=torch.Generator().manual_seed(seed))[:32].tolist()
    save(directory/"first32_ids.json",[ids[j] for j in first])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.0001)
    best=float("inf");best_epoch=0;history=[];started=time.perf_counter();partial=False
    for epoch in range(301):
        if epoch:
            model.train()
            for batch in torch.randperm(len(x),generator=generator).split(256):
                optimizer.zero_grad();loss=nn.functional.binary_cross_entropy_with_logits(model(x[batch]),y[batch]);loss.backward();optimizer.step()
        model.eval()
        with torch.no_grad():held=float(nn.functional.binary_cross_entropy_with_logits(model(hx),hy))
        history.append({"epoch":epoch,"held_BCE":held})
        if held<best-1e-6:
            best=held;best_epoch=epoch
            torch.save({"state":model.state_dict(),"mean":mean,"std":std,"feature_kind":feature_kind,"target_kind":target_kind,"seed":seed,
                        "source":source,"recipe":recipe,"epoch":epoch,"held_BCE":held},directory/"best.pt")
        partial=stop_requested(new_fit=True)
        if epoch-best_epoch>=30 or partial:break
    pd.DataFrame(history).to_csv(directory/"history.csv",index=False)
    torch.save({"state":model.state_dict(),"optimizer":optimizer.state_dict(),"epoch":epoch,"generator_state":generator.get_state()},directory/"last.pt")
    record={"source":source,"recipe":recipe,"feature_kind":feature_kind,"target_kind":target_kind,"seed":seed,"path":str(directory/"best.pt"),
            "epochs":epoch,"best_epoch":best_epoch,"held_BCE":best,"partial":partial,"seconds":time.perf_counter()-started,
            "initial_sha256":sha256(directory/"initial_state.pt"),"first32_sha256":sha256(directory/"first32_ids.json"),"checkpoint_sha256":sha256(directory/"best.pt")}
    save(directory/"completion.json",record);print(f"RANK FIT {source}/{recipe} {feature_kind}/{target_kind} seed{seed} best={best:.5f} e={epoch}",flush=True)
    return record


def main():
    plan();registration=register();train,_=joined_sources("train")
    role=pd.read_csv(ROOT/"models/experiments/r1_ap_followup_20261008/calibration_image_split.csv").set_index("image_id")
    results=[];records=[];selections=[];val_sources=None
    for source,recipe in POLICIES:
        entries=train[source];labels=label_entries(entries,recipe)
        held_ids=[i for i,e in enumerate(entries) if role.loc[e["image_id"],"role"]=="calibration_holdout"]
        he=[entries[i] for i in held_ids];hl=[labels[i] for i in held_ids]
        zero,_,_,_=evaluate(he,hl,dedup="scope_priority")
        prepared={(feature,target):prepare(entries,labels,role,feature,target) for feature,target in itertools.product(FEATURES,TARGETS)}
        for seed in SEEDS:
            pairs=list(itertools.product(FEATURES,TARGETS));np.random.default_rng(seed).shuffle(pairs)
            initial_hashes=[];order_hashes=[]
            for feature,target in pairs:
                if stop_requested(new_fit=True):break
                data,mean,std,ids=prepared[(feature,target)]
                record=fit(source,recipe,feature,target,seed,data,mean,std,ids);records.append(record)
                initial_hashes.append(record["initial_sha256"]);order_hashes.append(record["first32_sha256"])
                cp=torch.load(record["path"],map_location="cpu",weights_only=False);model=RankHead();model.load_state_dict(cp["state"]);model.eval()
                conf=rank_confidence(he,hl,cp,model);metrics,_,_,_=evaluate(he,hl,conf,"scope_priority")
                results.append(dict(partition="calibration_held",**record,**metrics))
                pd.DataFrame(records).to_csv(OUT/"rank_fit_ledger.csv",index=False)
                pd.DataFrame(results).to_csv(OUT/"rank_fit_metrics.csv",index=False)
            if len(initial_hashes)>1 and (len(set(initial_hashes))!=1 or len(set(order_hashes))!=1):raise ValueError("Paired initialization/order mismatch")
            if stop_requested(new_fit=True):break
        family=pd.DataFrame([r for r in results if r["source"]==source and r["recipe"]==recipe and r["partition"]=="calibration_held" and not r["partial"]])
        if len(family):
            means=family.groupby(["feature_kind","target_kind"]).agg(n=("seed","size"),AP50=("Mask_AP50","mean"),AP95=("Mask_AP50_95","mean"),UQ_escape=("UQ_escape","mean")).reset_index()
            eligible=means[(means.n==3)&(means.AP50>=zero["Mask_AP50"]-.002)&(means.AP95>=zero["Mask_AP50_95"]+.002)&(means.UQ_escape<=zero["UQ_escape"])]
            choice=None if not len(eligible) else eligible.sort_values(["AP95","AP50"],ascending=False,kind="stable").iloc[0].to_dict()
        else:means=pd.DataFrame();choice=None
        selection={"source":source,"recipe":recipe,"selected":choice,"zero_raw_NMS":zero,"fixed_seed":SEEDS[0],"selected_before_this_policy_val":True}
        selections.append(selection);save(OUT/"rank_internal_selections.json",selections)
        means.to_csv(OUT/f"rank_families_{source}_{recipe}.csv",index=False)
        if not stop_requested():
            if val_sources is None:val_sources,_=joined_sources("val")
            ve=val_sources[source];vl=label_entries(ve,recipe)
            for record in [r for r in records if r["source"]==source and r["recipe"]==recipe and not r["partial"]]:
                cp=torch.load(record["path"],map_location="cpu",weights_only=False);model=RankHead();model.load_state_dict(cp["state"]);model.eval()
                confidence=rank_confidence(ve,vl,cp,model);metrics,_,_,_=evaluate(ve,vl,confidence,"scope_priority")
                results.append(dict(partition="val",**record,**metrics))
            pd.DataFrame(results).to_csv(OUT/"rank_fit_metrics.csv",index=False)
        if stop_requested(new_fit=True):break
    save(OUT/"rank_fit_status.json",{"status":"completed" if len(records)==54 and not any(r["partial"] for r in records) else "budget_partial", "fits":len(records),"complete_fits":sum(not r["partial"] for r in records),"selection":selections,"test_accessed":False})


if __name__=="__main__":main()
