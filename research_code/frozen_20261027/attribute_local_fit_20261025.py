"""Paired2x2 view/tail experiment, fit-only balancing and held-only checkpoints."""
from attribute_local_common_20261025 import *
setup()
import copy
import numpy as np,pandas as pd,torch
from torch import nn
from train_r1a_hierarchical_grader import R1HierarchicalGrader
from validate_r1_closure_runtime import STEM

class AttributeModel(nn.Module):
    def __init__(self,source,view,adapt):
        super().__init__();self.tail=copy.deepcopy(source.visual[4:]);self.geometry=copy.deepcopy(source.geometry);self.fusion=copy.deepcopy(source.fusion)
        self.defect_head=copy.deepcopy(source.defect_head);self.stem_head=copy.deepcopy(source.stem_head);self.view=view;self.adapt=adapt
        for p in self.parameters():p.requires_grad_(False)
        for module in [self.defect_head,self.stem_head]+([self.tail] if adapt else []):
            for p in module.parameters():p.requires_grad_(True)
    def features(self,prefix,geometry):
        x=prefix if self.view=="global_local" else prefix[:,:1].expand(-1,2,-1,-1,-1)
        f=self.tail(x.reshape(-1,*x.shape[2:])).reshape(len(x),2,256).mean(1)
        return self.fusion(torch.cat([f,self.geometry(geometry)],1))
    def heads(self,f):return torch.stack([self.defect_head(f).squeeze(1),self.stem_head(f).squeeze(1)],1)
    def forward(self,prefix,geometry):return self.heads(self.features(prefix,geometry))

def original():
    source=R1HierarchicalGrader();cp=torch.load(STEM,map_location="cpu",weights_only=False);source.load_state_dict(cp["model_state"]);return source.eval()

def load_data():
    manifest=json.loads((BASE/"prefix_train_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"]!="completed":raise ValueError("Incomplete train cache")
    roles=pd.read_csv(ROOT/"models/experiments/r1_quality_followup_20261009/head_image_split.csv").set_index("image_id")
    packs={k:{"prefix":[],"geometry":[],"target":[],"ids":[]} for k in ["fit","held"]}
    for row in manifest["entries"]:
        if sha256(row["path"])!=row["sha256"]:raise ValueError("Prefix changed")
        e=torch.load(row["path"],map_location="cpu",weights_only=False);role="fit" if roles.loc[e["image_id"],"role"]=="head_fit" else "held"
        use=np.flatnonzero(e["gt_class"]>=0);p=packs[role]
        p["prefix"].append(e["prefix"][use]);p["geometry"].append(torch.tensor(e["geometry"][use]));p["target"].append(torch.tensor(np.stack([e["defect"],e["stem"]],1)[use]))
        p["ids"].extend([(e["image_id"],int(e["prediction_indices"][j]),e["instance_ids"][j]) for j in use])
    for p in packs.values():
        for k in ["prefix","geometry","target"]:p[k]=torch.cat(p[k]).cuda()
    fi={r[0] for r in packs["fit"]["ids"]};hi={r[0] for r in packs["held"]["ids"]}
    if fi&hi:raise ValueError("Image role overlap")
    y=packs["fit"]["target"];freq=torch.nanmean(y,0).clamp(.01,.99);positive=.5/freq;negative=.5/(1-freq)
    save(BASE/"fitting_population.json",dict(fit_instances=len(y),held_instances=len(packs["held"]["target"]),fit_images=len(fi),held_images=len(hi),class_positive_frequency=freq.cpu().tolist(),fit_ids=packs["fit"]["ids"],held_ids=packs["held"]["ids"],test_accessed=False))
    return packs,positive,negative

def balanced_loss(logits,target,positive,negative):
    valid=torch.isfinite(target);y=torch.nan_to_num(target)
    weight=torch.where(y>.5,positive,negative)
    losses=nn.functional.binary_cross_entropy_with_logits(logits,y,reduction="none")*weight
    return ((losses*valid).sum(0)/valid.sum(0).clamp_min(1)).sum()

def predict(model,prefix,geometry,batch=128):
    outputs=[]
    with torch.no_grad():
        for start in range(0,len(prefix),batch):outputs.append(model(prefix[start:start+batch],geometry[start:start+batch]))
    return torch.cat(outputs) if outputs else torch.empty((0,2),device=prefix.device)

def fit(view,adapt,seed,packs,positive,negative):
    name=f"{view}_{'finetune' if adapt else 'frozen'}_seed{seed}";directory=BASE/"fits"/name;completion=directory/"completion.json"
    if completion.exists():return json.loads(completion.read_text(encoding="utf-8"))
    if directory.exists():raise FileExistsError("Inspect partial model before restart "+str(directory))
    directory.mkdir(parents=True);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=AttributeModel(original(),view,adapt).cuda().eval();torch.save(model.state_dict(),directory/"initial.pt")
    groups=[dict(params=list(model.defect_head.parameters())+list(model.stem_head.parameters()),lr=.001)]
    if adapt:groups.append(dict(params=[p for p in model.tail.parameters() if p.requires_grad],lr=.00003))
    optimizer=torch.optim.AdamW(groups,weight_decay=.0001);generator=torch.Generator().manual_seed(seed)
    frozen={}
    if not adapt:
        with torch.no_grad():
            for role,p in packs.items():
                frozen[role]=torch.cat([model.features(p["prefix"][s:s+128],p["geometry"][s:s+128]) for s in range(0,len(p["prefix"]),128)])
    best=float("inf");best_epoch=0;history=[];started=time.perf_counter();partial=False
    for epoch in range(41):
        model.eval() # All BN stats and fusion dropout frozen even during adaptation.
        if epoch:
            p=packs["fit"]
            for ids in torch.randperm(len(p["target"]),generator=generator).split(128):
                ids=ids.cuda();optimizer.zero_grad(set_to_none=True)
                logits=model(p["prefix"][ids],p["geometry"][ids]) if adapt else model.heads(frozen["fit"][ids])
                loss=balanced_loss(logits,p["target"][ids],positive,negative);loss.backward();optimizer.step()
        with torch.no_grad():
            p=packs["held"];logits=predict(model,p["prefix"],p["geometry"]) if adapt else model.heads(frozen["held"])
            held=float(balanced_loss(logits,p["target"],positive,negative));prob=logits.sigmoid();valid=torch.isfinite(p["target"])
            error=(((prob>=.5)!=(p["target"]>.5))&valid).sum(0).cpu().tolist()
        history.append(dict(epoch=epoch,held_balanced_BCE=held,held_defect_errors=error[0],held_stem_errors=error[1]))
        if held<best-1e-6:
            best=held;best_epoch=epoch
            torch.save(dict(state=model.state_dict(),view=view,adapt=adapt,seed=seed,epoch=epoch,held_BCE=held,positive_weights=positive.cpu(),negative_weights=negative.cpu(),original_sha256=sha256(STEM)),directory/"best.pt")
        pd.DataFrame(history).to_csv(directory/"history.csv",index=False)
        if epoch%5==0:print(f"ATTRIBUTE FIT {name} epoch{epoch} held={held:.5f} errors={error} {time.perf_counter()-started:.1f}s",flush=True)
        partial=stop_requested(new_fit=True)
        if partial or epoch-best_epoch>=10:break
    torch.save(dict(state=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,generator_state=generator.get_state()),directory/"last.pt")
    r=dict(view=view,adapt=adapt,seed=seed,path=str(directory/"best.pt"),epochs=epoch,best_epoch=best_epoch,held_balanced_BCE=best,partial=partial,seconds=time.perf_counter()-started,checkpoint_sha256=sha256(directory/"best.pt"),initial_sha256=sha256(directory/"initial.pt"),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),allocated_parameters=sum(p.numel() for p in model.parameters()))
    save(completion,r);return r

def main():
    plan();packs,positive,negative=load_data();rows=[]
    for seed in SEEDS:
        choices=[("global_global",False),("global_local",False),("global_global",True),("global_local",True)]
        np.random.default_rng(seed).shuffle(choices)
        for view,adapt in choices:
            if stop_requested(new_fit=True):break
            rows.append(fit(view,adapt,seed,packs,positive,negative));pd.DataFrame(rows).to_csv(OUT/"fit_ledger.csv",index=False)
        if stop_requested(new_fit=True):break
    save(OUT/"fit_status.json",dict(status="completed" if len(rows)==12 and not any(r["partial"] for r in rows) else "partial",fits=len(rows),test_accessed=False))

if __name__=="__main__":main()
