"""Pointwise boundary head2x2: hard/soft coarse input x prototype absence/presence."""
from attribute_local_common_20261025 import *
setup()
import cv2,numpy as np,pandas as pd,torch
import torch.nn.functional as F
from torch import nn
from attribute_local_proto_cache_20261025 import register

class PointHead(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Conv2d(36,32,1),nn.SiLU(),nn.Conv2d(32,16,1),nn.SiLU(),nn.Conv2d(16,1,1));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,rgb,coarse,soft,proto,use_soft,use_proto):
        p=F.interpolate(proto,size=coarse.shape[-2:],mode="bilinear",align_corners=False) if use_proto else torch.zeros((len(rgb),32,*coarse.shape[-2:]),device=rgb.device)
        c=soft if use_soft else coarse*2-1
        return 6*torch.tanh(self.net(torch.cat([rgb,c,p],1)))

def data():
    m=json.loads((BASE/"prototype_train_manifest.json").read_text(encoding="utf-8"))
    if m["status"]!="completed":raise ValueError("Incomplete prototypecache")
    roles=pd.read_csv(ROOT/"models/experiments/r1_quality_followup_20261009/head_image_split.csv").set_index("image_id");records=[];meta=[]
    for row in m["entries"]:
        e=torch.load(row["path"],map_location="cpu",weights_only=False)
        for r in e["records"]:
            if r["gt_index"]<0:continue
            records.append(r);meta.append(dict(image_id=e["image_id"],role=roles.loc[e["image_id"],"role"],prediction_index=r["prediction_index"],instance_id=r["instance_id"]))
    p={"rgb":torch.stack([r["rgb"] for r in records]).cuda(),"coarse":torch.stack([r["coarse"] for r in records])[:,None].cuda(),"soft":torch.stack([r["soft"] for r in records]).cuda(),
       "proto":torch.stack([r["prototype"] for r in records]).cuda(),"band":torch.stack([r["band"] for r in records])[:,None].cuda(),"gt":torch.stack([r["gt"] for r in records])[:,None].cuda()}
    fi=torch.tensor([i for i,r in enumerate(meta) if r["role"]=="head_fit"]);hi=torch.tensor([i for i,r in enumerate(meta) if r["role"]=="head_holdout"])
    pd.DataFrame(meta).to_csv(BASE/"prototype_population.csv",index=False)
    if {meta[i]["image_id"] for i in fi}&{meta[i]["image_id"] for i in hi}:raise ValueError("Prototype image role overlap")
    return p,records,fi,hi

def forward(model,p,ids,soft,proto):
    return model(p["rgb"][ids].float()/255,p["coarse"][ids].float(),p["soft"][ids].float(),p["proto"][ids].float(),soft,proto)

def objective(delta,coarse,target,band):
    logits=(coarse*2-1)*2+delta*band;p=logits.sigmoid()
    bce=(F.binary_cross_entropy_with_logits(logits,target,reduction="none")*(1+4*band)).mean()
    dice=1-(2*(p*target).sum((1,2,3))+1)/(p.sum((1,2,3))+target.sum((1,2,3))+1)
    return bce+dice.mean()

def apply_native(record,delta):
    base=record["native_coarse"].numpy().astype(bool);band=record["native_band"].numpy()
    resized=cv2.resize(delta,(base.shape[1],base.shape[0]),interpolation=cv2.INTER_LINEAR)
    return ((base.astype(np.float32)*2-1)*2+resized*band)>0

def native_metrics(model,p,records,ids,soft,proto):
    ious=[];areas=[];model.eval()
    with torch.no_grad():
        for batch in ids.split(32):
            values=forward(model,p,batch.cuda(),soft,proto).cpu().numpy()[:,0]
            for idx,delta in zip(batch.tolist(),values):
                r=records[idx];base=r["native_coarse"].numpy().astype(bool);gt=r["native_gt"].numpy().astype(bool);band=r["native_band"].numpy()
                pred=apply_native(r,delta)
                inter=np.count_nonzero(pred & gt);union=np.count_nonzero(pred)+r["full_gt_area"]-inter
                ious.append(inter/max(union,1));areas.append(abs(np.count_nonzero(pred)-r["full_gt_area"])/max(r["full_gt_area"],1))
    return dict(IoU=float(np.mean(ious)),relative_area=float(np.mean(areas)))

def fit(soft,proto,seed,p,records,fi,hi):
    name=f"{'soft' if soft else 'hard'}_{'prototype' if proto else 'zero'}_seed{seed}";directory=BASE/"prototype_fits"/name
    if (directory/"completion.json").exists():return json.loads((directory/"completion.json").read_text(encoding="utf-8"))
    if directory.exists():raise FileExistsError("Partial prototypefit "+str(directory))
    directory.mkdir(parents=True);torch.manual_seed(seed);model=PointHead().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    torch.save(model.state_dict(),directory/"initial.pt");generator=torch.Generator().manual_seed(seed);best=(-1.,-float("inf"));best_epoch=0;rows=[];started=time.perf_counter();partial=False
    for epoch in range(31):
        if epoch:
            model.train()
            for batch in fi[torch.randperm(len(fi),generator=generator)].split(32):
                ids=batch.cuda();optimizer.zero_grad(set_to_none=True);delta=forward(model,p,ids,soft,proto)
                loss=objective(delta,p["coarse"][ids].float(),p["gt"][ids].float(),p["band"][ids].float());loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        metrics=native_metrics(model,p,records,hi,soft,proto);rows.append(dict(epoch=epoch,**metrics));key=(metrics["IoU"],-metrics["relative_area"])
        if key>best:
            best=key;best_epoch=epoch;torch.save(dict(state=model.state_dict(),soft=soft,prototype=proto,seed=seed,epoch=epoch,held_native_metrics=metrics),directory/"best.pt")
        pd.DataFrame(rows).to_csv(directory/"history.csv",index=False)
        if epoch%5==0:print(f"PROTOTYPE FIT {name} epoch{epoch} IoU={metrics['IoU']:.6f} area={metrics['relative_area']:.6f}",flush=True)
        partial=stop_requested(new_fit=True)
        if partial or epoch-best_epoch>=8:break
    torch.save(dict(state=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,generator_state=generator.get_state()),directory/"last.pt")
    r=dict(soft=soft,prototype=proto,seed=seed,path=str(directory/"best.pt"),epochs=epoch,best_epoch=best_epoch,held_IoU=best[0],held_relative_area=-best[1],partial=partial,seconds=time.perf_counter()-started,checkpoint_sha256=sha256(directory/"best.pt"),initial_sha256=sha256(directory/"initial.pt"))
    save(directory/"completion.json",r);return r

def main():
    plan();register();p,records,fi,hi=data();model=PointHead().cuda();reference=native_metrics(model,p,records,hi,False,False);rows=[]
    for seed in SEEDS:
        variants=[(s,p) for s in [False,True] for p in [False,True]];np.random.default_rng(seed).shuffle(variants)
        for soft,proto in variants:
            if stop_requested(new_fit=True):break
            rows.append(fit(soft,proto,seed,p,records,fi,hi));pd.DataFrame(rows).to_csv(OUT/"prototype_fit_ledger.csv",index=False)
        if stop_requested(new_fit=True):break
    f=pd.DataFrame(rows);selected=None
    if len(f):
        families=f[~f.partial].groupby(["soft","prototype"]).agg(seeds=("seed","nunique"),IoU=("held_IoU","mean"),relative_area=("held_relative_area","mean")).reset_index()
        families["eligible"]=(families.seeds==3)&(families.IoU>=reference["IoU"]+.001)&(families.relative_area<=reference["relative_area"]+.005)
        families.to_csv(OUT/"prototype_families.csv",index=False);good=families[families.eligible]
        if len(good):selected=good.sort_values(["IoU","relative_area"],ascending=[False,True],kind="stable").iloc[0].to_dict()
    save(OUT/"prototype_internal_selection.json",dict(selected=selected,reference=reference,fixed_seed=20261025,before_val=True,test_accessed=False))
    save(OUT/"prototype_fit_status.json",dict(status="completed" if len(rows)==12 else "partial",fits=len(rows),test_accessed=False))

if __name__=="__main__":main()
