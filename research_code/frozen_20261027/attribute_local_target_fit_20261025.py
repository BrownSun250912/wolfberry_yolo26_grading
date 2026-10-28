"""Target-specific adaptation:2targets x2losses x2backgrounds x3seeds."""
from attribute_local_common_20261025 import *
setup()
import numpy as np,pandas as pd,torch
from attribute_local_fit_20261025 import AttributeModel,original,load_data,balanced_loss,predict
from attribute_local_bg_20261025 import register

def background_data(packs):
    m=json.loads((BASE/"prefix_bg_train_manifest.json").read_text(encoding="utf-8"))
    if m["status"]!="completed":raise ValueError("Incomplete backgroundcache")
    mapping={};out={role:torch.empty_like(p["prefix"]) for role,p in packs.items()}
    for role,p in packs.items():
        for pos,(image,j,_) in enumerate(p["ids"]):mapping.setdefault(image,[]).append((role,int(j),pos))
    for row in m["entries"]:
        e=torch.load(row["path"],map_location="cpu",weights_only=False);indices={int(j):i for i,j in enumerate(e["prediction_indices"])}
        for role in packs:
            members=[(indices[j],pos) for rr,j,pos in mapping.get(e["image_id"],[]) if rr==role]
            if members:
                src,dest=zip(*members);out[role][list(dest)]=e["prefix"][list(src)].cuda()
    return out

def fit(target,balanced,paired,seed,packs,bg,pos,neg):
    name=f"{target}_{'balanced' if balanced else 'unweighted'}_{'neutral_pair' if paired else 'clean_duplicate'}_seed{seed}"
    directory=BASE/"target_fits"/name;complete=directory/"completion.json"
    if complete.exists():return json.loads(complete.read_text(encoding="utf-8"))
    if directory.exists():raise FileExistsError("Existing partial "+str(directory))
    directory.mkdir(parents=True);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=AttributeModel(original(),"global_global",True).cuda().eval();index=0 if target=="defect" else 1
    active=model.defect_head if index==0 else model.stem_head;unused=model.stem_head if index==0 else model.defect_head
    for p in unused.parameters():p.requires_grad_(False)
    torch.save(model.state_dict(),directory/"initial.pt")
    optimizer=torch.optim.AdamW([dict(params=active.parameters(),lr=.001),dict(params=model.tail.parameters(),lr=.00003)],weight_decay=.0001)
    positive=pos if balanced else torch.ones_like(pos);negative=neg if balanced else torch.ones_like(neg)
    def loss(z,y):return balanced_loss(z[:,index:index+1],y[:,index:index+1],positive[index:index+1],negative[index:index+1])
    best=float("inf");best_epoch=0;history=[];started=time.perf_counter();generator=torch.Generator().manual_seed(seed);partial=False
    for epoch in range(41):
        if epoch:
            p=packs["fit"]
            for ids in torch.randperm(len(p["target"]),generator=generator).split(128):
                ids=ids.cuda();optimizer.zero_grad(set_to_none=True)
                a=model(p["prefix"][ids],p["geometry"][ids]);b=model(bg["fit"][ids] if paired else p["prefix"][ids],p["geometry"][ids])
                objective=.5*(loss(a,p["target"][ids])+loss(b,p["target"][ids]))+.5*(a[:,index].sigmoid()-b[:,index].sigmoid()).square().mean()
                objective.backward();optimizer.step()
        with torch.no_grad():
            p=packs["held"];a=predict(model,p["prefix"],p["geometry"]);b=predict(model,bg["held"],p["geometry"])
            held=float(.5*(loss(a,p["target"])+loss(b,p["target"])));clean_errors=int((((a[:,index].sigmoid()>=.5)!=(p["target"][:,index]>.5))&torch.isfinite(p["target"][:,index])).sum())
        history.append(dict(epoch=epoch,held_two_condition_BCE=held,clean_attribute_errors=clean_errors))
        if held<best-1e-6:
            best=held;best_epoch=epoch
            torch.save(dict(state=model.state_dict(),view="global_global",adapt=True,target=target,balanced=balanced,paired=paired,seed=seed,epoch=epoch,held_BCE=held,positive_weights=positive.cpu(),negative_weights=negative.cpu(),original_sha256=sha256(ROOT/"models/experiments/r1a_hierarchical_grader_e200_resume_epoch150_seed20260929/best.pt")),directory/"best.pt")
        pd.DataFrame(history).to_csv(directory/"history.csv",index=False)
        if epoch%10==0:print(f"TARGET FIT {name} epoch{epoch} held={held:.5f} cleanerrors={clean_errors}",flush=True)
        partial=stop_requested(new_fit=True)
        if partial or epoch-best_epoch>=10:break
    torch.save(dict(state=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,generator_state=generator.get_state()),directory/"last.pt")
    r=dict(target=target,balanced=balanced,paired=paired,seed=seed,path=str(directory/"best.pt"),partial=partial,epochs=epoch,best_epoch=best_epoch,held_BCE=best,seconds=time.perf_counter()-started,checkpoint_sha256=sha256(directory/"best.pt"),initial_sha256=sha256(directory/"initial.pt"))
    save(complete,r);return r

def main():
    plan();register();packs,pos,neg=load_data();bg=background_data(packs);rows=[]
    for seed in SEEDS:
        choices=[(t,b,p) for t in ["defect","stem"] for b in [False,True] for p in [False,True]];np.random.default_rng(seed).shuffle(choices)
        for target,balanced,paired in choices:
            if stop_requested(new_fit=True):break
            rows.append(fit(target,balanced,paired,seed,packs,bg,pos,neg));pd.DataFrame(rows).to_csv(OUT/"target_fit_ledger.csv",index=False)
        if stop_requested(new_fit=True):break
    save(OUT/"target_fit_status.json",dict(status="completed" if len(rows)==24 and not any(r["partial"] for r in rows) else "partial",fits=len(rows),test_accessed=False))

if __name__=="__main__":main()
