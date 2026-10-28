"""Registered target-specific loss/background factorial follow-up."""
from attribute_local_common_20261025 import *
setup()
import math
import cv2,numpy as np,torch
from attribute_local_cache_20261025 import crop_tensor
from attribute_local_fit_20261025 import original
from run_r1_ap_followup import load_cache
from direction_experiments_20261023 import load_records
from analyze_val_instance_errors import polygon_to_mask

def register():
    p=BASE/"target_bg_plan.json"
    if not p.exists():save(p,dict(registered_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
      registered_after="Primary12fits and firstseed val comparison observed. Defect replacement reduces UQescapes14..17vs21 but qualified rejects25..28vs10; selected stem candidate447vs449. Adaptive followup, not independent confirmation.",
      question="Can task-specific adaptation separate defect/stem gradient competition, and can unweighted loss or protected-background pairing improve the error tradeoff?",
      fixed="global_global view and adaptive tail selected from primary internalheld; original teacher/prefixBN frozen; original geometry and continuousscore remain frozen",
      factors=dict(target=["defect","stem"],loss=["unweighted","balanced"],background=["clean_duplicate","neutral_pair"]),seeds=SEEDS,fits=24,
      dose="Same40epochs,maxpatience10,batch128,headlr.001,taillr3e-5; same sampleorder/init perseed. Onlytargethead plus tail trained. Allarms two supervised forwardpasses, clean duplicated in control.",
      image_operation="Neutral160 outside dilated union of original highconfidence predicted masks; dilation12imagepixels protects local context. Original boxes/masks retained during training augmentation. NoGT chooses protection or crop; no physical identity claim.",
      objective="Mean clean/paired supervisedtargetBCE + .5mean squared difference of target probabilities; clean duplicate has zero consistencyterm. Fitonlyweights for balancedarm; unweightedarm uses unitweights.",
      checkpoint="Minimum clean+neutral held targetBCE, epoch0 included; allfamilies use both held conditions, so cleancontrol is not transformation-blind during checkpointselection.",
      selection="Complete3seed mean clean postNMS correct/allGT>=rrreference,UQescape<=reference,qualifiedreject<=reference; maximize AP95 then correct. Keep previouscandidate as separate reference. Fixedseed25, valonlyafter newfamilychoice; reportall24models.",
      limits="Paired transforms are repeated observations of originalimages, not new independentfruit; upstream previously saw originaltrain. No calibratedprobability/safety guarantee.",test_accessed=False,primary_replaced=False))

def main():
    plan();register();entries=load_cache("train");records=load_records("train");model=original().cuda().eval();prefix=model.visual[:4]
    manifest=[];directory=BASE/"prefix_bg_train";directory.mkdir(exist_ok=True)
    with torch.inference_mode():
        for i,(e,r) in enumerate(zip(entries,records)):
            if stop_requested():break
            path=directory/(e["image_id"]+".pt")
            if not path.exists():
                base=torch.load(BASE/"prefix_train"/(e["image_id"]+".pt"),map_location="cpu",weights_only=False)
                image=cv2.imread(e["image_path"]);h,w=image.shape[:2];union=np.zeros((h,w),np.uint8)
                masks={}
                for j in np.flatnonzero(e["conf"]>=.25):
                    mask=polygon_to_mask(r["polygons"][j],h,w);union|=mask;masks[int(j)]=mask
                protected=cv2.dilate(union,np.ones((25,25),np.uint8))>0
                image=image.copy();image[~protected]=160;tensors=[]
                for j in base["prediction_indices"]:
                    for factor,minimum in [(.25,12),(.08,4)]:tensors.append(crop_tensor(image,masks[int(j)],160,factor,minimum)[0])
                values=[]
                for batch in np.array_split(np.stack(tensors),max(1,math.ceil(len(tensors)/128))) if tensors else []:values.append(prefix(torch.as_tensor(batch,device="cuda")).cpu())
                features=torch.cat(values).reshape(-1,2,192,10,10) if values else torch.empty((0,2,192,10,10))
                torch.save(dict(image_id=e["image_id"],prediction_indices=base["prediction_indices"],prefix=features),path)
            manifest.append(dict(image_id=e["image_id"],path=str(path),sha256=sha256(path)))
            if (i+1)%100==0 or i+1==len(entries):print(f"ATTRIBUTE BG PREFIX {i+1}/{len(entries)}",flush=True)
    save(BASE/"prefix_bg_train_manifest.json",dict(status="completed" if len(manifest)==966 else "partial",entries=manifest,test_accessed=False))

if __name__=="__main__":main()
