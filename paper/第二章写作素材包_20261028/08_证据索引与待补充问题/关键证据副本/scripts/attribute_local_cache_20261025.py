"""Cache frozen prefix tensors for paired attribute input/tail experiments."""
from attribute_local_common_20261025 import *
setup()
import argparse,math
import cv2,numpy as np,pandas as pd,torch
from train_r1a_hierarchical_grader import R1HierarchicalGrader,compute_geometry,letterbox
from validate_r1_closure_runtime import STEM,TARGETS
from run_r1_ap_followup import load_cache
from direction_experiments_20261023 import load_records
from analyze_val_instance_errors import polygon_to_mask,optimal_match

def crop_tensor(image,mask,size,pad_factor,pad_min):
    ys,xs=np.nonzero(mask)
    if not len(xs):raise ValueError("Empty mask")
    x,y=int(xs.min()),int(ys.min());w,h=int(xs.max()-x+1),int(ys.max()-y+1);pad=max(pad_min,math.ceil(max(w,h)*pad_factor))
    x0,y0=max(0,x-pad),max(0,y-pad);x1,y1=min(image.shape[1],x+w+pad),min(image.shape[0],y+h+pad)
    crop=image[y0:y1,x0:x1];mc=(mask[y0:y1,x0:x1]>0).astype(np.uint8)*255
    g=compute_geometry(mc,crop.shape[:2]);rgb,m=letterbox(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB),mc,size)
    rgb=(rgb.astype(np.float32)/255-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    return np.concatenate([rgb,(m/255).astype(np.float32)[...,None]],2).transpose(2,0,1),g

def main(split):
    plan();setup()
    if split=="val" and not (OUT/"internal_selection.json").exists():raise ValueError("Freeze train-held family selection before val processing")
    entries=load_cache(split);records=load_records(split);targets=pd.read_csv(TARGETS).set_index("instance_id")
    cp=torch.load(STEM,map_location="cpu",weights_only=False);model=R1HierarchicalGrader().cuda().eval();model.load_state_dict(cp["model_state"])
    prefix=model.visual[:4];directory=BASE/("prefix_"+split);directory.mkdir(exist_ok=True);manifest=[];start=time.perf_counter()
    with torch.inference_mode():
        for i,(e,r) in enumerate(zip(entries,records)):
            if stop_requested():break
            assert e["image_id"]==r["image_id"]
            path=directory/(e["image_id"]+".pt")
            if not path.exists():
                image=cv2.imread(e["image_path"]);high=np.flatnonzero(e["conf"]>=.25)
                tensors=[];geoms=[];ids=[];bad=[]
                for j in high:
                    mask=polygon_to_mask(r["polygons"][j],*image.shape[:2])
                    try:
                        global_t,g=crop_tensor(image,mask,cp["config"]["image_size"],.25,12);local_t,_=crop_tensor(image,mask,cp["config"]["image_size"],.08,4)
                        tensors.extend([global_t,local_t]);geoms.append(g);ids.append(int(j))
                    except ValueError:bad.append(int(j))
                prefixes=[]
                for batch in np.array_split(np.stack(tensors),max(1,math.ceil(len(tensors)/128))) if tensors else []:
                    prefixes.append(prefix(torch.as_tensor(batch,device="cuda")).cpu())
                features=torch.cat(prefixes).reshape(len(ids),2,192,10,10) if prefixes else torch.empty((0,2,192,10,10))
                geometry=(np.stack(geoms)-np.asarray(cp["geometry_mean"],np.float32))/np.asarray(cp["geometry_std"],np.float32) if ids else np.empty((0,6),np.float32)
                defect=np.full(len(ids),np.nan,np.float32);stem=defect.copy();gt_cls=np.full(len(ids),-1,np.int64);instance=[None]*len(ids)
                bypred={j:k for k,j in enumerate(ids)}
                for gi,pj,_ in optimal_match(e["mask_iou"][:,high],.5):
                    j=int(high[pj])
                    if j not in bypred:continue
                    k=bypred[j];row=targets.loc[e["gt_ids"][gi]];instance[k]=str(row.name);gt_cls[k]=int(row.final_class_id)
                    if row.defect_target_available:defect[k]=float(row.defect_target)
                    if row.stem_target_available:stem[k]=float(row.stem_target)
                record=dict(image_id=e["image_id"],source_group=e["source_group"],image_sha256=e["image_sha256"],split=split,prediction_indices=np.array(ids,int),
                     prefix=features,geometry=geometry.astype(np.float32),defect=defect,stem=stem,gt_class=gt_cls,instance_ids=instance,invalid_high=bad)
                torch.save(record,path)
            record=torch.load(path,map_location="cpu",weights_only=False)
            manifest.append(dict(image_id=e["image_id"],path=str(path),sha256=sha256(path),predictions=len(record["prediction_indices"]),matched=int((record["gt_class"]>=0).sum())))
            if (i+1)%50==0 or i+1==len(entries):print(f"ATTRIBUTE PREFIX {split} {i+1}/{len(entries)} {time.perf_counter()-start:.1f}s",flush=True)
    save(BASE/("prefix_"+split+"_manifest.json"),dict(status="completed" if len(manifest)==len(entries) else "partial",entries=manifest,images=len(manifest),test_accessed=False))
    save(OUT/("prefix_"+split+"_status.json"),dict(status="completed" if len(manifest)==len(entries) else "partial",images=len(manifest),predictions=sum(r["predictions"] for r in manifest),matched=sum(r["matched"] for r in manifest),test_accessed=False))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("split",choices=["train","val"]);main(p.parse_args().split)
