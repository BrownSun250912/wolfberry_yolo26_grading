"""Inference-only candidate evidence, bounded decisions, and true keep-index NMS."""
import numpy as np

LOW=37.40298225745565
HIGH=65.62684031710077


def pair_iou(boxes):
    b=np.asarray(boxes,np.float32)
    if not len(b):return np.empty((0,0),np.float32)
    lo=np.maximum(b[:,None,:2],b[None,:,:2]);hi=np.minimum(b[:,None,2:],b[None,:,2:])
    intersection=np.maximum(hi-lo,0).prod(-1)
    area=np.maximum(b[:,2:]-b[:,:2],0).prod(-1)
    return intersection/(area[:,None]+area[None,:]-intersection+1e-7)


def evidence(entry,kind="neighbour",iou=.85):
    base=np.asarray(entry["baseline"],int);conf=np.asarray(entry["conf"])
    n=len(base)
    if kind=="raw_same_candidate":
        value=np.asarray(entry["class_scores"],np.float32)
        if value.shape!=(n,4) or not np.isfinite(value).all():raise ValueError("Invalid raw class score lineage")
        return np.clip(value,1e-4,1-1e-4)
    value=np.full((n,4),1e-4,np.float32)
    if kind=="top1":
        value[np.arange(n),base]=conf
    elif kind=="neighbour":
        near=pair_iou(entry["boxes"])>=iou
        for c in range(4):
            values=np.where((base==c)[None,:]&near,conf[None,:],1e-4)
            if n:value[:,c]=values.max(1)
    else:raise ValueError(kind)
    return np.clip(value,1e-4,1-1e-4)


def logit(value):
    value=np.clip(value,1e-4,1-1e-4)
    return np.log(value/(1-value))


def decide(entry,kind="neighbour",boundary=True,stem=True,recover=False,alpha_rec=.5,alpha_stem=.5,lambda_b=1.5,sigma_s=2.5,uq_state=None):
    base=np.asarray(entry["baseline"],int);cls=base.copy();n=len(base)
    if not n:return cls
    score=np.asarray(entry["score"]);sp=np.asarray(entry["stem"])
    valid=(np.asarray(entry["conf"])>=.25)&np.isfinite(score)&np.isfinite(sp)
    if not boundary and not stem and not recover and uq_state is None:
        grade=np.where(score>=HIGH,1,np.where(score>=LOW,2,3))
        active=valid&(base!=0)
        cls[active]=np.where(sp[active]>=.5,3,grade[active]);return cls
    c=evidence(entry,kind);l=logit(c)
    uq=l[:,0]-l[:,1:].max(1)
    st=l[:,3]-np.maximum(l[:,1],l[:,2])
    d12=l[:,1]-l[:,2];d23=l[:,2]-l[:,3]
    is_uq=(base==0) if uq_state is None else np.asarray(uq_state,bool).copy()
    if is_uq.shape!=(n,):raise ValueError("UQ override alignment")
    if recover:
        defect=np.asarray(entry["defect_logit"])
        recovery=valid&(base==0)&np.isfinite(defect)&(defect < -3.)&(uq+alpha_rec*defect < 0)
        is_uq[recovery]=False
    m12=score-HIGH;m23=score-LOW
    if boundary:
        m12=m12+lambda_b*np.exp(-.5*((score-HIGH)/sigma_s)**2)*np.tanh(d12/3.)
        m23=m23+lambda_b*np.exp(-.5*((score-LOW)/sigma_s)**2)*np.tanh(d23/3.)
    grade=np.where(m12>=0,1,np.where(m23>=0,2,3))
    p=np.clip(np.nan_to_num(sp,nan=.5),1e-6,1-1e-6)
    missing=np.log(p/(1-p))+alpha_stem*st >= 0 if stem else sp>=.5
    cls[valid&is_uq]=0
    active=valid&~is_uq
    cls[active]=np.where(missing[active],3,grade[active])
    return cls


def consensus_decide(entry,kind="neighbour",mode="both",threshold=3.):
    """Require both local detector and original defect head to agree on gate changes.

    Adaptive follow-up to known development failures; this is not calibrated
    Bayesian fusion and does not guarantee absence of true-UQ escapes.
    """
    if mode not in ["recover","create","both","none"]:raise ValueError(mode)
    base=np.asarray(entry["baseline"],int);state=base==0
    c=evidence(entry,kind);l=logit(c);margin=l[:,0]-l[:,1:].max(1)
    defect=np.asarray(entry["defect_logit"])
    valid=(np.asarray(entry["conf"])>=.25)&np.isfinite(entry["score"])&np.isfinite(entry["stem"])&np.isfinite(defect)
    if mode in ["recover","both"]:state[valid&(base==0)&(defect < -threshold)&(margin < 0)]=False
    if mode in ["create","both"]:state[valid&(base!=0)&(defect > threshold)&(margin > 0)]=True
    return decide(entry,kind,True,True,False,uq_state=state)


def keep_indices(boxes,classes,ranking,threshold=.7,raw_scope=None):
    boxes=np.asarray(boxes);classes=np.asarray(classes);ranking=np.asarray(ranking)
    n=len(classes)
    if boxes.shape!=(n,4) or len(ranking)!=n or not np.isfinite(ranking).all():raise ValueError("NMS input alignment")
    if not n:return np.empty(0,int)
    # Stable ties preserve original candidate index. Optional scope has explicit priority.
    order=np.lexsort((np.arange(n),-ranking)) if raw_scope is None else np.lexsort((np.arange(n),-ranking,-np.asarray(raw_scope,int)))
    ious=pair_iou(boxes);keep=[]
    for j in order:
        if not any(classes[j]==classes[k] and ious[j,k]>=threshold for k in keep):keep.append(int(j))
    # Return original order for AP tie reproducibility and metadata correspondence.
    return np.array(sorted(keep),int)


def slice_prediction_fields(entry,keep):
    n=len(entry["baseline"]);out=dict(entry)
    for key in ["baseline","conf","score","stem","boxes","mask_area","defect_logit","class_scores","candidate_indices"]:
        if key in entry:
            if len(entry[key])!=n:raise ValueError("Prediction field mismatch: "+key)
            out[key]=np.asarray(entry[key])[keep]
    for key in ["mask_iou","box_iou"]:
        if key in entry:out[key]=entry[key][:,keep]
    for key in ["old_prediction_ids","polygons"]:
        if key in entry:out[key]=[entry[key][j] for j in keep]
    out["source_prediction_indices"]=np.asarray(keep,int)
    return out
