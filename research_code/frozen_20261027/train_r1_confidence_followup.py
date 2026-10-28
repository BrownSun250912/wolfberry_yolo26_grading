"""Train-only prediction-reliability ranking for fixed hierarchical grades.

Labels, boxes and masks are not learned or changed here. Detector class0 and
its confidence stay fixed. All checkpoint selection uses a train-image holdout.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from run_r1_ap_followup import EX,OUT,ROOT,LOW,HIGH,load_cache,policy,match_flags,evaluate,save,sha256

SEEDS=[20261008,20261009,20261010]
FEATURES=['detector_logit','score_01','boundary_distance_01','stem_logit_scaled',
          'log_mask_fraction','log_box_fraction','log_box_aspect','label_agreement',
          'pred_grade1','pred_grade2','pred_grade3','base_grade1','base_grade2','base_grade3']

def features(entry):
    cls=policy(entry,True,True,True)
    conf=np.clip(entry['conf'],1e-5,1-1e-5)
    score=np.nan_to_num(entry['score'],nan=50.)
    stem=np.clip(np.nan_to_num(entry['stem'],nan=.5),1e-6,1-1e-6)
    boxes=entry['boxes'];w=np.maximum(boxes[:,2]-boxes[:,0],1.);h=np.maximum(boxes[:,3]-boxes[:,1],1.)
    area=entry['image_height']*entry['image_width']
    values=[np.log(conf/(1-conf)),score/100,np.minimum(np.abs(score-LOW),np.abs(score-HIGH))/100,
        np.log(stem/(1-stem))/10,np.log(np.maximum(entry['mask_area']/area,1e-7)),np.log(np.maximum(w*h/area,1e-7)),np.log(w/h),(cls==entry['baseline']).astype(float)]
    values.extend([(cls==c).astype(float) for c in [1,2,3]])
    values.extend([(entry['baseline']==c).astype(float) for c in [1,2,3]])
    x=np.stack(values,axis=1).astype(np.float32)
    if x.shape[1]!=len(FEATURES) or not np.isfinite(x).all():raise ValueError('Invalid confidence features')
    return x,cls

class ReliabilityHead(nn.Module):
    def __init__(self,family):
        super().__init__()
        self.net=nn.Linear(len(FEATURES),1) if family=='linear_confidence' else nn.Sequential(nn.Linear(len(FEATURES),16),nn.SiLU(),nn.Linear(16,1))
    def forward(self,x):return self.net(x).squeeze(-1)

def prepare(train):
    ids=sorted(e['image_id'] for e in train)
    rng=np.random.default_rng(20261008)
    held=set(rng.permutation(ids)[:round(len(ids)*.2)].tolist())
    xfit,yfit,xheld,yheld,split_rows=[],[],[],[],[]
    for entry in train:
        x,cls=features(entry)
        use=entry['baseline']!=0
        y=match_flags(cls,entry['gt_cls'],entry['mask_iou'])[:,0].astype(np.float32)
        target_x,target_y=(xheld,yheld) if entry['image_id'] in held else (xfit,yfit)
        target_x.append(x[use]);target_y.append(y[use])
        split_rows.append({'image_id':entry['image_id'],'role':'calibration_holdout' if entry['image_id'] in held else 'calibration_fit','qualified_predictions':int(use.sum()),'positive_tp50':int(y[use].sum())})
    fit_x,fit_y=np.concatenate(xfit),np.concatenate(yfit)
    held_x,held_y=np.concatenate(xheld),np.concatenate(yheld)
    mean,std=fit_x.mean(0),fit_x.std(0);std[std<1e-6]=1.
    pd.DataFrame(split_rows).to_csv(EX/'calibration_image_split.csv',index=False)
    if not len(fit_x) or not len(held_x) or not 0<fit_y.mean()<1 or not 0<held_y.mean()<1:raise ValueError('Calibration population invalid')
    return fit_x,fit_y,held_x,held_y,mean,std

def fit(family,seed,data):
    torch.manual_seed(seed);np.random.seed(seed)
    fx,fy,hx,hy,mean,std=data
    x=torch.as_tensor((fx-mean)/std);y=torch.as_tensor(fy)
    vx=torch.as_tensor((hx-mean)/std);vy=torch.as_tensor(hy)
    head=ReliabilityHead(family)
    optimizer=torch.optim.AdamW(head.parameters(),lr=.003,weight_decay=.0001)
    loss_fn=nn.BCEWithLogitsLoss()
    generator=torch.Generator().manual_seed(seed)
    best,state,epoch_best=float('inf'),None,0
    history=[]
    for epoch in range(301):
        if epoch:
            head.train()
            for order in torch.randperm(len(x),generator=generator).split(256):
                optimizer.zero_grad();loss=loss_fn(head(x[order]),y[order]);loss.backward();optimizer.step()
        head.eval()
        with torch.no_grad():
            train_loss=float(loss_fn(head(x),y));val_loss=float(loss_fn(head(vx),vy))
        history.append({'epoch':epoch,'fit_bce':train_loss,'train_image_holdout_bce':val_loss})
        if val_loss<best-1e-6:
            best,epoch_best=val_loss,epoch;state={k:v.clone() for k,v in head.state_dict().items()}
        if epoch-epoch_best>=30:break
    head.load_state_dict(state)
    directory=EX/f'{family}_seed{seed}';directory.mkdir(exist_ok=False)
    pd.DataFrame(history).to_csv(directory/'history.csv',index=False)
    checkpoint={'family':family,'seed':seed,'head_state':state,'mean':mean,'std':std,'feature_names':FEATURES,'best_epoch':epoch_best,'epochs_completed':epoch,'holdout_bce':best,'policy':'score+stem only conf>=.25; class0 and its confidence unchanged','selection_data':'20% image-held-out subset of original train; no development val labels used in optimization','script_sha256':sha256(Path(__file__))}
    path=directory/'best.pt';torch.save(checkpoint,path)
    copy=ReliabilityHead(family);copy.load_state_dict(torch.load(path,weights_only=False)['head_state']);copy.eval()
    with torch.no_grad():
        if not torch.equal(copy(vx),head(vx)):raise ValueError('Checkpoint roundtrip mismatch')
    print(f'CONF FIT {family} seed={seed} BCE={best:.6f} epoch={epoch_best}/{epoch}',flush=True)
    return head,checkpoint,path

def confidences(head,checkpoint,cache,kind):
    result=[]
    with torch.inference_mode():
        for entry in cache:
            x,_=features(entry)
            p=torch.sigmoid(head(torch.as_tensor((x-checkpoint['mean'])/checkpoint['std']))).numpy()
            confidence=p if kind=='replace' else p*entry['conf']
            confidence=np.clip(confidence,0,1).astype(np.float64)
            confidence[entry['baseline']==0]=entry['conf'][entry['baseline']==0]
            result.append(confidence)
    return result

def main():
    torch.set_num_threads(4)
    plan_path=EX/'calibration_plan.json'
    if plan_path.exists():raise FileExistsError('Calibration stage already started; inspect artifacts before any retry')
    save(plan_path,{'script_sha256':sha256(Path(__file__)),'families':['linear_confidence','mlp_confidence'],'seeds':SEEDS,'maximum_fits':6,'features':FEATURES,'epochs':300,'patience':30,'lr':.003,'weight_decay':.0001,'loss':'unweighted BCE official candidate-class TP at mask IoU .5','fit_selection':'train-only image 80/20 seed20261008; minimum inner-holdout BCE','confidence_variants':['replace qualified confidences with learned reliability','multiply qualified detector confidence by learned reliability'],'class0_confidence_preserved':True,'grade_policy':'full hierarchy only conf>=.25, fixed before calibration evaluation','no_validation_labels_in_fit':True,'test_accessed':False,'no_automatic_model_promotion':True})
    train,val=load_cache('train'),load_cache('val')
    train_ids={e['image_id'] for e in train};val_ids={e['image_id'] for e in val}
    if train_ids&val_ids:raise ValueError('Split leakage')
    data=prepare(train)
    save(EX/'calibration_population.json',{'fit_predictions':len(data[0]),'fit_positives':int(data[1].sum()),'holdout_predictions':len(data[2]),'holdout_positives':int(data[3].sum()),'normalizer_fit_rows':len(data[0]),'train_images':len(train),'validation_images':len(val),'train_val_images_disjoint':True,'caveat':'Underlying detector and score/stem models previously learned original train, so this is only calibration holdout, not unseen-data performance'})
    reports,perclass,checks=[],[],[]
    labels=[policy(e,True,True,True) for e in val]
    for family in ['linear_confidence','mlp_confidence']:
        for seed in SEEDS:
            head,cp,path=fit(family,seed,data)
            for kind in ['replace','product']:
                name=f'{family}_seed{seed}_{kind}'
                confidence=confidences(head,cp,val,kind)
                report,classes,transitions=evaluate(val,name,labels,confidence)
                report.update({'family':family,'seed':seed,'confidence_kind':kind,'best_epoch':cp['best_epoch'],'holdout_bce':cp['holdout_bce']})
                reports.append(report);perclass.extend(classes)
                pd.DataFrame(transitions).to_csv(path.parent/f'predictions_{kind}.csv',index=False)
                checks.append({'variant':name,'checkpoint_sha256':sha256(path),'all_classes_unchanged':report['correct_500']==439,'finite_confidence':all(np.isfinite(c).all() for c in confidence),'checkpoint_cpu_roundtrip':'PASS'})
                print('CONF RESULT '+str(report),flush=True)
            pd.DataFrame(reports).to_csv(OUT/'calibration_metrics.csv',index=False)
    pd.DataFrame(perclass).to_csv(OUT/'calibration_class_ap.csv',index=False)
    save(OUT/'calibration_verification.json',checks)
    frame=pd.DataFrame(reports)
    means=frame.groupby(['family','confidence_kind']).agg(mask_ap50=('metrics/mAP50(M)','mean'),mask_ap95=('metrics/mAP50-95(M)','mean'),correct_500=('correct_500','mean'),min_mask_ap50=('metrics/mAP50(M)','min')).reset_index()
    baseline=json.loads((ROOT/'outputs/reports/r1_score_closure_20261008/deployment_ap.json').read_text(encoding='utf-8'))['baseline']
    means['accepted']=(means.mask_ap50>=baseline['metrics/mAP50(M)']+.005)&(means.mask_ap95>=baseline['metrics/mAP50-95(M)']-.002)&(means.correct_500>=439)
    means.to_csv(OUT/'calibration_family_summary.csv',index=False)
    save(OUT/'calibration_summary.json',{'status':'COMPLETED','new_fits':6,'evaluated_confidence_variants':12,'test_accessed':False,'primary_model_replaced':False,'family_results':means.to_dict(orient='records'),'selection':'predeclared practical gate, development-validation only; fixed deployment seed20261008, not best seed'})

if __name__=='__main__':main()
