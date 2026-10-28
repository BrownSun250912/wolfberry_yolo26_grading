"""Audit whether new rank gain needs learned attributes, rather than scope flag/old head."""
from attribute_local_common_20261025 import *
setup()
import numpy as np,pandas as pd,torch
from attribute_local_rank_20261025 import Rank,load,labels_for,prepare,fit,confidence
from joint_pipeline_evaluate_20261024 import evaluate

def main():
    p=BASE/'rank_information_control_plan.json'
    if not p.exists():save(p,dict(registered_utc=dt.datetime.now(dt.timezone.utc).isoformat(),trigger='All24newrank val outcomes observed. Primary new4 includes a rawscope availabilityflag and old-new defectdifference, absent fromzero4; gain cannot yet be attributed to learnedvisualevidence.',
      design='Only fixedRR22 gradepolicy and internally selected qualitytarget, two controls x3seeds=6fits. Same22->16->1, normalizationfitonly, init/order/max300patience30 as paired primary. Control1 zeroadditionalfeatures except availableflag; control2 originaldefectlogit/entropy/flag andzero4thdelta. Sameallocatedslots but effective-informationcapacitydiffers.',
      selection='New control checkpoints minimum sameheadheldBCE; all6held finish before inspecting theirval. No overwrite originalfamilyselection or bestvalfeature selection. Compare raw0/new4,flag-only,originaldefect; report all, including any gain explained by preexisting information.',
      limits='Adaptive mechanismaudit, repeatedval. Existingdefecthead/control can have less independent information; four extrafeatures are deterministic transformations, not independent labels.',test_accessed=False))
    role=pd.read_csv(ROOT/'models/experiments/r1_quality_followup_20261009/head_image_split.csv').set_index('image_id');ledger=[];rows=[]
    for seed in SEEDS:
        es,rs,ps,ds=load('train',seed);ls=labels_for(es,rs,ps,'RR22');ix=[i for i,e in enumerate(es) if role.loc[e['image_id'],'role']=='head_holdout'];he=[es[i] for i in ix];hl=[ls[i] for i in ix];hd=[ds[i] for i in ix]
        for kind in ['flag_only','original_defect']:
            if stop_requested(new_fit=True):break
            r=fit('RR22',kind,'independent_multi_iou',seed,prepare(es,ls,ds,role,kind,'independent_multi_iou'));ledger.append(r);cp=torch.load(r['path'],map_location='cpu',weights_only=False);model=Rank().eval();model.load_state_dict(cp['state']);c=confidence(he,hl,hd,cp,model);m,_,_,_=evaluate(he,hl,c,'scope_priority');rows.append(dict(partition='held',control=kind,seed=seed,**m));pd.DataFrame(ledger).to_csv(OUT/'attribute_rank_control_ledger.csv',index=False)
    save(OUT/'rank_information_control_frozen.json',dict(complete_fits=len(ledger),before_control_val=True,test_accessed=False))
    if len(ledger)!=6:raise RuntimeError('Incomplete controls; no val')
    for seed in SEEDS:
        es,rs,ps,ds=load('val',seed);ls=labels_for(es,rs,ps,'RR22')
        for r in [r for r in ledger if r['seed']==seed]:
            cp=torch.load(r['path'],map_location='cpu',weights_only=False);model=Rank().eval();model.load_state_dict(cp['state']);c=confidence(es,ls,ds,cp,model);m,_,cases,blocks=evaluate(es,ls,c,'scope_priority',case_rows=True);rows.append(dict(partition='val',control=r['extra'],seed=seed,**m));name=Path(r['path']).parent.name;torch.save(dict(labels=ls,confidence=c,blocks=blocks),BASE/'attribute_rank_val'/(name+'.pt'));pd.DataFrame(cases).to_csv(OUT/f'rank_control_{name}_cases.csv',index=False)
    pd.DataFrame(rows).to_csv(OUT/'attribute_rank_control_metrics.csv',index=False);save(OUT/'attribute_rank_control_status.json',dict(status='completed',fits=len(ledger),rows=len(rows),test_accessed=False));print('RANK INFORMATION CONTROLS',len(ledger),'fits',flush=True)

if __name__=='__main__':main()
