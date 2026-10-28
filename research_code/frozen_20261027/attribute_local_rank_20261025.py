"""OOF defect evidence for fixed-grade quality ranking; paired compact controls."""
from attribute_local_common_20261025 import *
setup()
import copy,numpy as np,pandas as pd,torch
from torch import nn
from attribute_local_evaluate_20261025 import load_records
from attribute_local_source_cv_20261025 import group
from attribute_local_target_evaluate_20261025 import gate_labels
from attribute_local_prior_20261025 import correct
from joint_pipeline_decision_matrix_20261024 import joined_sources
from joint_pipeline_policy_20261024 import decide,logit
from joint_pipeline_rank_fit_20261024 import inputs as base_inputs,targets
from joint_pipeline_evaluate_20261024 import evaluate
setup()

class Rank(nn.Module):
    def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(22,16),nn.SiLU(),nn.Linear(16,1))
    def forward(self,x):return self.net(x).squeeze(1)

def register():
    p=BASE/'attribute_rank_plan.json'
    if not p.exists():save(p,dict(registered_utc=dt.datetime.now(dt.timezone.utc).isoformat(),timing='Adaptive after attribute/boundary cleanval and audit outcomes, before any rank fits or rank-val outputs.',
      hypothesis='New defect evidence may help reliability ranking even where whole-population class replacement is unsafe.',
      design='2 fixed grade policies RR22 and correctedcreation99 x2 feature arms core18+zero4/core18+attribute4 x2 targets independentTP50/mean10IoUquality x3seeds=24 fits; original masks fixed. Architecture22->16->1, same allocated parameters and pairedinit/order; effective-information capacity differs.',
      attribute_features='Fit train uses source-left-out correcteddefect probability of correspondingseed, so no newly fitted defecthead used its evaluated source for fitting/checkpoint. Fourextra: correcteddefectlogit/10, entropy/log2, availableflag, differencefromoriginaldefectprob. Lowconfidence missingattribute zero-filled withflag0. Val uses corresponding fulltrain-role head, not bestval seed. Different OOF vs full-role sampledose disclosed.',
      roles='Original headfit/held roles. Feature normalization onlyrankfit eligible finalclass!=0 predictions, all lowconfidence/unmatched included. Held is only new rank checkpoint/selection, original encoder saw alltrain; sourceOOF onlynewdefectadaptation. Original grade scores already fittedoriginaltrain.',
      dose='CPUFP32 AdamW.003 wd.0001 max300patience30 batch256; epoch0 allowed. Positive/quality targets from GT trainingonly, neveronlineeligibility.',
      confidence='sigmoidrank * originalrawconf for finalclass!=0; all finalclass0 confidences unchanged. Actual sameclassNMS .7 and originalrawscopepriority; operating scope remains originalraw .25.',
      selection='For each gradepolicy, complete3seed heldmean AP50>=rawrank+.002, AP95>=rawrank, correct/allGT>=rawrank, UQescape<=rawrank, qualifiedreject<=rawrank, newlymisacceptedGT=0. MaxAP95 thenAP50, zeroextra wins ties. Fixedseed25. Ifnone retainraw. Evaluateall24 val onlyafter selections frozen; never promote basedval.',
      limits='Ranking does not create new segmentation localization; no calibrated probability claim. Many adaptive comparisons on reusedval, exploratory only. Candidateclass0 preservation does not guarantee GT safety after matching.',test_accessed=False,primary_replaced=False))

def load(split,seed):
    sources,_=joined_sources(split);es=sources['rule_anchored22'];rs=load_records(split)
    if split=='train':
        mapped={}
        for outer in range(4):
            directory=BASE/'source_cv'/f'source{outer}_exploratory_seed{seed}';cp=torch.load(directory/'best.pt',map_location='cpu',weights_only=False);pred=torch.load(directory/'outer_predictions.pt',map_location='cpu',weights_only=False);ratio=float(cp['positive_weights'][0]/cp['negative_weights'][0])
            for iid,p in zip(pred['image_ids'],pred['probabilities']):
                q=p.copy();v=np.clip(q[:,0],1e-6,1-1e-6);q[:,0]=1/(1+np.exp(-(np.log(v/(1-v))-np.log(ratio))));mapped[iid]=q
        ps=[mapped[e['image_id']] for e in es]
    else:
        name=f'defect_balanced_neutral_pair_seed{seed}';cp=torch.load(BASE/'target_fits'/name/'best.pt',map_location='cpu',weights_only=False);pred=torch.load(BASE/'target_outputs_val'/(name+'.pt'),map_location='cpu',weights_only=False)
        if pred['image_ids']!=[e['image_id'] for e in es]:raise ValueError('Attribute val order')
        ps=[correct(p,cp,True,False) for p in pred['probabilities']]
    defect=[]
    for e,r,p in zip(es,rs,ps):
        value=np.full(len(e['conf']),np.nan,np.float32);value[r['prediction_indices']]=p[:,0];defect.append(value)
    return es,rs,ps,defect

def labels_for(es,rs,ps,policy):
    return [decide(e,'neighbour',True,True,False) for e in es] if policy=='RR22' else gate_labels(es,rs,ps,'defect','creation_only.99')

def features(e,cls,prob,extra):
    x=base_inputs(e,cls,True);available=np.isfinite(prob);p=np.clip(np.nan_to_num(prob,nan=.5),1e-6,1-1e-6);old=1/(1+np.exp(-np.clip(np.nan_to_num(e['defect_logit'],nan=0),-80,80)))
    add=np.stack([np.log(p/(1-p))/10,-(p*np.log(p)+(1-p)*np.log1p(-p))/np.log(2),available.astype(float),p-old],1).astype(np.float32);add[~available]=0
    if extra=='flag_only':add[:,[0,1,3]]=0
    elif extra=='original_defect':
        q=np.clip(old,1e-6,1-1e-6);add=np.stack([np.log(q/(1-q))/10,-(q*np.log(q)+(1-q)*np.log1p(-q))/np.log(2),available.astype(float),np.zeros_like(q)],1).astype(np.float32);add[~available]=0
    result=np.concatenate([x,add if extra else np.zeros_like(add)],1)
    if result.shape[1]!=22 or not np.isfinite(result).all():raise ValueError('Nonfinite rank feature')
    return result

def confidence(es,ls,defects,cp,model):
    result=[]
    with torch.no_grad():
        for e,cls,p in zip(es,ls,defects):
            x=features(e,cls,p,cp['extra']);z=model(torch.tensor((x-cp['mean'])/cp['std']));c=z.sigmoid().numpy()*e['conf'];c[np.asarray(cls)==0]=e['conf'][np.asarray(cls)==0];result.append(c)
    return result

def prepare(es,ls,defects,role,extra,target):
    packs={'fit':[[],[],[]],'held':[[],[],[]]}
    for e,cls,p in zip(es,ls,defects):
        use=np.flatnonzero(cls!=0);x=features(e,cls,p,extra);y=targets(e,cls,target);key='fit' if role.loc[e['image_id'],'role']=='head_fit' else 'held';packs[key][0].append(x[use]);packs[key][1].append(y[use]);packs[key][2].extend([(e['image_id'],int(j)) for j in use])
    fx,fy=[np.concatenate(packs['fit'][i]) for i in [0,1]];hx,hy=[np.concatenate(packs['held'][i]) for i in [0,1]];mean=fx.mean(0);std=fx.std(0);std[std<1e-6]=1
    return [torch.tensor(v) for v in [(fx-mean)/std,fy,(hx-mean)/std,hy]],mean,std,packs

def fit(policy,extra,target,seed,prepared):
    kind=extra if isinstance(extra,str) else ('attribute4' if extra else 'zero4');directory=BASE/'attribute_rank_fits'/f'{policy}_{kind}_{target}_seed{seed}';complete=directory/'completion.json'
    if complete.exists():return json.loads(complete.read_text(encoding='utf-8'))
    if directory.exists():raise FileExistsError(str(directory))
    directory.mkdir(parents=True);torch.manual_seed(seed);model=Rank();torch.save(model.state_dict(),directory/'initial.pt');(x,y,hx,hy),mean,std,packs=prepared;gen=torch.Generator().manual_seed(seed);optim=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.0001);best=float('inf');best_epoch=0;rows=[];partial=False;started=time.perf_counter()
    save(directory/'population.json',dict(fit=packs['fit'][2],held=packs['held'][2]))
    for epoch in range(301):
        if epoch:
            model.train()
            for ix in torch.randperm(len(x),generator=gen).split(256):
                optim.zero_grad(set_to_none=True);loss=nn.functional.binary_cross_entropy_with_logits(model(x[ix]),y[ix]);loss.backward();optim.step()
        model.eval()
        with torch.no_grad():held=float(nn.functional.binary_cross_entropy_with_logits(model(hx),hy))
        rows.append(dict(epoch=epoch,held_BCE=held))
        if held<best-1e-6:
            best=held;best_epoch=epoch;torch.save(dict(state=model.state_dict(),mean=mean,std=std,extra=extra,target=target,seed=seed,epoch=epoch,policy=policy,held_BCE=held),directory/'best.pt')
        partial=stop_requested(new_fit=True)
        if partial or epoch-best_epoch>=30:break
    torch.save(dict(state=model.state_dict(),optimizer=optim.state_dict(),epoch=epoch,generator=gen.get_state()),directory/'last.pt');pd.DataFrame(rows).to_csv(directory/'history.csv',index=False)
    r=dict(policy=policy,extra=extra,target=target,seed=seed,path=str(directory/'best.pt'),epochs=epoch,best_epoch=best_epoch,held_BCE=best,partial=partial,seconds=time.perf_counter()-started,checkpoint_sha256=sha256(directory/'best.pt'),initial_sha256=sha256(directory/'initial.pt'));save(complete,r);print('ATTRIBUTE RANK FIT',directory.name,epoch,flush=True);return r

def main():
    plan();register();role=pd.read_csv(ROOT/'models/experiments/r1_quality_followup_20261009/head_image_split.csv').set_index('image_id');ledger=[];rows=[]
    for seed in SEEDS:
        es,rs,ps,ds=load('train',seed);held=[i for i,e in enumerate(es) if role.loc[e['image_id'],'role']=='head_holdout'];he=[es[i] for i in held];hd=[ds[i] for i in held];choices=[(p,e,t) for p in ['RR22','correctedcreation99'] for e in [False,True] for t in ['independent_tp50','independent_multi_iou']];np.random.default_rng(seed).shuffle(choices)
        for policy,extra,target in choices:
            if stop_requested(new_fit=True):break
            ls=labels_for(es,rs,ps,policy);hl=[ls[i] for i in held];ref,_,rc,_=evaluate(he,hl,dedup='scope_priority',case_rows=True);safe={c['instance_id'] for c in rc if c['gt_class']==0 and c['pred_class']==0}
            r=fit(policy,extra,target,seed,prepare(es,ls,ds,role,extra,target));ledger.append(r);cp=torch.load(r['path'],map_location='cpu',weights_only=False);model=Rank().eval();model.load_state_dict(cp['state']);rank=confidence(he,hl,hd,cp,model);m,_,cases,_=evaluate(he,hl,rank,'scope_priority',case_rows=True)
            rows.append(dict(policy=policy,extra=extra,target=target,seed=seed,reference_AP50=ref['Mask_AP50'],reference_AP95=ref['Mask_AP50_95'],reference_correct=ref['correct_per_all_GT'],reference_UQ=ref['UQ_escape'],reference_rejected=ref['qualified_rejected'],new_UQ_escape=sum(c['instance_id'] in safe and c['pred_class']!=0 for c in cases if c['gt_class']==0),**m));pd.DataFrame(ledger).to_csv(OUT/'attribute_rank_ledger.csv',index=False);pd.DataFrame(rows).to_csv(OUT/'attribute_rank_held_metrics.csv',index=False)
    f=pd.DataFrame(rows);families=f.groupby(['policy','extra','target']).agg(seeds=('seed','nunique'),AP50=('Mask_AP50','mean'),AP95=('Mask_AP50_95','mean'),correct=('correct_per_all_GT','mean'),UQ=('UQ_escape','mean'),rejected=('qualified_rejected','mean'),new_UQ=('new_UQ_escape','mean'),ref50=('reference_AP50','mean'),ref95=('reference_AP95','mean'),refcorrect=('reference_correct','mean'),refUQ=('reference_UQ','mean'),refrejected=('reference_rejected','mean')).reset_index();families['eligible']=(families.seeds==3)&(families.AP50>=families.ref50+.002)&(families.AP95>=families.ref95)&(families.correct>=families.refcorrect)&(families.UQ<=families.refUQ)&(families.rejected<=families.refrejected)&(families.new_UQ==0);families.to_csv(OUT/'attribute_rank_families.csv',index=False);selections=[]
    for policy,g in families.groupby('policy'):
        good=g[g.eligible];best=None if not len(good) else good.sort_values(['AP95','AP50','extra'],ascending=[False,False,True],kind='stable').iloc[0].to_dict();selections.append(dict(policy=policy,selected=best,fixed_seed=20261025,before_rank_val=True))
    save(OUT/'attribute_rank_selections.json',selections);save(OUT/'attribute_rank_fit_status.json',dict(status='completed' if len(ledger)==24 and not any(r['partial'] for r in ledger) else 'partial',fits=len(ledger),test_accessed=False))

if __name__=='__main__':main()
