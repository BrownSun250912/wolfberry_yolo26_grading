"""Adaptive source-block audit of two fixed recipes, never a whole-system OOD test."""
from attribute_local_common_20261025 import *
setup()
import numpy as np,pandas as pd,torch
from attribute_local_fit_20261025 import AttributeModel,original,load_data,balanced_loss,predict
from attribute_local_target_fit_20261025 import background_data
from attribute_local_evaluate_20261025 import load_records,outputs
from attribute_local_target_evaluate_20261025 import gate_labels
from joint_pipeline_decision_matrix_20261024 import joined_sources
from joint_pipeline_policy_20261024 import decide
from joint_pipeline_evaluate_20261024 import evaluate
setup()

def group(image):
    n=int(image.rsplit('_',1)[1])
    return 0 if n<=150 else 1 if n<=500 else 2 if n<=900 else 3

def register():
    p=BASE/'source_cv_plan.json'
    if not p.exists():save(p,dict(registered_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
      timing='Adaptive after primary and target val results; no outer-source results observed for these new fits.',
      recipes={'internal_selected':'unweighted clean duplicate + hysteresis .05/.95','exploratory':'balanced neutral pair + creation-only .99'},
      design='4 outer source index ranges x2 fixed recipes x3 seeds=24 new fits; outer excluded from new-head fitting and checkpoint selection. Remaining original head_fit/holdout roles determine inner fitting/checkpoints. Same initialization, max40 epochs/patience10, AdamW head.001 tail.00003, batch128.',
      selection='No further family selection. Report all outer results and aggregate disjoint images per seed. Fit weights recomputed only on inner fit. No val or test access.',
      limits='Original detector, visual prefix, original tail initialization, geometry normalization and old score heads saw all original train. This isolates additional adaptation only, cannot support whole-system unseen-source generalization. Source index ranges are proxies, not proven independent physical-fruit groups. Recipe comparison jointly changes balancing/background/policy and is not a causal isolated-factor contrast.',test_accessed=False))

def subset(p,keep):
    ids=[i for i,r in enumerate(p['ids']) if keep(group(r[0]))];idx=torch.tensor(ids,device='cuda')
    return {k:([p[k][i] for i in ids] if k=='ids' else p[k][idx]) for k in p}

def fit(outer,recipe,seed,packs,bg):
    name=f'source{outer}_{recipe}_seed{seed}';directory=BASE/'source_cv'/name
    complete=directory/'completion.json'
    if complete.exists():return json.loads(complete.read_text(encoding='utf-8'))
    if directory.exists():raise FileExistsError(str(directory))
    directory.mkdir(parents=True);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=AttributeModel(original(),'global_global',True).cuda().eval()
    for p in model.stem_head.parameters():p.requires_grad_(False)
    torch.save(model.state_dict(),directory/'initial.pt')
    optim=torch.optim.AdamW([dict(params=model.defect_head.parameters(),lr=.001),dict(params=model.tail.parameters(),lr=.00003)],weight_decay=.0001)
    paired=recipe=='exploratory';freq=torch.nanmean(packs['fit']['target'][:,0]).clamp(.01,.99)
    pos=(.5/freq if paired else torch.ones_like(freq)).reshape(1);neg=(.5/(1-freq) if paired else torch.ones_like(freq)).reshape(1)
    def loss(z,y):return balanced_loss(z[:,:1],y[:,:1],pos,neg)
    generator=torch.Generator().manual_seed(seed);best=float('inf');best_epoch=0;history=[];partial=False;started=time.perf_counter()
    for epoch in range(41):
        if epoch:
            p=packs['fit']
            for ids in torch.randperm(len(p['target']),generator=generator).split(128):
                ids=ids.cuda();optim.zero_grad(set_to_none=True)
                a=model(p['prefix'][ids],p['geometry'][ids]);b=model(bg['fit'][ids] if paired else p['prefix'][ids],p['geometry'][ids])
                objective=.5*(loss(a,p['target'][ids])+loss(b,p['target'][ids]))+.5*(a[:,0].sigmoid()-b[:,0].sigmoid()).square().mean()
                objective.backward();optim.step()
        with torch.no_grad():
            p=packs['held'];a=predict(model,p['prefix'],p['geometry']);b=predict(model,bg['held'],p['geometry'])
            held=float(.5*(loss(a,p['target'])+loss(b,p['target'])))
        history.append(dict(epoch=epoch,held_BCE=held))
        if held<best-1e-6:
            best=held;best_epoch=epoch;torch.save(dict(state=model.state_dict(),view='global_global',adapt=True,seed=seed,epoch=epoch,positive_weights=pos.cpu(),negative_weights=neg.cpu()),directory/'best.pt')
        pd.DataFrame(history).to_csv(directory/'history.csv',index=False)
        partial=stop_requested(new_fit=True)
        if partial or epoch-best_epoch>=10:break
    torch.save(dict(state=model.state_dict(),optimizer=optim.state_dict(),epoch=epoch,generator=generator.get_state()),directory/'last.pt')
    result=dict(outer=outer,recipe=recipe,seed=seed,path=str(directory/'best.pt'),epochs=epoch,best_epoch=best_epoch,partial=partial,seconds=time.perf_counter()-started,fit_instances=len(packs['fit']['ids']),held_instances=len(packs['held']['ids']),fit_positive_frequency=float(freq),checkpoint_sha256=sha256(directory/'best.pt'))
    save(directory/'population.json',dict(fit=packs['fit']['ids'],held=packs['held']['ids'],outer=outer))
    save(complete,result);print('SOURCE FIT',name,'epoch',epoch,'best',best_epoch,flush=True);return result

def main():
    plan();register();allpacks,_,_=load_data();allbg=background_data(allpacks);records=load_records('train');sources,_=joined_sources('train');entries=sources['rule_anchored22'];ledger=[];rows=[]
    for outer in range(4):
        packs={role:subset(p,lambda g:g!=outer) for role,p in allpacks.items()}
        bg={role:allbg[role][torch.tensor([i for i,r in enumerate(allpacks[role]['ids']) if group(r[0])!=outer],device='cuda')] for role in allpacks}
        take=[i for i,e in enumerate(entries) if group(e['image_id'])==outer];es=[entries[i] for i in take];rs=[records[i] for i in take]
        ref,_,rc,_=evaluate(es,[decide(e,'neighbour',True,True,False) for e in es],dedup='scope_priority',case_rows=True)
        safe={c['instance_id'] for c in rc if c['gt_class']==0 and c['pred_class']==0}
        for seed in SEEDS:
            recipes=['internal_selected','exploratory'];np.random.default_rng(seed+outer).shuffle(recipes)
            for recipe in recipes:
                if stop_requested(new_fit=True):break
                r=fit(outer,recipe,seed,packs,bg);ledger.append(r);pd.DataFrame(ledger).to_csv(OUT/'source_cv_fit_ledger.csv',index=False)
                if r['partial']:break
                cp=torch.load(r['path'],map_location='cpu',weights_only=False);pp=outputs(cp,rs)
                variant='creation_only.99' if recipe=='exploratory' else 'hysteresis.05_.95';labels=gate_labels(es,rs,pp,'defect',variant)
                m,_,cases,_=evaluate(es,labels,dedup='scope_priority',case_rows=True)
                rows.append(dict(outer=outer,recipe=recipe,seed=seed,new_UQ_escape=sum(c['instance_id'] in safe and c['pred_class']!=0 for c in cases if c['gt_class']==0),reference_AP50=ref['Mask_AP50'],reference_correct=ref['correct_per_all_GT'],reference_UQ_escape=ref['UQ_escape'],reference_rejected=ref['qualified_rejected'],**m))
                torch.save(dict(image_ids=[e['image_id'] for e in es],probabilities=pp,labels=labels,checkpoint_sha256=r['checkpoint_sha256']),Path(r['path']).parent/'outer_predictions.pt')
                pd.DataFrame(cases).to_csv(OUT/f'source{outer}_{recipe}_seed{seed}_cases.csv',index=False)
                pd.DataFrame(rows).to_csv(OUT/'source_cv_metrics.csv',index=False)
    save(OUT/'source_cv_status.json',dict(status='completed' if len(rows)==24 else 'partial',fits=len(ledger),evaluations=len(rows),test_accessed=False))

if __name__=='__main__':main()
