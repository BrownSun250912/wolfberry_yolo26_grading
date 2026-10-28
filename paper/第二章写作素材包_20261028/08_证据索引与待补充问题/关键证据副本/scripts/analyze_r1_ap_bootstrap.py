"""Paired image-block, development-set uncertainty; no fitting or test access."""
from __future__ import annotations
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils.metrics import ap_per_class
from run_r1_ap_followup import EX, OUT, ROOT, load_cache, policy, save
from train_r1_confidence_followup import ReliabilityHead, confidences


def pure_match(classes, gt, ious):
    """Official matching without constructor-created run directories."""
    return BaseValidator.match_predictions(
        SimpleNamespace(iouv=torch.linspace(.5,.95,10)),
        torch.as_tensor(classes),torch.as_tensor(gt),torch.as_tensor(ious)).numpy()


def ap_value(blocks, indices):
    tp,conf,cls,gt=(np.concatenate([blocks[i][j] for i in indices]) for j in range(4))
    ap=ap_per_class(tp,conf,cls,gt,plot=False)[5]
    if ap.shape != (4,10):
        raise ValueError('Bootstrap sample has missing target classes; do not silently change AP denominator')
    return np.array([ap[:,0].mean(),ap.mean()])


def build_blocks(cache, labels, confidence):
    return [(pure_match(cls,e['gt_cls'],e['mask_iou']),c,cls,e['gt_cls'])
            for e,cls,c in zip(cache,labels,confidence)]


def main():
    torch.set_num_threads(4)
    save(OUT/'bootstrap_plan.json',{'iterations':2000,'seed':20261008,
        'unit':'original image; all fruit/predictions stay together',
        'strata':'four pre-existing source ranges; preserve image count within each range',
        'paired':True,'interval':'percentile 95%, descriptive exploratory',
        'fixed_checkpoint_seed':20261008,'models':['baseline','hierarchy_high','linear_product','mlp_product','mlp_replace'],
        'no_new_training':True,'test_accessed':False,
        'limitations':'Development set reused in earlier model selection; image independence not guaranteed to imply independent physical fruit or harvest batches. These intervals do not establish external generalization or remove selection bias.'})
    cache=load_cache('val');base=[e['baseline'] for e in cache]
    full=[policy(e,True,True,True) for e in cache];raw=[e['conf'] for e in cache]
    blocks={'baseline':build_blocks(cache,base,raw),'hierarchy_high':build_blocks(cache,full,raw)}
    for family,kind,name in [('linear_confidence','product','linear_product'),('mlp_confidence','product','mlp_product'),('mlp_confidence','replace','mlp_replace')]:
        cp=torch.load(EX/f'{family}_seed20261008/best.pt',map_location='cpu',weights_only=False)
        head=ReliabilityHead(family);head.load_state_dict(cp['head_state']);head.eval()
        blocks[name]=build_blocks(cache,full,confidences(head,cp,cache,kind))
    indices=np.arange(len(cache));point={name:ap_value(b,indices) for name,b in blocks.items()}
    expected=pd.read_csv(OUT/'calibration_metrics.csv').set_index('variant')
    for name,variant in [('linear_product','linear_confidence_seed20261008_product'),('mlp_product','mlp_confidence_seed20261008_product'),('mlp_replace','mlp_confidence_seed20261008_replace')]:
        ref=expected.loc[variant,['metrics/mAP50(M)','metrics/mAP50-95(M)']].to_numpy(float)
        if not np.allclose(point[name],ref,atol=1e-10,rtol=0):raise ValueError('AP point reproduction failed')
    baseline=json.loads((ROOT/'outputs/reports/r1_score_closure_20261008/deployment_ap.json').read_text(encoding='utf-8'))['baseline']
    if abs(point['baseline'][0]-baseline['metrics/mAP50(M)'])>1e-10:raise ValueError('Baseline point mismatch')
    # Matched grading accuracy is a separate, conditional endpoint (500, not 518).
    fixed=pd.read_csv(ROOT/'outputs/reports/h1_valfix_replay_20261006/variant_instance_outcomes.csv')
    fixed=fixed[(fixed.variant=='baseline')&(fixed.localization_matched==True)].set_index('prediction_id')
    grading=[]
    for e,cls in zip(cache,full):
        rows=[(j,int(fixed.loc[pid,'gt_class_id'])) for j,pid in enumerate(e['old_prediction_ids']) if pid in fixed.index]
        grading.append([len(rows),sum(int(e['baseline'][j]==gt) for j,gt in rows),sum(int(cls[j]==gt) for j,gt in rows)])
    grading=np.asarray(grading)
    if grading.sum(0).tolist()!=[500,416,439]:raise ValueError('Fixed grading cohort changed')
    strata=[np.array([i for i,e in enumerate(cache) if e['source_group']==group]) for group in sorted({e['source_group'] for e in cache})]
    rng=np.random.default_rng(20261008);draws=[]
    for k in range(2000):
        sample=np.concatenate([rng.choice(group,len(group),replace=True) for group in strata])
        # Sorting retains original image order for tied confidences, without altering multiplicities.
        sample.sort();ap={name:ap_value(b,sample) for name,b in blocks.items()}
        n,bc,hc=grading[sample].sum(0)
        row={'replicate':k,'matched_n':int(n),'grading_delta':float((hc-bc)/n)}
        for name,value in ap.items():
            row[name+'_ap50']=value[0];row[name+'_ap95']=value[1]
            if name!='baseline':
                row[name+'_delta50']=value[0]-ap['baseline'][0]
                row[name+'_delta95']=value[1]-ap['baseline'][1]
        draws.append(row)
        if (k+1)%250==0:print(f'BOOTSTRAP {k+1}/2000',flush=True)
    frame=pd.DataFrame(draws);frame.to_csv(OUT/'bootstrap_draws.csv',index=False)
    summary=[]
    for name in blocks:
        if name=='baseline':continue
        for suffix,index in [('50',0),('95',1)]:
            values=frame[name+'_delta'+suffix].to_numpy()
            summary.append({'comparison':name+' minus baseline','endpoint':'mask_ap'+suffix,'delta':float(point[name][index]-point['baseline'][index]),'ci95_low':float(np.quantile(values,.025)),'ci95_high':float(np.quantile(values,.975)),'bootstrap_fraction_gt_zero':float((values>0).mean())})
    values=frame.grading_delta.to_numpy()
    summary.append({'comparison':'hierarchy minus baseline','endpoint':'conditional_grading_accuracy','delta':23/500,'ci95_low':float(np.quantile(values,.025)),'ci95_high':float(np.quantile(values,.975)),'bootstrap_fraction_gt_zero':float((values>0).mean())})
    pd.DataFrame(summary).to_csv(OUT/'bootstrap_summary.csv',index=False)
    save(OUT/'bootstrap_summary.json',{'status':'COMPLETED','point_metrics':{k:v.tolist() for k,v in point.items()},'intervals':summary,'point_reproduction':'PASS','test_accessed':False,'inference':'exploratory paired development-image intervals, not confirmatory significance; fraction_gt_zero is not a Bayesian probability or a p-value'})
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
