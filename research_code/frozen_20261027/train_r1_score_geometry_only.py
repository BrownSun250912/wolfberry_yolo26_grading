"""Geometry-only score ablation on the immutable S2 train/validation cache."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from train_r1_score_geometry_s2 import GeometryScoreHead, GEOMETRY_NAMES, fit_standardizer
from train_r1_score_specialist import ROOT, available_scores, evaluate_scores, save_json
from train_r1a_hierarchical_grader import set_determinism
from evaluate_r1a_frozen_predictions import sha256, paired_change

SOURCE = ROOT/'models/experiments/r1_score_geometry_s2_20261007_compatfix'
OUTPUT = ROOT/'models/experiments/r1_score_geometry_only_20261008'


class GeometryOnlyScoreHead(nn.Module):
    """Keep S2 initialization/capacity; the public input contains geometry only."""
    def __init__(self):
        super().__init__()
        self.head = GeometryScoreHead(use_geometry=True)

    def forward(self, geometry):
        if geometry.ndim != 2 or geometry.shape[1] != 8:
            raise ValueError('Expected N x 8 geometry, never R1 features')
        return self.head(geometry.new_zeros((len(geometry),256)), geometry)


def check_inputs(inputs):
    for name,item in inputs.items():
        if sha256(Path(item['path'])) != item['sha256']:
            raise ValueError(f'Input changed: {name}')


def load_data(inputs):
    check_inputs(inputs)
    source = json.loads(Path(inputs['s2_summary']['path']).read_text(encoding='utf-8'))
    geometry = torch.load(inputs['geometry_cache']['path'],map_location='cpu',weights_only=False)
    if geometry['inputs'] != source['inputs'] or tuple(geometry['geometry_names']) != tuple(GEOMETRY_NAMES):
        raise ValueError('Geometry cache lineage/schema mismatch')
    targets = pd.read_csv(inputs['source_s1_targets']['path'])
    if targets.instance_id.duplicated().any():
        raise ValueError('Duplicate targets')
    indexed = targets.set_index('instance_id',drop=False)
    train = indexed.loc[geometry['train_ids']].reset_index(drop=True)
    val = indexed.loc[geometry['val_ids']].reset_index(drop=True)
    for split,frame,n in [('train',train,4277),('val',val,518)]:
        if len(frame) != n or not frame.split.eq(split).all():
            raise ValueError(f'{split} population mismatch')
    if set(train.image_path)&set(val.image_path):
        raise ValueError('Train/validation image overlap')
    for split,n in [('train',4277),('val',518),('predictions',549)]:
        tensor = geometry[split]
        if tuple(tensor.shape) != (n,8) or not torch.isfinite(tensor).all():
            raise ValueError(f'Invalid geometry {split}')
    train_mask,_ = available_scores(train)
    val_mask,_ = available_scores(val)
    if (int(train_mask.sum()),int(val_mask.sum())) != (3905,464):
        raise ValueError('Score eligibility changed')
    raw = pd.read_csv(inputs['train_geometry_csv']['path']).set_index('instance_id')
    values = raw.loc[train.instance_id,list(GEOMETRY_NAMES)].to_numpy(dtype=np.float32)
    mean,std = fit_standardizer(values,train_mask)
    np.testing.assert_allclose(mean,geometry['mean'],rtol=1e-6,atol=1e-6)
    np.testing.assert_allclose(std,geometry['std'],rtol=1e-6,atol=1e-6)
    h1 = pd.read_csv(inputs['source_s1_h1']['path']).set_index('prediction_id',drop=False)
    h1 = h1.loc[geometry['prediction_ids']].reset_index(drop=True)
    fixed = pd.read_csv(inputs['source_fixed']['path']).set_index('prediction_id',drop=False)
    fixed = fixed.loc[geometry['prediction_ids']].reset_index(drop=True)
    if len(fixed) != 549 or not fixed.prediction_id.is_unique or not fixed.image_path.map(lambda p: Path(p).parent.name == 'val').all():
        raise ValueError('Fixed prediction IDs changed')
    outcomes = pd.read_csv(inputs['source_s1_outcomes']['path'])
    base = outcomes[outcomes.variant.eq('baseline')].copy()
    cases = pd.read_csv(inputs['source_s1_score_cases']['path'])
    if len(cases) != 448:
        raise ValueError('Matched score population changed')
    return geometry,train,val,h1,base,cases


def fit(geometry,train,val,args,output,inputs,deadline):
    set_determinism(args.seed)
    model = GeometryOnlyScoreHead().to(args.device)
    train_mask,train_y = available_scores(train)
    val_mask,val_y = available_scores(val)
    tx = geometry['train'][torch.from_numpy(train_mask.copy())].to(args.device)
    vx = geometry['val'][torch.from_numpy(val_mask.copy())].to(args.device)
    ty,vy = torch.from_numpy(train_y.copy()).to(args.device),torch.from_numpy(val_y.copy()).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss(beta=.05)
    generator = torch.Generator().manual_seed(args.seed)
    history,best,best_epoch,state = [],float('inf'),0,None
    for epoch in range(args.epochs+1):
        if time.monotonic() > deadline:
            raise TimeoutError('Geometry-only training budget exceeded')
        model.train()
        if epoch:
            for batch in torch.randperm(len(tx),generator=generator).to(args.device).split(args.batch_size):
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(tx[batch]),ty[batch])
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),5.)
                optimizer.step()
        model.eval()
        with torch.no_grad():
            train_mae = float((model(tx)-ty).abs().mean().cpu())*100
            val_mae = float((model(vx)-vy).abs().mean().cpu())*100
        history.append({'epoch':epoch,'train_mae':train_mae,'val_gt_crop_mae':val_mae})
        if val_mae < best-1e-6:
            best,best_epoch = val_mae,epoch
            state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        pd.DataFrame(history).to_csv(output/'history.csv',index=False)
        if epoch%10 == 0:
            print(f'geometry_only epoch={epoch} train_MAE={train_mae:.4f} val_MAE={val_mae:.4f} best={best:.4f}@{best_epoch}',flush=True)
        if epoch-best_epoch >= args.patience:
            break
    model.load_state_dict(state)
    record = {'best_epoch':best_epoch,'epochs_completed':epoch,'val_gt_crop_mae':best,
              'train_mae_at_best':history[best_epoch]['train_mae'],
              'parameters':sum(p.numel() for p in model.parameters()),
              'zero_input_weight_parameters':256*128,'input_features':8}
    torch.save({'head_state':state,'geometry_names':GEOMETRY_NAMES,'mean':geometry['mean'],
                'std':geometry['std'],'record':record,'seed':args.seed,'inputs':inputs,
                'requires_R1_feature_values':False},output/'best.pt')
    with torch.no_grad():
        scores = model(geometry['predictions'].to(args.device)).cpu().numpy()*100
        val_scores = model(geometry['val'].to(args.device)).cpu().numpy()*100
    table = val[['instance_id','score_target','score_target_available']].copy()
    table['score_prediction'] = val_scores
    table.to_csv(output/'val_gt_predictions.csv',index=False)
    pd.DataFrame({'prediction_id':geometry['prediction_ids'],'score_prediction':scores}).to_csv(output/'prediction_scores.csv',index=False)
    return scores,record


def verify(output):
    target = output/'checkpoint_verification.json'
    if target.exists():
        raise FileExistsError(target)
    summary = json.loads((output/'summary.json').read_text(encoding='utf-8'))
    inputs = summary['inputs']
    geometry,train,val,h1,base,cases = load_data(inputs)
    checkpoint = torch.load(output/'best.pt',map_location='cpu',weights_only=False)
    if checkpoint['inputs'] != inputs or checkpoint['requires_R1_feature_values']:
        raise ValueError('Checkpoint lineage/input mismatch')
    model = GeometryOnlyScoreHead().eval()
    model.load_state_dict(checkpoint['head_state'])
    with torch.no_grad():
        vp = model(geometry['val']).numpy()*100
        pp = model(geometry['predictions']).numpy()*100
    old = pd.read_csv(output/'val_gt_predictions.csv')
    delta = float(np.max(np.abs(vp-old.score_prediction.to_numpy())))
    if delta > .001:
        raise ValueError(f'Checkpoint prediction delta {delta}')
    metrics,stats,_,_,_ = evaluate_scores('geometry_only',pp,h1,base,cases)
    for key in ['correct_instances','uq_mismatches','grade_1_3_errors','score_gt_10']:
        if metrics[key] != summary['metrics'][key]:
            raise ValueError(f'Replay changed: {key}')
    if abs(stats['mae']-summary['score_metrics']['mae']) > .0001:
        raise ValueError('Replay score MAE changed')
    check_inputs(inputs)
    save_json(target,{'status':'PASS','scope':'CPU checkpoint/cache replay, not independent retraining',
                     'max_val_prediction_delta':delta,'replay_score_mae':stats['mae'],
                     'train_only_standardization_verified':True,'inputs_unchanged':True,
                     'R1_feature_values_loaded':False,'test_accessed':False,'checkpoint_sha256':sha256(output/'best.pt')})
    print('Checkpoint verification PASS',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--seed',type=int,default=20261007)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--epochs',type=int,default=200)
    parser.add_argument('--patience',type=int,default=40)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--learning-rate',type=float,default=.001)
    parser.add_argument('--timeout-seconds',type=int,default=1800)
    parser.add_argument('--verify',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    output = args.output.resolve()
    if args.verify:
        verify(output)
        return
    if output.exists():
        raise FileExistsError(f'Refusing overwrite: {output}')
    if min(args.epochs,args.patience,args.batch_size,args.timeout_seconds) < 1:
        raise ValueError('Invalid budget')
    source = json.loads((SOURCE/'summary.json').read_text(encoding='utf-8'))
    check_inputs(source['inputs'])
    paths = {'s2_summary':SOURCE/'summary.json','geometry_cache':SOURCE/'geometry_cache.pt',
             'train_geometry_csv':SOURCE/'train_geometry.csv','s2_metrics':SOURCE/'replay_metrics.csv',
             's2_score_metrics':SOURCE/'score_metrics.csv','s2_outcomes':SOURCE/'variant_instance_outcomes.csv',
             'reference_outcomes':Path(source['inputs']['s1_summary']['path']).parent/'variant_instance_outcomes.csv',
             'script':Path(__file__)}
    inputs = {k:{'path':str(p),'sha256':sha256(p)} for k,p in paths.items()}
    inputs.update({f'source_{k}':v for k,v in source['inputs'].items()})
    data = load_data(inputs)
    output.mkdir(parents=True)
    save_json(output/'config.json',{'date':'2026-10-08','args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                                  'inputs':inputs,'seed_reason':'Match S2 seed and minibatch order',
                                  'preflight_note':'Initial preflight stopped before output/training: GT and detector image paths use different roots. Fixed by matching immutable prediction IDs as S2 does.',
                                  'R1_feature_values_loaded':False})
    started = time.monotonic()
    try:
        geometry,train,val,h1,base,cases = data
        scores,record = fit(geometry,train,val,args,output,inputs,started+args.timeout_seconds)
        metrics,stats,outcomes,classes,points = evaluate_scores('geometry_only',scores,h1,base,cases)
        outcomes.to_csv(output/'variant_instance_outcomes.csv',index=False)
        classes.to_csv(output/'class_metrics.csv',index=False)
        points.to_csv(output/'matched_score_predictions.csv',index=False)
        pd.concat([pd.read_csv(paths['s2_metrics']),pd.DataFrame([metrics])],ignore_index=True).to_csv(output/'replay_metrics.csv',index=False)
        pd.concat([pd.read_csv(paths['s2_score_metrics']),pd.DataFrame([{'variant':'geometry_only',**stats}])],ignore_index=True).to_csv(output/'score_metrics.csv',index=False)
        all_outcomes = pd.concat([pd.read_csv(paths['reference_outcomes']),pd.read_csv(paths['s2_outcomes']),outcomes],ignore_index=True)
        paired,tables = {},[]
        for ref in ['S0_cached_learned','S1_mlp','H1_rule_score','S2_zero_control','S2_geometry']:
            counts,table = paired_change(all_outcomes,ref,'geometry_only')
            paired[ref] = counts
            tables.append(table)
        pd.concat(tables,ignore_index=True).to_csv(output/'paired_changes.csv',index=False)
        check_inputs(inputs)
        save_json(output/'summary.json',{'status':'COMPLETED','origin_skill':'experiment-agent','mode':'run',
                   'verification_status':'ANALYZED','date':'2026-10-08','record':record,'metrics':metrics,
                   'score_metrics':stats,'paired_changes':paired,'inputs':inputs,
                   'elapsed_seconds':time.monotonic()-started,'labels_changed':False,'test_accessed':False,
                   'thresholds_changed':False,'R1_feature_values_loaded':False,'gates_reused':True})
        print(json.dumps({'record':record,'metrics':metrics,'paired_changes':paired},ensure_ascii=False,indent=2),flush=True)
    except Exception as exc:
        save_json(output/'failure.json',{'status':'FAILED','error':repr(exc),'inputs':inputs})
        raise


if __name__ == '__main__':
    main()
