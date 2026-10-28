"""S3: S2 score head plus fixed-threshold ordinal auxiliary; val development only.

Two fresh heads share architecture, initialization, data, batches and selection.
The control has zero auxiliary weight. No new ordinal inference head is added.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from train_r1_score_geometry_only import SOURCE, load_data, check_inputs
from train_r1_score_geometry_s2 import GeometryScoreHead, GEOMETRY_NAMES
from train_r1_score_specialist import ROOT, available_scores, evaluate_scores, save_json
from train_r1a_hierarchical_grader import set_determinism
from diagnose_r1_score_replay import LOWER, UPPER
from evaluate_r1a_frozen_predictions import sha256, paired_change

OUTPUT = ROOT/'models/experiments/r1_score_ordinal_s3_20261008'


def ordinal_targets(scores_100):
    """Reference score thresholds, not final labels (which include gates)."""
    values = torch.as_tensor(scores_100,dtype=torch.float64)
    if values.ndim != 1 or not torch.isfinite(values).all() or ((values<0)|(values>100)).any():
        raise ValueError('Expected finite reference scores in [0,100]')
    thresholds = values.new_tensor([LOWER,UPPER])
    return (values[:,None] >= thresholds).to(torch.float32)


def ordinal_logits(prediction,temperature):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Ordinal temperature must be finite and positive')
    if prediction.ndim != 1:
        raise ValueError('Expected one normalized score per instance')
    return (prediction[:,None]-prediction.new_tensor([LOWER/100,UPPER/100]))/temperature


def objective(prediction,target,ordinal_target,weight,temperature):
    if not np.isfinite(weight) or weight < 0:
        raise ValueError('Invalid ordinal weight')
    regression = F.smooth_l1_loss(prediction,target,beta=.05)
    auxiliary = F.binary_cross_entropy_with_logits(ordinal_logits(prediction,temperature),ordinal_target)
    total = regression if weight == 0 else regression+weight*auxiliary
    return total,regression,auxiliary


def load_all(inputs):
    geometry,train,val,h1,base,cases = load_data(inputs)
    cache = torch.load(inputs['source_feature_cache']['path'],map_location='cpu',weights_only=False)
    s1 = json.loads(Path(inputs['source_s1_summary']['path']).read_text(encoding='utf-8'))
    if cache['inputs'] != s1['inputs']:
        raise ValueError('Frozen R1 feature lineage changed')
    for key in ['train_ids','val_ids','prediction_ids']:
        if cache[key] != geometry[key]:
            raise ValueError(f'Geometry/feature alignment changed: {key}')
    for split,n in [('train',4277),('val',518),('predictions',549)]:
        if tuple(cache[split].shape) != (n,256) or not torch.isfinite(cache[split]).all():
            raise ValueError(f'Invalid R1 feature tensor: {split}')
    return cache,geometry,train,val,h1,base,cases


def fit(kind,weight,cache,geometry,train,val,args,output,inputs,deadline):
    set_determinism(args.seed)
    head = GeometryScoreHead(True).to(args.device)
    tm,ty_np = available_scores(train)
    vm,vy_np = available_scores(val)
    ti,vi = torch.from_numpy(tm.copy()),torch.from_numpy(vm.copy())
    tx,vx = cache['train'][ti].to(args.device),cache['val'][vi].to(args.device)
    tg,vg = geometry['train'][ti].to(args.device),geometry['val'][vi].to(args.device)
    ty,vy = torch.from_numpy(ty_np.copy()).to(args.device),torch.from_numpy(vy_np.copy()).to(args.device)
    to = ordinal_targets(train.loc[tm,'score_target'].to_numpy()).to(args.device)
    vo = ordinal_targets(val.loc[vm,'score_target'].to_numpy()).to(args.device)
    optimizer = torch.optim.AdamW(head.parameters(),lr=args.learning_rate,weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed)
    history,best,best_epoch,state = [],float('inf'),0,None
    for epoch in range(args.epochs+1):
        if time.monotonic() > deadline:
            raise TimeoutError('S3 training timeout')
        head.train()
        if epoch:
            order = torch.randperm(len(tx),generator=generator).to(args.device)
            for batch in order.split(args.batch_size):
                optimizer.zero_grad(set_to_none=True)
                loss,_,_ = objective(head(tx[batch],tg[batch]),ty[batch],to[batch],weight,args.temperature)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite loss')
                loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(),5.)
                optimizer.step()
        head.eval()
        with torch.no_grad():
            tp,vp = head(tx,tg),head(vx,vg)
            train_mae = float((tp-ty).abs().mean().cpu())*100
            val_mae = float((vp-vy).abs().mean().cpu())*100
            train_loss,regression,auxiliary = objective(tp,ty,to,weight,args.temperature)
            crossing = int(((ordinal_logits(vp,args.temperature)>=0) != vo.bool()).any(dim=1).sum().cpu())
        history.append({'epoch':epoch,'train_mae':train_mae,'val_gt_crop_mae':val_mae,
                        'train_total_loss':float(train_loss.cpu()),'train_regression_loss':float(regression.cpu()),
                        'train_ordinal_loss':float(auxiliary.cpu()),'val_score_grade_errors':crossing})
        if val_mae < best-1e-6:
            best,best_epoch = val_mae,epoch
            state = {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        pd.DataFrame(history).to_csv(output/f'{kind}_history.csv',index=False)
        if epoch%10 == 0:
            print(f'S3 {kind} epoch={epoch} train_MAE={train_mae:.4f} val_MAE={val_mae:.4f} best={best:.4f}@{best_epoch}',flush=True)
        if epoch-best_epoch >= args.patience:
            break
    head.load_state_dict(state)
    record = {'kind':kind,'ordinal_weight':weight,'temperature':args.temperature,
              'best_epoch':best_epoch,'epochs_completed':epoch,'val_gt_crop_mae':best,
              'train_mae_at_best':history[best_epoch]['train_mae'],
              'val_gt_score_grade_errors_at_best':history[best_epoch]['val_score_grade_errors'],
              'parameters':sum(p.numel() for p in head.parameters())}
    torch.save({'head_state':state,'geometry_names':GEOMETRY_NAMES,'mean':geometry['mean'],
                'std':geometry['std'],'record':record,'seed':args.seed,'inputs':inputs,
                'requires_frozen_R1_features':True,'thresholds':[LOWER,UPPER]},output/f'{kind}_best.pt')
    with torch.no_grad():
        pp = head(cache['predictions'].to(args.device),geometry['predictions'].to(args.device)).cpu().numpy()*100
        vp = head(cache['val'].to(args.device),geometry['val'].to(args.device)).cpu().numpy()*100
    table = val[['instance_id','score_target','score_target_available']].copy()
    table['score_prediction'] = vp
    table.to_csv(output/f'{kind}_val_gt_predictions.csv',index=False)
    pd.DataFrame({'prediction_id':cache['prediction_ids'],'score_prediction':pp}).to_csv(output/f'{kind}_prediction_scores.csv',index=False)
    return pp,record


def verify(output):
    target = output/'checkpoint_verification.json'
    if target.exists():
        raise FileExistsError(target)
    summary = json.loads((output/'summary.json').read_text(encoding='utf-8'))
    inputs = summary['inputs']
    cache,geometry,train,val,h1,base,cases = load_all(inputs)
    results = {}
    for kind,variant in [('control','S3_control'),('ordinal','S3_ordinal')]:
        checkpoint = torch.load(output/f'{kind}_best.pt',map_location='cpu',weights_only=False)
        if checkpoint['inputs'] != inputs or checkpoint['thresholds'] != [LOWER,UPPER]:
            raise ValueError('Checkpoint lineage mismatch')
        head = GeometryScoreHead(True).eval()
        head.load_state_dict(checkpoint['head_state'])
        with torch.no_grad():
            vp = head(cache['val'],geometry['val']).numpy()*100
            pp = head(cache['predictions'],geometry['predictions']).numpy()*100
        saved = pd.read_csv(output/f'{kind}_val_gt_predictions.csv')
        if saved.instance_id.tolist() != cache['val_ids']:
            raise ValueError('Saved validation IDs changed')
        delta = float(np.max(np.abs(vp-saved.score_prediction.to_numpy())))
        if delta > .001:
            raise ValueError('CPU/GPU prediction delta too large')
        mask,y = available_scores(val)
        mae = float(np.mean(np.abs(vp[mask]-y*100)))
        if abs(mae-summary['runs'][kind]['val_gt_crop_mae']) > .0001:
            raise ValueError('Validation MAE changed')
        metrics,stats,_,_,_ = evaluate_scores(variant,pp,h1,base,cases)
        expected = next(m for m in summary['metrics'] if m['variant'] == variant)
        for key in ['correct_instances','uq_mismatches','grade_1_3_errors','score_gt_10']:
            if metrics[key] != expected[key]:
                raise ValueError(f'Replay mismatch {kind}: {key}')
        if abs(stats['mae']-expected['score_mae']) > .0001:
            raise ValueError('Replay MAE changed')
        results[kind] = {'max_val_prediction_delta':delta,'val_gt_crop_mae':mae,
                         'replay_score_mae':stats['mae'],'checkpoint_sha256':sha256(output/f'{kind}_best.pt')}
    check_inputs(inputs)
    save_json(target,{'status':'PASS','scope':'CPU checkpoint/cache replay, not independent retraining',
                     'runs':results,'inputs_unchanged':True,'train_only_standardization_verified':True,
                     'test_accessed':False})
    print('S3 checkpoint verification PASS',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--seed',type=int,default=20261007)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--epochs',type=int,default=200)
    parser.add_argument('--patience',type=int,default=40)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--learning-rate',type=float,default=.001)
    parser.add_argument('--ordinal-weight',type=float,default=.05)
    parser.add_argument('--temperature',type=float,default=.05)
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
    if min(args.epochs,args.patience,args.batch_size,args.timeout_seconds) < 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError('Invalid training budget')
    if not np.isfinite(args.ordinal_weight) or args.ordinal_weight <= 0:
        raise ValueError('Positive ordinal weight required for candidate')
    ordinal_logits(torch.zeros(1),args.temperature)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    source = json.loads((SOURCE/'summary.json').read_text(encoding='utf-8'))
    check_inputs(source['inputs'])
    paths = {'s2_summary':SOURCE/'summary.json','geometry_cache':SOURCE/'geometry_cache.pt',
             'train_geometry_csv':SOURCE/'train_geometry.csv','s2_metrics':SOURCE/'replay_metrics.csv',
             's2_score_metrics':SOURCE/'score_metrics.csv','s2_outcomes':SOURCE/'variant_instance_outcomes.csv',
             's2_prediction_scores':SOURCE/'geometry_val_gt_predictions.csv',
             'reference_outcomes':Path(source['inputs']['s1_summary']['path']).parent/'variant_instance_outcomes.csv',
             'geometry_loader_script':Path(__file__).with_name('train_r1_score_geometry_only.py'),
             'script':Path(__file__)}
    inputs = {k:{'path':str(p),'sha256':sha256(p)} for k,p in paths.items()}
    inputs.update({f'source_{k}':v for k,v in source['inputs'].items()})
    data = load_all(inputs)
    output.mkdir(parents=True)
    save_json(output/'config.json',{'date':'2026-10-08','command':sys.argv,
              'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'inputs':inputs,
              'thresholds':[LOWER,UPPER],'auxiliary_targets':'reference score >= threshold; never final grade',
              'loss':'SmoothL1(score/100,beta=.05) + weight*mean BCEWithLogits((pred-threshold/100)/temperature)',
              'selection':'minimum validation GT-crop score MAE, same as S2',
              'label_version':'v3_valfix_20261006_first_7_only','test_accessed':False,
              'seed_reason':'Match S2 initialization and minibatch order','precision':'FP32','augmentation':'none'})
    started = time.monotonic()
    try:
        cache,geometry,train,val,h1,base,cases = data
        runs,metrics,statistics,frames = {},[],[],[]
        for kind,weight in [('control',0.),('ordinal',args.ordinal_weight)]:
            scores,record = fit(kind,weight,cache,geometry,train,val,args,output,inputs,started+args.timeout_seconds)
            runs[kind] = record
            variant = f'S3_{kind}'
            row,stats,outcomes,classes,points = evaluate_scores(variant,scores,h1,base,cases)
            metrics.append(row)
            statistics.append({'variant':variant,**stats})
            frames.append(outcomes)
            classes.to_csv(output/f'{kind}_class_metrics.csv',index=False)
            points.to_csv(output/f'{kind}_matched_score_predictions.csv',index=False)
        all_outcomes = pd.concat([pd.read_csv(paths['reference_outcomes']),pd.read_csv(paths['s2_outcomes']),*frames],ignore_index=True)
        counts,tables = {},[]
        for ref in ['S0_cached_learned','S1_mlp','H1_rule_score','S2_geometry','S3_control']:
            count,table = paired_change(all_outcomes,ref,'S3_ordinal')
            if len(table) != 518:
                raise ValueError('Paired evaluation population changed')
            counts[ref] = count
            tables.append(table)
        pd.concat(tables,ignore_index=True).to_csv(output/'paired_changes.csv',index=False)
        pd.concat(frames,ignore_index=True).to_csv(output/'variant_instance_outcomes.csv',index=False)
        pd.concat([pd.read_csv(paths['s2_metrics']),pd.DataFrame(metrics)],ignore_index=True).to_csv(output/'replay_metrics.csv',index=False)
        pd.concat([pd.read_csv(paths['s2_score_metrics']),pd.DataFrame(statistics)],ignore_index=True).to_csv(output/'score_metrics.csv',index=False)
        previous = pd.read_csv(paths['s2_prediction_scores'])
        current = pd.read_csv(output/'control_val_gt_predictions.csv')
        if previous.instance_id.tolist() != current.instance_id.tolist():
            raise ValueError('Control validation alignment changed')
        control_delta = float(np.max(np.abs(previous.score_prediction-current.score_prediction)))
        check_inputs(inputs)
        result = {'status':'COMPLETED','origin_skill':'experiment-agent','mode':'run','verification_status':'ANALYZED',
                  'date':'2026-10-08','runs':runs,'metrics':metrics,'score_metrics':statistics,'paired_changes':counts,
                  'control_max_val_prediction_delta_vs_S2':control_delta,'inputs':inputs,'elapsed_seconds':time.monotonic()-started,
                  'labels_changed':False,'thresholds_changed':False,'test_accessed':False,'gates_reused':True,
                  'interpretation':'Single-seed development validation, not independent generalization evidence'}
        save_json(output/'summary.json',result)
        print(pd.DataFrame(metrics)[['variant','score_mae','correct_instances','macro_f1','score_gt_10']].to_string(index=False),flush=True)
        print(f'S3 COMPLETED; control max validation delta vs S2 = {control_delta:.8f}',flush=True)
    except Exception as exc:
        save_json(output/'failure.json',{'status':'FAILED','error':repr(exc),'inputs':inputs})
        raise


if __name__ == '__main__':
    main()
