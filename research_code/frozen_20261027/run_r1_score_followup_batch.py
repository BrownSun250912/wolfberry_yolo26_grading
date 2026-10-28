"""Five bounded follow-up groups: paired seeds, ECDF heads, fixed ensembles."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from train_r1_score_ordinal_s3 import load_all, fit
from train_r1_score_geometry_s2 import GeometryScoreHead, fit_standardizer
from train_r1_score_geometry_only import check_inputs
from train_r1_score_specialist import ROOT, available_scores, evaluate_scores, save_json
from evaluate_r1a_frozen_predictions import sha256, paired_change

S3 = ROOT/'models/experiments/r1_score_ordinal_s3_20261008'
OUTPUT = ROOT/'models/experiments/r1_score_followup_batch_20261008'
SEEDS = (20261007,20261008,20261009)


def percentile_values(values,references):
    values = np.asarray(values,dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all() or len(references) != values.shape[1]:
        raise ValueError('Invalid percentile inputs')
    columns = []
    for index,reference in enumerate(references):
        ref = np.asarray(reference,dtype=np.float32)
        if ref.ndim != 1 or not len(ref) or not np.isfinite(ref).all() or (np.diff(ref)<0).any():
            raise ValueError('Invalid training reference')
        left = np.searchsorted(ref,values[:,index],side='left')
        right = np.searchsorted(ref,values[:,index],side='right')
        columns.append((left+right)/(2.*len(ref)))
    return np.column_stack(columns).astype(np.float32)


def fit_percentile_transform(train,eligible):
    train = np.asarray(train,dtype=np.float32)
    eligible = np.asarray(eligible,bool)
    if train.ndim != 2 or eligible.shape != (len(train),) or not eligible.any():
        raise ValueError('Invalid training eligibility')
    references = [np.sort(train[eligible,index]).copy() for index in range(train.shape[1])]
    percentiles = percentile_values(train,references)
    mean,std = fit_standardizer(percentiles,eligible)
    return {'references':references,'mean':mean,'std':std,'fit_rows':int(eligible.sum())}


def apply_percentile_transform(values,state):
    p = percentile_values(values,state['references'])
    return torch.from_numpy(((p-state['mean'])/state['std']).astype(np.float32))


def aligned_average(tables,id_column,expected_ids):
    arrays = []
    for table in tables:
        if not table[id_column].is_unique or set(table[id_column]) != set(expected_ids):
            raise ValueError('Ensemble population mismatch')
        values = table.set_index(id_column).loc[expected_ids,'score_prediction'].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite ensemble score')
        arrays.append(values)
    if len(arrays) != 3:
        raise ValueError('This experiment requires exactly three equally weighted seeds')
    return np.mean(np.stack(arrays),axis=0)


def run_child(command,log):
    with log.open('w',encoding='utf-8') as handle:
        result = subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,timeout=600,check=False)
    if result.returncode:
        raise RuntimeError(f'Child exit={result.returncode}; retained log {log}')
    print('\n'.join(log.read_text(encoding='utf-8',errors='replace').splitlines()[-5:]),flush=True)


def verify_rank_checkpoint(kind,cache,geometry,val,h1,base,cases,output,expected):
    directory = output/kind
    checkpoint = torch.load(directory/f'{kind}_best.pt',map_location='cpu',weights_only=False)
    head = GeometryScoreHead(True).eval()
    head.load_state_dict(checkpoint['head_state'])
    with torch.no_grad():
        vp = head(cache['val'],geometry['val']).numpy()*100
        pp = head(cache['predictions'],geometry['predictions']).numpy()*100
    old = pd.read_csv(directory/f'{kind}_val_gt_predictions.csv')
    if old.instance_id.tolist() != val.instance_id.tolist():
        raise ValueError('Rank validation IDs changed')
    delta = float(np.max(np.abs(vp-old.score_prediction.to_numpy())))
    if delta > .001:
        raise ValueError('Rank checkpoint prediction mismatch')
    metrics,stats,_,_,_ = evaluate_scores(kind,pp,h1,base,cases)
    for field in ['correct_instances','uq_mismatches','grade_1_3_errors','score_gt_10']:
        if metrics[field] != expected[field]:
            raise ValueError(f'Rank replay mismatch: {field}')
    if abs(stats['mae']-expected['score_mae']) > .0001:
        raise ValueError('Rank replay MAE mismatch')
    record = {'status':'PASS','scope':'CPU saved-checkpoint replay, not independent retraining',
              'max_val_prediction_delta':delta,'checkpoint_sha256':sha256(directory/f'{kind}_best.pt')}
    save_json(directory/'checkpoint_verification.json',record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing overwrite: {output}')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    source = json.loads((S3/'summary.json').read_text(encoding='utf-8'))
    cache,geometry,train,val,h1,base,cases = load_all(source['inputs'])
    paths = {'source_s3_summary':S3/'summary.json','batch_script':Path(__file__),
             'diagnosis':ROOT/'outputs/reports/r1_score_error_ceiling_20261008/summary.json'}
    inputs = {k:{'path':str(p),'sha256':sha256(p)} for k,p in paths.items()}
    inputs.update({f'prior_{k}':v for k,v in source['inputs'].items()})
    output.mkdir(parents=True)
    plan = {'date':'2026-10-08','origin_skill':'experiment-agent','mode':'run','inputs':inputs,
            'groups':['paired S2/S3 seed 20261008','paired S2/S3 seed 20261009',
                      'train-ECDF geometry + frozen R1 features, seed 20261007',
                      'train-ECDF geometry only, seed 20261007',
                      'fixed equally weighted three-seed S2 and S3 score ensembles'],
            'seeds':SEEDS,'selection':'minimum validation GT score MAE, same frozen populations',
            'budget':'200 epochs and patience 40 per head; 600 seconds per child; 1800 seconds rank training',
            'no_validation_hyperparameter_sweep':True,'labels_changed':False,'thresholds_changed':False,
            'test_accessed':False,'deployment_changed':False,
            'stop_after':'Five groups regardless of outcome; no post-result candidate tuning'}
    save_json(output/'plan.json',plan)
    started = time.monotonic()
    try:
        sources = [S3]
        derived_inputs = {}
        for seed in SEEDS[1:]:
            directory = output/f'seed_{seed}'
            print(f'GROUP paired seed {seed}: train control and ordinal',flush=True)
            command = [sys.executable,str(ROOT/'scripts/train_r1_score_ordinal_s3.py'),
                       '--output',str(directory),'--seed',str(seed),'--timeout-seconds','600']
            run_child(command,output/f'seed_{seed}_train.log')
            run_child([sys.executable,str(ROOT/'scripts/train_r1_score_ordinal_s3.py'),
                       '--output',str(directory),'--verify'],output/f'seed_{seed}_verify.log')
            sources.append(directory)
        tm,_ = available_scores(train)
        transform = fit_percentile_transform(geometry['train'].numpy(),tm)
        transform['input_geometry_mean'] = geometry['mean']
        transform['input_geometry_std'] = geometry['std']
        transform['scope'] = 'ECDF fitted to 3905 eligible training rows only; no score targets as features'
        torch.save(transform,output/'percentile_transform.pt')
        ranked_geometry = {split:apply_percentile_transform(geometry[split].numpy(),transform)
                           for split in ['train','val','predictions']}
        ranked_geometry.update({'mean':transform['mean'],'std':transform['std']})
        rank_inputs = dict(inputs)
        rank_inputs['percentile_transform'] = {'path':str(output/'percentile_transform.pt'),
                                              'sha256':sha256(output/'percentile_transform.pt')}
        metrics,score_metrics,frames,records,verification = [],[],[],{},{}
        training_args = SimpleNamespace(seed=SEEDS[0],device='cuda:0',learning_rate=.001,
                    temperature=.05,epochs=200,patience=40,batch_size=128)
        for kind in ['rank_fusion','rank_geometry_only']:
            print(f'GROUP {kind}: train-only percentile input, original score targets',flush=True)
            directory = output/kind
            directory.mkdir()
            used_cache = cache if kind == 'rank_fusion' else {
                **cache,**{split:torch.zeros_like(cache[split]) for split in ['train','val','predictions']}}
            pp,record = fit(kind,0.,used_cache,ranked_geometry,train,val,training_args,directory,rank_inputs,time.monotonic()+1800)
            checkpoint_path = directory/f'{kind}_best.pt'
            checkpoint = torch.load(checkpoint_path,map_location='cpu',weights_only=False)
            checkpoint['requires_frozen_R1_features'] = kind == 'rank_fusion'
            checkpoint['geometry_transform'] = str(output/'percentile_transform.pt')
            checkpoint['zero_R1_feature_input'] = kind == 'rank_geometry_only'
            torch.save(checkpoint,checkpoint_path)
            row,stats,outcomes,classes,points = evaluate_scores(kind,pp,h1,base,cases)
            metrics.append(row)
            score_metrics.append({'variant':kind,**stats})
            frames.append(outcomes)
            record['zero_R1_feature_input'] = kind == 'rank_geometry_only'
            records[kind] = record
            classes.to_csv(directory/'class_metrics.csv',index=False)
            points.to_csv(directory/'matched_score_predictions.csv',index=False)
            outcomes.to_csv(directory/'variant_instance_outcomes.csv',index=False)
            verification[kind] = verify_rank_checkpoint(kind,used_cache,ranked_geometry,val,h1,base,cases,output,row)
            print(f'{kind}: MAE={stats["mae"]:.4f}, correct={row["correct_instances"]}/500',flush=True)
        repeat_rows,repeat_score_rows,repeat_frames = [],[],[]
        for seed,directory in zip(SEEDS,sources):
            run_summary = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
            if seed != SEEDS[0]:
                for row in run_summary['metrics']:
                    repeat_rows.append({**row,'variant':f'repeat_{seed}_{row["variant"]}'})
                for row in run_summary['score_metrics']:
                    repeat_score_rows.append({**row,'variant':f'repeat_{seed}_{row["variant"]}'})
                outcomes = pd.read_csv(directory/'variant_instance_outcomes.csv')
                outcomes['variant'] = outcomes.variant.map(lambda v:f'repeat_{seed}_{v}')
                repeat_frames.append(outcomes)
            for filename in ['summary.json','control_prediction_scores.csv','ordinal_prediction_scores.csv',
                             'control_val_gt_predictions.csv','ordinal_val_gt_predictions.csv','checkpoint_verification.json']:
                path = directory/filename
                derived_inputs[f'{seed}_{filename}'] = {'path':str(path),'sha256':sha256(path)}
        print('GROUP ensemble: fixed equal-weight average, no fitted weights',flush=True)
        for kind in ['control','ordinal']:
            variant = f'ensemble_{kind}'
            pp = aligned_average([pd.read_csv(p/f'{kind}_prediction_scores.csv') for p in sources],
                                 'prediction_id',cache['prediction_ids'])
            vp = aligned_average([pd.read_csv(p/f'{kind}_val_gt_predictions.csv') for p in sources],
                                 'instance_id',cache['val_ids'])
            row,stats,outcomes,classes,points = evaluate_scores(variant,pp,h1,base,cases)
            metrics.append(row)
            score_metrics.append({'variant':variant,**stats})
            frames.append(outcomes)
            vm,vy = available_scores(val)
            records[variant] = {'seeds':SEEDS,'weights':[1/3]*3,'no_fitted_weights':True,
                               'val_gt_crop_mae':float(np.mean(np.abs(vp[vm]-vy*100)))}
            pd.DataFrame({'prediction_id':cache['prediction_ids'],'score_prediction':pp}).to_csv(output/f'{variant}_prediction_scores.csv',index=False)
            pd.DataFrame({'instance_id':cache['val_ids'],'score_prediction':vp}).to_csv(output/f'{variant}_val_gt_predictions.csv',index=False)
            classes.to_csv(output/f'{variant}_class_metrics.csv',index=False)
            points.to_csv(output/f'{variant}_matched_score_predictions.csv',index=False)
            print(f'{variant}: MAE={stats["mae"]:.4f}, correct={row["correct_instances"]}/500',flush=True)
        seed_rows = []
        for seed,directory in zip(SEEDS,sources):
            s = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
            for row in s['metrics']:
                seed_rows.append({'seed':seed,**row})
        seed_table = pd.DataFrame(seed_rows)
        seed_table.to_csv(output/'paired_seed_metrics.csv',index=False)
        aggregates = []
        for kind,group in seed_table.groupby('variant'):
            aggregate = {'variant':kind,'n_seeds':len(group)}
            for field in ['score_mae','correct_instances','macro_f1','score_gt_10']:
                aggregate[f'{field}_mean'] = float(group[field].mean())
                aggregate[f'{field}_sample_std'] = float(group[field].std(ddof=1))
            aggregates.append(aggregate)
        pd.DataFrame(aggregates).to_csv(output/'seed_aggregate_metrics.csv',index=False)
        prior_outcomes = [pd.read_csv(source['inputs'][key]['path']) for key in ['reference_outcomes','s2_outcomes']]
        all_outcomes = pd.concat([*prior_outcomes,pd.read_csv(S3/'variant_instance_outcomes.csv'),*repeat_frames,*frames],ignore_index=True)
        paired,tables = {},[]
        for row in [*repeat_rows,*metrics]:
            variant = row['variant']
            paired[variant] = {}
            for ref in ['S2_geometry','S3_ordinal','S1_mlp','H1_rule_score']:
                counts,table = paired_change(all_outcomes,ref,variant)
                if len(table) != 518:
                    raise ValueError('Paired population changed')
                paired[variant][ref] = counts
                tables.append(table)
        pd.concat(tables,ignore_index=True).to_csv(output/'paired_changes.csv',index=False)
        pd.concat([*repeat_frames,*frames],ignore_index=True).to_csv(output/'variant_instance_outcomes.csv',index=False)
        previous = pd.read_csv(source['inputs']['s2_metrics']['path'])
        pd.concat([previous,pd.DataFrame(source['metrics']),pd.DataFrame(repeat_rows+metrics)],ignore_index=True).to_csv(output/'replay_metrics.csv',index=False)
        pd.concat([pd.read_csv(source['inputs']['s2_score_metrics']['path']),pd.DataFrame(source['score_metrics']),
                   pd.DataFrame(repeat_score_rows+score_metrics)],ignore_index=True).to_csv(output/'score_metrics.csv',index=False)
        check_inputs(inputs)
        check_inputs(rank_inputs)
        check_inputs(derived_inputs)
        save_json(output/'summary.json',{'status':'COMPLETED','origin_skill':'experiment-agent','mode':'run',
                  'verification_status':'ANALYZED','date':'2026-10-08','groups_completed':5,
                  'metrics':repeat_rows+metrics,'runs':records,'seed_aggregates':aggregates,'paired_changes':paired,
                  'checkpoint_verification':verification,'inputs':inputs,'rank_inputs':rank_inputs,
                  'derived_inputs':derived_inputs,'elapsed_seconds':time.monotonic()-started,
                  'labels_changed':False,'thresholds_changed':False,'test_accessed':False,'deployment_changed':False,
                  'interpretation':'Development validation after multiple candidates; not independent test or significance evidence'})
        print(pd.DataFrame(repeat_rows+metrics)[['variant','score_mae','correct_instances','score_gt_10']].to_string(index=False),flush=True)
        print(f'COMPLETED five groups in {time.monotonic()-started:.1f}s',flush=True)
    except Exception as exc:
        save_json(output/'failure.json',{'status':'FAILED','error':repr(exc),'inputs':inputs})
        raise


if __name__ == '__main__':
    main()
