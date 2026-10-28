"""S1/S1b: train independent score heads on immutable R1 features, val only.

No detector, labels, thresholds or original checkpoint are changed. Cached
features use deterministic unaugmented crops (a documented strategy change
relative to original R1). Both heads share exactly the same feature cache.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from train_r1a_hierarchical_grader import (
    ROOT, R1Dataset, R1HierarchicalGrader, scan_geometry, set_determinism,
)
from evaluate_r1a_frozen_predictions import (
    DEFAULT_R0, prediction_input, replace_classes, sha256, variant_summary,
)
from diagnose_r1_score_replay import compose_variants, score_stats

CHECKPOINT = ROOT / 'models/experiments/r1a_hierarchical_grader_e200_resume_epoch150_seed20260929/best.pt'
TARGETS = ROOT / 'data/grading_four_class_v3_valfix_20261006/audit/r1_multitask_targets.csv'
H1 = ROOT / 'outputs/reports/h1_frozen_replay_20261006/h1_predictions.csv'
OUTCOMES = ROOT / 'outputs/reports/h1_valfix_replay_20261006/variant_instance_outcomes.csv'
CASES = ROOT / 'outputs/reports/r1_score_diagnostic_20261007/score_cases_ranked.csv'
DEFAULT_OUTPUT = ROOT / 'models/experiments/r1_score_specialist_20261007'


def freeze_backbone(model):
    model.eval()
    model.requires_grad_(False)
    return model


def fused_features(model, image, geometry):
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('R1 must remain entirely frozen and in eval mode')
    with torch.no_grad():
        return model.fusion(torch.cat([model.visual(image), model.geometry(geometry)], dim=1))


def make_head(kind, original):
    if kind == 'linear':
        head = copy.deepcopy(original)
    elif kind == 'mlp':
        head = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 32),
                             nn.SiLU(), nn.Linear(32, 1))
    else:
        raise ValueError(kind)
    return head.requires_grad_(True)


def score_prediction(head, features):
    return torch.sigmoid(head(features).squeeze(-1))


def available_scores(frame):
    eligible = frame.score_target_available.eq(1).to_numpy()
    values = frame.score_target.to_numpy(dtype=np.float32)
    if not eligible.any() or not np.isfinite(values[eligible]).all():
        raise ValueError('Missing/nonfinite available targets')
    if ((values[eligible] < 0) | (values[eligible] > 100)).any():
        raise ValueError('Score target outside [0,100]')
    if frame.loc[eligible, 'defect_target'].ne(0).any():
        raise ValueError('Unqualified instance has available score')
    return eligible, values[eligible] / 100.0


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def extract_gt(model, frame, checkpoint, device, batch_size, name):
    for row in frame.itertuples(index=False):
        for field in ('crop', 'mask'):
            if sha256(ROOT / getattr(row, field + '_path')) != getattr(row, field + '_sha256'):
                raise ValueError(f'Changed {field}: {row.instance_id}')
    geometry = scan_geometry(frame)
    dataset = R1Dataset(frame, geometry, np.asarray(checkpoint['geometry_mean'], np.float32),
                        np.asarray(checkpoint['geometry_std'], np.float32),
                        checkpoint['config']['image_size'], False, 20261007)
    features = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for index, batch in enumerate(loader):
        features.append(fused_features(model, batch['image'].to(device), batch['geometry'].to(device)).cpu())
        if index % 10 == 0 or index + 1 == len(loader):
            print(f'cache {name}: {min((index+1)*batch_size, len(frame))}/{len(frame)}', flush=True)
    return torch.cat(features)


def extract_predictions(model, fixed, checkpoint, device, batch_size):
    features = []
    mean = np.asarray(checkpoint['geometry_mean'], np.float32)
    std = np.asarray(checkpoint['geometry_std'], np.float32)
    for start in range(0, len(fixed), batch_size):
        items = [prediction_input(row, DEFAULT_R0, checkpoint['config']['image_size'])
                 for row in fixed.iloc[start:start+batch_size].itertuples(index=False)]
        image = torch.from_numpy(np.stack([x[0] for x in items])).to(device)
        geometry = torch.from_numpy((np.stack([x[1] for x in items])-mean)/std).to(device)
        features.append(fused_features(model, image, geometry).cpu())
        print(f'cache frozen predictions: {min(start+batch_size, len(fixed))}/{len(fixed)}', flush=True)
    return torch.cat(features)


def fit_head(kind, original, train_features, train_frame, val_features, val_frame, args, output, started):
    set_determinism(args.seed)
    head = make_head(kind, original).to(args.device)
    train_mask, train_y = available_scores(train_frame)
    val_mask, val_y = available_scores(val_frame)
    x = train_features[train_mask].to(args.device)
    y = torch.from_numpy(train_y).to(args.device)
    vx = val_features[val_mask].to(args.device)
    vy = torch.from_numpy(val_y).to(args.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss(beta=.05)
    history = []
    best_mae, best_epoch, best_state = float('inf'), 0, None
    generator = torch.Generator().manual_seed(args.seed)
    for epoch in range(args.epochs+1):
        if time.monotonic()-started > args.timeout_seconds:
            raise TimeoutError('Experiment hard timeout reached')
        head.train()
        if epoch:
            indices = torch.randperm(len(x), generator=generator).to(x.device)
            for batch in indices.split(args.head_batch_size):
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(score_prediction(head, x[batch]), y[batch])
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(), 5.0)
                optimizer.step()
        head.eval()
        with torch.no_grad():
            train_mae = float((score_prediction(head, x)-y).abs().mean().cpu())*100
            val_mae = float((score_prediction(head, vx)-vy).abs().mean().cpu())*100
        history.append({'epoch': epoch, 'train_mae': train_mae, 'val_gt_crop_mae': val_mae})
        if val_mae < best_mae-1e-6:
            best_mae, best_epoch = val_mae, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        pd.DataFrame(history).to_csv(output / f'{kind}_history.csv', index=False)
        if epoch % 10 == 0:
            print(f'{kind} epoch={epoch} train_MAE={train_mae:.4f} val_MAE={val_mae:.4f} best={best_mae:.4f}@{best_epoch}', flush=True)
        if epoch-best_epoch >= args.patience:
            break
    head.load_state_dict(best_state)
    torch.save({'head_kind': kind, 'head_state': best_state, 'epoch': best_epoch,
                'val_gt_crop_mae': best_mae, 'backbone_checkpoint': str(CHECKPOINT),
                'backbone_sha256': sha256(CHECKPOINT), 'config': vars(args),
                'requires_frozen_R1_features': True}, output / f'{kind}_best.pt')
    return head, {'best_epoch': best_epoch, 'epochs_completed': epoch,
                  'best_val_gt_crop_mae': best_mae, 'initial_val_gt_crop_mae': history[0]['val_gt_crop_mae'],
                  'train_eligible_n': int(train_mask.sum()), 'val_eligible_n': int(val_mask.sum()),
                  'train_mae_at_best': history[best_epoch]['train_mae']}


def evaluate_scores(name, scores, h1, base, cases):
    pred = h1.copy()
    pred['score_prediction'] = scores
    classes = compose_variants(pred)['fixed_gate_learned_score']
    replacement = pd.DataFrame({'prediction_id': pred.prediction_id, 'r1a_class_id': classes})
    outcomes = replace_classes(base, replacement)
    outcomes['variant'] = name
    metrics, per_class = variant_summary(outcomes, name)
    if tuple(metrics[k] for k in ('gt_instances', 'localized_instances', 'predictions', 'false_positive_predictions')) != (518,500,549,49):
        raise ValueError('Frozen replay population changed')
    matched = cases[['prediction_id', 'instance_id', 'score_target']].merge(
        pred[['prediction_id', 'score_prediction']], on='prediction_id', validate='one_to_one')
    if len(matched) != 448:
        raise ValueError('Score population changed')
    stats = score_stats(matched.score_target, matched.score_prediction)
    matched['absolute_error'] = (matched.score_prediction-matched.score_target).abs()
    gt = outcomes.loc[outcomes.record_type.eq('ground_truth') & outcomes.localization_matched.eq(True)]
    a, b = gt.gt_class_id.to_numpy(int), gt.pred_class_id.to_numpy(int)
    metrics.update({'uq_mismatches': int(((a == 0) != (b == 0)).sum()),
                    'grade_1_3_errors': int((((a == 1)&(b == 3))|((a == 3)&(b == 1))).sum()),
                    'score_mae': stats['mae'], 'score_gt_10': stats['absolute_error_gt_10']})
    return metrics, stats, outcomes, per_class, matched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--seed', type=int, default=20261007)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--head-batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--timeout-seconds', type=int, default=1800)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or min(args.batch_size, args.head_batch_size) < 1:
        raise ValueError('Invalid training budget')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(4)
    set_determinism(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.monotonic()
    paths = {'checkpoint': CHECKPOINT, 'targets': TARGETS, 'h1': H1, 'outcomes': OUTCOMES,
             'score_cases': CASES, 'fixed': DEFAULT_R0/'fixed_predictions.csv',
             'script': Path(__file__)}
    inputs = {k: {'path': str(p), 'sha256': sha256(p)} for k,p in paths.items()}
    data = pd.read_csv(TARGETS)
    if len(data) != 4795 or set(data.split) != {'train','val'} or not data.instance_id.is_unique:
        raise ValueError('Target split or identity changed')
    train = data.loc[data.split.eq('train')].reset_index(drop=True)
    val = data.loc[data.split.eq('val')].reset_index(drop=True)
    if (len(train),len(val)) != (4277,518) or set(train.image_id)&set(val.image_id):
        raise ValueError('Frozen split changed or image leakage')
    available_scores(train)
    available_scores(val)
    fixed = pd.read_csv(paths['fixed'])
    h1 = pd.read_csv(H1)
    if len(fixed) != 549 or not fixed.prediction_id.is_unique or not h1.prediction_id.is_unique:
        raise ValueError('Frozen prediction identities changed')
    if not fixed.image_path.map(lambda p: Path(p).parent.name == 'val').all():
        raise ValueError('Only validation replay allowed')
    if set(fixed.prediction_id) != set(h1.prediction_id):
        raise ValueError('Frozen prediction IDs differ')
    h1 = h1.set_index('prediction_id').loc[fixed.prediction_id].reset_index()
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    model = freeze_backbone(R1HierarchicalGrader())
    model.load_state_dict(checkpoint['model_state'], strict=True)
    original_state = {k:v.clone() for k,v in model.state_dict().items()}
    model.to(device)
    output.mkdir(parents=True)
    save_json(output/'config.json', {'args': {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'created_utc': datetime.now(timezone.utc).isoformat(), 'command': sys.argv,
        'inputs': inputs, 'selection': 'lowest val GT-crop score MAE; earlier epoch on ties',
        'label_version': 'v3_valfix_20261006_first_7_only', 'augmentation': 'none; fixed cached features',
        'precision': 'FP32 no AMP', 'test_accessed': False, 'head_kinds': ['linear','mlp'],
        'torch': torch.__version__, 'device': torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu'})
    print(f'Start S1/S1b on {device}; output={output}', flush=True)
    train_features = extract_gt(model, train, checkpoint, device, args.batch_size, 'train')
    val_features = extract_gt(model, val, checkpoint, device, args.batch_size, 'val')
    pred_features = extract_predictions(model, fixed, checkpoint, device, args.batch_size)
    with torch.no_grad():
        baseline_scores = score_prediction(model.score_head, pred_features.to(device)).cpu().numpy()*100
        stem = torch.sigmoid(model.stem_head(pred_features.to(device)).squeeze(1)).cpu().numpy()
    stem_decisions_equal = np.array_equal(stem >= .5, h1.stem_missing_probability.to_numpy() >= .5)
    score_delta = float(np.max(np.abs(baseline_scores-h1.score_prediction.to_numpy())))
    if not stem_decisions_equal or score_delta > .1:
        raise ValueError(f'Frozen baseline reproduction failed: stem={stem_decisions_equal}, score_delta={score_delta}')
    torch.save({'train': train_features, 'val': val_features, 'predictions': pred_features,
                'train_ids': train.instance_id.tolist(), 'val_ids': val.instance_id.tolist(),
                'prediction_ids': fixed.prediction_id.tolist(), 'inputs': inputs}, output/'feature_cache.pt')
    original_outcomes = pd.read_csv(OUTCOMES)
    base = original_outcomes.loc[original_outcomes.variant.eq('baseline')].copy()
    cases = pd.read_csv(CASES)
    rows, score_rows, runs, outcome_frames, matched_frames = [], [], {}, [], []
    candidates = [('S0_cached_learned', h1.score_prediction.to_numpy()),
                  ('S0_recomputed_fp32', baseline_scores)]
    for kind in ('linear','mlp'):
        head, run = fit_head(kind, model.score_head, train_features, train, val_features, val,
                             args, output, started)
        runs[kind] = run
        with torch.no_grad():
            prediction = score_prediction(head, pred_features.to(device)).cpu().numpy()*100
            val_prediction = score_prediction(head, val_features.to(device)).cpu().numpy()*100
        table = val[['instance_id','score_target','score_target_available']].copy()
        table['score_prediction'] = val_prediction
        table.to_csv(output/f'{kind}_val_gt_predictions.csv', index=False, encoding='utf-8-sig')
        candidates.append((f'S1_{kind}',prediction))
    # Rule score is evaluated through the identical fixed gates for comparison.
    candidates.append(('H1_rule_score', h1.rule_score.to_numpy()))
    for name, prediction in candidates:
        metrics, stats, outcomes, per_class, matched = evaluate_scores(name,prediction,h1,base,cases)
        rows.append(metrics)
        score_rows.append({'variant':name, **stats})
        outcome_frames.append(outcomes)
        matched_frames.append(matched.assign(variant=name))
        per_class.to_csv(output/f'{name}_class_metrics.csv',index=False,encoding='utf-8-sig')
    unchanged = all(torch.equal(v.cpu(),original_state[k]) for k,v in model.state_dict().items())
    if not unchanged or any(p.requires_grad for p in model.parameters()) or model.training:
        raise ValueError('Frozen backbone/buffers changed')
    if not all(sha256(Path(v['path'])) == v['sha256'] for v in inputs.values()):
        raise ValueError('An input changed during the experiment')
    pd.DataFrame(rows).to_csv(output/'replay_metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(score_rows).to_csv(output/'score_metrics.csv',index=False,encoding='utf-8-sig')
    pd.concat(outcome_frames).to_csv(output/'variant_instance_outcomes.csv',index=False,encoding='utf-8-sig')
    pd.concat(matched_frames).to_csv(output/'matched_score_predictions.csv',index=False,encoding='utf-8-sig')
    result = {'status':'COMPLETED','origin_skill':'experiment-agent','mode':'run',
              'verification_status':'ANALYZED','runs':runs,'metrics':rows,'score_metrics':score_rows,
              'elapsed_seconds':time.monotonic()-started,'seed':args.seed,'inputs':inputs,
              'backbone_and_buffers_unchanged':unchanged,'stem_decisions_unchanged':stem_decisions_equal,
              'baseline_max_score_delta_fp32':score_delta,'test_accessed':False,'labels_changed':False,
              'thresholds_changed':False,'precision':'FP32','augmentation':'none; immutable feature cache',
              'interpretation':'Single-seed development validation; not independent test or significance evidence.'}
    save_json(output/'summary.json',result)
    print(pd.DataFrame(rows)[['variant','correct_instances','macro_f1','score_mae']].to_string(index=False),flush=True)
    print(f'COMPLETED in {result["elapsed_seconds"]:.1f}s: {output}',flush=True)


if __name__ == '__main__':
    main()
