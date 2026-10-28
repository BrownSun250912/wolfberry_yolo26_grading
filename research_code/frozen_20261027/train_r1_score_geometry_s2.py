"""S2: add mask geometry to frozen R1 features, with same-capacity zero control.

Uses the immutable S1 feature cache. No reference score/grade is an input.
Mask geometry is measured before letterboxing, without new mask refinement.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn

from train_r1_score_specialist import (
    ROOT, DEFAULT_OUTPUT as S1_OUTPUT, available_scores, evaluate_scores, save_json,
)
from train_r1a_hierarchical_grader import set_determinism
from evaluate_r1a_frozen_predictions import DEFAULT_R0, load_fixed_mask, sha256, paired_change
from extract_features import fit_axes, main_contour

DEFAULT_OUTPUT = ROOT/'models/experiments/r1_score_geometry_s2_20261007'
GEOMETRY_NAMES = ('log_mask_area','log_major_axis','log_minor_axis','axis_ratio',
                  'circularity','solidity','eccentricity','log_perimeter')


def mask_geometry(mask):
    """Pixel area; main external contour for axes and shape, as in extract_features.

    This uses the current input mask, NOT historical border-refined features.
    Translation/canvas size do not change any feature. Degenerate masks fail.
    """
    # Importing extract_features imports Ultralytics, whose imread wrapper
    # returns HxWx1 for grayscale in this environment. Never squeeze RGB.
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]
    if mask.ndim != 2:
        raise ValueError('Expected 2D mask')
    binary = np.ascontiguousarray((mask > 0).astype(np.uint8)*255)
    area = int(np.count_nonzero(binary))
    contour = main_contour(binary)
    if not area or contour is None:
        raise ValueError('Empty mask')
    perimeter = float(cv2.arcLength(contour,True))
    contour_area = float(cv2.contourArea(contour))
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    if min(perimeter,contour_area,hull_area) <= 0:
        raise ValueError('Degenerate contour')
    axes = fit_axes(contour)
    major, minor = float(axes['major_axis']),float(axes['minor_axis'])
    values = np.asarray([math.log1p(area),math.log1p(major),math.log1p(minor),
                         major/minor,4*math.pi*contour_area/(perimeter**2),
                         contour_area/hull_area,math.sqrt(max(0.,1-(minor/major)**2)),
                         math.log1p(perimeter)],dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite mask geometry')
    return values


def fit_standardizer(features, eligible):
    values = np.asarray(features,np.float32)[np.asarray(eligible,bool)]
    if values.ndim != 2 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError('Invalid training-only geometry statistics')
    mean,std = values.mean(axis=0),values.std(axis=0)
    std[std < 1e-6] = 1.
    return mean,std


class GeometryScoreHead(nn.Module):
    def __init__(self, use_geometry=True):
        super().__init__()
        self.use_geometry = use_geometry
        self.geometry_encoder = nn.Sequential(nn.Linear(8,16),nn.SiLU(),nn.Linear(16,16),nn.SiLU())
        self.score_mlp = nn.Sequential(nn.Linear(272,128),nn.SiLU(),nn.Linear(128,32),
                                       nn.SiLU(),nn.Linear(32,1))

    def forward(self, features, geometry):
        if not self.use_geometry:
            geometry = torch.zeros_like(geometry)
        encoded = self.geometry_encoder(geometry)
        return torch.sigmoid(self.score_mlp(torch.cat([features,encoded],dim=1)).squeeze(1))


def extract_geometry(frame, output, split, deadline):
    values = []
    for index,row in enumerate(frame.itertuples(index=False)):
        if time.monotonic() > deadline:
            raise TimeoutError('S2 feature extraction timeout')
        path = ROOT/row.mask_path
        if sha256(path) != row.mask_sha256:
            raise ValueError(f'Changed mask: {row.instance_id}')
        mask = cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(path)
        values.append(mask_geometry(mask))
        if (index+1)%1000 == 0 or index+1 == len(frame):
            print(f'geometry {split}: {index+1}/{len(frame)}',flush=True)
    result = np.stack(values)
    table = pd.DataFrame(result,columns=GEOMETRY_NAMES)
    table.insert(0,'instance_id',frame.instance_id.to_numpy())
    table.to_csv(output/f'{split}_geometry.csv',index=False,encoding='utf-8-sig')
    return result


def fit(kind, cache, geometry, train, val, args, output, inputs, deadline):
    set_determinism(args.seed)
    head = GeometryScoreHead(kind == 'geometry').to(args.device)
    train_mask,train_y = available_scores(train)
    val_mask,val_y = available_scores(val)
    tx = cache['train'][torch.from_numpy(train_mask.copy())].to(args.device)
    vx = cache['val'][torch.from_numpy(val_mask.copy())].to(args.device)
    tg = geometry['train'][torch.from_numpy(train_mask.copy())].to(args.device)
    vg = geometry['val'][torch.from_numpy(val_mask.copy())].to(args.device)
    ty,vy = torch.from_numpy(train_y.copy()).to(args.device),torch.from_numpy(val_y.copy()).to(args.device)
    optimizer = torch.optim.AdamW(head.parameters(),lr=args.learning_rate,weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss(beta=.05)
    generator = torch.Generator().manual_seed(args.seed)
    history = []
    best,epoch_best,state = float('inf'),0,None
    for epoch in range(args.epochs+1):
        if time.monotonic() > deadline:
            raise TimeoutError('S2 training timeout')
        head.train()
        if epoch:
            order = torch.randperm(len(tx),generator=generator).to(args.device)
            for batch in order.split(args.batch_size):
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(head(tx[batch],tg[batch]),ty[batch])
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite loss')
                loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(),5.)
                optimizer.step()
        head.eval()
        with torch.no_grad():
            train_mae = float((head(tx,tg)-ty).abs().mean().cpu())*100
            val_mae = float((head(vx,vg)-vy).abs().mean().cpu())*100
        history.append({'epoch':epoch,'train_mae':train_mae,'val_gt_crop_mae':val_mae})
        if val_mae < best-1e-6:
            best,epoch_best = val_mae,epoch
            state = {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        pd.DataFrame(history).to_csv(output/f'{kind}_history.csv',index=False)
        if epoch%10 == 0:
            print(f'S2 {kind} epoch={epoch} train_MAE={train_mae:.4f} val_MAE={val_mae:.4f} best={best:.4f}@{epoch_best}',flush=True)
        if epoch-epoch_best >= args.patience:
            break
    head.load_state_dict(state)
    record = {'kind':kind,'best_epoch':epoch_best,'epochs_completed':epoch,
              'val_gt_crop_mae':best,'train_mae_at_best':history[epoch_best]['train_mae'],
              'parameters':sum(p.numel() for p in head.parameters())}
    torch.save({'head_state':state,'head_kind':kind,'geometry_names':GEOMETRY_NAMES,
                'mean':geometry['mean'].tolist(),'std':geometry['std'].tolist(),
                'record':record,'seed':args.seed,'inputs':inputs,
                'requires_frozen_R1_features':True},output/f'{kind}_best.pt')
    with torch.no_grad():
        prediction = head(cache['predictions'].to(args.device),geometry['predictions'].to(args.device)).cpu().numpy()*100
        val_prediction = head(cache['val'].to(args.device),geometry['val'].to(args.device)).cpu().numpy()*100
    table = val[['instance_id','score_target','score_target_available']].copy()
    table['score_prediction'] = val_prediction
    table.to_csv(output/f'{kind}_val_gt_predictions.csv',index=False,encoding='utf-8-sig')
    return prediction,record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--seed',type=int,default=20261007)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--epochs',type=int,default=200)
    parser.add_argument('--patience',type=int,default=40)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--learning-rate',type=float,default=.001)
    parser.add_argument('--timeout-seconds',type=int,default=1800)
    args = parser.parse_args()
    if min(args.epochs,args.patience,args.batch_size,args.timeout_seconds) < 1:
        raise ValueError('Invalid budget')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite: {output}')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    started = time.monotonic()
    deadline = started+args.timeout_seconds
    torch.set_num_threads(4)
    set_determinism(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    s1 = json.loads((S1_OUTPUT/'summary.json').read_text(encoding='utf-8'))
    source_inputs = s1['inputs']
    for name,item in source_inputs.items():
        if sha256(Path(item['path'])) != item['sha256']:
            raise ValueError(f'S1 source changed: {name}')
    paths = {'feature_cache':S1_OUTPUT/'feature_cache.pt','s1_summary':S1_OUTPUT/'summary.json',
             's1_replay_metrics':S1_OUTPUT/'replay_metrics.csv',
             's1_score_metrics':S1_OUTPUT/'score_metrics.csv','s1_mlp':S1_OUTPUT/'mlp_best.pt',
             'fixed':DEFAULT_R0/'fixed_predictions.csv','script':Path(__file__),
             'geometry_definition':ROOT/'scripts/extract_features.py',
             'evaluation_definition':ROOT/'scripts/evaluate_r1a_frozen_predictions.py',
             'composition_definition':ROOT/'scripts/diagnose_r1_score_replay.py'}
    inputs = {k:{'path':str(p),'sha256':sha256(p)} for k,p in paths.items()}
    inputs.update({f's1_{k}':v for k,v in source_inputs.items()})
    cache = torch.load(paths['feature_cache'],map_location='cpu',weights_only=False)
    if cache['inputs'] != source_inputs:
        raise ValueError('S1 cache lineage differs')
    data = pd.read_csv(source_inputs['targets']['path'])
    if len(data) != 4795 or not data.instance_id.is_unique or set(data.split) != {'train','val'}:
        raise ValueError('Unexpected target package')
    data = data.set_index('instance_id')
    train = data.loc[cache['train_ids']].reset_index()
    val = data.loc[cache['val_ids']].reset_index()
    if (len(train),len(val)) != (4277,518) or not train.split.eq('train').all() or not val.split.eq('val').all():
        raise ValueError('Unexpected frozen splits')
    if set(train.image_id)&set(val.image_id):
        raise ValueError('Image leakage')
    eligible,_ = available_scores(train)
    val_eligible,_ = available_scores(val)
    if (int(eligible.sum()),int(val_eligible.sum())) != (3905,464):
        raise ValueError('Eligible score population changed')
    fixed = pd.read_csv(paths['fixed']).set_index('prediction_id').loc[cache['prediction_ids']].reset_index()
    if len(fixed) != 549 or not fixed.prediction_id.is_unique or not fixed.image_path.map(lambda p: Path(p).parent.name == 'val').all():
        raise ValueError('Only frozen validation predictions permitted')
    h1 = pd.read_csv(source_inputs['h1']['path']).set_index('prediction_id').loc[cache['prediction_ids']].reset_index()
    output.mkdir(parents=True)
    save_json(output/'config.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
              'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              'command':sys.argv,'inputs':inputs,'geometry_names':GEOMETRY_NAMES,
              'geometry_scope':'current raw input masks before letterbox; no border refinement or empirical score transformation',
              'normalization':'3905 eligible training instances only',
              'selection':'minimum val GT-crop score MAE, early epoch on ties',
              'precision':'FP32','augmentation':'none; reuse S1 frozen feature cache',
              'label_version':'v3_valfix_20261006_first_7_only','test_accessed':False})
    print(f'S2 started seed={args.seed} output={output}',flush=True)
    tg = extract_geometry(train,output,'train',deadline)
    vg = extract_geometry(val,output,'val',deadline)
    pg = []
    for index,row in enumerate(fixed.itertuples(index=False)):
        if time.monotonic() > deadline:
            raise TimeoutError('S2 prediction geometry timeout')
        pg.append(mask_geometry(load_fixed_mask(row,DEFAULT_R0)))
        if (index+1)%100 == 0 or index+1 == len(fixed):
            print(f'geometry fixed predictions: {index+1}/{len(fixed)}',flush=True)
    pg = np.stack(pg)
    table = pd.DataFrame(pg,columns=GEOMETRY_NAMES)
    table.insert(0,'prediction_id',fixed.prediction_id.to_numpy())
    table.to_csv(output/'prediction_geometry.csv',index=False,encoding='utf-8-sig')
    mean,std = fit_standardizer(tg,eligible)
    geometry = {k:torch.from_numpy(((v-mean)/std).astype(np.float32))
                for k,v in (('train',tg),('val',vg),('predictions',pg))}
    geometry.update({'mean':mean,'std':std})
    torch.save({'train':geometry['train'],'val':geometry['val'],'predictions':geometry['predictions'],
                'mean':mean.tolist(),'std':std.tolist(),'geometry_names':GEOMETRY_NAMES,
                'train_ids':cache['train_ids'],'val_ids':cache['val_ids'],
                'prediction_ids':cache['prediction_ids'],'inputs':inputs},output/'geometry_cache.pt')
    save_json(output/'geometry_stats.json',{'names':GEOMETRY_NAMES,'mean':mean.tolist(),
               'std':std.tolist(),'fit_split':'eligible_train_only','fit_n':3905})
    old = pd.read_csv(source_inputs['outcomes']['path'])
    base = old.loc[old.variant.eq('baseline')].copy()
    cases = pd.read_csv(source_inputs['score_cases']['path'])
    metrics,score_metrics,outcomes,matched,runs = [],[],[],[],{}
    for kind in ('zero_control','geometry'):
        predictions,record = fit(kind,cache,geometry,train,val,args,output,inputs,deadline)
        runs[kind] = record
        name = f'S2_{kind}'
        row,stats,frame,per_class,points = evaluate_scores(name,predictions,h1,base,cases)
        metrics.append(row)
        score_metrics.append({'variant':name,**stats})
        outcomes.append(frame)
        matched.append(points.assign(variant=name))
        per_class.to_csv(output/f'{kind}_class_metrics.csv',index=False,encoding='utf-8-sig')
        pd.DataFrame({'prediction_id':cache['prediction_ids'],'score_prediction':predictions}).to_csv(
            output/f'{kind}_prediction_scores.csv',index=False,encoding='utf-8-sig')
    combined = pd.concat(outcomes,ignore_index=True)
    pairs,changes = {},[]
    reference_outcomes = pd.read_csv(S1_OUTPUT/'variant_instance_outcomes.csv')
    reference_outcomes = reference_outcomes.loc[reference_outcomes.variant.isin(['S0_cached_learned','S1_mlp'])]
    all_pairs = pd.concat([reference_outcomes,combined],ignore_index=True)
    for reference,candidate in (('S0_cached_learned','S2_geometry'),('S1_mlp','S2_geometry'),
                                ('S2_zero_control','S2_geometry')):
        stats,points = paired_change(all_pairs,reference,candidate)
        pairs[f'{reference} -> {candidate}'] = stats
        changes.append(points)
    pd.concat(changes).to_csv(output/'paired_instance_changes.csv',index=False,encoding='utf-8-sig')
    pd.concat([pd.read_csv(paths['s1_replay_metrics']),pd.DataFrame(metrics)],ignore_index=True).to_csv(
        output/'replay_metrics.csv',index=False,encoding='utf-8-sig')
    pd.concat([pd.read_csv(paths['s1_score_metrics']),pd.DataFrame(score_metrics)],ignore_index=True).to_csv(
        output/'score_metrics.csv',index=False,encoding='utf-8-sig')
    combined.to_csv(output/'variant_instance_outcomes.csv',index=False,encoding='utf-8-sig')
    pd.concat(matched).to_csv(output/'matched_score_predictions.csv',index=False,encoding='utf-8-sig')
    if not all(sha256(Path(v['path'])) == v['sha256'] for v in inputs.values()):
        raise ValueError('An input changed during S2')
    result = {'status':'COMPLETED','origin_skill':'experiment-agent','mode':'run','verification_status':'ANALYZED',
              'runs':runs,'metrics':metrics,'score_metrics':score_metrics,'paired_changes':pairs,
              'elapsed_seconds':time.monotonic()-started,'seed':args.seed,'inputs':inputs,
              'backbone_not_updated':True,'gate_and_stem_from_unchanged_cache':True,
              'test_accessed':False,'labels_changed':False,'thresholds_changed':False,
              'interpretation':'Single-seed development validation, no independent generalization claim.'}
    save_json(output/'summary.json',result)
    print(pd.DataFrame(metrics)[['variant','correct_instances','macro_f1','score_mae','score_gt_10']].to_string(index=False),flush=True)
    print(f'S2 COMPLETED in {result["elapsed_seconds"]:.1f}s',flush=True)


if __name__ == '__main__':
    main()
