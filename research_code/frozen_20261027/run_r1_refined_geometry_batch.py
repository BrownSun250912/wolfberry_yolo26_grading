"""Audit eleven validation cases and compare consistent refined-mask score inputs.

No label, segmentation, gate, threshold or primary model is changed. All
standardizers/ECDFs fit qualified training rows only. Test is never read.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageDraw

from train_r1_score_ordinal_s3 import load_all, fit
from train_r1_score_geometry_only import check_inputs
from train_r1_score_geometry_s2 import mask_geometry, fit_standardizer, GeometryScoreHead, GEOMETRY_NAMES
from train_r1_score_specialist import ROOT, available_scores, evaluate_scores, save_json
from run_r1_score_followup_batch import fit_percentile_transform, apply_percentile_transform
from evaluate_r1a_frozen_predictions import DEFAULT_R0, load_fixed_mask, sha256
from extract_features import clean_analysis_mask, refine_geometry_mask
from build_r1_multitask_targets import parse_polygon

OUTPUT = ROOT/'models/experiments/r1_score_refined_geometry_20261008'
S3 = ROOT/'models/experiments/r1_score_ordinal_s3_20261008'


def read_mask(path):
    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if value is None:
        raise FileNotFoundError(path)
    if value.ndim == 3 and value.shape[2] == 1:
        value = value[:, :, 0]
    if value.ndim != 2:
        raise ValueError('Expected a single channel mask')
    return value


def refine(image, mask, config):
    if image is None or image.shape[:2] != mask.shape:
        raise ValueError('Image/mask shape mismatch')
    cleaned, _, _ = clean_analysis_mask(mask, 20, True)
    result = refine_geometry_mask(image, cleaned, config)
    return result[0], float(result[3]), result[6]


def extract_gt(frame, config, output, split, deadline):
    values, rows, hashes = [], [], []
    for i, row in enumerate(frame.itertuples(index=False)):
        if time.monotonic() > deadline:
            raise TimeoutError('Refined extraction time budget exceeded')
        cp, mp = ROOT/row.crop_path, ROOT/row.mask_path
        if sha256(cp) != row.crop_sha256 or sha256(mp) != row.mask_sha256:
            raise ValueError(f'Changed crop/mask {row.instance_id}')
        image, mask = cv2.imread(str(cp), cv2.IMREAD_COLOR), read_mask(mp)
        refined, removed, status = refine(image, mask, config)
        values.append(mask_geometry(refined))
        rows.append({'instance_id':row.instance_id, 'raw_area':int((mask>0).sum()),
                     'refined_area':int((refined>0).sum()), 'border_removed_ratio':removed,
                     'refinement_status':status})
        hashes.extend([{'path':str(cp),'sha256':row.crop_sha256},
                       {'path':str(mp),'sha256':row.mask_sha256}])
        if (i+1)%500 == 0 or i+1 == len(frame):
            print(f'REFINE {split} {i+1}/{len(frame)}', flush=True)
    pd.DataFrame(rows).to_csv(output/f'{split}_refinement.csv',index=False)
    return np.stack(values), rows, hashes


def extract_predictions(cache, fixed, config, output):
    feature_table = pd.read_csv(DEFAULT_R0/'prediction_features.csv').set_index('prediction_id')
    feature_root = DEFAULT_R0/'replay_workspace/outputs/features_frozen_v1/val'
    values, rows, hashes = [], [], []
    images = {}
    for pid in cache['prediction_ids']:
        row, feature = fixed.loc[pid], feature_table.loc[pid]
        if row.image_id != feature.image_id:
            raise ValueError('Prediction feature image mismatch')
        path = feature_root/feature.mask_path
        refined = read_mask(path)
        value = mask_geometry(refined)
        np.testing.assert_allclose(np.expm1(value[:3]),
            [feature.area,feature.major_axis,feature.minor_axis], rtol=2e-6,atol=.001)
        # Independent full frozen-mask reconstruction: verify cached refinement
        # exactly, not just its numeric feature table.
        if row.image_id not in images:
            images[row.image_id] = cv2.imread(str(row.image_path),cv2.IMREAD_COLOR)
            hashes.append({'path':str(row.image_path),'sha256':sha256(Path(row.image_path))})
        raw = load_fixed_mask(row,DEFAULT_R0)
        computed, ratio, status = refine(images[row.image_id],raw,config)
        roi = computed[int(feature.roi_y1):int(feature.roi_y2),int(feature.roi_x1):int(feature.roi_x2)]
        if not np.array_equal(roi,refined) or np.count_nonzero(computed)!=np.count_nonzero(refined):
            raise ValueError(f'Cached refinement reconstruction mismatch {pid}')
        values.append(value)
        hashes.extend([{'path':str(path),'sha256':sha256(path)},
                       {'path':str(DEFAULT_R0/row.fixed_mask_path),'sha256':row.fixed_mask_sha256}])
        rows.append({'prediction_id':pid,'raw_area':int((raw>0).sum()),
                     'refined_area':int((computed>0).sum()),'border_removed_ratio':ratio,
                     'refinement_status':status,'cached_refinement_equal':True})
    pd.DataFrame(rows).to_csv(output/'prediction_refinement.csv',index=False)
    return np.stack(values), rows, hashes


def audit_cases(train, val, fixed, geometry_raw, details, inputs, output, config):
    candidates = pd.read_csv(inputs['candidate_table']['path']).set_index('instance_id')
    historical = pd.read_csv(inputs['historical_features']['path']).set_index('instance_id')
    source = pd.read_csv(inputs['diagnosis']['path'])
    chosen = source[source.correctable_by_score_alone & (source.reference_threshold_distance>5)]
    if len(chosen)!=11:
        raise ValueError('Eleven-case cohort changed')
    targets = val.set_index('instance_id')
    rows = []
    sheet = Image.new('RGB',(1100,len(chosen)*190),'white')
    draw = ImageDraw.Draw(sheet)
    for case in chosen.itertuples(index=False):
        target, teacher = targets.loc[case.instance_id], candidates.loc[case.instance_id]
        if teacher.image_id != target.image_id or abs(teacher.manual_calibrated_score-target.score_target)>1e-10:
            raise ValueError('Score source join mismatch')
        original = historical.loc[case.instance_id]
        for field in ['area','major_axis','minor_axis','circularity','solidity','border_removed_ratio']:
            if abs(float(original[field])-float(teacher[field]))>1e-9:
                raise ValueError('Candidate versus original feature table mismatch')
        image = cv2.imread(str(ROOT/target.image_path),cv2.IMREAD_COLOR)
        if sha256(ROOT/target.image_path)!=sha256(Path(teacher.image_path)):
            raise ValueError('Historical/current source images differ')
        lines = (ROOT/target.label_path).read_text().splitlines()
        _, points = parse_polygon(lines[int(target.source_polygon_row)-1],image.shape[1],image.shape[0])
        full = np.zeros(image.shape[:2],np.uint8)
        cv2.fillPoly(full,[points],255)
        x1,y1,x2,y2 = map(int,[target.crop_x1,target.crop_y1,target.crop_x2,target.crop_y2])
        if not np.array_equal(full[y1:y2,x1:x2],read_mask(ROOT/target.mask_path)):
            raise ValueError('Target polygon/crop mask mismatch')
        if not np.array_equal(image[y1:y2,x1:x2],cv2.imread(str(ROOT/target.crop_path),cv2.IMREAD_COLOR)):
            raise ValueError('Target RGB crop mismatch')
        full_refined, _, _ = refine(image,full,config)
        crop_refined, _, _ = refine(image[y1:y2,x1:x2],full[y1:y2,x1:x2],config)
        if not np.array_equal(full_refined[y1:y2,x1:x2],crop_refined):
            raise ValueError('Full-image/crop refinement differs')
        teacher_path = ROOT/'outputs/features_frozen_v1/val'/teacher.mask_path
        teacher_iou, teacher_hash = None, None
        if teacher_path.exists():
            teacher_mask = read_mask(teacher_path)
            teacher_values = mask_geometry(teacher_mask)
            np.testing.assert_allclose(np.expm1(teacher_values[:3]),
                [teacher.area,teacher.major_axis,teacher.minor_axis],rtol=2e-6,atol=.001)
            embedded = np.zeros(full.shape,np.uint8)
            embedded[int(teacher.roi_y1):int(teacher.roi_y2),int(teacher.roi_x1):int(teacher.roi_x2)] = teacher_mask
            teacher_iou = float(((embedded>0)&(full>0)).sum()/((embedded>0)|(full>0)).sum())
            teacher_hash = sha256(teacher_path)
            if teacher_iou < .5:
                raise ValueError('Historical teacher instance correspondence suspect')
        # Spatial metadata can be checked even when old raster artifacts were removed.
        cx,cy = float(teacher.centroid_x),float(teacher.centroid_y)
        gx,gy,gw,gh = cv2.boundingRect(points)
        if not (gx<=cx<gx+gw and gy<=cy<gy+gh):
            raise ValueError('Historical teacher centroid outside GT bounding box')
        index = val.index[val.instance_id.eq(case.instance_id)][0]
        pred_index = list(fixed.index).index(case.prediction_id)
        row = case._asdict()
        row.update({'score_source_equal':True,'source_image_equal':True,'crop_polygon_equal':True,
                    'crop_refinement_equal':True,'teacher_mask_iou_vs_gt':teacher_iou,
                    'teacher_refined_area':float(teacher.area),'teacher_removed_ratio':float(teacher.border_removed_ratio),
                    'gt_raw_area':details['val'][index]['raw_area'],
                    'gt_refined_area':details['val'][index]['refined_area'],
                    'prediction_refined_area':details['predictions'][pred_index]['refined_area'],
                    'audited_cause':'stem_false_negative_not_score_error' if case.reference_grade_differs_final else 'score_threshold_crossing',
                    'teacher_mask_available':teacher_path.exists(),'teacher_mask_sha256':teacher_hash,
                    'teacher_feature_table_equal':True,'teacher_centroid_in_gt_bbox':True})
        pred = fixed.loc[case.prediction_id]
        pred_mask = load_fixed_mask(pred,DEFAULT_R0)
        overlay = image.copy()
        overlay[(full>0)] = (.65*overlay[(full>0)]+.35*np.array([0,255,0])).astype(np.uint8)
        overlay[(pred_mask>0)&(full==0)] = [255,0,255]
        patch = cv2.cvtColor(overlay[y1:y2,x1:x2],cv2.COLOR_BGR2RGB)
        thumbnail = Image.fromarray(patch)
        thumbnail.thumbnail((260,165))
        top = len(rows)*190
        sheet.paste(thumbnail,(0,top+18))
        draw.text((275,top+20),f'{case.instance_id} / {case.prediction_id}',fill='black')
        draw.text((275,top+50),f'ref={case.score_target:.3f} S3={case.s3_score:.3f} GT={int(case.gt_class_id)} pred={int(case.pred_class_id)}',fill='black')
        draw.text((275,top+80),f'mask IoU={case.mask_iou:.3f}; old refinement removed={teacher.border_removed_ratio:.1%}',fill='black')
        draw.text((275,top+110),row['audited_cause'],fill='black')
        draw.text((275,top+140),'Green: GT; magenta: prediction outside GT. No relabeling.',fill='black')
        rows.append(row)
    pd.DataFrame(rows).to_csv(output/'eleven_case_audit.csv',index=False)
    sheet.save(output/'eleven_case_audit.png')
    corrected = source[source.correctable_by_score_alone & ~source.reference_grade_differs_final]
    return {'cases':len(rows),'score_threshold_crossings':sum(r['audited_cause']=='score_threshold_crossing' for r in rows),
            'stem_false_negatives':sum(r['audited_cause']=='stem_false_negative_not_score_error' for r in rows),
            'historical_masks_available':sum(r['teacher_mask_available'] for r in rows),
            'historical_mask_pixel_verification':'Unavailable where old raster artifacts are missing; no substitute IoU claimed',
            'all_active_score_errors_consistent_with_reference_grade':len(corrected),
            'scope':'Automated source/mask/crop audit, not human visual or expert review'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    source = json.loads((S3/'summary.json').read_text(encoding='utf-8'))
    cache, old_geometry, train, val, h1, base, cases = load_all(source['inputs'])
    paths = {'source_summary':S3/'summary.json','script':Path(__file__),
             'candidate_table':ROOT/'outputs/grading_standard_manual_v4_final/grading_labels_candidate_v3.csv',
             'historical_features':ROOT/'outputs/strict_rot_exclusion_v2/features_with_manual_rot.csv',
             'feature_config':ROOT/'configs/feature_extraction_frozen_v1_train.yaml',
             'val_feature_config':ROOT/'configs/feature_extraction_frozen_v1_val.yaml',
             'feature_script':ROOT/'scripts/extract_features.py',
             'fit_script':ROOT/'scripts/train_r1_score_ordinal_s3.py',
             'percentile_script':ROOT/'scripts/run_r1_score_followup_batch.py',
             'prediction_features':DEFAULT_R0/'prediction_features.csv',
             'diagnosis':ROOT/'outputs/reports/r1_score_error_ceiling_20261008/qualified_score_diagnosis.csv'}
    inputs = {k:{'path':str(p),'sha256':sha256(p)} for k,p in paths.items()}
    inputs.update({f'prior_{k}':v for k,v in source['inputs'].items()})
    config_all = yaml.safe_load(paths['feature_config'].read_text(encoding='utf-8'))
    val_config = yaml.safe_load(paths['val_feature_config'].read_text(encoding='utf-8'))
    if config_all['geometry_refinement'] != val_config['geometry_refinement'] or config_all['analysis_mask'] != val_config['analysis_mask']:
        raise ValueError('Train/validation feature pipelines differ')
    config = config_all['geometry_refinement']
    output.mkdir(parents=True)
    started = time.monotonic()
    deadline = started+2400
    profiles = ['refined_fused','refined_only','refined_ecdf_fused','refined_ecdf_only']
    save_json(output/'plan.json',{'inputs':inputs,'date':'2026-10-08','profiles':profiles,
        'labels_changed':False,'test_accessed':False,'gates_reused':True,'thresholds_changed':False,
        'seed':20261007,'epochs':200,'patience':40,'timeout_seconds':2400,
        'selection':'minimum GT validation score MAE; development only',
        'stop_after':'Four prespecified profiles plus two extra seeds of refined_fused, independent of results'})
    try:
        raw, details, hashes = {}, {}, []
        for split,frame in [('train',train),('val',val)]:
            raw[split], details[split], hs = extract_gt(frame,config,output,split,deadline)
            hashes.extend(hs)
        fixed = pd.read_csv(source['inputs']['source_fixed']['path']).set_index('prediction_id',drop=False).loc[cache['prediction_ids']]
        raw['predictions'], details['predictions'], hs = extract_predictions(cache,fixed,config,output)
        hashes.extend(hs)
        audit = audit_cases(train,val,fixed,raw,details,inputs,output,config)
        save_json(output/'audit_summary.json',audit)
        print('AUDIT '+json.dumps(audit),flush=True)
        eligible,_ = available_scores(train)
        mean,std = fit_standardizer(raw['train'],eligible)
        geometry = {k:torch.from_numpy(((v-mean)/std).astype(np.float32)) for k,v in raw.items()}
        geometry.update({'mean':mean,'std':std,'train_ids':cache['train_ids'],
                         'val_ids':cache['val_ids'],'prediction_ids':cache['prediction_ids']})
        state = fit_percentile_transform(raw['train'],eligible)
        rank_geometry = {k:apply_percentile_transform(v,state) for k,v in raw.items()}
        rank_geometry.update({'mean':state['mean'],'std':state['std']})
        torch.save({'raw':raw,'geometry':geometry,'rank_geometry':rank_geometry,'percentile_state':state,
                    'inputs':inputs,'artifact_hashes':hashes,'config':config},output/'refined_geometry_cache.pt')
        for split in raw:
            table = pd.DataFrame(raw[split],columns=GEOMETRY_NAMES)
            table.insert(0,'id',cache['prediction_ids' if split=='predictions' else split+'_ids'])
            table.to_csv(output/f'{split}_geometry.csv',index=False)
        save_json(output/'artifact_hashes.json',hashes)
        runs, metrics, statistics, outcomes, verification = {}, [], [], [], {}
        jobs = [(profile,20261007) for profile in profiles]+[('refined_fused',20261008),('refined_fused',20261009)]
        for profile,seed in jobs:
            kind = f'{profile}_seed{seed}'
            directory = output/kind
            directory.mkdir()
            only = profile.endswith('_only')
            current_cache = dict(cache)
            if only:
                current_cache.update({k:torch.zeros_like(cache[k]) for k in raw})
            current_geometry = rank_geometry if 'ecdf' in profile else geometry
            options = SimpleNamespace(seed=seed,device='cuda:0',learning_rate=1e-3,
                temperature=.05,epochs=200,patience=40,batch_size=128)
            print(f'GROUP {kind}',flush=True)
            scores, record = fit(kind,0.,current_cache,current_geometry,train,val,options,directory,inputs,deadline)
            checkpoint_path = directory/f'{kind}_best.pt'
            checkpoint = torch.load(checkpoint_path,map_location='cpu',weights_only=False)
            checkpoint.update({'requires_frozen_R1_features':not only,'zero_R1_features':only,
                               'geometry_preprocessing':'historical pale-boundary refinement',
                               'transform':'train_ecdf' if 'ecdf' in profile else 'train_zscore',
                               'geometry_cache_sha256':sha256(output/'refined_geometry_cache.pt')})
            torch.save(checkpoint,checkpoint_path)
            row,stats,frame,classes,points = evaluate_scores(kind,scores,h1,base,cases)
            classes.to_csv(directory/'class_metrics.csv',index=False)
            points.to_csv(directory/'matched_score_predictions.csv',index=False)
            runs[kind],metrics,statistics,outcomes = record,metrics+[row],statistics+[{'variant':kind,**stats}],outcomes+[frame]
            # Saved checkpoint CPU replay, independent from GPU output arrays.
            head = GeometryScoreHead(True).eval()
            head.load_state_dict(torch.load(checkpoint_path,map_location='cpu',weights_only=False)['head_state'])
            with torch.no_grad():
                cpu_scores = head(current_cache['predictions'],current_geometry['predictions']).numpy()*100
                cpu_val = head(current_cache['val'],current_geometry['val']).numpy()*100
            saved = pd.read_csv(directory/f'{kind}_val_gt_predictions.csv')
            if saved.instance_id.tolist()!=cache['val_ids']:
                raise ValueError('Validation ID alignment changed')
            delta = max(float(np.max(np.abs(cpu_scores-scores))),float(np.max(np.abs(cpu_val-saved.score_prediction))))
            if delta>.001:
                raise ValueError('CPU checkpoint replay mismatch')
            replay,_,_,_,_ = evaluate_scores(kind,cpu_scores,h1,base,cases)
            for field in ['correct_instances','uq_mismatches','grade_1_3_errors','score_gt_10']:
                if replay[field]!=row[field]:
                    raise ValueError(f'CPU metric mismatch {field}')
            verification[kind] = {'status':'PASS','max_prediction_delta':delta,'checkpoint_sha256':sha256(checkpoint_path)}
            pd.DataFrame(metrics).to_csv(output/'replay_metrics.csv',index=False)
            print('RESULT '+json.dumps(row),flush=True)
        check_inputs(inputs)
        for entry in hashes:
            if sha256(Path(entry['path'])) != entry['sha256']:
                raise ValueError('Feature artifact changed during run')
        pd.concat(outcomes,ignore_index=True).to_csv(output/'variant_instance_outcomes.csv',index=False)
        pd.DataFrame(statistics).to_csv(output/'score_metrics.csv',index=False)
        save_json(output/'checkpoint_verification.json',verification)
        save_json(output/'summary.json',{'status':'COMPLETED','date':'2026-10-08','origin_skill':'experiment-agent',
            'mode':'run','verification_status':'ANALYZED','inputs':inputs,'audit':audit,'runs':runs,
            'metrics':metrics,'score_metrics':statistics,'elapsed_seconds':time.monotonic()-started,
            'labels_changed':False,'test_accessed':False,'thresholds_changed':False,'gates_reused':True,
            'deployment_changed':False,'scope':'Adaptive development validation; not independent generalization evidence'})
        print('COMPLETED refined geometry batch',flush=True)
    except Exception as exc:
        save_json(output/'failure.json',{'status':'FAILED','error':repr(exc),'inputs':inputs})
        raise


if __name__=='__main__':
    main()
