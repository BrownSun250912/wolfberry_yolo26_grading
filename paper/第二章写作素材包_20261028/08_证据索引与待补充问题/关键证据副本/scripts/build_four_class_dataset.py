"""Trace grades to human polygons; quarantine incomplete images; export without training."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment
from skimage.metrics import structural_similarity


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_polygons(root, source, rec, config):
    stem, split = str(rec['new_stem']), str(rec['split'])
    txt = root / source / 'labels' / split / f'{stem}.txt'
    jp = root / source / 'labelme_json' / split / f'{stem}.json'
    doc = json.loads(jp.read_text(encoding='utf-8'))
    width, height = int(rec['image_width']), int(rec['image_height'])
    if (doc['imageWidth'], doc['imageHeight']) != (width, height):
        raise ValueError('json_image_size_mismatch')
    lines = [line.split() for line in txt.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if len(lines) != len(doc['shapes']):
        raise ValueError('json_txt_count_mismatch')
    polys, masks, duplicates = [], [], []
    for row_index, (line, shape) in enumerate(zip(lines, doc['shapes'])):
        if int(line[0]) != 0 or shape['label'] != 'wolfberry' or shape['shape_type'] != 'polygon':
            raise ValueError('unexpected_source_class_or_shape')
        xy = np.asarray(line[1:], dtype=float).reshape(-1, 2)
        if len(xy) < 3 or not np.isfinite(xy).all() or np.any(xy < 0) or np.any(xy > 1):
            raise ValueError('invalid_polygon_coordinates')
        points = xy * [width, height]
        other = np.asarray(shape['points'], dtype=float)
        if points.shape != other.shape or np.max(np.abs(points-other)) > config['json_txt_pixel_tolerance']:
            raise ValueError('json_txt_geometry_mismatch')
        if abs(cv2.contourArea(points.astype(np.float32))) < config['minimum_polygon_area_px2']:
            raise ValueError('degenerate_source_polygon')
        mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
        match = next((i for i, prior in enumerate(masks) if np.array_equal(mask, prior)), None)
        if match is not None:
            duplicates.append({'image_id': stem, 'duplicate_source_row': row_index + 1,
                               'retained_source_row': polys[match]['source_row']})
            continue
        polys.append({'source_row': row_index + 1, 'normalized': xy, 'points': points,
                      'source_tokens': line[1:], 'shape': shape})
        masks.append(mask)
    return polys, masks, duplicates, doc


def assign_instances(rows, polygons, gt_masks, root, config):
    n = len(rows)
    matrix = np.zeros((n, len(polygons)), dtype=float)
    for i, row in enumerate(rows):
        path = root / 'outputs/features_frozen_v1' / row['split'] / row['raw_mask_path']
        raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise ValueError('missing_raw_mask')
        x1, y1, x2, y2 = (int(row[k]) for k in ('roi_x1', 'roi_y1', 'roi_x2', 'roi_y2'))
        if raw.shape != (y2-y1, x2-x1):
            raise ValueError('mask_roi_shape_mismatch')
        pred = raw > 0
        if not pred.any():
            raise ValueError('empty_predicted_mask')
        for j, truth in enumerate(gt_masks):
            inter = int(np.count_nonzero(pred & (truth[y1:y2, x1:x2] > 0)))
            union = int(np.count_nonzero(pred)) + int(np.count_nonzero(truth)) - inter
            matrix[i,j] = inter / max(union,1)
    ri, ci = linear_sum_assignment(-matrix)
    assignments = []
    for i, j in zip(ri, ci):
        alt_row = max(np.delete(matrix[i], j), default=0.0)
        alt_col = max(np.delete(matrix[:,j], i), default=0.0)
        margin = matrix[i,j] - max(alt_row, alt_col)
        assignments.append({**rows[i], 'source_row': polygons[j]['source_row'],
                            'polygon_index': int(j), 'match_iou': float(matrix[i,j]),
                            'match_margin': float(margin)})
    safe = (len(assignments) == len(rows) == len(polygons) and
            all(r['match_iou'] >= config['minimum_iou'] and
                r['match_margin'] >= config['minimum_ambiguity_margin'] for r in assignments))
    return sorted(assignments, key=lambda r:r['source_row']), safe


def phash(gray):
    coeff = cv2.dct(cv2.resize(gray, (32,32)).astype(np.float32))[:8,:8].flatten()[1:]
    median = np.median(coeff)
    return sum(int(value > median) << i for i, value in enumerate(coeff))


def near_duplicate_pairs(root, manifest, source, config):
    """Use all source train/val images, not test, as leakage reference."""
    samples = []
    for rec in manifest.loc[manifest['split'].isin(['train','val'])].to_dict('records'):
        stem, split = rec['new_stem'], rec['split']
        path = root / source / 'images' / split / f'{stem}.png'
        im = cv2.imread(str(path))
        if im is None:
            raise ValueError(f'missing source image {path}')
        gray = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (160,90), interpolation=cv2.INTER_AREA)
        layout = np.zeros((90,160), np.uint8)
        txt = root / source / 'labels' / split / f'{stem}.txt'
        for line in txt.read_text(encoding='utf-8-sig').splitlines():
            tokens = line.split()
            if not tokens:
                continue
            xy = np.asarray(tokens[1:],float).reshape(-1,2)
            cv2.fillPoly(layout,[np.rint(xy*[160,90]).astype(np.int32)],1)
        samples.append({'image_id':stem,'split':split,'phash':phash(gray),'gray':gray,
                        'layout':layout.astype(bool),'pixel_hash':hashlib.sha256(im.tobytes()).hexdigest()})
    pairs=[]
    for i,a in enumerate(samples):
        for b in samples[i+1:]:
            exact=a['pixel_hash']==b['pixel_hash']
            distance=(a['phash']^b['phash']).bit_count()
            if not exact and distance > config['phash_max_distance']:
                continue
            union=np.count_nonzero(a['layout']|b['layout'])
            liou=np.count_nonzero(a['layout']&b['layout'])/max(union,1)
            if not exact and liou < config['foreground_layout_iou_min']:
                continue
            ssim=float(structural_similarity(a['gray'],b['gray'],data_range=255))
            if exact or ssim >= config['gray_ssim_min']:
                pairs.append({'image_a':a['image_id'],'split_a':a['split'],
                              'image_b':b['image_id'],'split_b':b['split'],
                              'exact_pixels':exact,'phash_distance':distance,
                              'layout_iou':float(liou),'gray_ssim':ssim})
    return pairs


def audit(root, config):
    manifest=pd.read_csv(root/config['source_manifest'])
    selected=pd.read_csv(root/config['candidate_manifest'])
    full=pd.read_csv(root/config['all_features'])
    if selected.duplicated(['image_id','instance_id']).any() or full.duplicated(['image_id','instance_id']).any():
        raise ValueError('duplicate instance keys')
    cols=['image_id','instance_id','raw_mask_path','roi_x1','roi_y1','roi_x2','roi_y2']
    rows=selected.merge(full[cols],on=['image_id','instance_id'],how='left',validate='one_to_one')
    if rows['raw_mask_path'].isna().any():
        raise ValueError('missing feature join')
    if not selected.four_class_label.isin(config['classes']).all():
        raise ValueError('unknown grade class')
    truth=selected.merge(full[['image_id','instance_id','manual_rot_exclude']],on=['image_id','instance_id'],validate='one_to_one')
    rot=truth.manual_rot_exclude.astype(str).str.upper().isin(['TRUE','1','YES'])
    if not (rot==truth.four_class_label.eq('UNQUALIFIED')).all():
        raise ValueError('rot_grade_inconsistency')
    source=manifest.set_index('new_stem')
    audit_rows=[]
    matching_rows=[]
    exact_duplicates=[]
    assets={}
    for idx,(image_id, group) in enumerate(rows.groupby('image_id',sort=True)):
        rec=source.loc[image_id].to_dict() | {'new_stem':image_id}
        result={'image_id':image_id,'original_split':rec['split'],'candidate_instances':len(group),
                'original_manifest_instances':int(rec['instance_count']),'source_unique_polygons':None,
                'status':'HOLD','reason':'','match_min_iou':None,'match_min_margin':None}
        try:
            if rec['split']=='test' or set(group['split'])!={rec['split']}:
                raise ValueError('unexpected_split')
            polygons, masks, duplicates, doc=read_polygons(root,config['source_dataset'],rec,config['matching'])
            exact_duplicates.extend(duplicates)
            result['source_unique_polygons']=len(polygons)
            if len(group)!=len(polygons):
                raise ValueError('original_polygon_count_differs_from_graded_instances')
            assigned,safe=assign_instances(group.to_dict('records'),polygons,masks,root,config['matching'])
            matching_rows.extend(assigned)
            result['match_min_iou']=min(r['match_iou'] for r in assigned)
            result['match_min_margin']=min(r['match_margin'] for r in assigned)
            if not safe:
                raise ValueError('low_iou_or_ambiguous_instance_match')
            assets[image_id]={'rec':rec,'polygons':polygons,'assignments':assigned,'json':doc}
            result['status']='MATCHED'
        except (ValueError,KeyError,FileNotFoundError) as e:
            result['reason']=str(e)
        audit_rows.append(result)
        if (idx+1)%150==0:
            print(f'matched audit {idx+1} images',flush=True)
    print('checking near-duplicates in original train/val',flush=True)
    pairs=near_duplicate_pairs(root,manifest,config['source_dataset'],config['duplicates'])
    # Group transitively so a near-duplicate chain cannot bridge training and validation.
    parent={x:x for x in manifest['new_stem']}
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]
            x=parent[x]
        return x
    for p in pairs:
        a,b=find(p['image_a']),find(p['image_b'])
        parent[max(a,b)]=min(a,b)
    boundary=pd.read_csv(root/config['boundary_audit'])
    boundary_ids=set(boundary['image_id'].astype(str))
    train_sources=set(manifest.loc[manifest['split'].eq('train'),'new_stem'])|boundary_ids
    train_groups={find(x) for x in train_sources}
    for row in audit_rows:
        image_id=row['image_id']
        row['duplicate_group']=find(image_id)
        row['boundary_review_used']=image_id in boundary_ids
        row['dataset_split']=row['original_split']
        if row['status']!='MATCHED':
            continue
        if row['original_split']=='val' and image_id in boundary_ids:
            row['dataset_split']='train'
        elif row['original_split']=='val' and find(image_id) in train_groups:
            row.update(status='HOLD',reason='near_duplicate_of_original_train_or_calibration_image')
        if row['status']=='MATCHED':
            row['status']='EXPORT'
    return manifest,audit_rows,matching_rows,exact_duplicates,pairs,assets


COLORS = {0:(180,60,180), 1:(55,180,40), 2:(235,140,0), 3:(0,130,240)}


def qc_panel(image, asset, image_id, split, names, class_map):
    marked=image.copy()
    width=image.shape[1]
    height=image.shape[0]
    placed=[]
    for row in asset['assignments']:
        polygon=asset['polygons'][row['polygon_index']]
        points=np.rint(polygon['points']).astype(np.int32)
        class_id=int(class_map[row['four_class_label']])
        color=COLORS[class_id]
        cv2.polylines(marked,[points],True,color,3,cv2.LINE_AA)
        center=np.mean(points,axis=0).astype(int)
        label=f"{names[class_id]} {row['instance_id'].split('__')[-1]}"
        (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.55,2)
        initial_y=int(points[:,1].min())-6
        candidates=[(center[0]-tw/2,initial_y-dy) for dy in (0,th+14,2*(th+14),3*(th+14))]
        candidates += [(center[0]-tw/2,int(points[:,1].max())+th+12),
                       (int(points[:,0].max())+10,center[1]),
                       (int(points[:,0].min())-tw-10,center[1])]
        for proposed_x,proposed_y in candidates:
            x=int(np.clip(proposed_x,0,width-tw-6))
            y=int(np.clip(proposed_y,th+6,height-6))
            rect=(x,y-th-5,x+tw+5,y+4)
            if not any(rect[0]<r[2]+5 and rect[2]>r[0]-5 and rect[1]<r[3]+5 and rect[3]>r[1]-5 for r in placed):
                break
        placed.append(rect)
        if abs(y-initial_y)>5:
            cv2.line(marked,(x+tw//2,y+4),(int(center[0]),int(points[:,1].min())),color,1,cv2.LINE_AA)
        cv2.rectangle(marked,(x,y-th-5),(x+tw+5,y+4),(20,20,20),-1)
        cv2.putText(marked,label,(x+2,y),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2,cv2.LINE_AA)
    canvas=np.full((image.shape[0]+80,2*width,3),24,np.uint8)
    canvas[80:,:width]=image
    canvas[80:,width:]=marked
    cv2.putText(canvas,f'{image_id} | {split} | Original (left) / human polygons + grade (right)',
                (18,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(245,245,245),2,cv2.LINE_AA)
    for i in range(4):
        cv2.putText(canvas,f'{i}: {names[i]}',(18+i*310,62),cv2.FONT_HERSHEY_SIMPLEX,.7,COLORS[i],2,cv2.LINE_AA)
    return canvas


def refresh_qc(out,config):
    index=pd.read_csv(out/'audit/qc_index.csv')
    mapping=pd.read_csv(out/'audit/label_manifest.csv')
    inverse={int(v):k for k,v in config['classes'].items()}
    for row in index.to_dict('records'):
        image_id,split=row['image_id'],row['dataset_split']
        im=cv2.imread(str(out/'images'/split/f'{image_id}.png'))
        lines=[s.split() for s in (out/'labels'/split/f'{image_id}.txt').read_text().splitlines() if s.strip()]
        assigned=mapping.loc[mapping.image_id.eq(image_id)].sort_values('output_label_row').to_dict('records')
        polygons=[]
        for i,(line,record) in enumerate(zip(lines,assigned)):
            polygons.append({'points':np.asarray(line[1:],float).reshape(-1,2)*[im.shape[1],im.shape[0]]})
            record['polygon_index']=i
            record['four_class_label']=inverse[int(line[0])]
        panel=qc_panel(im,{'polygons':polygons,'assignments':assigned},image_id,split,config['names'],config['classes'])
        assert cv2.imwrite(str(out/row['qc_file']),panel,[cv2.IMWRITE_JPEG_QUALITY,93])
    print(f'Refreshed {len(index)} QC images; dataset labels unchanged.',flush=True)


def export_dataset(root, out, config, source_manifest, audit_rows, assets, summary, config_path):
    selected=[row for row in audit_rows if row['status']=='EXPORT']
    if not selected or not any(row['dataset_split']=='val' for row in selected):
        raise ValueError('No safe train/val split remains; cannot export training config.')
    for split in ('train','val'):
        for folder in ('images','labels','labelme_json'):
            (out/folder/split).mkdir(parents=True,exist_ok=True)
    qcdir=out/'qc_visualization'
    qcdir.mkdir(exist_ok=True)
    records=[]
    image_records=[]
    source=Path(config['source_dataset'])
    names={int(k):v for k,v in config['names'].items()}
    for number, row in enumerate(selected,1):
        image_id, split=row['image_id'],row['dataset_split']
        asset=assets[image_id]
        original_split=asset['rec']['split']
        src=root/source/'images'/original_split/f'{image_id}.png'
        dst=out/'images'/split/src.name
        shutil.copy2(src,dst)
        source_hash=digest(src)
        if digest(dst)!=source_hash:
            raise ValueError('copy_checksum_mismatch')
        lines=[]
        doc=copy.deepcopy(asset['json'])
        doc['imageData']=None
        doc['imagePath']=f'../../images/{split}/{image_id}.png'
        doc['shapes']=[]
        for output_row, assigned in enumerate(asset['assignments'],1):
            poly=asset['polygons'][assigned['polygon_index']]
            class_id=int(config['classes'][assigned['four_class_label']])
            lines.append(str(class_id)+' '+' '.join(poly['source_tokens']))
            shape=copy.deepcopy(poly['shape'])
            shape['label']=names[class_id]
            shape['description']=f"instance_id={assigned['instance_id']}; source_row={poly['source_row']}; original_label=wolfberry"
            doc['shapes'].append(shape)
            records.append({'image_id':image_id,'instance_id':assigned['instance_id'],
                            'original_split':original_split,'dataset_split':split,
                            'source_polygon_row':poly['source_row'],'output_label_row':output_row,
                            'class_id':class_id,'class_name':names[class_id],
                            'match_iou':assigned['match_iou'],'match_margin':assigned['match_margin'],
                            'grade_source':assigned.get('label_source_v3') if class_id else 'strict_manual_rot_review',
                            'stem_decision':assigned.get('stem_decision'),
                            'mask_label_source':'original_human_polygon'})
        (out/'labels'/split/f'{image_id}.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        (out/'labelme_json'/split/f'{image_id}.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding='utf-8')
        src_txt=root/source/'labels'/original_split/f'{image_id}.txt'
        src_json=root/source/'labelme_json'/original_split/f'{image_id}.json'
        image_records.append({**row,'source_image':str(src),'image_sha256':source_hash,
                              'source_txt_sha256':digest(src_txt),'source_json_sha256':digest(src_json),
                              'output_image':f'images/{split}/{image_id}.png',
                              'output_label':f'labels/{split}/{image_id}.txt'})
        if number%150==0:
            print(f'exported {number}/{len(selected)} images',flush=True)
    records=pd.DataFrame(records)
    pd.DataFrame(image_records).to_csv(out/'audit/image_manifest.csv',index=False,encoding='utf-8-sig')
    records.to_csv(out/'audit/label_manifest.csv',index=False,encoding='utf-8-sig')
    pending=source_manifest.loc[source_manifest['split'].eq('test')].copy()
    pending['status']='FROZEN_GRADE_LABELS_PENDING_NOT_EXPORTED'
    pending.to_csv(out/'audit/test_pending.csv',index=False,encoding='utf-8-sig')
    data_yaml={'path':out.resolve().as_posix(),'train':'images/train','val':'images/val','names':names}
    yaml_text='# Candidate grade labels; independent test grades are not yet available.\n'
    (out/'data.yaml').write_text(yaml_text+yaml.safe_dump(data_yaml,sort_keys=False,allow_unicode=True),encoding='utf-8')
    (out/'classes.txt').write_text('\n'.join(names[i] for i in range(4))+'\n',encoding='utf-8')
    # Visual samples: deterministic per-split random samples plus the weakest geometric matches.
    rng=np.random.default_rng(config['qc_seed'])
    chosen=[]
    for split in ('train','val'):
        pool=[r['image_id'] for r in selected if r['dataset_split']==split]
        n=min(len(pool),int(config['qc_images'])//2)
        chosen.extend(rng.choice(sorted(pool),n,replace=False).tolist())
    weakest=sorted(selected,key=lambda r:r['match_min_iou'])[:6]
    chosen=list(dict.fromkeys(chosen+[r['image_id'] for r in weakest]))
    by_id={r['image_id']:r for r in selected}
    qc_rows=[]
    for image_id in chosen:
        split=by_id[image_id]['dataset_split']
        im=cv2.imread(str(out/'images'/split/f'{image_id}.png'))
        panel=qc_panel(im,assets[image_id],image_id,split,names,config['classes'])
        if not cv2.imwrite(str(qcdir/f'{image_id}.jpg'),panel,[cv2.IMWRITE_JPEG_QUALITY,93]):
            raise IOError('QC image write failed')
        qc_rows.append({'image_id':image_id,'dataset_split':split,'qc_file':f'qc_visualization/{image_id}.jpg',
                        'review_grade_ok':'','review_polygon_ok':'','review_no_missing_fruit':'','review_comment':''})
    pd.DataFrame(qc_rows).to_csv(out/'audit/qc_index.csv',index=False,encoding='utf-8-sig')
    distribution=records.groupby(['dataset_split','class_id','class_name']).size().reset_index(name='instances')
    distribution.to_csv(out/'audit/class_distribution.csv',index=False,encoding='utf-8-sig')
    inputs={}
    for key in ('candidate_manifest','all_features','source_manifest','boundary_audit'):
        path=root/config[key]
        inputs[key]={'path':str(path),'sha256':digest(path)}
    inputs['export_config']={'path':str(config_path),'sha256':digest(config_path)}
    frozen_config=root/config['grading_standard']
    inputs['grading_standard']={'path':str(frozen_config),'sha256':digest(frozen_config)}
    shutil.copy2(frozen_config,out/'audit/grading_standard_config.json')
    shutil.copy2(config_path,out/'audit/export_config.yaml')
    summary.update({'built':True,'validated':False,'created_utc':datetime.now(timezone.utc).isoformat(),
                    'instances':len(records),'class_counts':records['class_name'].value_counts().to_dict(),
                    'split_instance_counts':records['dataset_split'].value_counts().to_dict(),
                    'qc_images':len(chosen),'inputs':inputs,
                    'test_available':False,'label_policy':'human polygons + manual/rule-derived candidate grades'})
    (out/'audit/build_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--config',default='configs/grading_dataset_v1.yaml')
    ap.add_argument('--build',action='store_true')
    ap.add_argument('--refresh-qc',action='store_true')
    args=ap.parse_args()
    root=args.project_root.resolve()
    config=yaml.safe_load((root/args.config).read_text(encoding='utf-8'))
    out=root/config['output']
    if out.resolve()==root or root not in out.resolve().parents:
        raise ValueError('Output must be a dedicated subdirectory of this project.')
    if args.refresh_qc:
        refresh_qc(out,config)
        return
    if args.build and (out/'images').exists():
        raise FileExistsError('Dataset images already exist; choose a new output version.')
    auditdir=out/'audit'
    auditdir.mkdir(parents=True,exist_ok=True)
    manifest,ar,mr,duplicates,pairs,assets=audit(root,config)
    pd.DataFrame(ar).to_csv(auditdir/'image_audit.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(mr).drop(columns=['polygon_index'],errors='ignore').to_csv(auditdir/'instance_mapping.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(duplicates,columns=['image_id','duplicate_source_row','retained_source_row']).to_csv(auditdir/'exact_duplicate_polygons.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(pairs,columns=['image_a','split_a','image_b','split_b','exact_pixels','phash_distance','layout_iou','gray_ssim']).to_csv(auditdir/'near_duplicate_pairs.csv',index=False,encoding='utf-8-sig')
    frame=pd.DataFrame(ar)
    summary={'input_images':len(frame),'image_status_counts':frame['status'].value_counts().to_dict(),
             'hold_reasons':frame.loc[frame.status.eq('HOLD'),'reason'].value_counts().to_dict(),
             'export_split_counts':frame.loc[frame.status.eq('EXPORT'),'dataset_split'].value_counts().to_dict(),
             'near_duplicate_pairs':len(pairs),'exact_duplicate_polygons':len(duplicates),
             'original_test_images_reserved':int(manifest['split'].eq('test').sum()),
             'test_available':False,'built':False}
    if args.build:
        summary=export_dataset(root,out,config,manifest,ar,assets,summary,root/args.config)
    (auditdir/'preflight_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
