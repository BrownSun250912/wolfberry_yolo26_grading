"""Freeze existing test inputs and verify provenance without model inference."""
from pathlib import Path
from collections import Counter,defaultdict
import csv,datetime as dt,hashlib,json,sys
import cv2,numpy as np,yaml

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/reports/final_test_preflight_20261027'
DATA=ROOT/'data/grading_four_class_v3'
OVERLAY=ROOT/'data/grading_four_class_v3_valfix_20261006'
PROTOCOL=ROOT/'outputs/reports/attribute_local_20261025/FROZEN_CANDIDATES_AND_TEST_PROTOCOL.json'
SUFFIXES={'.png','.jpg','.jpeg','.bmp','.tif','.tiff','.webp'}
CLASS_NAMES=['unqualified','grade_1','grade_2','grade_3']
ARMS=['raw_YOLO','raw_YOLO_realNMS','RR22_reference','original_defect_rank','original_defect_rank_prototype']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def csvrows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def label_path(split,iid):return (OVERLAY if split=='val' else DATA)/'labels'/split/(iid+'.txt')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    checks=[];errors=[];warnings=[];image_records=[];labels_by_image={};counts={};hash_groups=defaultdict(list)
    def check(name,value,detail=None):
        checks.append(dict(check=name,passed=bool(value),detail=detail))
        if not value:errors.append(name)
    protocol=json.loads(PROTOCOL.read_text(encoding='utf-8'))
    names=yaml.safe_load((DATA/'data.yaml').read_text(encoding='utf-8'))['names']
    check('Class IDs and names agree with frozen four-class protocol',[names[i] for i in range(4)]==CLASS_NAMES)
    check('Five primary arm identities match frozen protocol',protocol['main_controls']==[
        'raw_YOLO no extraNMS','raw_YOLO realsameclassNMS','RR22 boundary/stem/neighbour realNMS',
        'old_defect_rank','old_defect_rank + prototypeoutput'])
    artifacts=[]
    for name,item in protocol['weights_and_sources'].items():
        p=Path(item['path']);value=sha(p) if p.is_file() else None
        check('Frozen artifact '+name,value==item['sha256'])
        artifacts.append(dict(name=name,path=str(p),sha256=value))
    frozen_env=json.loads((PROTOCOL.parent/'runtime_environment.json').read_text(encoding='utf-8'))
    ops=frozen_env['versions']['installed_ultralytics_ops']
    check('Native-mask implementation hash',sha(ops['path'])==ops['sha256'])

    image_manifest=csvrows(DATA/'audit/image_manifest.csv')
    exported_images={r['image_id']:r for r in image_manifest if r.get('dataset_split') in ['train','val','test'] and r.get('output_image')}
    label_manifest=csvrows(DATA/'audit/label_manifest.csv')
    label_rows=defaultdict(list)
    for r in label_manifest:label_rows[(r['dataset_split'],r['image_id'])].append(r)
    targets=csvrows(OVERLAY/'audit/r1_multitask_targets.csv')
    check('Multitask target manifest contains train/val only',{r['split'] for r in targets}=={'train','val'})
    target_map={r['instance_id']:r for r in targets}
    test_final_path=ROOT/'review/three_person_audit_v1/04_审核结果汇总/test_final_labels.json'
    adjudicated={r['instance_id']:r for r in json.loads(test_final_path.read_text(encoding='utf-8'))}
    class_ids={'UNQUALIFIED':0,'GRADE1':1,'GRADE2':2,'GRADE3':3,'GRADE_1':1,'GRADE_2':2,'GRADE_3':3}
    test_sources=Counter();val_changes=0;seen_ids={};pixel_checks=0
    for split in ['train','val','test']:
        folder=DATA/'images'/split
        images=sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES)
        ids=[p.stem for p in images];seen_ids[split]=set(ids)
        check(split+' unique image stems',len(ids)==len(set(ids)))
        actual_labels={p.stem for p in label_path(split,'_').parent.glob('*.txt')}
        check(split+' one label file per image',set(ids)==actual_labels)
        classes=Counter();instances=0
        for p in images:
            iid=p.stem;lp=label_path(split,iid);ish=sha(p);lsh=sha(lp)
            hash_groups[ish].append((split,iid))
            if iid not in exported_images:
                check('Exported image provenance '+iid,False);continue
            original=exported_images[iid]
            check('Image SHA '+iid,ish==original['image_sha256'])
            meta=sorted(label_rows[(split,iid)],key=lambda r:int(r['output_label_row']))
            raw=[line.split() for line in lp.read_text(encoding='utf-8').splitlines() if line.strip()]
            check('Manifest row count '+iid,len(raw)==len(meta))
            parsed=[]
            if split=='val':parent=[line.split() for line in (DATA/'labels/val'/(iid+'.txt')).read_text(encoding='utf-8').splitlines() if line.strip()]
            # Decode only evaluation partitions. Train byte hashes still support exact-duplicate checks.
            shape=None
            if split in ['val','test']:
                image=cv2.imdecode(np.fromfile(p,dtype=np.uint8),cv2.IMREAD_COLOR)
                check('Image decodes '+iid,image is not None)
                if image is None:continue
                shape=list(image.shape[:2]);h,w=shape
            for j,parts in enumerate(raw):
                numeric=np.array([float(v) for v in parts],dtype=np.float64)
                good=len(parts)>=7 and (len(parts)-1)%2==0 and np.isfinite(numeric).all() and numeric[0] in [0,1,2,3] and ((numeric[1:]>=0)&(numeric[1:]<=1)).all()
                check(f'Valid polygon {iid}:{j+1}',good)
                if not good:continue
                cls=int(numeric[0]);points=numeric[1:].reshape(-1,2);classes[cls]+=1;instances+=1
                record=dict(row=j+1,class_id=cls,polygon=points.tolist())
                if j<len(meta):
                    m=meta[j];record['instance_id']=m['instance_id']
                    if split=='test':
                        check(f'Test original class {iid}:{j+1}',cls==int(m['class_id']))
                        human=adjudicated.get(m['instance_id'])
                        check(f'Test adjudication {iid}:{j+1}',human is not None and class_ids.get(human['final_label'])==cls)
                        if human:test_sources[human['label_source']]+=1
                    else:
                        target=target_map.get(m['instance_id'])
                        check(f'Target class {iid}:{j+1}',target is not None and int(target['final_class_id'])==cls)
                if split=='val':
                    check(f'Overlay geometry unchanged {iid}:{j+1}',len(parent)==len(raw) and parent[j][1:]==parts[1:])
                    val_changes+=int(parent[j][0]!=parts[0])
                if shape:
                    pts=np.rint(points*np.array([w,h])).astype(np.int32)
                    pts[:,0]=np.clip(pts[:,0],0,w-1);pts[:,1]=np.clip(pts[:,1],0,h-1)
                    mask=np.zeros((h,w),np.uint8);cv2.fillPoly(mask,[pts],1)
                    check(f'Nonempty original-grid polygon {iid}:{j+1}',mask.any());pixel_checks+=1
                parsed.append(record)
            labels_by_image[(split,iid)]=parsed
            image_records.append(dict(split=split,image_id=iid,image_path=str(p),label_path=str(lp),
                image_sha256=ish,label_sha256=lsh,shape=shape,instances=len(parsed),source_group=source_group(iid)))
        counts[split]=dict(images=len(images),instances=instances,class_counts={str(c):classes[c] for c in range(4)})
        print('AUDIT',split,counts[split],flush=True)
    expected={'train':(966,4277,[372,945,1138,1822]),'val':(116,518,[54,109,150,205]),'test':(129,576,[25,126,167,258])}
    for split,(n,k,c) in expected.items():
        check(split+' expected image count',counts[split]['images']==n)
        check(split+' expected instance count',counts[split]['instances']==k)
        check(split+' expected class counts',[counts[split]['class_counts'][str(j)] for j in range(4)]==c)
    check('Overlay changes exactly seven class labels',val_changes==7)
    crosses=[v for v in hash_groups.values() if len({a for a,b in v})>1]
    check('No byte-identical image across train/val/test',len(crosses)==0)
    for a,b in [('train','val'),('train','test'),('val','test')]:check(a+'/'+b+' IDs disjoint',not (seen_ids[a]&seen_ids[b]))
    dedup=json.loads((DATA/'audit/test_polygon_deduplication.json').read_text(encoding='utf-8'))
    check('Six historical duplicate removals remain documented',dedup['dropped_instances']==6 and dedup['retained_test_instances']==576)
    check('Adjudication records account for retained and historically removed instances',len(adjudicated)==576+6)
    source_files=[PROTOCOL,DATA/'data.yaml',DATA/'audit/label_manifest.csv',DATA/'audit/image_manifest.csv',
                  DATA/'audit/test_polygon_deduplication.json',DATA/'audit/build_summary.json',test_final_path,OVERLAY/'version.json',
                  OVERLAY/'audit/r1_multitask_targets.csv',Path(ops['path'])]
    snapshot=dict(schema=1,protocol_sha256=sha(PROTOCOL),classes=CLASS_NAMES,arms=ARMS,
        data_policy='Original v3 images; validation-only valfix labels; unchanged original v3 test labels.',
        images=image_records,artifacts=artifacts,metadata=[dict(path=str(p),sha256=sha(p)) for p in source_files],
        test_prediction_or_metric_computed=False,historical_test_use=protocol['historical_test_use'])
    frozen=OUT/'FROZEN_INPUTS.json'
    if frozen.exists():check('Frozen input snapshot unchanged',json.loads(frozen.read_text(encoding='utf-8'))==snapshot)
    elif not errors:write(frozen,snapshot)
    warnings.extend(['Byte hashes do not establish distinct physical fruits or independent acquisition batches.',
        'Only 25 unqualified test instances; report its uncertainty and support counts.',
        'Test labels mix independent review and frozen-rule fallback; grade accuracy is not an independent commercial-quality gold standard.',
        'Validation overlay is a historical seven-label, single-reviewer correction; test is not silently overlaid.',
        'No independent score/defect/stem test target manifest is present in the frozen multitask targets; omit those extra target-specific metrics unless separately verified.'])
    status=dict(status='PASS_DATA_AND_FROZEN_ARTIFACTS' if not errors else 'FAIL',
        checks=len(checks),failed_checks=len(errors),errors=errors,warnings=warnings,counts=counts,
        test_label_sources=dict(test_sources),test_cross_split_exact_duplicates=crosses,
        original_grid_polygon_checks=pixel_checks,val_overlay_changed_classes=val_changes,
        test_images_and_annotations_read=True,test_model_inference_runs=0,test_performance_metrics_computed=False,
        frozen_input_manifest=str(frozen) if frozen.exists() else None,generated_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    write(OUT/'data_preflight.json',status);write(OUT/'data_checks.json',checks)
    print(json.dumps({k:v for k,v in status.items() if k not in ['warnings','test_cross_split_exact_duplicates']},ensure_ascii=False,indent=2))
    if errors:sys.exit(1)

def source_group(iid):
    n=int(iid.rsplit('_',1)[1])
    return '001-150' if n<=150 else '151-500' if n<=500 else '501-900' if n<=900 else '901-1300'

if __name__=='__main__':main()
