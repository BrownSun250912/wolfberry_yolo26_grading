"""Read-only primary-file extraction for the methods writing material pack.

Loads trusted local checkpoint objects on CPU without constructing a runner,
calling prediction, fitting a model, or changing original inputs.
"""
from pathlib import Path
from collections import Counter,defaultdict
from PIL import Image
import csv,ctypes,hashlib,importlib.metadata,json,os,platform,sys,time,winreg
import numpy as np,torch,yaml

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/reports/materials_audit_20261028';OUT.mkdir(parents=True,exist_ok=True)
sources={}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def register(path):
    path=Path(path).resolve()
    if path not in sources and path.is_file():
        sources[path]=dict(path=str(path),relative_path=path.relative_to(ROOT).as_posix() if ROOT in path.parents else os.path.relpath(path,ROOT).replace('\\','/'),
            bytes=path.stat().st_size,sha256=sha(path))
    return path


def js(path):
    p=register(path);return json.loads(p.read_text(encoding='utf-8-sig'))


def rows(path):
    p=register(path)
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def save(name,v):
    (OUT/name).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def simple(value,depth=0):
    if depth>8:return str(type(value).__name__)
    if value is None or isinstance(value,(str,int,float,bool)):return value
    if isinstance(value,Path):return str(value)
    if isinstance(value,(np.integer,np.floating)):return value.item()
    if isinstance(value,torch.Tensor):
        v=value.detach().cpu()
        return dict(shape=list(v.shape),dtype=str(v.dtype),values=v.tolist() if v.numel()<=32 else None)
    if isinstance(value,np.ndarray):return dict(shape=list(value.shape),values=value.tolist() if value.size<=32 else None)
    if isinstance(value,dict):return {str(k):simple(v,depth+1) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [simple(v,depth+1) for v in value]
    return str(type(value).__name__)


def main():
    data={};original=rows(ROOT/'data/manifests/dataset_manifest.csv')
    splitcounts=Counter(r['split'] for r in original)
    raw_counts={s:sum(int(r['instance_count']) for r in original if r['split']==s) for s in splitcounts}
    data['original_manifest']=dict(images=len(original),images_per_split=dict(splitcounts),instances_per_split=raw_counts,
        instances=sum(raw_counts.values()),resolutions=dict(Counter(r['image_width']+'x'+r['image_height'] for r in original)),
        source_groups=dict(Counter(r['source_group'] for r in original)),physical_independence_verified=False)
    versions=Counter();labels=Counter();shapes=Counter();rawhashes=defaultdict(list);png_info=Counter();exifkeys=Counter();duplicates=0;polygons=0;source_inventory=[]
    for i,r in enumerate(original,1):
        image=Path(r['original_image']);annotation=Path(r['original_json']);txt=Path(r['original_txt'])
        obj=json.loads(annotation.read_text(encoding='utf-8-sig'))
        versions[str(obj.get('version'))]+=1
        for s in obj['shapes']:labels[str(s.get('label'))]+=1;shapes[str(s.get('shape_type'))]+=1
        raw=[line.split() for line in txt.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        boxes=[]
        for line in raw:
            p=np.array(line[1:],float).reshape(-1,2);boxes.append(tuple(np.r_[p.min(0),p.max(0)]))
        duplicates+=len(boxes)-len(set(boxes));polygons+=len(raw)
        with Image.open(image) as im:
            png_info.update(im.info.keys());exifkeys.update(str(k) for k in im.getexif().keys())
            shape=list(im.size)
        digest=sha(image);rawhashes[digest].append((r['split'],r['new_stem']))
        source_inventory.append(dict(image_id=r['new_stem'],source_image=os.path.relpath(image,ROOT).replace('\\','/'),source_json=os.path.relpath(annotation,ROOT).replace('\\','/'),
            source_label=os.path.relpath(txt,ROOT).replace('\\','/'),split=r['split'],image_sha256=digest,json_sha256=sha(annotation),label_sha256=sha(txt),instances=len(raw),width=shape[0],height=shape[1]))
        if i%250==0:print('READ RAW SOURCES',i,'/',len(original),flush=True)
    data['raw_actual_files']=dict(images=len(source_inventory),JSON_versions=dict(versions),shape_labels=dict(labels),shape_types=dict(shapes),
        TXT_instances=polygons,duplicate_bbox_rows_exact_normalized=duplicates,png_info_key_counts=dict(png_info),EXIF_tag_counts=dict(exifkeys),
        exact_image_duplicate_groups=[v for v in rawhashes.values() if len(v)>1],
        cross_split_exact_duplicate_groups=[v for v in rawhashes.values() if len({x[0] for x in v})>1])
    with (OUT/'raw_source_inventory.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(source_inventory[0]));writer.writeheader();writer.writerows(source_inventory)
    final={}
    for version in ['grading_four_class_v3','grading_four_class_v3_valfix_20261006']:
        final[version]={}
        for split in ['train','val','test']:
            folder=ROOT/'data'/version/'labels'/split
            if not folder.is_dir():continue
            counter=Counter();n=0
            for p in sorted(folder.glob('*.txt')):
                for line in p.read_text(encoding='utf-8-sig').splitlines():
                    if line.strip():counter[int(line.split()[0])]+=1;n+=1
            final[version][split]=dict(images=len(list(folder.glob('*.txt'))),instances=n,class_counts={str(k):counter[k] for k in range(4)})
    data['label_counts_actual']=final
    data['build_summary']=js(ROOT/'data/grading_four_class_v3/audit/build_summary.json')
    data['val_overlay_version']=js(ROOT/'data/grading_four_class_v3_valfix_20261006/version.json')
    data['val_overlay_corrections']=js(ROOT/'data/grading_four_class_v3_valfix_20261006/audit/corrections.json')
    data['hold_source_polygons']=sum(int(r['instance_count']) for r in original if r['new_stem'] not in {p.stem for s in ['train','val','test'] for p in (ROOT/'data/grading_four_class_v3/images'/s).glob('*.png')})
    auditrows=rows(ROOT/'data/grading_four_class_v3/audit/image_audit.csv')
    data['image_audit_fields']=list(auditrows[0]);data['image_audit_status']=dict(Counter(r.get('status','') for r in auditrows))
    data['test_label_sources']=dict(Counter(r['label_source'] for r in js(ROOT/'review/three_person_audit_v1/04_审核结果汇总/test_final_labels_deduplicated.json')))
    targets=rows(ROOT/'data/grading_four_class_v3_valfix_20261006/audit/r1_multitask_targets.csv')
    data['multitask_targets']=dict(rows=len(targets),columns=list(targets[0]),splits=dict(Counter(r['split'] for r in targets)),
        missing={k:sum(r[k] in ['','nan','NaN','None'] for r in targets) for k in targets[0] if any(x in k for x in ['target','available'])},
        availability={k:dict(Counter(r[k] for r in targets)) for k in targets[0] if 'available' in k})
    aug=ROOT/'data/grading_four_class_v3_augmented_delivery'
    data['augmentation_config']=js(aug/'metadata/segmentation_augmentation_config.json')
    augrows=rows(aug/'metadata/segmentation_augmentation_manifest.csv')
    data['augmentation_manifest']=dict(rows=len(augrows),columns=list(augrows[0]),record_types=dict(Counter(r['record_type'] for r in augrows)),first_three=augrows[:3])
    data['augmented_actual']={}
    for split in ['train','val','test']:
        folder=aug/'images'/split
        if not folder.exists():continue
        c=Counter()
        for p in (aug/'labels'/split).glob('*.txt'):
            for line in p.read_text(encoding='utf-8-sig').splitlines():
                if line.strip():c[int(line.split()[0])]+=1
        data['augmented_actual'][split]=dict(images=len(list(folder.glob('*.png'))),instances=sum(c.values()),class_counts=dict(c))
    roles=rows(ROOT/'models/experiments/r1_quality_followup_20261009/head_image_split.csv')
    data['internal_roles']=dict(images=len(roles),counts=dict(Counter(r['role'] for r in roles)),columns=list(roles[0]))
    pop=ROOT/'models/experiments/attribute_local_20261025/prototype_population.csv'
    if pop.exists():
        pp=rows(pop);data['prototype_population']=dict(instances=len(pp),by_role=dict(Counter(r['role'] for r in pp)))
    rankpop=js(ROOT/'models/experiments/attribute_local_20261025/attribute_rank_fits/RR22_original_defect_independent_multi_iou_seed20261025/population.json')
    data['rank_population']={k:len(v) for k,v in rankpop.items()}
    scorecfg=js(ROOT/'outputs/grading_standard_manual_v4_final/grading_standard_config.json');data['grading_config']=scorecfg
    boundary=rows(ROOT/'outputs/grading_standard_manual_v4_final/audit/boundary_review_audit.csv')
    data['boundary_audit']=dict(rows=len(boundary),columns=list(boundary[0]),split_counts=dict(Counter(r.get('split','') for r in boundary)))
    candidates=rows(ROOT/'outputs/grading_standard_manual_v4_final/grading_labels_candidate_v3.csv')
    data['frozen_candidate_reference']=dict(rows=len(candidates),columns=list(candidates[0]),split_counts=dict(Counter(r.get('split','') for r in candidates)))
    save('data_facts.json',data)
    print('DATA FACTS SAVED',flush=True)
    frozen=js(ROOT/'outputs/reports/attribute_local_20261025/FROZEN_CANDIDATES_AND_TEST_PROTOCOL.json')
    weights={};model_layers=[]
    selected=['detector','attribute_encoder','RR22_residual','old_defect_rank','prototype_output_refiner','legacy_geometry_score','legacy_learned_residual','legacy_rank_loaded']
    for name in selected+['pretrained']:
        path=ROOT/'models/pretrained/yolo26n-seg.pt' if name=='pretrained' else Path(frozen['weights_and_sources'][name]['path'])
        register(path);cp=torch.load(path,map_location='cpu',weights_only=False)
        rec=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size,keys=list(cp))
        for k in ['epoch','seed','version','date','config','train_args','kind','family','extra','target','policy','held','held_BCE','held_native_metrics','geometry_mean','geometry_std']:
            if k in cp:rec[k]=simple(cp[k])
        for k in ['state','model_state','head_state']:
            if k in cp and isinstance(cp[k],dict):
                rec[k+'_shapes']={x:list(v.shape) for x,v in cp[k].items() if isinstance(v,torch.Tensor)}
                rec[k+'_elements']=sum(v.numel() for v in cp[k].values() if isinstance(v,torch.Tensor))
        model=cp.get('model')
        if model is not None:
            rec['model_type']=type(model).__module__+'.'+type(model).__name__;rec['parameters']=sum(p.numel() for p in model.parameters())
            rec['yaml']=simple(model.yaml);rec['names']=simple(getattr(model,'names',None));rec['stride']=simple(getattr(model,'stride',None));rec['end2end']=simple(getattr(model,'end2end',None))
            head=model.model[-1];rec['head_attributes']={k:simple(getattr(head,k,None)) for k in ['nc','nm','npr','reg_max','max_det','nl','end2end','legacy']}
            if name=='detector':
                for layer in model.model:
                    model_layers.append(dict(index=getattr(layer,'i',None),from_layer=simple(getattr(layer,'f',None)),type=type(layer).__name__,parameters=sum(p.numel() for p in layer.parameters()),structure=str(layer)))
        weights[name]=rec
        del cp
        print('CHECKPOINT READ ONLY',name,flush=True)
    save('weights_metadata.json',weights);save('detector_layers.json',model_layers)
    runs={}
    dirs=[
        'models/experiments/yolo26n_seg_offlineaug_e200_mask50_seed20260915',
        'models/experiments/r1a_hierarchical_grader_seed20260929',
        'models/experiments/r1a_hierarchical_grader_e150_resume_epoch40_seed20260929',
        'models/experiments/r1a_hierarchical_grader_e200_resume_epoch150_seed20260929',
        'models/experiments/refine_module_20261022/rule_anchored_residual/rule_base_residual_seed20261022',
        'models/experiments/attribute_local_20261025/attribute_rank_fits/RR22_original_defect_independent_multi_iou_seed20261025',
        'models/experiments/attribute_local_20261025/prototype_fits/soft_prototype_seed20261025',
        'models/experiments/r1_score_refined_geometry_20261008_artifactfix/refined_only_seed20261007']
    for name in dirs:
        p=ROOT/name;rec=dict(files=[f.name for f in p.iterdir() if f.is_file()])
        for filename in ['training_config.json','training_summary.json','completion.json','args.yaml','config.json']:
            f=p/filename
            if f.exists():
                register(f);rec[filename]=yaml.safe_load(f.read_text(encoding='utf-8-sig')) if f.suffix=='.yaml' else js(f)
        for filename in ['results.csv','history.csv','mask50_selection_trace.csv']:
            f=p/filename
            if f.exists():
                rr=rows(f);rec[filename]=dict(rows=len(rr),columns=list(rr[0]) if rr else [],first=rr[0] if rr else None,last=rr[-1] if rr else None)
                if filename=='mask50_selection_trace.csv':rec['best_selection_trace']=[r for r in rr if r['is_new_best']=='1'][-1]
        runs[name]=rec
    save('training_records.json',runs)
    hardware={}
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:hardware['CPU']=winreg.QueryValueEx(key,'ProcessorNameString')[0]
    except OSError:hardware['CPU']='未找到记录'
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(x,ctypes.c_ulonglong) for x in ['total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','extended']]
    mem=Memory();mem.length=ctypes.sizeof(Memory);ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
    hardware.update(RAM_bytes=mem.total_phys,OS=platform.platform(),Python=platform.python_version(),torch=torch.__version__,torch_compiled_cuda=torch.version.cuda,
        GPU=torch.cuda.get_device_name(0),GPU_memory_bytes=torch.cuda.get_device_properties(0).total_memory,logical_CPUs=os.cpu_count(),
        software={p:importlib.metadata.version(p) for p in ['torch','torchvision','ultralytics','numpy','pandas','scipy','opencv-python','Pillow']},
        scope='Current environment on 2026-10-28; historical training configs retain their own recorded versions.')
    save('current_hardware_software.json',hardware)
    save('primary_read_manifest.json',list(sources.values()))
    print(json.dumps(dict(raw=data['original_manifest'],raw_actual=data['raw_actual_files'],label_counts=final,
        augmentation=data['augmented_actual'],internal_roles=data['internal_roles'],rank_population=data['rank_population'],
        prototype_population=data.get('prototype_population'),checkpoint_summary={k:{x:v.get(x) for x in ['epoch','config','extra','target','policy','parameters','head_attributes']} for k,v in weights.items()},
        hardware=hardware,sources=len(sources)),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
