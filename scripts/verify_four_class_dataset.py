"""Read-only label checks plus a no-training Ultralytics loading smoke test.

Only the new dataset's audit result and Ultralytics label caches are written.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import yaml

from build_four_class_dataset import digest


def _portable_input_path(recorded_path, project_root):
    """Resolve a recorded build input on either the original or a cloned checkout."""
    recorded = Path(recorded_path)
    if recorded.exists():
        return recorded
    lowered = [part.lower() for part in recorded.parts]
    if 'wolfberry_yolo26_cv' in lowered:
        index = lowered.index('wolfberry_yolo26_cv')
        candidate = project_root.joinpath(*recorded.parts[index + 1:])
        if candidate.exists():
            return candidate
    return None


def verify(out, smoke=True, strict_provenance=False):
    from ultralytics.data.utils import verify_image_label
    project_root = Path(__file__).resolve().parents[1]
    cfg=yaml.safe_load((out/'data.yaml').read_text(encoding='utf-8'))
    images=pd.read_csv(out/'audit/image_manifest.csv')
    labels=pd.read_csv(out/'audit/label_manifest.csv')
    build=json.loads((out/'audit/build_summary.json').read_text(encoding='utf-8'))
    assert len(images)==images.image_id.nunique(), 'duplicate images'
    assert len(labels)==labels.instance_id.nunique(), 'duplicate instance IDs'
    assert set(labels.image_id)==set(images.image_id)
    test_graded=bool(build.get('test_graded',False))
    if test_graded:
        assert cfg.get('test')=='images/test', 'graded test path missing from active YAML'
    else:
        assert 'test' not in cfg, 'ungraded test must not appear in active YAML'
    allowed_splits={'train','val','test'} if test_graded else {'train','val'}
    assert not set(images.original_split)-allowed_splits
    assert not ((images.dataset_split=='val') & images.boundary_review_used).any()
    assert not set(images['dataset_split'])-allowed_splits
    # No group may appear in multiple partitions, even via a chain of near duplicates.
    assert images.groupby('duplicate_group').dataset_split.nunique().max()==1
    hash_groups=images.groupby('image_sha256').dataset_split.nunique()
    assert hash_groups.max()==1
    by_image={k:g.sort_values('output_label_row') for k,g in labels.groupby('image_id')}
    native_instances=0
    active_splits=('train','val','test') if test_graded else ('train','val')
    for split in active_splits:
        expected=set(images.loc[images.dataset_split.eq(split),'image_id'])
        assert {p.stem for p in (out/'images'/split).glob('*.png')}==expected
        assert {p.stem for p in (out/'labels'/split).glob('*.txt')}==expected
        assert {p.stem for p in (out/'labelme_json'/split).glob('*.json')}==expected
    for i,row in enumerate(images.to_dict('records'),1):
        im=out/row['output_image']
        label=out/row['output_label']
        mapped=by_image[row['image_id']]
        assert digest(im)==row['image_sha256']
        source=project_root/'data/split_v1'
        source_image=source/'images'/row['original_split']/f"{row['image_id']}.png"
        if source_image.exists():
            assert digest(source_image)==row['image_sha256']
        elif strict_provenance:
            raise FileNotFoundError(f'Missing source image required by strict provenance: {source_image}')
        old_txt=source/'labels'/row['original_split']/f"{row['image_id']}.txt"
        old_json=source/'labelme_json'/row['original_split']/f"{row['image_id']}.json"
        assert digest(old_txt)==row['source_txt_sha256']
        assert digest(old_json)==row['source_json_sha256']
        old_lines=[s.split() for s in old_txt.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
        lines=[s.split() for s in label.read_text(encoding='utf-8').splitlines() if s.strip()]
        assert len(lines)==len(mapped)==row['candidate_instances']==row['source_unique_polygons']
        jp=out/'labelme_json'/row['dataset_split']/f"{row['image_id']}.json"
        doc=json.loads(jp.read_text(encoding='utf-8'))
        assert (jp.parent/doc['imagePath']).resolve()==im.resolve()
        assert doc['imageData'] is None and len(doc['shapes'])==len(lines)
        for line,record,shape in zip(lines,mapped.to_dict('records'),doc['shapes']):
            assert int(line[0])==record['class_id'] and len(line)>=7 and len(line)%2==1
            xy=np.asarray(line[1:],float).reshape(-1,2)
            assert np.isfinite(xy).all() and np.all((xy>=0)&(xy<=1))
            assert line[1:]==old_lines[int(record['source_polygon_row'])-1][1:]
            assert shape['label']==cfg['names'][int(line[0])]
            assert np.allclose(xy*[doc['imageWidth'],doc['imageHeight']],shape['points'],atol=.001)
        native=verify_image_label((str(im),str(label),'',False,4,0,0,False))
        assert native[0] is not None and native[5:9]==(0,1,0,0), str(native[-1])
        assert len(native[1])==len(lines)==len(native[3]), str(native[-1])
        assert not native[-1], str(native[-1])
        native_instances+=len(native[1])
        if i%150==0:
            print(f'verified {i}/{len(images)} image-label pairs',flush=True)
    assert native_instances==len(labels)
    checked_inputs=[]
    unavailable_inputs=[]
    for name,entry in build['inputs'].items():
        resolved=_portable_input_path(entry['path'],project_root)
        if resolved is None:
            unavailable_inputs.append(name)
            if strict_provenance:
                raise FileNotFoundError(f"Missing build input required by strict provenance: {entry['path']}")
            continue
        assert digest(resolved)==entry['sha256'], f"Input changed: {resolved}"
        checked_inputs.append(name)
    smoke_results={}
    if smoke:
        from ultralytics.data.dataset import YOLODataset
        for split in active_splits:
            ds=YOLODataset(img_path=str(out/'images'/split),imgsz=640,batch_size=2,
                           augment=False,cache=False,data=cfg,task='segment',prefix=split+': ')
            count=sum(len(x['cls']) for x in ds.labels)
            assert count==int(labels.dataset_split.eq(split).sum())
            assert len(ds)==int(images.dataset_split.eq(split).sum())
            sample=ds[0]
            assert 'masks' in sample and sample['masks'].numel()>0
            smoke_results[split]={'images':len(ds),'instances':int(count),
                                  'sample_image_shape':list(sample['img'].shape),
                                  'sample_mask_shape':list(sample['masks'].shape)}
    result={'passed':True,'image_pairs':len(images),'instances':len(labels),
            'strict_provenance':strict_provenance,
            'build_inputs_checked':checked_inputs,'build_inputs_unavailable':unavailable_inputs,
            'original_polygon_coordinates_preserved':True,
            'labels_dropped_by_ultralytics':0,'test_graded':test_graded,
            'sample_visual_review_is_not_all_images_human_qc':True,'loader_smoke':smoke_results}
    (out/'audit/verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    build['validated']=True
    (out/'audit/build_summary.json').write_text(json.dumps(build,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',type=Path,default=Path(__file__).resolve().parents[1]/'data/grading_four_class_v3')
    ap.add_argument('--no-loader-smoke',action='store_true')
    ap.add_argument('--strict-provenance',action='store_true',
                    help='Require all historical source images and build inputs to be present.')
    args=ap.parse_args()
    verify(args.dataset.resolve(),not args.no_loader_smoke,args.strict_provenance)
