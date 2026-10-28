from pathlib import Path
from collections import Counter
from PIL import Image,ImageDraw
import csv,hashlib,json
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/reports/materials_audit_20261028'
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
orig=rows(ROOT/'data/manifests/dataset_manifest.csv');mapping={r['new_stem']:r for r in orig}
boundary=rows(ROOT/'outputs/grading_standard_manual_v4_final/audit/boundary_review_audit.csv')
eligible=[r for r in boundary if r['calibration_eligible'].lower() in ['true','1']]
counts={};retained_changes=[]
for split in ['train','val','test']:
    samebox=0;actual=0
    for p in (ROOT/'data/grading_four_class_v3/labels'/split).glob('*.txt'):
        rr=[l.split() for l in p.read_text(encoding='utf-8').splitlines() if l.strip()];bb=[]
        for r in rr:
            poly=np.array(r[1:],float).reshape(-1,2);bb.append(tuple(np.r_[poly.min(0),poly.max(0)]))
        samebox+=len(bb)-len(set(bb));actual+=len(rr)
        if len(rr)!=int(mapping[p.stem]['instance_count']):retained_changes.append(dict(image_id=p.stem,split=split,raw_instances=int(mapping[p.stem]['instance_count']),formal_instances=len(rr)))
    counts[split]=dict(instances=actual,duplicate_bboxes=samebox)
points=['wb_000002','wb_000151','wb_000501','wb_000911']
contact=Image.new('RGB',(1280,2*(360+34)),(245,245,245));draw=ImageDraw.Draw(contact);samples=[]
for i,iid in enumerate(points):
    row=mapping[iid];path=Path(row['original_image'])
    with Image.open(path) as im:thumb=im.convert('RGB').resize((640,360))
    x=(i%2)*640;y=(i//2)*394;contact.paste(thumb,(x,y+34));draw.text((x+12,y+8),iid+' / '+row['source_group']+' / '+row['split'],fill='black')
    samples.append(dict(image_id=iid,original_source=row['original_image'],split=row['split'],original_number=row['original_number']))
contact.save(OUT/'raw_contact_preview.png')
res=dict(boundary_total=len(boundary),boundary_eligible=len(eligible),boundary_eligible_splits=dict(Counter(mapping[r['image_id']]['split'] for r in eligible)),
    boundary_all_splits=dict(Counter(mapping[r['image_id']]['split'] for r in boundary)),formal_duplicate_bbox_counts=counts,
    retained_image_instance_changes=retained_changes,sample_candidates=samples,
    hold_raw_instances=sum(int(r['instance_count']) for r in orig if not any((ROOT/'data/grading_four_class_v3/labels'/s/(r['new_stem']+'.txt')).exists() for s in ['train','val','test'])))
feat=rows(ROOT/'outputs/strict_rot_exclusion_v2/features_with_manual_rot.csv')
res['feature_manifest_counts']=dict(rows=len(feat),manual_rot_label=dict(Counter(r['manual_rot_label'] for r in feat)),
    manual_rot_exclude=dict(Counter(r['manual_rot_exclude'] for r in feat)),splits=dict(Counter(r['split'] for r in feat)))
hw=Counter()
for p in (ROOT/'outputs/reports/final_test_preflight_20261027/test_replay/predictions').glob('*.pt'):
    cached=torch.load(p,map_location='cpu',weights_only=False);hw[str(cached['entry'].get('input_hw'))]+=1
res['saved_test_network_input_shapes']=dict(hw)
(OUT/'additional_facts.json').write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(res,ensure_ascii=False,indent=2))
