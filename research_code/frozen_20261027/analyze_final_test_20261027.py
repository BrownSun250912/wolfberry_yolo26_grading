"""Frozen-test analysis from saved predictions only; never run a model or fit parameters."""
from pathlib import Path
from collections import Counter, defaultdict
import datetime as dt, hashlib, json, time
import numpy as np, pandas as pd, torch
from ultralytics.utils.metrics import ap_per_class
from analyze_r1_ap_bootstrap import pure_match
from joint_pipeline_policy_20261024 import keep_indices

ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / 'outputs/reports/final_test_preflight_20261027'
RUN = PREFLIGHT / 'test_replay'
OUT = PREFLIGHT / 'test_analysis'
ARMS = ['raw_YOLO', 'raw_YOLO_realNMS', 'RR22_reference', 'original_defect_rank', 'original_defect_rank_prototype']
ENDPOINTS = ['Mask_AP50', 'Mask_AP50_95', 'conditional_grade_accuracy', 'correct_per_all_GT',
             'operating_macro_F1_rawscope', 'operating_macro_recall_rawscope',
             'UQ_escape_per_all_UQ_GT', 'qualified_rejected_per_qualified_GT',
             'new_UQ_escape_per_all_UQ_GT', 'UQ_any_qualified_emission_per_all_UQ_GT']
PAIRS = [(ARMS[1], ARMS[0]), (ARMS[2], ARMS[1]), (ARMS[3], ARMS[2]),
         (ARMS[3], ARMS[0]), (ARMS[3], ARMS[1]), (ARMS[4], ARMS[3])]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p, v): Path(p).write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def ratio(n, d): return float(n / d) if d else float('nan')


def ap(blocks, indices):
    tp, confidence, cls, gt = (np.concatenate([blocks[i][j] for i in indices]) for j in range(4))
    if len(np.unique(gt)) != 4: return np.array([np.nan, np.nan])
    values = ap_per_class(tp, confidence, cls, gt, plot=False)[5]
    if values.shape != (4, 10): raise ValueError('Four-class AP denominator changed')
    return np.array([values[:, 0].mean(), values.mean()])


def count_values(v):
    # GT, localized, correct, UQ_GT, UQ_escape, qualified_rejected, new_escape,
    # any_escape, then per-class TP, prediction count, and GT count.
    g, loc, correct, uq, esc, rejected, new, anyesc = v[:8]
    tp, pred, classes = v[8:12], v[12:16], v[16:20]
    return np.array([ratio(correct, loc), ratio(correct, g),
        float((2 * tp / np.maximum(pred + classes, 1)).mean()),
        float((tp / np.maximum(classes, 1)).mean()), ratio(esc, uq),
        ratio(rejected, g-uq), ratio(new, uq), ratio(anyesc, uq)])


def main():
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    run = read(RUN / 'replay_report.json')
    if run['status'] != 'PASS' or run['split'] != 'test' or run['images'] != 129: raise ValueError('Incomplete test output')
    frozen = read(PREFLIGHT / 'FROZEN_INPUTS.json')
    selected = [r for r in frozen['images'] if r['split'] == 'test']
    plan = dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(), iterations=2000, seed=20261027,
        registered_before_test=False, timing='Analysis details fixed after viewing point estimates, before drawing bootstrap samples. Arm identities and primary endpoint were frozen before test.',
        primary_comparison=[ARMS[3], ARMS[2]], primary_endpoint='Mask_AP50', all_comparisons=PAIRS,
        endpoints=ENDPOINTS, unit='Original image, all GT and predictions kept together; paired multiplicities shared across all five arms.',
        strata='Four pre-existing filename/source ranges, preserving each range size. These are not verified independent harvest batches.',
        interval='Pointwise percentile 95%, descriptive; no p-values or multiplicity-adjusted simultaneous inference. No threshold/checkpoint selection.',
        tie_handling='Sort sampled image indices to preserve original order of tied scores; use unchanged official AP implementation.',
        missing_class='AP undefined if a draw lacks a GT class; retain draw and report valid counts, do not redraw.',
        test_model_inference_this_analysis=0, sources={str(RUN/'replay_report.json'):sha(RUN/'replay_report.json'),
            str(RUN/'metrics.csv'):sha(RUN/'metrics.csv'), str(__file__):sha(__file__)})
    write(OUT / 'analysis_plan.json', plan)
    with (RUN / 'metrics.csv').open(encoding='utf-8', newline='') as f: metrics = pd.read_csv(f).set_index('arm')
    details = read(RUN / 'details.json')
    caches = [torch.load(RUN / 'predictions' / (r['image_id']+'.pt'), map_location='cpu', weights_only=False) for r in selected]
    if [c['entry']['image_id'] for c in caches] != [r['image_id'] for r in selected]: raise ValueError('Image order changed')
    blocks = {}; counts = {}; cases_by_arm = {}
    safe = {r['instance_id'] for r in details['RR22_reference']['cases'] if r['gt_class'] == r['pred_class'] == 0}
    for arm in ARMS:
        saved = torch.load(RUN / (arm+'_evaluation.pt'), map_location='cpu', weights_only=False)
        blocks[arm] = saved['blocks']
        if len(blocks[arm]) != 129: raise ValueError('Block count changed')
        by_image = defaultdict(list)
        for row in details[arm]['cases']: by_image[row['image_id']].append(row)
        cases_by_arm[arm] = by_image
        per_image = []
        for item, cache in zip(selected, caches):
            e = cache['entry']; cls = e['baseline'] if arm.startswith('raw_YOLO') else cache['labels']
            rank = cache['confidence'] if arm.startswith('original_defect_rank') else e['conf']
            ious = cache['refined_mask_iou'] if arm.endswith('_prototype') else e['mask_iou']
            keep = np.arange(len(cls)) if arm == 'raw_YOLO' else keep_indices(e['boxes'], cls, rank, .7, e['conf'] >= .25)
            high = keep[e['conf'][keep] >= .25]
            flags = pure_match(cls[high], e['gt_cls'], ious[:, high])[:, 0]
            cc = by_image[item['image_id']]
            g = len(e['gt_cls']); uq = int((e['gt_cls'] == 0).sum())
            correct = sum(r['correct'] for r in cc)
            esc = sum(r['gt_class'] == 0 and r['pred_class'] != 0 for r in cc)
            rejected = sum(r['gt_class'] != 0 and r['pred_class'] == 0 for r in cc)
            new = sum(r['instance_id'] in safe and r['pred_class'] != 0 for r in cc if r['gt_class'] == 0)
            qualified = high[cls[high] > 0]
            anyesc = int((ious[e['gt_cls'] == 0][:, qualified] >= .5).any(1).sum())
            tp = [int(flags[cls[high] == c].sum()) for c in range(4)]
            preds = [int((cls[high] == c).sum()) for c in range(4)]
            gt = [int((e['gt_cls'] == c).sum()) for c in range(4)]
            per_image.append([g, len(cc), correct, uq, esc, rejected, new, anyesc] + tp + preds + gt)
        counts[arm] = np.array(per_image, dtype=float)
    ids = np.arange(129)
    points = {a:np.r_[ap(blocks[a], ids), count_values(counts[a].sum(0))] for a in ARMS}
    for a in ARMS:
        for i, key in enumerate(ENDPOINTS[:6]):
            if abs(points[a][i] - float(metrics.loc[a, key])) > 1e-12: raise ValueError('Point metric reproduction failed: '+a+' '+key)
        total = counts[a].sum(0)
        for index, key in [(0,'GT'), (1,'localized'), (2,'correct_grade'), (3,'UQ_GT'), (4,'UQ_escape'),
                           (5,'qualified_rejected'), (6,'new_UQ_escape'), (7,'UQ_any_qualified_emission')]:
            if total[index] != metrics.loc[a, key]: raise ValueError('Counter reproduction failed: '+a+' '+key)
    groups = sorted(set(r['source_group'] for r in selected))
    strata = [np.array([i for i,r in enumerate(selected) if r['source_group']==g]) for g in groups]
    write(OUT / 'point_reproduction.json', dict(status='PASS', arms=ARMS, endpoints=ENDPOINTS,
        points={a:points[a].tolist() for a in ARMS}, strata_sizes={g:len(s) for g,s in zip(groups,strata)}))
    rng = np.random.default_rng(plan['seed'])
    draws = np.empty((plan['iterations'], len(ARMS), len(ENDPOINTS)), dtype=float)
    start = time.perf_counter()
    for k in range(plan['iterations']):
        sample = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata]); sample.sort()
        for j,a in enumerate(ARMS): draws[k,j] = np.r_[ap(blocks[a], sample), count_values(counts[a][sample].sum(0))]
        if (k+1)%250 == 0: print('TEST BOOTSTRAP', k+1, '/', plan['iterations'], 'elapsed_s', round(time.perf_counter()-start,1), flush=True)
    np.savez_compressed(OUT / 'bootstrap_draws.npz', draws=draws, arms=np.array(ARMS), endpoints=np.array(ENDPOINTS))
    absolute = []; contrasts = []
    def interval(v):
        ok = np.isfinite(v); good = v[ok]
        return dict(valid_replicates=int(ok.sum()), invalid_replicates=int((~ok).sum()),
                    ci95_low=float(np.quantile(good, .025)), ci95_high=float(np.quantile(good, .975)))
    for j,a in enumerate(ARMS):
        for k,name in enumerate(ENDPOINTS): absolute.append(dict(arm=a, endpoint=name, point=float(points[a][k]), **interval(draws[:,j,k])))
    for a,b in PAIRS:
        ai,bi = ARMS.index(a),ARMS.index(b)
        for k,name in enumerate(ENDPOINTS):
            contrasts.append(dict(candidate=a, reference=b, endpoint=name, point_delta=float(points[a][k]-points[b][k]),
                primary=(a,b)==(ARMS[3],ARMS[2]) and name=='Mask_AP50', **interval(draws[:,ai,k]-draws[:,bi,k])))
    pd.DataFrame(absolute).to_csv(OUT/'absolute_intervals.csv',index=False)
    pd.DataFrame(contrasts).to_csv(OUT/'paired_contrasts.csv',index=False)
    changes = []
    original = {r['instance_id']:r for r in details[ARMS[3]]['cases']}
    refined = {r['instance_id']:r for r in details[ARMS[4]]['cases']}
    for iid in sorted(set(original)|set(refined)):
        before,after = original.get(iid),refined.get(iid)
        if before is None or after is None or before['prediction_index'] != after['prediction_index'] or before['correct'] != after['correct']:
            changes.append(dict(instance_id=iid, original_match=before, refined_match=after))
    write(OUT/'refined_matching_changes.json', dict(original_correct=481, refined_correct=480,
        inference_labels_identical=True, explanation='The same frozen per-candidate labels/ranks are used; refined geometry changes class-agnostic GT-to-prediction assignment. Do not describe as invariant end-to-end grading correctness.', changes=changes))
    subgroup = []
    for group,ix in zip(groups,strata):
        present=np.unique(np.concatenate([caches[i]['entry']['gt_cls'] for i in ix])).tolist()
        for a in ARMS:
            v=counts[a][ix].sum(0)
            subgroup.append(dict(source_group=group, arm=a, images=len(ix), GT=int(v[0]), localized=int(v[1]), correct=int(v[2]),
                UQ_GT=int(v[3]), UQ_escape=int(v[4]), qualified_rejected=int(v[5]), present_classes=present,
                correct_per_all_GT=ratio(v[2],v[0]), conditional_grade_accuracy=ratio(v[2],v[1])))
    pd.DataFrame(subgroup).to_csv(OUT/'source_group_counts.csv',index=False)
    if sha(RUN/'replay_report.json') != plan['sources'][str(RUN/'replay_report.json')] or sha(RUN/'metrics.csv') != plan['sources'][str(RUN/'metrics.csv')]:
        raise ValueError('Frozen test output changed during analysis')
    result = dict(status='COMPLETE', iterations=plan['iterations'], seed=plan['seed'], test_inference_repeated=False,
        point_reproduction='PASS', intervals_are='Descriptive pointwise image-paired source-stratified percentile 95%.',
        primary=[c for c in contrasts if c['primary']][0],
        cumulative=[c for c in contrasts if c['candidate']==ARMS[3] and c['reference']==ARMS[0] and c['endpoint']=='Mask_AP50'][0],
        output_refinement=[c for c in contrasts if c['candidate']==ARMS[4] and c['endpoint']=='Mask_AP50_95'][0],
        no_missing_class_AP_draws=bool(np.isfinite(draws[:,:,:2]).all()),
        limitations=['Image-level pairing does not establish physical-fruit/batch independence.',
            'Early baseline test use and mixed annotation provenance remain.',
            'No multiplicity-adjusted tests or causal/commercial deployment conclusions.',
            '25 UQ test instances; zero new escape delta is not zero absolute escape risk.'],
        completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    write(OUT/'analysis_status.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__': main()
