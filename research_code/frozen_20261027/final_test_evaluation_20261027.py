"""Replay frozen five-arm evaluation on val; final test requires an explicit opt-in.

No fitting, threshold selection, or test-dependent choice is performed here.
All arms reuse the same detector forward. This is an accuracy evaluation, not a
per-arm latency benchmark. Existing research-stage test guards stay unchanged.
"""
from pathlib import Path
from collections import defaultdict
import argparse, csv, datetime as dt, hashlib, json, os, sys, time, traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/reports/final_test_preflight_20261027'
FROZEN = OUT / 'FROZEN_INPUTS.json'
PROTOCOL = ROOT / 'outputs/reports/attribute_local_20261025/FROZEN_CANDIDATES_AND_TEST_PROTOCOL.json'
ARMS = ['raw_YOLO', 'raw_YOLO_realNMS', 'RR22_reference',
        'original_defect_rank', 'original_defect_rank_prototype']
FLOAT_TOLERANCE = 1e-4  # Matches the previously verified runtime AP tolerance.
EXACT_COLUMNS = ['GT', 'localized', 'correct_grade', 'predictions_before',
                 'predictions_after', 'high_before', 'high_after',
                 'class_aware_TP50', 'UQ_GT', 'UQ_localized', 'UQ_correct',
                 'UQ_escape', 'qualified_rejected', 'new_UQ_escape',
                 'UQ_any_qualified_emission']
FLOAT_COLUMNS = ['Mask_AP50', 'Mask_AP50_95', 'conditional_grade_accuracy',
                 'correct_per_all_GT', 'operating_macro_F1_rawscope',
                 'operating_micro_F1_rawscope', 'operating_macro_recall_rawscope']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def sources():
    """Record every loaded project Python module, this entry, and native-mask code."""
    paths = {Path(__file__).resolve(), ROOT / '.venv/Lib/site-packages/ultralytics/utils/ops.py'}
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            path = Path(name).resolve()
            if path.suffix == '.py' and (ROOT / 'scripts') in path.parents: paths.add(path)
    return {str(p): sha(p) for p in sorted(paths)}


def check_hashes(items):
    for item in items:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Frozen input changed: ' + item['path'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--execute-final-test', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.split == 'test' and not args.execute_final_test:
        parser.error('Test inference is disabled without --execute-final-test and a passed val replay.')
    if args.split == 'val' and args.execute_final_test:
        parser.error('--execute-final-test is only valid for --split test.')
    audit = json.loads((OUT / 'data_preflight.json').read_text(encoding='utf-8'))
    if audit['status'] != 'PASS_DATA_AND_FROZEN_ARTIFACTS': raise ValueError('Data preflight did not pass')
    frozen = json.loads(FROZEN.read_text(encoding='utf-8'))
    if frozen['arms'] != ARMS: raise ValueError('Frozen arms changed')
    check_hashes(frozen['artifacts'] + frozen['metadata'])
    selected = [r for r in frozen['images'] if r['split'] == args.split]
    for item in selected:
        if sha(item['image_path']) != item['image_sha256'] or sha(item['label_path']) != item['label_sha256']:
            raise ValueError('Image/label changed: ' + item['image_id'])
    if len(selected) != {'val': 116, 'test': 129}[args.split]: raise ValueError('Incomplete split')
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    reference = ROOT / 'outputs/reports/attribute_local_20261025/headline_results.csv'
    # Old experiment setup() is called by imports. Redirect settings output immediately.
    os.environ['YOLO_CONFIG_DIR'] = str(OUT / 'ultralytics_config')
    import cv2, numpy as np, torch
    from attribute_local_runtime_20261025 import RankedPipeline
    from analyze_val_instance_errors import Instance, read_yolo_instances, mask_iou_matrix
    from joint_pipeline_policy_20261024 import keep_indices
    from joint_pipeline_evaluate_20261024 import evaluate
    from analyze_r1_ap_bootstrap import pure_match
    os.environ['YOLO_CONFIG_DIR'] = str(OUT / 'ultralytics_config')
    torch.set_num_threads(4); cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    if not torch.cuda.is_available(): raise RuntimeError('Frozen GPU runtime is unavailable')
    # Import the lazily loaded rank feature module before the source snapshot.
    import attribute_local_rank_20261025
    code_before = sources()
    old_critical = [reference, PROTOCOL, PROTOCOL.parent / 'SEALED.json']
    old_critical = {str(p): sha(p) for p in old_critical if p.is_file()}
    if args.split == 'test':
        gate = json.loads((OUT / 'VAL_REPLAY_PASS.json').read_text(encoding='utf-8'))
        if gate['frozen_inputs_sha256'] != sha(FROZEN): raise ValueError('Val gate belongs to other inputs')
        if gate['source_hashes'] != code_before: raise ValueError('Code changed since passed val replay')
        if sha(gate['report_path']) != gate['report_sha256']: raise ValueError('Val report changed')
    dest = (args.output or OUT / (args.split + '_replay')).resolve()
    if OUT.resolve() not in dest.parents: raise ValueError('Output must be under the preflight report directory')
    dest.mkdir(exist_ok=False, parents=True)
    (dest / 'predictions').mkdir()
    if args.split == 'test':
        with (OUT / 'FINAL_TEST_STARTED.json').open('x', encoding='utf-8') as stream:
            json.dump(dict(started_utc=dt.datetime.now(dt.timezone.utc).isoformat(), output=str(dest)), stream)
    label_metadata = defaultdict(list)
    for r in rows(ROOT / 'data/grading_four_class_v3/audit/label_manifest.csv'):
        if r['dataset_split'] == args.split: label_metadata[r['image_id']].append(r)
    pipeline = None
    try:
        pipeline = RankedPipeline(output_mask='prototype', seed=20261025,
                                  rank_checkpoint=protocol['weights_and_sources']['old_defect_rank']['path'])
        entries = []; refined_entries = []; labels = []; ranks = []; mask_hashes = []
        start = time.perf_counter()
        for number, item in enumerate(selected, 1):
            image = cv2.imdecode(np.fromfile(item['image_path'], dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None or list(image.shape[:2]) != item['shape']: raise ValueError('Image decode/shape changed')
            # Inference happens before the GT is loaded; no target reaches the predictor.
            prediction = pipeline.infer(image)
            e = prediction['entry']; e['image_id'] = item['image_id']
            gt = read_yolo_instances(Path(item['label_path']), *image.shape[:2])
            meta = sorted(label_metadata[item['image_id']], key=lambda r: int(r['output_label_row']))
            if len(gt) != len(meta): raise ValueError('GT lineage changed')
            e.update(gt_cls=np.array([g.class_id for g in gt], dtype=int),
                     gt_ids=[r['instance_id'] for r in meta], source_group=item['source_group'])
            def ious(masks):
                return mask_iou_matrix(gt, [Instance(class_id=int(c), mask=m) for c, m in zip(e['baseline'], masks)])
            e['mask_iou'] = ious(prediction['grading_masks'])
            refined = dict(e, mask_iou=ious(prediction['output_masks']))
            ls = np.asarray(prediction['labels_all'], int); rank = np.asarray(prediction['ranking_all'])
            keep = keep_indices(e['boxes'], ls, rank, .7, e['conf'] >= .25)
            if not np.array_equal(keep, prediction['keep_indices']): raise ValueError('Online/evaluation NMS mismatch')
            if not np.array_equal(rank[ls == 0], e['conf'][ls == 0]): raise ValueError('UQ confidence changed')
            if not (len(ls) == len(rank) == len(e['baseline']) == len(prediction['grading_masks']) == len(prediction['output_masks'])):
                raise ValueError('Prediction row alignment changed')
            hashes = {name: [hashlib.sha256(np.packbits(mask).tobytes()).hexdigest() for mask in prediction[key]]
                      for name, key in [('grading', 'grading_masks'), ('output', 'output_masks')]}
            cache = dict(entry=e, refined_mask_iou=refined['mask_iou'], labels=ls, confidence=rank,
                         keep_indices=keep, mask_sha256=hashes, image_sha256=item['image_sha256'],
                         label_sha256=item['label_sha256'])
            torch.save(cache, dest / 'predictions' / (item['image_id'] + '.pt'))
            entries.append(e); refined_entries.append(refined); labels.append(ls); ranks.append(rank)
            mask_hashes.append(dict(image_id=item['image_id'], **hashes))
            if number == 1 or number % 10 == 0 or number == len(selected):
                print('REPLAY', args.split, number, '/', len(selected), 'elapsed_s', round(time.perf_counter()-start, 1), flush=True)
        pipeline.close(); pipeline = None
        specs = [
            (ARMS[0], entries, [e['baseline'] for e in entries], None, 'none'),
            (ARMS[1], entries, [e['baseline'] for e in entries], None, 'scope_priority'),
            (ARMS[2], entries, labels, None, 'scope_priority'),
            (ARMS[3], entries, labels, ranks, 'scope_priority'),
            (ARMS[4], refined_entries, labels, ranks, 'scope_priority')]
        results = []; details = {}; safe = set()
        for name, es, ls, scores, mode in specs:
            rank = [e['conf'] for e in es] if scores is None else scores
            metrics, cm, cases, blocks = evaluate(es, ls, rank, mode, case_rows=True)
            if name == 'RR22_reference': safe = {r['instance_id'] for r in cases if r['gt_class'] == r['pred_class'] == 0}
            tp = np.zeros(4, int); pred = np.zeros(4, int); gtcounts = np.zeros(4, int); escape = 0
            for e, cls, conf in zip(es, ls, rank):
                keep = np.arange(len(cls)) if mode == 'none' else keep_indices(e['boxes'], cls, conf, .7, e['conf'] >= .25)
                high = keep[e['conf'][keep] >= .25]
                flags = pure_match(cls[high], e['gt_cls'], e['mask_iou'][:, high])[:, 0]
                for c in range(4):
                    tp[c] += int(flags[cls[high] == c].sum()); pred[c] += int((cls[high] == c).sum())
                    gtcounts[c] += int((e['gt_cls'] == c).sum())
                qualified = high[cls[high] > 0]
                escape += int((e['mask_iou'][e['gt_cls'] == 0][:, qualified] >= .5).any(1).sum())
            metrics.update(arm=name, operating_macro_F1_rawscope=float((2 * tp / np.maximum(pred + gtcounts, 1)).mean()),
                operating_micro_F1_rawscope=float(2 * tp.sum() / max(pred.sum() + gtcounts.sum(), 1)),
                operating_macro_recall_rawscope=float((tp / np.maximum(gtcounts, 1)).mean()), UQ_any_qualified_emission=escape)
            details[name] = dict(cases=cases, confusion_gt_row_pred_col=cm.tolist(),
                                 per_class_TP=tp.tolist(), per_class_predictions=pred.tolist(), per_class_GT=gtcounts.tolist())
            results.append(metrics)
            torch.save(dict(blocks=blocks, metrics=metrics, cases=cases), dest / (name + '_evaluation.pt'))
        for result in results:
            result['new_UQ_escape'] = sum(c['instance_id'] in safe and c['pred_class'] != 0
                for c in details[result['arm']]['cases'] if c['gt_class'] == 0)
        code_after = sources()
        if code_after != code_before: raise ValueError('Loaded source set or contents changed during evaluation')
        for path, before in old_critical.items():
            if sha(path) != before: raise ValueError('Sealed output changed: ' + path)
        check_hashes(frozen['artifacts'] + frozen['metadata'])
        comparisons = []
        if args.split == 'val':
            expected = {r['arm']: r for r in rows(reference)}
            for actual in results:
                for column in EXACT_COLUMNS + FLOAT_COLUMNS:
                    delta = abs(float(actual[column]) - float(expected[actual['arm']][column]))
                    tolerance = 0 if column in EXACT_COLUMNS else FLOAT_TOLERANCE
                    comparisons.append(dict(arm=actual['arm'], metric=column, actual=actual[column],
                        expected=float(expected[actual['arm']][column]), absolute_delta=delta,
                        tolerance=tolerance, passed=delta <= tolerance))
            passed = all(c['passed'] for c in comparisons)
        else: passed = True
        import pandas as pd
        pd.DataFrame(results).to_csv(dest / 'metrics.csv', index=False)
        pd.DataFrame(comparisons).to_csv(dest / 'val_parity_checks.csv', index=False)
        write(dest / 'details.json', details); write(dest / 'mask_hashes.json', mask_hashes)
        report = dict(status='PASS' if passed else 'FAIL_METRIC_PARITY', split=args.split,
            images=len(selected), arms=ARMS, metrics=results, comparison_count=len(comparisons),
            failed_comparisons=[c for c in comparisons if not c['passed']],
            floating_metric_tolerance=FLOAT_TOLERANCE, exact_counters_required=True,
            frozen_inputs_sha256=sha(FROZEN), source_hashes=code_before, sealed_sources_unchanged=True,
            test_model_inference_runs=1 if args.split == 'test' else 0,
            test_performance_metrics_computed=args.split == 'test',
            inference='One shared detector forward per image after one initial predictor warmup; all five frozen arms derived. No per-arm latency claims.',
            GT='Fresh file-based GT polygons; validation-only overlay for val, original v3 for test.',
            timing_note='Elapsed wall time includes GT I/O, mask IoU, checkpoint/cache writes and warmup. Not publication latency.',
            score_MAE='Unavailable: no independently verified test score target manifest in frozen protocol.',
            runtime=dict(python=sys.version, torch=torch.__version__, GPU=torch.cuda.get_device_name(0),
                         tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32),
            generated_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write(dest / 'replay_report.json', report)
        if passed and args.split == 'val':
            write(OUT / 'VAL_REPLAY_PASS.json', dict(report_path=str(dest / 'replay_report.json'),
                report_sha256=sha(dest / 'replay_report.json'), frozen_inputs_sha256=sha(FROZEN), source_hashes=code_before))
        print(json.dumps(dict(status=report['status'], split=args.split, images=len(selected),
            failed_comparisons=report['failed_comparisons']), ensure_ascii=False, indent=2), flush=True)
        if not passed: raise RuntimeError('Validation parity did not pass; see preserved report')
    except Exception:
        write(dest / 'failure.json', dict(error=traceback.format_exc(), split=args.split,
            test_model_inference_attempted=args.split == 'test', generated_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        raise
    finally:
        if pipeline is not None: pipeline.close()


if __name__ == '__main__': main()
