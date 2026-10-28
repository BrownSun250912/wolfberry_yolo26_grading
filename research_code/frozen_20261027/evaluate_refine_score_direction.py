"""Developer evaluation only after train-held score family selection is saved."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from refine_score_direction import STAGE, KINDS, SEEDS, binding, load_original, DirectionScore
from run_refine_module_experiments import OUT
from cache_r1_encoder_inputs import load_inputs, decode
from run_r1_ap_followup import load_cache, TARGETS, save, sha256, policy, evaluate
import run_r1_ap_followup as harness
from analyze_r1_ap_bootstrap import pure_match
from analyze_val_instance_errors import optimal_match


def score_diagnostics(cache):
    target = pd.read_csv(TARGETS).set_index("instance_id")
    errors = []; confusion = np.zeros((4, 4), int); severe = adjacent = uq = 0
    for e in cache:
        labels = policy(e, True, True, True)
        high = np.flatnonzero(e["conf"] >= .25)
        for gi, pj, _ in optimal_match(e["mask_iou"][:, high], .5):
            j = int(high[pj]); a = int(e["gt_cls"][gi]); b = int(labels[j])
            confusion[a, b] += 1
            severe += int({a, b} == {1, 3}); adjacent += int(a>0 and b>0 and abs(a-b)==1); uq += int((a==0)!=(b==0))
            row = target.loc[e["gt_ids"][gi]]
            if int(row.score_target_available) == 1 and np.isfinite(e["score"][j]):
                errors.append(float(e["score"][j]) - float(row.score_target))
    per_class = [2*confusion[i, i]/max(int(confusion[i, :].sum()+confusion[:, i].sum()), 1) for i in range(4)]
    return {"score_MAE": float(np.mean(np.abs(errors))), "score_RMSE": float(np.sqrt(np.mean(np.square(errors)))),
            "score_n": len(errors), "score_abs_gt10": int((np.abs(errors)>10).sum()),
            "localized_macro_f1": float(np.mean(per_class)), "adjacent_errors": adjacent,
            "grade1_vs_grade3_errors": severe, "qualified_vs_unqualified_errors": uq,
            "localized_confusion": confusion.tolist()}


def main():
    torch.set_num_threads(4); harness.match_flags = pure_match
    harness.OUT = OUT / "score_direction_metrics"
    harness.OUT.mkdir(parents=True, exist_ok=True)
    choice_path = OUT / "score_direction_selection.json"
    if not choice_path.exists():
        raise ValueError("Internal selection must precede development evaluation")
    inputs = load_inputs("val"); cache = load_cache("val")
    original = load_original("cuda:0")
    cached = torch.load(STAGE / "train_prefix.pt", map_location="cpu", weights_only=False)
    if cached["binding"] != binding():
        raise ValueError("Score cache lineage drift")
    prefix_path = STAGE / "development_prefix.pt"
    if prefix_path.exists():
        prefixes = torch.load(prefix_path, map_location="cpu", weights_only=False)
        if prefixes["binding"] != binding():
            raise ValueError("Development prefix lineage drift")
        prefixes = prefixes["prefixes"]
    else:
        prefixes = []
        with torch.inference_mode():
            for item, e in zip(inputs, cache):
                if item["image_id"] != e["image_id"]:
                    raise ValueError("Development row mismatch")
                prefix = torch.cat([original.visual[:4](decode(batch).to("cuda:0")).cpu()
                                    for batch in item["image_uint8"].split(32)])
                prefixes.append(prefix)
        torch.save({"binding": binding(), "prefixes": prefixes}, prefix_path)
    baseline, _, _ = evaluate(cache, "original_frozen_R1_raw_ranking", [policy(e, True, True, True) for e in cache])
    baseline.update(score_diagnostics(cache))
    save(OUT / "score_direction_development_baseline.json", baseline)
    reports = []; classes = []
    for kind in KINDS:
        for seed in SEEDS:
            directory = STAGE / f"{kind}_seed{seed}"
            complete = json.loads((directory / "completion.json").read_text(encoding="utf-8"))
            if complete["budget_partial"]:
                raise ValueError("Budget partial fit not eligible for family comparison")
            path = directory / "best.pt"
            if sha256(path) != complete["checkpoint_sha256"]:
                raise ValueError("Score checkpoint drift")
            cp = torch.load(path, map_location="cpu", weights_only=False)
            model = DirectionScore(kind, original, cached["mean"], cached["std"]).to("cuda:0")
            model.load_state_dict(cp["state"]); model.eval()
            revised = []
            with torch.inference_mode():
                for e, item, prefix in zip(cache, inputs, prefixes):
                    values = e["score"].copy(); idx = item["prediction_indices"]
                    base = torch.tensor(np.nan_to_num(values[idx], nan=0).astype(np.float32)/100, device="cuda:0")
                    pred = torch.cat([model(prefix[i].to("cuda:0"), item["geometry"][i].to("cuda:0"), base[i])[0].cpu()
                                      for i in torch.arange(len(idx)).split(128)]).numpy()*100
                    use = (e["baseline"][idx] != 0) & (e["conf"][idx] >= .25) & np.isfinite(values[idx])
                    values[idx[use]] = pred[use]
                    revised.append(dict(e, score=values))
            report, class_rows, _ = evaluate(revised, f"{kind}_seed{seed}", [policy(e, True, True, True) for e in revised])
            report.update(score_diagnostics(revised)); report.update({"kind": kind, "seed": seed, "held_MAE": cp["held"]["MAE"]})
            reports.append(report); classes.extend(class_rows)
            pd.DataFrame(reports).to_csv(OUT / "score_direction_development.csv", index=False)
            pd.DataFrame(classes).to_csv(OUT / "score_direction_class_ap.csv", index=False)
            print(f"SCORE DEV {kind} seed{seed} correct={report['correct_500']} MAE={report['score_MAE']:.5f} AP50={report['metrics/mAP50(M)']:.7f}", flush=True)
            del model; torch.cuda.empty_cache()
    save(OUT / "score_direction_development_summary.json", {"baseline": baseline, "internal_selection": json.loads(choice_path.read_text(encoding="utf-8")),
         "rows": reports, "ranking": "Original YOLO confidences for every arm; NO old class-dependent calibration reuse",
         "new_fits": 0, "test_accessed": False, "primary_model_replaced": False})


if __name__ == "__main__":
    main()
