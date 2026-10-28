"""Generate the paper result sheet once the frozen 200-epoch run completes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "models/experiments/yolo26n_seg_offlineaug_e200_mask50_seed20260915"
REPORT = ROOT / "docs/YOLO26_OFFLINEAUG_E200_RESULTS_20260918.md"
SUMMARY = RUN / "e200_evidence_summary.json"


def rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def inputs():
    result = rows(RUN / "results.csv")
    trace = rows(RUN / "mask50_selection_trace.csv")
    if len(result) != 200 or len(trace) != 200:
        return None
    expected = list(range(1, 201))
    if [int(r["epoch"]) for r in result] != expected or [int(t["epoch"]) for t in trace] != expected:
        raise ValueError("Missing or duplicated epochs")
    if not all((RUN / ("weights/" + name)).is_file() for name in ("best.pt", "last.pt")):
        return None
    return result, trace


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate(result: list[dict[str, str]], trace: list[dict[str, str]]) -> None:
    if REPORT.exists() or SUMMARY.exists():
        raise FileExistsError(f"Refusing to overwrite result artifact: {REPORT}")
    selected = [t for t in trace if t["is_new_best"] == "1"][-1]
    index = int(selected["epoch"]) - 1
    best, last = result[index], result[-1]
    if not all(float(t["selection_fitness"]) == float(r["metrics/mAP50(M)"]) for t, r in zip(trace, result)):
        raise ValueError("Checkpoint selection does not match recorded Mask mAP50")
    tail = [float(r["metrics/mAP50(M)"]) for r in result[-20:]]
    evidence = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "epochs": 200,
        "best_epoch": index + 1,
        "best_val_mask_map50": float(best["metrics/mAP50(M)"]),
        "best_val_mask_map50_95": float(best["metrics/mAP50-95(M)"]),
        "last_val_mask_map50": float(last["metrics/mAP50(M)"]),
        "last_val_mask_map50_95": float(last["metrics/mAP50-95(M)"]),
        "last20_mask_map50_mean": statistics.mean(tail),
        "last20_mask_map50_sd": statistics.stdev(tail),
        "last20_mask_map50_range": max(tail) - min(tail),
        "best_pt_sha256": fingerprint(RUN / "weights/best.pt"),
        "last_pt_sha256": fingerprint(RUN / "weights/last.pt"),
        "test_evaluated": False,
    }
    losses = ("train/box_loss", "train/seg_loss", "train/cls_loss", "val/box_loss", "val/seg_loss", "val/cls_loss")
    loss_lines = [f"| {key} | {float(result[0][key]):.5f} | {float(best[key]):.5f} | {float(last[key]):.5f} |" for key in losses]
    content = [
        "# 离线增强 200 轮冻结基线：实测结果", "",
        f"_由原始逐轮记录生成，时间 {evidence['generated_at']}；协议见 [冻结实验记录](YOLO26_OFFLINEAUG_E200_FROZEN_PROTOCOL_20260918.md)。_", "",
        "## 🧪 运行与选模", "",
        "- 实际完成 200 轮，固定种子 20260915；原始 val，未使用 test。",
        f"- 最佳轮次 {index + 1}：val Mask mAP50 = {evidence['best_val_mask_map50']:.5f}；Mask mAP50–95 = {evidence['best_val_mask_map50_95']:.5f}。",
        f"- 末轮 200：val Mask mAP50 = {evidence['last_val_mask_map50']:.5f}；Mask mAP50–95 = {evidence['last_val_mask_map50_95']:.5f}。",
        f"- 最后 20 轮 Mask mAP50 均值 {evidence['last20_mask_map50_mean']:.5f}，标准差 {evidence['last20_mask_map50_sd']:.5f}，极差 {evidence['last20_mask_map50_range']:.5f}。这些描述量不独自证明收敛。", "",
        "## 📉 损失轨迹", "",
        "| 项目 | 第 1 轮 | 最佳轮 | 第 200 轮 |", "| --- | ---: | ---: | ---: |", *loss_lines, "",
        "## 🔗 可追溯证据与局限", "",
        f"- [逐轮结果](../models/experiments/{RUN.name}/results.csv) · [Mask mAP50 选模轨迹](../models/experiments/{RUN.name}/mask50_selection_trace.csv) · [实际参数](../models/experiments/{RUN.name}/args.yaml) · [曲线图](../models/experiments/{RUN.name}/results.png)。",
        f"- best.pt SHA-256：{evidence['best_pt_sha256']}；last.pt SHA-256：{evidence['last_pt_sha256']}。",
        "- 仅报告 val 选模表现；这不是独立 test 结论。收敛与否应人工结合整条曲线、验证波动和误差实例判断。", "",
    ]
    REPORT.write_text("\n".join(content), encoding="utf-8")
    SUMMARY.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"RESULT WRITTEN: best epoch {index + 1}, Mask mAP50 {evidence['best_val_mask_map50']:.5f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true", help="Wait at most eight hours")
    args = parser.parse_args()
    deadline = time.monotonic() + (8 * 3600 if args.wait else 0)
    while True:
        ready = inputs()
        if ready is not None:
            if args.wait:  # allow the library's final best.pt validation/strip to finish
                time.sleep(120)
            generate(*ready)
            return
        if time.monotonic() >= deadline:
            raise RuntimeError("Training is not complete; no result was generated")
        time.sleep(60)


if __name__ == "__main__":
    main()
