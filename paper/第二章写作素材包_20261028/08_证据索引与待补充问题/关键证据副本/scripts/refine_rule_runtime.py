"""Opt-in rule/learned-correction runtime; no production entry replacement.

Fast rule geometry crops only the irrelevant zero-mask background, then restores
GLOBAL contour coordinates before fitting axes. Frozen pixel/formula semantics
are preserved and tested against the full-image reference, including fallbacks.
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

import extract_features as extraction
import run_r0_fixed_prediction_replay as rules
import refine_rule_reliability as direct
import refine_rule_anchored_residual as anchor
from r1_research_runtime import ResearchGrader
from run_r1_ap_followup import policy, sha256


def fast_shape_features(image, mask, config):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("Direct rule has no valid contour")
    padding = max(12, math.ceil(float(config["geometry_refinement"]["max_boundary_depth_px"]))+2)
    x0, y0 = max(0, int(xs.min())-padding), max(0, int(ys.min())-padding)
    x2, y2 = min(image.shape[1], int(xs.max())+padding+1), min(image.shape[0], int(ys.max())+padding+1)
    cropped = (np.asarray(mask[y0:y2, x0:x2]) > 0).astype(np.uint8)*255
    analysis = config["analysis_mask"]
    cleaned = extraction.clean_analysis_mask(cropped, int(analysis["min_component_area_px"]),
                                              bool(analysis["keep_largest_component"]))[0]
    refined = extraction.refine_geometry_mask(image[y0:y2, x0:x2], cleaned, config["geometry_refinement"])[0]
    contour = extraction.main_contour(refined)
    if contour is None:
        raise ValueError("Direct rule has no valid contour")
    contour = contour.copy(); contour[:, :, 0] += x0; contour[:, :, 1] += y0
    perimeter = float(cv2.arcLength(contour, True)); area = float(cv2.contourArea(contour))
    hull = float(cv2.contourArea(cv2.convexHull(contour))); axes = extraction.fit_axes(contour)
    if perimeter <= 0 or hull <= 0:
        raise ValueError("Direct rule geometry degenerate")
    return {"area": float(np.count_nonzero(refined)), "minor_axis": float(axes["minor_axis"]),
            "aspect_ratio": float(axes["major_axis"]/axes["minor_axis"]),
            "circularity": 4*math.pi*area/perimeter**2, "solidity": area/hull}


class RuleGrader(ResearchGrader):
    def __init__(self, hybrid=False, fast=True):
        self.hybrid = hybrid; self.fast = fast
        rank = (anchor.BASE / "reliability/rule_base_score20261022/rank_seed20261008/best.pt" if hybrid
                else direct.BASE / "reliability/direct_rule/rank_seed20261008/best.pt")
        super().__init__(rank)
        self.rule_config = yaml.safe_load(direct.CFG.read_text(encoding="utf-8"))
        self.grading = json.loads(rules.DEFAULT_GRADING_CONFIG.read_text(encoding="utf-8"))
        self.references, _, _ = rules.recover_references_from_candidate_ecdf(
            pd.read_csv(rules.DEFAULT_OLD_CANDIDATES), rules.DEFAULT_REFERENCE_SIZE)
        if not hybrid:
            expected = self.reliability_checkpoint["context"]["plan"]["sources"]["refine_rule_reliability.py"]
            if expected != sha256(Path(direct.__file__)):
                raise ValueError("Direct rule runtime differs from fitted rule implementation")
        self.captured = []; self.handle = None; self.rule_fallbacks = 0
        if hybrid:
            path = anchor.BASE / "rule_base_residual_seed20261022/best.pt"
            cp = torch.load(path, map_location="cpu", weights_only=False)
            if cp["kind"] != "rule_base_residual" or cp["seed"] != 20261022:
                raise ValueError("Rule residual runtime only fixed seed22")
            if self.reliability_checkpoint["context"]["score_checkpoint_sha256"] != sha256(path):
                raise ValueError("Rule residual/ranking checkpoint mismatch")
            self.residual = anchor.Residual(cp["config"], cp["state"]["mean"], cp["state"]["std"])
            self.residual.load_state_dict(cp["state"]); self.residual.to(self.device).eval()
            self.handle = self.stem.fusion.register_forward_hook(self._capture)

    def _capture(self, module, args, value):
        self.captured.append(value.detach())

    @torch.inference_mode()
    def classify(self, image, masks, baseline, original_confidence):
        self.captured.clear()
        _, scores, stems = super().classify(image, masks, baseline)
        base = np.asarray(baseline); conf = np.asarray(original_confidence)
        eligible = np.flatnonzero((conf >= .25) & np.isfinite(scores) & np.isfinite(stems))
        features = []; indices = []
        function = fast_shape_features if self.fast else direct.shape_features
        for j in eligible:
            try:
                features.append(function(image, masks[j], self.rule_config)); indices.append(int(j))
            except ValueError:
                self.rule_fallbacks += 1
        if features:
            values = rules.apply_frozen_score(pd.DataFrame(features), self.references, self.grading)
            if not values.score_available.all():
                raise ValueError("Rule runtime scoring nonfinite")
            scores[indices] = values.rule_score.to_numpy(float)
        if self.hybrid:
            valid = np.flatnonzero(np.isfinite(stems))
            if len(valid):
                fusion = torch.cat(self.captured)
                if len(fusion) != len(valid):
                    raise ValueError("Rule residual fusion index mismatch")
                active = (base[valid] != 0) & (conf[valid] >= .25) & np.isfinite(scores[valid])
                ids = valid[active]
                if len(ids):
                    values = torch.tensor(scores[ids].astype(np.float32)/100, device=self.device)
                    idx = torch.tensor(np.flatnonzero(active), device=self.device)
                    scores[ids] = self.residual(fusion[idx], values)[0].cpu().numpy()*100
        self.captured.clear()
        return policy({"baseline": base, "conf": conf, "score": scores, "stem": stems}, True, True, True), scores, stems

    def close(self):
        if self.handle is not None:
            self.handle.remove(); self.handle = None
        self.captured.clear()
