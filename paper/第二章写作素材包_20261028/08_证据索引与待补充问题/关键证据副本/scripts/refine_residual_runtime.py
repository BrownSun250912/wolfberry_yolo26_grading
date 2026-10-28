"""Opt-in bounded residual inference using the EXISTING stem fusion pass.

No second visual encoder pass, no production model replacement. Raw detector
confidence (not calibrated reliability) defines the fixed .25 update scope.
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn

from r1_research_runtime import ResearchGrader
from run_r1_ap_followup import policy, sha256
from refine_score_direction import STAGE as SCORE_STAGE
from refine_score_ranking import BASE as RANK_STAGE


class FusionResidual(nn.Module):
    def __init__(self, cp):
        super().__init__()
        self.delta = nn.Sequential(nn.Linear(256,32),nn.SiLU(),nn.Linear(32,1))
        self.delta.load_state_dict({k[6:]:v for k,v in cp["state"].items() if k.startswith("delta.")})
        self.register_buffer("mean",cp["state"]["mean"].clone())
        self.register_buffer("std",cp["state"]["std"].clone())

    def forward(self, features, base):
        return (base+.05*torch.tanh(self.delta((features-self.mean)/self.std).squeeze(1))).clamp(0,1)


class ResidualGrader(ResearchGrader):
    def __init__(self, score_checkpoint=None, rank_checkpoint=None):
        score_checkpoint = Path(score_checkpoint or SCORE_STAGE / "control_seed20261022/best.pt")
        rank_checkpoint = Path(rank_checkpoint or RANK_STAGE / "control_score20261022/rank_seed20261008/best.pt")
        super().__init__(checkpoint=rank_checkpoint)
        cp = torch.load(score_checkpoint,map_location="cpu",weights_only=False)
        if cp["kind"] != "control" or cp["seed"] != 20261022:
            raise ValueError("This runtime is bound to fixed internally selected CONTROL seed22")
        expected = self.reliability_checkpoint["context"]["score_checkpoint_sha256"]
        if sha256(score_checkpoint) != expected:
            raise ValueError("Residual and reliability checkpoints do not belong together")
        self.residual = FusionResidual(cp).to(self.device).eval()
        self.captured = []
        self.capture_handle = self.stem.fusion.register_forward_hook(self._capture)

    def _capture(self, module, args, value):
        self.captured.append(value.detach())

    @torch.inference_mode()
    def classify(self,image,masks,baseline,original_confidence):
        self.captured.clear()
        _,scores,stems = super().classify(image,masks,baseline)
        valid = np.flatnonzero(np.isfinite(stems))
        if len(valid):
            features = torch.cat(self.captured)
            if len(features) != len(valid):
                raise ValueError("Runtime frozen-fusion prediction alignment differs")
            active = (np.asarray(baseline)[valid]!=0) & (np.asarray(original_confidence)[valid]>=.25) & np.isfinite(scores[valid])
            if active.any():
                tensor_index = torch.tensor(np.flatnonzero(active),device=self.device)
                ids = valid[active]
                base = torch.tensor(scores[ids].astype(np.float32)/100,device=self.device)
                scores[ids] = self.residual(features[tensor_index],base).cpu().numpy()*100
        self.captured.clear()
        entry = {"baseline":np.asarray(baseline),"conf":np.asarray(original_confidence),"score":scores,"stem":stems}
        return policy(entry,True,True,True),scores,stems

    def close(self):
        self.capture_handle.remove(); self.captured.clear()
