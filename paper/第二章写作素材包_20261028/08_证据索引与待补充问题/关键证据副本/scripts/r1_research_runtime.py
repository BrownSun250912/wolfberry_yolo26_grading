"""Opt-in inference for the frozen research candidate; never edits main models.

Detection confidence is calibrated AFTER unchanged detection/NMS. The original
confidence controls the .25 grading scope. Output score is TP ranking reliability,
not the fruit quality score, and not a claim of calibrated probability.
"""
from __future__ import annotations
import copy
import numpy as np
import torch
from torch import nn
from validate_r1_closure_runtime import Grader
from train_r1_confidence_followup import ReliabilityHead,confidences
from run_r1_ap_followup import EX,policy


class CompactGeometryScore(nn.Module):
    """Algebraically discard columns always multiplied by zero (no retraining)."""
    def __init__(self,source):
        super().__init__()
        if not source.use_geometry:raise ValueError('Requires geometry-only research checkpoint')
        self.geometry_encoder=copy.deepcopy(source.geometry_encoder)
        first=source.score_mlp[0]
        narrow=nn.Linear(16,first.out_features,device=first.weight.device,dtype=first.weight.dtype)
        with torch.no_grad():
            narrow.weight.copy_(first.weight[:,256:]);narrow.bias.copy_(first.bias)
        self.score_mlp=nn.Sequential(narrow,*[copy.deepcopy(layer) for layer in list(source.score_mlp)[1:]])

    def forward(self,unused_features,geometry):
        return torch.sigmoid(self.score_mlp(self.geometry_encoder(geometry)).squeeze(1))


class StemOnly(nn.Module):
    def __init__(self,source):
        super().__init__()
        self.visual=source.visual;self.geometry=source.geometry
        self.fusion=source.fusion;self.stem_head=source.stem_head

    def forward(self,image,geometry):
        features=self.fusion(torch.cat([self.visual(image),self.geometry(geometry)],dim=1))
        return {'stem_logit':self.stem_head(features).squeeze(1)}


class ResearchGrader(Grader):
    def __init__(self,checkpoint=None):
        super().__init__()
        self.score=CompactGeometryScore(self.score).to(self.device).eval()
        self.stem=StemOnly(self.stem).to(self.device).eval()
        path=checkpoint or EX/'mlp_confidence_seed20261008/best.pt'
        self.reliability_checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        if self.reliability_checkpoint['family']!='mlp_confidence' or self.reliability_checkpoint['seed']!=20261008:
            raise ValueError('Only predeclared fixed-seed MLP research candidate is allowed')
        self.reliability=ReliabilityHead('mlp_confidence')
        self.reliability.load_state_dict(self.reliability_checkpoint['head_state']);self.reliability.eval()

    @torch.inference_mode()
    def classify(self,image,masks,baseline):
        baseline=np.asarray(baseline,dtype=np.int64)
        if len(baseline)!=len(masks):raise ValueError('Mask and class count differ')
        # Keep the exact original batch composition. Skipping class0 would be
        # logically safe for grade gating, but changed convolution numerics by
        # 5.26e-5 in stem probability and failed the predeclared 1e-5 parity gate.
        return super().classify(image,masks,baseline)

    def finish(self,image,masks,baseline,original_confidence,boxes,scores,stems):
        entry={'baseline':np.asarray(baseline,dtype=np.int64),'conf':np.asarray(original_confidence),
            'score':np.asarray(scores),'stem':np.asarray(stems),'boxes':np.asarray(boxes),
            'mask_area':np.asarray([np.count_nonzero(m) for m in masks],np.float32),
            'image_height':image.shape[0],'image_width':image.shape[1]}
        # Class0 and <.25 grades remain detector grades. Reliability still ranks
        # low-confidence qualified predictions, as in the fixed train-only fit.
        labels=policy(entry,True,True,True)
        confidence=confidences(self.reliability,self.reliability_checkpoint,[entry],'product')[0]
        return labels,confidence
