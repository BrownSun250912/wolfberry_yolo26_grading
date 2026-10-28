"""Opt-in shared single-encoder runtime for joint grading research candidates."""
from joint_pipeline_common_20261024 import ROOT
import numpy as np
import torch
from torch import nn
from r1_research_runtime import ResearchGrader
from refine_residual_runtime import FusionResidual
from refine_rule_runtime import fast_shape_features
import refine_rule_reliability as direct
import refine_rule_anchored_residual as anchor
from refine_score_direction import STAGE as SCORE
from validate_r1_closure_runtime import STEM


class SharedGrader(ResearchGrader):
    def __init__(self,rule_sources=True):
        super().__init__()
        self.captured=[]
        self.handle=self.stem.fusion.register_forward_hook(lambda module,args,value:self.captured.append(value.detach()))
        cp=torch.load(SCORE/"control_seed20261022/best.pt",map_location="cpu",weights_only=False)
        self.learned=FusionResidual(cp).to(self.device).eval()
        cp_stem=torch.load(STEM,map_location="cpu",weights_only=False)
        self.defect=nn.Linear(256,1).to(self.device).eval()
        self.defect.load_state_dict({"weight":cp_stem["model_state"]["defect_head.weight"],"bias":cp_stem["model_state"]["defect_head.bias"]})
        self.rule_sources=rule_sources
        if rule_sources:
            import json,yaml,pandas as pd
            import run_r0_fixed_prediction_replay as rules
            self.rules=rules
            self.rule_config=yaml.safe_load(direct.CFG.read_text(encoding="utf-8"))
            self.grading=json.loads(rules.DEFAULT_GRADING_CONFIG.read_text(encoding="utf-8"))
            self.references,_,_=rules.recover_references_from_candidate_ecdf(pd.read_csv(rules.DEFAULT_OLD_CANDIDATES),rules.DEFAULT_REFERENCE_SIZE)
            cp=torch.load(anchor.BASE/"rule_base_residual_seed20261022/best.pt",map_location="cpu",weights_only=False)
            self.anchored=anchor.Residual(cp["config"],cp["state"]["mean"],cp["state"]["std"]).to(self.device).eval()
            self.anchored.load_state_dict(cp["state"])
        self.last={};self.rule_fallbacks=0

    @torch.inference_mode()
    def infer_sources(self,image,masks,baseline,confidence,return_features=False):
        self.captured.clear()
        _,original,stem=super().classify(image,masks,baseline)
        valid=np.flatnonzero(np.isfinite(stem));n=len(baseline)
        features=torch.cat(self.captured) if self.captured else torch.empty((0,256),device=self.device)
        if len(features)!=len(valid):raise ValueError("Fusion/crop lineage changed")
        defect=np.full(n,np.nan)
        if len(valid):defect[valid]=self.defect(features).squeeze(1).cpu().numpy()
        scores={"original":original,"selected22":original.copy()}
        active=(np.asarray(baseline)[valid]!=0)&(np.asarray(confidence)[valid]>=.25)&np.isfinite(original[valid])
        ids=valid[active];tensor_idx=torch.tensor(np.flatnonzero(active),device=self.device)
        if len(ids):
            base=torch.tensor(original[ids].astype(np.float32)/100,device=self.device)
            scores["selected22"][ids]=self.learned(features[tensor_idx],base).cpu().numpy()*100
        if self.rule_sources:
            import pandas as pd
            rule=original.copy();geometry=[];rule_ids=[]
            eligible=np.flatnonzero((np.asarray(confidence)>=.25)&np.isfinite(original)&np.isfinite(stem))
            for j in eligible:
                try:geometry.append(fast_shape_features(image,masks[j],self.rule_config));rule_ids.append(int(j))
                except ValueError:self.rule_fallbacks+=1
            if geometry:
                values=self.rules.apply_frozen_score(pd.DataFrame(geometry),self.references,self.grading)
                if not values.score_available.all():raise ValueError("Nonfinite rule score")
                rule[rule_ids]=values.rule_score.to_numpy(float)
            scores["direct_rule"]=rule;scores["rule_anchored22"]=rule.copy()
            if len(ids):
                base=torch.tensor(rule[ids].astype(np.float32)/100,device=self.device)
                scores["rule_anchored22"][ids]=self.anchored(features[tensor_idx],base)[0].cpu().numpy()*100
        result={"score_sources":scores,"stem":stem,"defect_logit":defect}
        if return_features:
            ff=np.zeros((n,256),np.float32);ff[valid]=features.cpu().numpy();result["features"]=ff
        self.captured.clear();self.last=result
        return result

    def close(self):
        self.handle.remove();self.captured.clear()
