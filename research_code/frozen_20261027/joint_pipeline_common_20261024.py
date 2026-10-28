"""Isolated quota-bounded experiment stage for joint decisions and deduplication."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import time

ROOT=Path(__file__).resolve().parents[1]
NAME="joint_pipeline_20261024"
BASE=ROOT/"models/experiments"/NAME
OUT=ROOT/"outputs/reports"/NAME
DOC=ROOT/"docs/experiments"/NAME
RESET_AT=1792796516
START_PERCENT=27
START_CREDIT_BALANCE="549.3160820000"


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    os.replace(tmp,path)


def setup():
    for p in [BASE,OUT,DOC,OUT/"ultralytics_config"]:p.mkdir(parents=True,exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"]=str(OUT/"ultralytics_config")
    import torch
    import cv2
    torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True


def stop_requested(new_fit=False):
    if (BASE/"STOP_AFTER_CURRENT_EPOCH").exists():return True
    if time.time()>=RESET_AT-180:return True
    p=OUT/"quota_latest.json"
    if not p.exists():return True
    q=json.loads(p.read_text(encoding="utf-8"))
    if q["reset_at"]!=RESET_AT:return True
    if time.time()-q["recorded_epoch"]>1800:return True
    return q["used_percent"] >= (95 if new_fit else 98)


def plan():
    setup()
    from validate_r1_closure_runtime import DETECTOR,STEM,TARGETS
    inputs={"detector":DETECTOR,"stem":STEM,"targets":TARGETS,
       "head_roles":ROOT/"models/experiments/r1_quality_followup_20261009/head_image_split.csv",
       "ranking_roles":ROOT/"models/experiments/r1_ap_followup_20261008/calibration_image_split.csv",
       "review":ROOT/"outputs/reports/pipeline_breakthrough_audit_20261024/核验意见与集成计划.md"}
    value={"created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
       "authorization":"执行你的计划进行进一步的实验吧，直到5h额度跑完为止",
       "registered_after":"Independent audit already observed val452/500, duplicate suppression and unsafe UQ recovery; this is follow-up development, not preregistered independent confirmation.",
       "budget":{"starting_used_percent":START_PERCENT,"reset_at":RESET_AT,"credit_balance":START_CREDIT_BALANCE,
          "stop_new_fits_percent":95,"wrapup_percent":90,"stop_work_percent":98,"no_reset_credit_or_paid_spillover":True,
          "quota_file_max_age_s":1800},
       "sources":{k:{"path":str(p),"sha256":sha256(p)} for k,p in inputs.items()},
       "scope":"train and val only; original weights, labels, thresholds and production entry points remain available",
       "stages":[
          {"name":"P0","question":"Separate baseline duplicate suppression from new class decisions using physically retained predictions.",
           "scores":["original","selected22","direct_rule","rule_anchored22","rule_anchored23","rule_anchored24"],
           "confidence":["raw","own_rank08"],"dedup":["none","same_class_raw_or_rank"],
           "additional_controls":["raw_YOLO","scope_priority_NMS prevents low-raw candidate displacing high-raw candidate"],
           "statistics":"Fixed old500 labels diagnostic separately from post-keep rawconf>=.25 rematching, all518GT AP; no zero-confidence fake deletion"},
          {"name":"P1","question":"Compare same-candidate raw four-class evidence with neighbourhood candidate aggregation.",
           "capture":"Same Segment26 forward before top-k, retain candidate index, score vector and exact public-result row lineage; GPU FP32 original preprocessing",
           "evidence":["top1_only","raw_same_candidate","neighbour_iou085_max"],
           "fallback":"Retain original label if score/stem/evidence invalid; no GT determines online eligibility"},
          {"name":"P2","question":"Separate boundary, stem fusion and class0 recovery, with full cost/error reporting.",
           "fixed_parameters":{"alpha_rec":.5,"alpha_stem":.5,"lambda_b":1.5,"sigma_s":2.5,"temperature":3.,"recover_defect_logit":-3.,"neighbour_iou":.85,"nms_iou":.7},
           "recipes":["standard","boundary","stem","boundary_stem","recovery","full"],
           "held_candidate":{"alpha_rec":.35,"alpha_stem":.5,"lambda_b":2.},
           "primary_protection":"No default promotion of UQ recovery if it creates extra true-UQ escapes; all gains/losses and subgroup counts retained.",
           "possible_fits":"Only after separate saved amendment: train-only low-dimensional fusion or new-decision ranker, matched architecture/dose and three fixed seeds; never rank-transplant and call calibrated."},
          {"name":"P3","question":"Check normal and predefined photometric/background failure conditions, raw-runtime parity and paired complete latency.",
           "conditions":["clean","dark070","colorcast","background_colorcast","foreground_colorcast","foreground_dark070","background_neutral160"],
           "timing":"Same confidence/max_det/device/precision/disk scope, warmup and randomized arm order; report components and total, no inherited old latency points"}],
       "splits":"Original train773/193 head roles and original ranking split; original detector/R1 saw original train, so internal held is not a full-system independent set.",
       "selection":"All val results exploratory; later changes saved as amendments before evaluation; never choose best val seed or add per-instance exceptions.",
       "report":"Durable Chinese report with all conditions, negative results, reproduction and limitations; fit counts distinct from deterministic evaluations.",
       "test_accessed":False,"primary_replaced":False}
    p=BASE/"plan.json"
    if p.exists():return json.loads(p.read_text(encoding="utf-8"))
    save(p,value)
    save(OUT/"quota_latest.json",{"used_percent":START_PERCENT,"reset_at":RESET_AT,"recorded_epoch":time.time(),"credit_balance":START_CREDIT_BALANCE})
    save(OUT/"execution_status.json",{"status":"registered","stage":"P0","new_fits":0,"test_accessed":False})
    return value


if __name__=="__main__":plan()
