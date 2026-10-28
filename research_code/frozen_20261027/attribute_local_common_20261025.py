"""New window, isolated attribute experiments; previous stages remain sealed."""
from pathlib import Path
import datetime as dt
import hashlib,json,os,time
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"models/experiments/attribute_local_20261025"
OUT=ROOT/"outputs/reports/attribute_local_20261025"
DOC=ROOT/"docs/experiments/attribute_local_20261025"
RESET=1792880841
SEEDS=[20261025,20261026,20261027]

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8");os.replace(tmp,path)

def setup():
    for p in [BASE,OUT,DOC,OUT/"ultralytics_config"]:p.mkdir(parents=True,exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"]=str(OUT/"ultralytics_config")
    import torch,cv2
    torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True

def stop_requested(new_fit=False):
    qpath=OUT/"quota_latest.json"
    if (BASE/"STOP_AFTER_CURRENT_EPOCH").exists() or time.time()>=RESET-180 or not qpath.exists():return True
    q=json.loads(qpath.read_text(encoding="utf-8"))
    return q["reset_at"]!=RESET or time.time()-q["recorded_epoch"]>1800 or q["used_percent"] >= (95 if new_fit else 98)

def plan():
    setup();p=BASE/"plan.json"
    if p.exists():return json.loads(p.read_text(encoding="utf-8"))
    from validate_r1_closure_runtime import STEM,DETECTOR,TARGETS
    inputs={"attribute_encoder":STEM,"detector":DETECTOR,"targets":TARGETS,"roles":ROOT/"models/experiments/r1_quality_followup_20261009/head_image_split.csv"}
    value={"date":"2026-10-25","registered_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"authorization":"你给我做相关实验吧，依旧跑完5h额度",
      "budget":{"starting_used_percent":2,"reset_at":RESET,"credit_balance":"549.3160820000","new_fit_stop":95,"wrapup":90,"stop":98,"no_reset_or_paid_spillover":True},
      "question":"Does local image information and last-block visual adaptation improve defect/stem attributes and final grading beyond frozen representations?",
      "sources":{k:{"path":str(v),"sha256":sha256(v)} for k,v in inputs.items()},
      "factors":{"input":["global_global","global_local"],"visual_tail":["frozen","finetune"]},"seeds":SEEDS,"primary_fits":12,
      "architecture":"Pretrained R1 visual prefix blocks0..3 frozen; block4 optional adaptation; shared encoder two views; average256dim visual vectors, original geometry32 and fusion256; same allocated defect/stem heads. Control repeats global input, local uses tighter predicted-mask crop. Both2encoding passes, sameparameters, optional tail trainability is explicit factor.",
      "crop":"Global padmax(12,.25longestside),local padmax(4,.08longestside); mask and rawRGB cropped from predicted mask bounds, neverGT. Globalcontext remains to retain pedicel beyond tightcrop.",
      "supervision":"Frozen train detections rawconf>=.25, class-agnostic optimal maskIoU>=.5 GT pairing. Defect and available missing-stem targets from original audited labels; no synthesized spatial annotations. Unmatched predictions remain evaluated but not assigned false labels for fitting.",
      "roles":"Original773head_fit/193head_holdout image roles. All instances and views stay together. Original frozen encoder saw all originaltrain; held isolates only new fitting, not whole-system unseen data.",
      "normalization":"Original pretrained geometry normalization preserved; no new val normalization. FP32 prefix cache is exact-computation storage, not a new image sample.",
      "dose":{"max_epochs":40,"patience":10,"batch":128,"head_lr":.001,"tail_lr":.00003,"weight_decay":.0001,"BN":"All running statistics frozen/eval; fusion dropout off in allarms","loss":"Fit-only class-balanced BCE for defect + available stem; same examples/order/init perseed across4arms"},
      "checkpoint":"Minimum sum of class-balanced held defect/stem BCE; epoch0 included. No val checkpoint choices.",
      "pipeline":"Keep original detector, masks, score22/rr22 values and raw confidence fixed. New attribute tail separate from legacy score feature path; no reuse of adapted features in an old scorehead without validation. Actual added cost measured later.",
      "selection":"After12fits evaluate held same-real-NMS. Fixed thresholds defect.5 and stem.5 first; policies replace_stem,replace_defect,replace_both; additional default boundary+neighbour uses fixed oldparameters. Full3seed family mean correct/allGT>=rr22reference, UQescape<=reference, qualified_rejected<=reference, prioritize AP95 then correct/allGT then less adaptation. If none passes retain oldreference. Fixedseed25; allval explored only after familychoice, reportallseeds.",
      "later":"Amend before subsequent runs: held-only threshold calibration if necessary, photometric stress on original7+3combinations, independent boundary direction if evidence supports, runtime parity/timing and report",
      "primary_replaced":False,"test_accessed":False,"limits":"Adaptive development after extensive reused-val knowledge; no external generalization or causal physical-fruit independence claim; seed repetitions not new datasets."}
    save(p,value);return value

if __name__=="__main__":plan()
