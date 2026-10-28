"""Paired timing of complete ranked graphs, with matched recomputation control."""
from attribute_local_common_20261025 import *
setup()
import cv2,numpy as np,pandas as pd,torch
from ultralytics.utils import LOGGER
from attribute_local_runtime_20261025 import AttributePipeline,RankedPipeline
from joint_pipeline_inference_20261024 import JointPipeline
from joint_pipeline_decision_matrix_20261024 import joined_sources

def main():
    plan();LOGGER.setLevel(40);sources,_=joined_sources('val');es=sources['rule_anchored22'];chosen=[]
    for source in sorted({e['source_group'] for e in es}):chosen.extend([e for e in es if e['source_group']==source][:6])
    images=[(e['image_id'],cv2.imread(e['image_path'])) for e in chosen];old=BASE/'attribute_rank_fits/RR22_original_defect_independent_multi_iou_seed20261025/best.pt';specs=[('raw_YOLO_API',lambda:JointPipeline('raw_YOLO','standard',dedup='none')),('lean_RR22',lambda:JointPipeline('rule_anchored22','boundary_stem')),('shared_reference',lambda:AttributePipeline()),('newrank_strict_replay',lambda:RankedPipeline()),('newrank_global_recompute',lambda:RankedPipeline(recompute_global=True)),('newrank_prefix_reuse',lambda:RankedPipeline(reuse_prefix=True)),('newrank_proto_reuse',lambda:RankedPipeline(output_mask='prototype',reuse_prefix=True)),('olddefect_rank',lambda:RankedPipeline(rank_checkpoint=old)),('olddefect_rank_proto',lambda:RankedPipeline(rank_checkpoint=old,output_mask='prototype')),('selected_creation_rank',lambda:RankedPipeline('correctedcreation99'))]
    p=BASE/'rank_latency_plan.json'
    if not p.exists():save(p,dict(registered_utc=dt.datetime.now(dt.timezone.utc).isoformat(),reason='Benchmark new finalrank graphs and originaldefectcontrol. Strict interleavedcache replay encodes unusedlocalviews for batchidentity, so speedup versusstrict replay alone cannot be attributed tofeature reuse; add single-global recomputation control.',
      images=[i for i,_ in images],repeats=3,arms=[n for n,_ in specs],protocol='FP32 conf.001 maxdet300 retina masks; all10graphs loaded, warmup3images each; fixedseed25 randomgraphorder perpairedimage/repeat; cuda sync before/after, sourceimages predecoded. Actual fullROI geometry/refinement, grader, features, rank, masks andNMS included. No other experimentworker while timed.',
      units='Image-cluster bootstrap3000 on withinimage repeatmean. RawYOLO API includes normal publicprediction formatting/review flags, not claimed minimalengine time. Originalimage decode/model loading excluded, timing comparable onlysameprotocol/device.',test_accessed=False,new_fits=0))
    pipelines={};rows=[]
    try:
        for name,create in specs:
            pipelines[name]=create()
            for _,image in images[:3]:pipelines[name].infer(image)
        rng=np.random.default_rng(20261025)
        for repeat in range(3):
            for pos,(iid,image) in enumerate(images):
                names=list(pipelines);rng.shuffle(names)
                for name in names:
                    if stop_requested():raise RuntimeError('Budgetguard')
                    torch.cuda.synchronize();start=time.perf_counter();result=pipelines[name].infer(image);torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000;rows.append(dict(arm=name,image_id=iid,repeat=repeat,milliseconds=ms,predictions=len(result['labels_all']),retained=len(result['keep_indices'])))
                if (pos+1)%6==0:print('RANK LATENCY',repeat,pos+1,'/',len(images),flush=True)
            pd.DataFrame(rows).to_csv(OUT/'rank_latency_samples.csv',index=False)
    finally:
        for pipe in pipelines.values():pipe.close()
    f=pd.DataFrame(rows);summary=f.groupby('arm').milliseconds.agg(['mean','median','std','count',lambda x:x.quantile(.95)]).rename(columns={'<lambda_0>':'p95'}).reset_index();summary['FPS_from_mean']=1000/summary['mean'];summary.to_csv(OUT/'rank_latency_summary.csv',index=False);m=f.groupby(['image_id','arm']).milliseconds.mean().unstack('arm');intervals=[];rng=np.random.default_rng(20261025)
    for reference in ['lean_RR22','shared_reference','newrank_global_recompute','newrank_strict_replay']:
        for name in m:
            if name==reference:continue
            delta=(m[name]-m[reference]).to_numpy();draws=np.array([delta[rng.integers(len(delta),size=len(delta))].mean() for _ in range(3000)]);intervals.append(dict(arm=name,reference=reference,mean_added_ms=float(delta.mean()),ci95_low=float(np.quantile(draws,.025)),ci95_high=float(np.quantile(draws,.975)),images=len(m)))
    pd.DataFrame(intervals).to_csv(OUT/'rank_latency_intervals.csv',index=False);save(OUT/'rank_latency_status.json',dict(status='completed',images=len(images),repeats=3,arms=len(pipelines),rows=len(f),device=torch.cuda.get_device_name(),torch=torch.__version__,test_accessed=False));print(summary.to_string(index=False),flush=True)

if __name__=='__main__':main()
