"""Explicit opt-in research CLI; no annotation/cache lookup inside inference."""
from attribute_local_common_20261025 import *
setup()
import argparse,cv2,numpy as np,torch
from ultralytics.utils import LOGGER
from attribute_local_runtime_20261025 import RankedPipeline,AttributePipeline

def main():
    p=argparse.ArgumentParser();p.add_argument('--image',required=True);p.add_argument('--preset',choices=['RR22','selected-new-rank','old-defect-rank','exploratory-creation-rank'],default='old-defect-rank');p.add_argument('--refine-output-mask',action='store_true');p.add_argument('--reuse-prefix',action='store_true');p.add_argument('--output',default=str(OUT/'inference_examples'));args=p.parse_args();path=Path(args.image).resolve()
    if any(x.lower()=='test' for x in path.parts):raise ValueError('This experiment stage does not authorize final test evaluation')
    image=cv2.imread(str(path))
    if image is None:raise ValueError('Unreadable image')
    LOGGER.setLevel(40);mask='prototype' if args.refine_output_mask else 'original';checkpoint=None
    if args.preset=='RR22':pipe=AttributePipeline(output_mask=mask)
    else:
        if args.preset=='old-defect-rank':checkpoint=BASE/'attribute_rank_fits/RR22_original_defect_independent_multi_iou_seed20261025/best.pt'
        pipe=RankedPipeline('correctedcreation99' if args.preset=='exploratory-creation-rank' else 'RR22',mask,reuse_prefix=args.reuse_prefix,rank_checkpoint=checkpoint)
    try:result=pipe.infer(image)
    finally:pipe.close()
    directory=Path(args.output);directory.mkdir(parents=True,exist_ok=True);ranking=result.get('ranking_all',result['entry']['conf']);canvas=image.copy();colors=[(30,60,200),(150,110,10),(20,135,205),(160,75,130)];rows=[];saved=[]
    for j in result['keep_indices']:
        cls=int(result['labels_all'][j]);raw=float(result['entry']['conf'][j]);rows.append(dict(prediction_index=int(j),class_id=cls,raw_detector_confidence=raw,ranking_score=float(ranking[j]),quality_score=float(result['entry']['score'][j]) if np.isfinite(result['entry']['score'][j]) else None,xyxy=result['entry']['boxes'][j].tolist()))
        if raw>=.25:
            contours,_=cv2.findContours(result['output_masks'][j].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(canvas,contours,-1,colors[cls],1);x0,y0,x1,y1=map(int,result['entry']['boxes'][j]);cv2.rectangle(canvas,(x0,y0),(x1,y1),colors[cls],1);cv2.putText(canvas,f'C{cls} rank={ranking[j]:.2f}',(x0,max(14,y0-3)),cv2.FONT_HERSHEY_SIMPLEX,.4,colors[cls],1,cv2.LINE_AA);saved.append((int(j),np.packbits(result['output_masks'][j].astype(bool).reshape(-1))))
    save(directory/(path.stem+'_prediction.json'),dict(input_path=str(path),input_sha256=sha256(path),preset=args.preset,seed=20261025,output_mask=mask,reuse_prefix=args.reuse_prefix,raw_scope=.25,detector_pool=.001,ranking_is_not_calibrated_probability=True,research_only=True,primary_entry_replaced=False,predictions=rows,test_accessed=False));np.savez_compressed(directory/(path.stem+'_high_masks.npz'),shape=np.array(image.shape[:2]),prediction_indices=np.array([j for j,_ in saved],int),**{f'mask_{j}':bits for j,bits in saved});cv2.imwrite(str(directory/(path.stem+'_overlay.png')),canvas);print('Saved research inference:',directory,flush=True)

if __name__=='__main__':main()
