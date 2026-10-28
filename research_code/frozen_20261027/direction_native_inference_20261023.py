"""Opt-in original-grid boundary module; prediction-only inputs, no GT dependency."""
import json
import cv2
import numpy as np
import torch
from direction_experiments_20261023 import STAGE,OUT,SIZE,BoundaryRefiner,crop_bounds,boundary_band


def prediction_records(image,masks,confidence):
    records=[]
    for j in np.flatnonzero(np.asarray(confidence)>=.25):
        bb=crop_bounds(masks[j])
        if bb is None:continue
        x0,y0,x1,y1=bb;rgb=cv2.cvtColor(image[y0:y1,x0:x1],cv2.COLOR_BGR2RGB)
        cm=cv2.resize(masks[j][y0:y1,x0:x1].astype(np.uint8),(SIZE,SIZE),interpolation=cv2.INTER_NEAREST)
        records.append({'prediction_index':int(j),'bounds':bb,
                        'rgb':torch.from_numpy(cv2.resize(rgb,(SIZE,SIZE)).transpose(2,0,1).copy()),
                        'coarse':torch.from_numpy(cm),'band':torch.from_numpy(boundary_band(cm))})
    return records


class NativeBoundary:
    def __init__(self):
        self.kind=json.loads((OUT/'boundary_selection.json').read_text(encoding='utf-8'))['selected']['kind']
        self.model=BoundaryRefiner().eval()
        self.model.load_state_dict(torch.load(STAGE/f'boundary_{self.kind}_seed20261023/best.pt',map_location='cpu',weights_only=False)['state'])

    @torch.inference_mode()
    def refine(self,image,masks,confidence):
        masks=[m.copy() for m in masks];rs=prediction_records(image,masks,confidence)
        if not rs:return masks
        x=torch.stack([torch.cat((r['rgb'],r['coarse'][None]*255)) for r in rs]).float()/255
        band=torch.stack([r['band'] for r in rs])[:,None].float()
        delta=torch.cat([self.model(x[ix],band[ix],self.kind)-(x[ix,3:4]*2-1)*2 for ix in torch.arange(len(rs)).split(32)]).numpy()[:,0]
        for r,d in zip(rs,delta):
            x0,y0,x1,y1=r['bounds'];roi=masks[r['prediction_index']][y0:y1,x0:x1]
            correction=cv2.resize(d,(x1-x0,y1-y0),interpolation=cv2.INTER_LINEAR)
            roi[:]=(((roi.astype(np.float32)*2-1)*2+correction)>0).astype(roi.dtype)
        return masks
