"""ECA-inspired identity-initialized adapter; published ECA is prior art.

Q. Wang et al., ECA-Net, CVPR2020/arXiv1910.03151. We adapt the gate to
2*sigmoid(z), with zero convolution weights, to preserve a trained model at
installation. This scaled/identity initialization is not the canonical gate.
"""
import math
import torch
from torch import nn
from ultralytics.nn.modules.head import Segment26
from ultralytics.models.yolo.segment.train import SegmentationTrainer


class IdentityECA(nn.Module):
    def __init__(self,channels):
        super().__init__();size=int(abs((math.log2(channels)+1)/2));size=size if size%2 else size+1
        self.channels=channels;self.conv=nn.Conv1d(1,1,size,padding=size//2,bias=False);nn.init.zeros_(self.conv.weight)
    def forward(self,x):
        if x.shape[1]!=self.channels:raise ValueError('ECA channel mismatch')
        pooled=x.mean((2,3)).unsqueeze(1);gate=2*torch.sigmoid(self.conv(pooled)).squeeze(1)[:,:,None,None]
        return x*gate


class IdentityECASegment26(Segment26):
    def forward(self,x):
        if len(x)!=len(self.identity_eca):raise ValueError('ECA feature-level count differs')
        return super().forward([module(value) for module,value in zip(self.identity_eca,x)])


def install(model):
    head=model.model[-1]
    if not isinstance(head,Segment26):raise ValueError('Expected frozen YOLO26 Segment26')
    if isinstance(head,IdentityECASegment26):return model
    # Zero initialization must not shift paired-control loader/random streams.
    state=torch.get_rng_state()
    head.__class__=IdentityECASegment26;head.identity_eca=nn.ModuleList([IdentityECA(c) for c in [64,128,256]])
    torch.set_rng_state(state)
    return model


class NativeMask50Trainer(SegmentationTrainer):
    use_attention=False
    def get_model(self,cfg=None,weights=None,verbose=True):
        model=super().get_model(cfg=cfg,weights=None,verbose=verbose)
        if self.use_attention:install(model)
        if weights:model.load(weights)
        return model

    def validate(self):
        previous=self.best_fitness;metrics,_=super().validate()
        if metrics is None:raise ValueError('Native validation missing')
        fitness=float(metrics['metrics/mAP50(M)']);self.best_fitness=fitness if previous is None else max(previous,fitness)
        return metrics,fitness
