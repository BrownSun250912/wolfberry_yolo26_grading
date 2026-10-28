"""Explicit seeded shuffle for installed Ultralytics InfiniteDataLoader."""
import torch
from pathlib import Path
from yolo26_identity_eca import NativeMask50Trainer
from run_r1_ap_followup import save,sha256


def seed_training_loader(loader,seed):
    if not isinstance(loader.sampler,torch.utils.data.RandomSampler):raise ValueError('Expected single-device shuffled random sampler')
    generator=torch.Generator().manual_seed(int(seed))
    loader.generator=generator;loader.sampler.generator=generator
    # InfiniteDataLoader already created an iterator/prefetch during __init__.
    # Its installed reset() closes old workers and creates a fresh iterator.
    loader.reset()
    return loader


class SeededMask50Trainer(NativeMask50Trainer):
    def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode='train'):
        loader=super().get_dataloader(dataset_path,batch_size,rank,mode)
        if mode=='train':
            seed_training_loader(loader,self.args.seed)
            self._sample_trace=[]
            save(Path(self.save_dir)/'sampler_context.json',{'seed':int(self.args.seed),'source_sha256':sha256(Path(__file__)),
                'loader':'Installed InfiniteDataLoader; RandomSampler and worker generator explicitly share seed; reset closes prefetched fixed-seed iterator',
                'sampler':type(loader.sampler).__name__,'rank':rank,'validation_sampler_not_changed':True})
        return loader

    def preprocess_batch(self,batch):
        if hasattr(self,'_sample_trace') and len(self._sample_trace)<32:
            self._sample_trace.extend([Path(p).name for p in batch['im_file']])
            save(Path(self.save_dir)/'first32_training_images.json',{'seed':int(self.args.seed),'images':self._sample_trace[:32],
                'captured_actual_preprocessed_training_order':True,'test_accessed':False})
        return super().preprocess_batch(batch)


class SeededECAHeadTrainer(SeededMask50Trainer):
    use_attention=True
