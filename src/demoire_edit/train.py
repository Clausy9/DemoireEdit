"""Train on an explicitly supplied, relative-path paired manifest."""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

def main():
    parser = argparse.ArgumentParser(description='DiT LoRA training with the maintained five-component objective')
    parser.add_argument('--manifest',type=Path,required=True,help='CSV with relative lq and gt paths')
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('outputs/training'))
    parser.add_argument('--model-paths',type=Path,required=True,help='JSON list of local model files or shard lists')
    parser.add_argument('--processor',type=Path,required=True)
    parser.add_argument('--initialize-lora',type=Path)
    parser.add_argument('--micro-batch',type=int,default=1)
    parser.add_argument('--accumulation',type=int,default=24)
    parser.add_argument('--steps',type=int,default=3000)
    parser.add_argument('--seed',type=int,default=2026)
    args = parser.parse_args()
    if min(args.micro_batch,args.accumulation,args.steps) <= 0: raise ValueError('Batch, accumulation and steps must be positive')
    import accelerate
    import torch
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image, ImageOps
    from accelerate.utils import set_seed
    from diffsynth.pipelines.qwen_image import ModelConfig
    from diffsynth.diffusion import ModelLogger, launch_training_task
    from .training import PaperTrainingModule
    import random
    set_seed(args.seed)
    root = args.data_root.resolve()
    rows = list(csv.DictReader(args.manifest.open(newline='')))
    if not rows: raise ValueError('Manifest contains no samples')
    def source_path(value):
        if Path(value).is_absolute(): raise ValueError('Manifest paths must be relative')
        p = (root / value).resolve()
        if not p.is_relative_to(root): raise ValueError('Manifest path escapes data root')
        if not p.is_file(): raise FileNotFoundError('Missing manifest image')
        return p
    pairs = [(source_path(row['lq']),source_path(row['gt'])) for row in rows]
    class Pairs(Dataset):
        def __len__(self): return len(pairs)
        def __getitem__(self,index):
            with Image.open(pairs[index][0]) as f: lq = ImageOps.exif_transpose(f).convert('RGB')
            with Image.open(pairs[index][1]) as f: gt = ImageOps.exif_transpose(f).convert('RGB')
            if lq.size != gt.size or min(lq.size) < 256: raise ValueError('Paired images must share dimensions and support a 256 crop')
            x=random.randrange(lq.width-256+1);y=random.randrange(lq.height-256+1)
            box=(x,y,x+256,y+256); lq=lq.crop(box);gt=gt.crop(box)
            for transform in [Image.Transpose.FLIP_LEFT_RIGHT,Image.Transpose.FLIP_TOP_BOTTOM]:
                if random.random()<.5: lq=lq.transpose(transform);gt=gt.transpose(transform)
            rotations=random.randrange(4)
            if rotations: lq=lq.rotate(rotations*90);gt=gt.rotate(rotations*90)
            # Match the selected training loader: LQ supplies the editing reference.
            return {'lq':lq,'edit_image':lq,'gt_image':gt}
    def collate(samples):
        if len(samples)==1: return samples[0]
        return {key:[sample[key] for sample in samples] for key in samples[0]}
    dataset=Pairs(); loader=DataLoader(dataset,batch_size=args.micro_batch,shuffle=True,num_workers=0,collate_fn=collate)
    accelerator=accelerate.Accelerator(gradient_accumulation_steps=args.accumulation,kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=True)])
    model_configs=[ModelConfig(path=path) for path in json.loads(args.model_paths.read_text())]
    model=PaperTrainingModule(model_configs,ModelConfig(path=str(args.processor)),args.initialize_lora)
    logger=ModelLogger(str(args.output),remove_prefix_in_ckpt='pipe.dit.')
    settings=SimpleNamespace(learning_rate=1e-4,weight_decay=.01,dataset_num_workers=0,save_steps=500,num_epochs=args.steps, max_train_steps=args.steps,lr_scheduler='cosine',lr_warmup_steps=0,lr_min_ratio=.001,resume_training_state=None,output_path=str(args.output))
    set_seed(args.seed,device_specific=True)
    launch_training_task(accelerator,dataset,model,logger,args=settings,dataloader=loader)

if __name__ == '__main__': main()
