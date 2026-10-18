"""Portable CLI for the selected dual-scale inference route."""
import argparse
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

EXPECTED_SHA256 = '15ad05bfb1b8c9d975605b18195abcf8485f28a26bb9b07dc350740a465319fd'

def verify_lora(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): digest.update(block)
    if path.stat().st_size != 472047184 or digest.hexdigest() != EXPECTED_SHA256:
        raise ValueError('LoRA size or SHA-256 does not match the selected checkpoint')
    return path

def parse_args():
    p = argparse.ArgumentParser(description='Dual-Scale Condition-Shift Editing inference')
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lora',type=Path,default=Path(os.environ.get('DEMOIRE_EDIT_WEIGHTS_PATH','weights/demoireedit.safetensors')))
    p.add_argument('--device',default='cuda')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--no-offload',action='store_true')
    p.add_argument('--debug',action='store_true')
    return p.parse_args()

def restore_image(image, lora, device='cuda', seed=0, offload=True, debug=False):
    from .inference_backend import StrategyConfig, build_pipeline, infer_one_progressive_2stage
    lora = verify_lora(lora)
    strategy = StrategyConfig('paper','Qwen/Qwen-Image-Edit-2509',lora,False)
    pipe = build_pipeline(strategy,device,vram_management=offload)
    args = SimpleNamespace(device=device,seed=seed,debug=debug,cfg_scale=1.0,
        stage1_cfg_scale=None,mid_steps=10,num_inference_steps=10,target_short_side=1024,
        stage1_ref_mode='black',progressive_edit_ratio=.3,tile_size=128,tile_stride=64)
    return infer_one_progressive_2stage(pipe,image.convert('RGB'),args,strategy)

def main():
    args = parse_args()
    if args.input.resolve() == args.output.resolve(): raise ValueError('Output must differ from input')
    from PIL import Image, ImageOps
    with Image.open(args.input) as image:
        if image.width * image.height > 40_000_000: raise ValueError('Input exceeds the 40 megapixel limit')
        image = ImageOps.exif_transpose(image).convert('RGB')
        output = restore_image(image,args.lora,args.device,args.seed,not args.no_offload,args.debug)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    output.save(args.output)

if __name__ == '__main__': main()
