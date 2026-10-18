"""Check selected LoRA tensors against a metadata-only DiT skeleton."""
import argparse,json,hashlib
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--lora',type=Path,required=True);p.add_argument('--report',type=Path,required=True);args=p.parse_args()
    import torch
    from safetensors import safe_open
    from demoire_edit.infer import verify_lora
    from diffsynth.core import load_state_dict
    from diffsynth.models.qwen_image_dit import QwenImageDiT
    from diffsynth.utils.lora.general import GeneralLoRALoader
    verify_lora(args.lora)
    with safe_open(str(args.lora),framework='pt',device='cpu') as f:
        metadata_keys=sorted((f.metadata() or {}).keys())
    if any(k!='format' for k in metadata_keys):raise ValueError('Unreviewed checkpoint metadata keys')
    state=load_state_dict(str(args.lora),device='cpu')
    if len(state)!=1440 or any(v.dtype!=torch.bfloat16 for v in state.values()):raise ValueError('Unexpected selected LoRA tensor inventory')
    if any(not torch.isfinite(v).all() for v in state.values()):raise ValueError('Nonfinite LoRA tensor')
    converted=GeneralLoRALoader().convert_state_dict(state)
    with torch.device('meta'):model=QwenImageDiT()
    modules=dict(model.named_modules());matched=0
    for key,b in converted.items():
        if not key.endswith('.lora_B.weight'):continue
        name=key.removesuffix('.lora_B.weight');a=converted[name+'.lora_A.weight'];module=modules.get(name)
        if module is None or not isinstance(module,torch.nn.Linear):raise ValueError('LoRA target module does not exist')
        if a.ndim!=2 or b.ndim!=2 or a.shape[0]!=b.shape[1] or tuple(module.weight.shape)!=(b.shape[0],a.shape[1]):raise ValueError('LoRA target dimensions mismatch')
        matched+=1
    if matched!=720:raise ValueError('Unexpected LoRA adapter count')
    report={'status':'passed','tensor_count':len(state),'dtype':'BF16','matched_linear_adapters':matched,'metadata_keys':metadata_keys,'scope':'actual checkpoint tensors validated against a metadata-only DiT skeleton; full pretrained weight loading and GPU inference remain separate'}
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
