"""Loopback-only Gradio interface for dual-scale restoration."""
import argparse
import os
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description='Local Dual-Scale Condition-Shift Editing interface')
    p.add_argument('--lora',type=Path,default=Path(os.environ.get('DEMOIRE_EDIT_WEIGHTS_PATH','weights/demoireedit.safetensors')))
    p.add_argument('--device',default='cuda')
    p.add_argument('--port',type=int,default=7860)
    args=p.parse_args()
    os.environ.setdefault('GRADIO_ANALYTICS_ENABLED','False')
    cache=Path(os.environ.get('GRADIO_TEMP_DIR','outputs/.demo-cache'))
    cache.mkdir(parents=True,exist_ok=True,mode=0o700)
    os.environ['GRADIO_TEMP_DIR']=str(cache.resolve())
    import gradio as gr
    from .infer import restore_image, verify_lora
    verify_lora(args.lora)
    def restore(image,seed):
        if image is None: raise gr.Error('Choose an input image first.')
        if image.width*image.height>40_000_000: raise gr.Error('Input exceeds 40 megapixels.')
        try:return restore_image(image,args.lora,args.device,int(seed),offload=True)
        except Exception:
            raise gr.Error('Restoration failed. Check available memory, model files and dependencies in the runtime.') from None
    with gr.Blocks(title='Dual-Scale Condition-Shift Editing',analytics_enabled=False) as app:
        gr.Markdown('# Dual-Scale Condition-Shift Editing\nDual-scale condition-shift restoration. Upload an image to run two-stage restoration. Results may retain color artifacts or soften detail.')
        with gr.Row():
            input_image=gr.Image(type='pil',label='Input photograph',format='png')
            output=gr.Image(type='pil',label='Restored image',format='png')
        seed=gr.Number(value=0,precision=0,label='Seed')
        button=gr.Button('Restore')
        button.click(restore,[input_image,seed],output,concurrency_limit=1)
    app.queue(max_size=4).launch(server_name='127.0.0.1',server_port=args.port,share=False,show_error=False)

if __name__=='__main__':main()
