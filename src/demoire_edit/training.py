"""Selected DiT LoRA training module; owner-selected objective."""
import torch
from diffsynth.diffusion import DiffusionTrainingModule
from diffsynth.pipelines.qwen_image import QwenImagePipeline
from diffsynth.utils.controlnet.controlnet_input import ControlNetInput

LORA_TARGETS = "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1"

class PaperTrainingModule(DiffusionTrainingModule):
    def __init__(self, model_configs, processor_config, lora_checkpoint=None):
        super().__init__()
        self.pipe = QwenImagePipeline.from_pretrained(torch_dtype=torch.bfloat16,device="cpu",model_configs=model_configs,tokenizer_config=None,processor_config=processor_config)
        self.switch_pipe_to_training_mode(self.pipe,trainable_models=None,lora_base_model="dit",lora_target_modules=LORA_TARGETS,lora_rank=32,lora_checkpoint=lora_checkpoint,task="sft")
        self.pipe.configure_demoire_loss_weights(velocity_weight=1.0,latent_mse_weight=1.0,l1_weight=.5,lpips_weight=.3,ffl_weight=.2)
        self.pipe._ensure_loss_models_on_device("cpu")
        self.use_gradient_checkpointing = True
        self.use_gradient_checkpointing_offload = False
        self.extra_inputs = ["edit_image", "gt_image"]
        self.zero_cond_t = False

    def forward(self, data):
        inputs = self.forward_preprocess(data)
        return self.pipe.demoire_training_loss(num_denoising_steps=5,**inputs)

    def forward_preprocess(self, data):
        FIXED_POSITIVE_PROMPT = "Remove moiré patterns, color aliasing and color distoration from the image while preserving all details, colors, and textures of the original content."
        FIXED_NEGATIVE_PROMPT = "blurry, low quality, artifacts, distorted, oversaturated, moiré, aliased"
        input_images = data["lq"]
        batch_size = len(input_images) if isinstance(input_images, list) else 1
        reference_image = input_images[0] if isinstance(input_images, list) else input_images
    
        # CFG-sensitive parameters
        inputs_posi = {"prompt": [FIXED_POSITIVE_PROMPT] * batch_size if batch_size > 1 else FIXED_POSITIVE_PROMPT}
        inputs_nega = {"negative_prompt": [FIXED_NEGATIVE_PROMPT] * batch_size if batch_size > 1 else FIXED_NEGATIVE_PROMPT}
        
        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_image": input_images,
            "edit_image": data["edit_image"],
            "gt_image": data["gt_image"],          # Clean 图像
            "height": reference_image.size[1],
            "width": reference_image.size[0],
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "edit_image_auto_resize": True,
            "zero_cond_t": self.zero_cond_t,  # Add zero_cond_t parameter for 2511
        }
        
        # Extra inputs
        controlnet_input, blockwise_controlnet_input = {}, {}
        for extra_input in self.extra_inputs:
            if extra_input.startswith("blockwise_controlnet_"):
                blockwise_controlnet_input[extra_input.replace("blockwise_controlnet_", "")] = data[extra_input]
            elif extra_input.startswith("controlnet_"):
                controlnet_input[extra_input.replace("controlnet_", "")] = data[extra_input]
            else:
                # print(data.keys())
                inputs_shared[extra_input] = data[extra_input]
        if len(controlnet_input) > 0:
            inputs_shared["controlnet_inputs"] = [ControlNetInput(**controlnet_input)]
        if len(blockwise_controlnet_input) > 0:
            inputs_shared["blockwise_controlnet_inputs"] = [ControlNetInput(**blockwise_controlnet_input)]
        
        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}

