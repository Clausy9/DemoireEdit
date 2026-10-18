import json
import os
import torch, math
from PIL import Image
from typing import Union
from tqdm import tqdm
from einops import rearrange
import numpy as np
from math import prod

from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig, gradient_checkpoint_forward
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit, ControlNetInput
from ..utils.lora.merge import merge_lora

from ..models.qwen_image_dit import QwenImageDiT
from ..models.qwen_image_text_encoder import QwenImageTextEncoder
from ..models.qwen_image_vae import QwenImageVAE
from ..models.qwen_image_controlnet import QwenImageBlockWiseControlNet
try:
    from ..models.siglip2_image_encoder import Siglip2ImageEncoder
except Exception:
    Siglip2ImageEncoder = None

try:
    from ..models.dinov3_image_encoder import DINOv3ImageEncoder
except Exception:
    DINOv3ImageEncoder = None

try:
    from ..models.qwen_image_image2lora import QwenImageImage2LoRAModel
except Exception:
    QwenImageImage2LoRAModel = None

try:
    from focal_frequency_loss import FocalFrequencyLoss as FFL
    FFL_SOURCE = "package"
except Exception:
    try:
        from ..utils.focal_frequency_loss_compat import FocalFrequencyLoss as FFL
        FFL_SOURCE = "compat"
    except Exception:
        FFL = None
        FFL_SOURCE = None


class QwenImagePipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16,
        )
        from transformers import Qwen2Tokenizer, Qwen2VLProcessor
        
        self.scheduler = FlowMatchScheduler("Qwen-Image")
        self.text_encoder: QwenImageTextEncoder = None
        self.dit: QwenImageDiT = None
        self.vae: QwenImageVAE = None
        self.blockwise_controlnet: QwenImageBlockwiseMultiControlNet = None
        self.tokenizer: Qwen2Tokenizer = None
        self.siglip2_image_encoder: Siglip2ImageEncoder = None
        self.dinov3_image_encoder: DINOv3ImageEncoder = None
        self.image2lora_style: QwenImageImage2LoRAModel = None
        self.image2lora_coarse: QwenImageImage2LoRAModel = None
        self.image2lora_fine: QwenImageImage2LoRAModel = None
        self.processor: Qwen2VLProcessor = None
        self.in_iteration_models = ("dit", "blockwise_controlnet")
        self.units = [
            QwenImageUnit_ShapeChecker(),
            QwenImageUnit_NoiseInitializer(),
            QwenImageUnit_InputImageEmbedder(),
            QwenImageUnit_Inpaint(),
            QwenImageUnit_EditImageEmbedder(),
            QwenImageUnit_CleanImageEmbedder(),  # For demoire training: encode GT clean image
            QwenImageUnit_LayerInputImageEmbedder(),
            QwenImageUnit_ContextImageEmbedder(),
            QwenImageUnit_PromptEmbedder(),
            QwenImageUnit_EntityControl(),
            QwenImageUnit_BlockwiseControlNet(),
        ]
        self.model_fn = model_fn_qwen_image

        # Loss networks are training-only and initialized lazily.
        self.lpips_loss_fn = None
        self.ffl = None
        self.velocity_weight = 1.0
        self.latent_mse_weight = 1.0
        self.l1_weight = 0.5
        self.lpips_weight = 0.3
        self.ffl_weight = 0.2

        # Loss printing control
        self.print_loss = True
        self.print_loss_interval = 1
        self.print_loss_step = 0
        self.loss_print_format = "detailed"
        self.latest_loss_info = None
    

    def configure_demoire_loss_weights(self, velocity_weight=1.0, latent_mse_weight=1.0, l1_weight=0.5, lpips_weight=0.3, ffl_weight=0.2):
        for name, value in locals().copy().items():
            if name == "self":
                continue
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Loss weights must be finite and positive")
            setattr(self, name, float(value))





    
    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
        processor_config: ModelConfig = None,
        vram_limit: float = None,
    ):
        # Initialize pipeline
        pipe = QwenImagePipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        
        # Fetch models
        pipe.text_encoder = model_pool.fetch_model("qwen_image_text_encoder")
        pipe.dit = model_pool.fetch_model("qwen_image_dit")
        pipe.vae = model_pool.fetch_model("qwen_image_vae")
        pipe.blockwise_controlnet = QwenImageBlockwiseMultiControlNet(model_pool.fetch_model("qwen_image_blockwise_controlnet", index="all"))
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            from transformers import Qwen2Tokenizer
            pipe.tokenizer = Qwen2Tokenizer.from_pretrained(tokenizer_config.path)
        if processor_config is not None:
            processor_config.download_if_necessary()
            from transformers import Qwen2VLProcessor
            pipe.processor = Qwen2VLProcessor.from_pretrained(processor_config.path)
        pipe.siglip2_image_encoder = model_pool.fetch_model("siglip2_image_encoder")
        pipe.dinov3_image_encoder = model_pool.fetch_model("dinov3_image_encoder")
        pipe.image2lora_style = model_pool.fetch_model("qwen_image_image2lora_style")
        pipe.image2lora_coarse = model_pool.fetch_model("qwen_image_image2lora_coarse")
        pipe.image2lora_fine = model_pool.fetch_model("qwen_image_image2lora_fine")
        
        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe

    def enable_frozen_vae_gate(self):
        if self.vae is None:
            raise ValueError("VAE is not initialized.")
        self.vae.enable_input_gated_decode()

    def _prepare_image_batch_tensor(self, images, device=None, dtype=None):
        if images is None:
            return None
        device = self.device if device is None else device
        dtype = self.torch_dtype if dtype is None else dtype
        if isinstance(images, torch.Tensor):
            return images.to(device=device, dtype=dtype)
        if isinstance(images, Image.Image):
            images = [images]
        if isinstance(images, list):
            tensors = []
            for image in images:
                if isinstance(image, torch.Tensor):
                    tensor = image.to(device=device, dtype=dtype)
                    if tensor.dim() == 3:
                        tensor = tensor.unsqueeze(0)
                else:
                    tensor = self.preprocess_image(image).to(device=device, dtype=dtype)
                tensors.append(tensor)
            return torch.cat(tensors, dim=0)
        raise TypeError(f"Unsupported image batch type: {type(images)}")

    def prepare_vae_guidance_cache(self, images, device=None, dtype=None):
        if self.vae is None or not getattr(self.vae, "gated_decode_enabled", False):
            return None
        image_tensor = self._prepare_image_batch_tensor(images, device=device, dtype=dtype)
        if image_tensor is None:
            return None
        return self.vae.prepare_decode_guidance(image_tensor)
    
    
    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: str = "",
        cfg_scale: float = 4.0,
        # Image
        input_image: Image.Image = None,
        denoising_strength: float = 1.0,
        # Inpaint
        inpaint_mask: Image.Image = None,
        inpaint_blur_size: int = None,
        inpaint_blur_sigma: float = None,
        # Shape
        height: int = 1328,
        width: int = 1328,
        # Randomness
        seed: int = None,
        rand_device: str = "cpu",
        # Steps
        num_inference_steps: int = 30,
        exponential_shift_mu: float = None,
        enable_fp8_attention: bool = False,
        # Blockwise ControlNet
        blockwise_controlnet_inputs: list[ControlNetInput] = None,
        # EliGen
        eligen_entity_prompts: list[str] = None,
        eligen_entity_masks: list[Image.Image] = None,
        eligen_enable_on_negative: bool = False,
        # Qwen-Image-Edit
        edit_image: Image.Image = None,
        edit_image_auto_resize: bool = True,
        edit_rope_interpolation: bool = False,
        # Qwen-Image-Edit-2511
        zero_cond_t: bool = False,
        # Qwen-Image-Layered
        layer_input_image: Image.Image = None,
        layer_num: int = None,
        # In-context control
        context_image: Image.Image = None,
        # Tile
        tiled: bool = False,
        tile_size: int = 128,
        tile_stride: int = 64,
        # Progress bar
        progress_bar_cmd = tqdm,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, dynamic_shift_len=(height // 16) * (width // 16), exponential_shift_mu=exponential_shift_mu)
        
        # Parameters
        inputs_posi = {
            "prompt": prompt,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
        }
        inputs_shared = {
            "cfg_scale": cfg_scale,
            "input_image": input_image, "denoising_strength": denoising_strength,
            "inpaint_mask": inpaint_mask, "inpaint_blur_size": inpaint_blur_size, "inpaint_blur_sigma": inpaint_blur_sigma,
            "height": height, "width": width,
            "seed": seed, "rand_device": rand_device,
            "num_inference_steps": num_inference_steps,
            "enable_fp8_attention": enable_fp8_attention,
            "blockwise_controlnet_inputs": blockwise_controlnet_inputs,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "eligen_entity_prompts": eligen_entity_prompts, "eligen_entity_masks": eligen_entity_masks, "eligen_enable_on_negative": eligen_enable_on_negative,
            "edit_image": edit_image, "edit_image_auto_resize": edit_image_auto_resize, "edit_rope_interpolation": edit_rope_interpolation, 
            "context_image": context_image,
            "zero_cond_t": zero_cond_t,
            "layer_input_image": layer_input_image,
            "layer_num": layer_num,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            noise_pred = self.cfg_guided_model_fn(
                self.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            if not torch.isfinite(noise_pred).all():
                raise FloatingPointError("Nonfinite Stage 1 noise prediction")
            inputs_shared["latents"] = self.step(self.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared)
            if not torch.isfinite(inputs_shared["latents"]).all():
                raise FloatingPointError("Nonfinite Stage 1 latent")
        
        # Decode
        self.load_models_to_device(['vae'])
        guidance_images = input_image if input_image is not None else edit_image
        if isinstance(guidance_images, list) and len(guidance_images) > 1:
            guidance_images = guidance_images[:1]
        guidance_cache = self.prepare_vae_guidance_cache(guidance_images, device=self.device, dtype=self.torch_dtype)
        image = self.vae.decode(
            inputs_shared["latents"],
            guidance_cache=guidance_cache,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        if not torch.isfinite(image).all():
            raise FloatingPointError("Nonfinite Stage 1 VAE output")
        if layer_num is None:
            image = self.vae_output_to_image(image)
        else:
            image = [self.vae_output_to_image(i, pattern="C H W") for i in image]
        self.load_models_to_device([])

        return image

    def _ensure_loss_models_on_device(self, target_device):
        if self.lpips_loss_fn is None:
            import lpips
            self.lpips_loss_fn = lpips.LPIPS(net="vgg").eval()
            self.lpips_loss_fn.requires_grad_(False)
        if self.ffl is None:
            if FFL is None:
                raise RuntimeError("Focal frequency loss is required for the paper objective")
            self.ffl = FFL(loss_weight=1.0, alpha=1.0)
        self.lpips_loss_fn.to(target_device)
        self.ffl.to(target_device)


    def _debug_tensor(self, name, tensor, verbose=False):
        """Debug tensor, check for NaN/Inf"""
        if tensor is None:
            return f"{name}: None"
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        if has_nan or has_inf or verbose:
            info = f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}"
            info += f", min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, mean={tensor.float().mean().item():.4f}"
            info += f", NaN={torch.isnan(tensor).sum().item()}, Inf={torch.isinf(tensor).sum().item()}"
            return info
        return None

    def _print_loss(self, loss_type, loss_dict, inputs):
        self.latest_loss_info = {"type": loss_type, "values": dict(loss_dict)}
        self.print_loss_step += 1
        if self.print_loss and self.print_loss_step % self.print_loss_interval == 0:
            print(json.dumps(self.latest_loss_info, sort_keys=True))


    def demoire_training_loss(self, num_denoising_steps=5, **inputs):
        """
        Demoire specialized training loss:
        1. Add noise to the clean target latent at the initial training timestep
        2. Denoise under moire prior guidance for specified steps
        3. Use clean image as supervision target

        Args:
            num_denoising_steps: Number of denoising steps, default 5
            inputs: Contains input_latents (lq), edit_latents (moire), target_latents (clean)
        """
        # 0. Get actual computation device (inferred from input data)
        lq_latents = inputs["input_latents"]
        clean_latents = inputs["target_latents"]
        actual_device = lq_latents.device

        # Ensure auxiliary loss models are on correct device
        self._ensure_loss_models_on_device(actual_device)

        # Debug: Check inputs for NaN
        debug_info = self._debug_tensor("input_latents", lq_latents)
        if debug_info:
            print(f"[DEBUG] {debug_info}")
        debug_info = self._debug_tensor("target_latents", clean_latents)
        if debug_info:
            print(f"[DEBUG] {debug_info}")

        # 1. Set scheduler to use only specified number of steps
        self.scheduler.set_timesteps(num_denoising_steps, denoising_strength=1.0)

        # 2. Add noise to LQ latents (corresponding to first step noise level)
        # Select first timestep (maximum noise) - use actual device not self.device
        timestep = self.scheduler.timesteps[0].unsqueeze(0).to(
            dtype=self.torch_dtype, device=actual_device
        )

        # Generate noise
        noise = torch.randn_like(lq_latents)
        trace_limit = int(os.environ.get("DIFFSYNTH_TRACE_BATCHES", "0"))
        trace_count = getattr(self, "_demoire_noise_trace_count", 0)
        if trace_count < trace_limit:
            flat_noise = noise.detach().float().reshape(-1)
            record = {
                "rank": int(os.environ.get("RANK", "0")),
                "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
                "trace_index": trace_count,
                "torch_initial_seed": int(torch.initial_seed()),
                "noise_mean": float(flat_noise.mean().cpu()),
                "noise_std": float(flat_noise.std().cpu()),
                "noise_prefix": flat_noise[:8].cpu().tolist(),
            }
            print(f"[trajectory-noise] {json.dumps(record, sort_keys=True)}", flush=True)
            self._demoire_noise_trace_count = trace_count + 1

        # Add noise: noisy_lq = lq + noise * sigma
        noisy_lq_latents = self.scheduler.add_noise(clean_latents, noise, timestep)

        # Debug: Check noisy latents
        debug_info = self._debug_tensor("noisy_lq_latents", noisy_lq_latents)
        if debug_info:
            print(f"[DEBUG] {debug_info}")

        # 3. Execute denoising process
        inputs["latents"] = noisy_lq_latents
        models = {name: getattr(self, name) for name in self.in_iteration_models}

        velocity_target = self.scheduler.training_target(clean_latents, noise, timestep)
        velocity_losses = []

        for progress_id, timestep in enumerate(self.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=actual_device)

            # Model prediction
            noise_pred = self.model_fn(
                **models, **inputs,
                timestep=timestep,
                progress_id=progress_id
            )

            # Debug: Check noise_pred
            debug_info = self._debug_tensor(f"noise_pred[step={progress_id}]", noise_pred)
            if debug_info:
                print(f"[DEBUG] {debug_info}")

            velocity_losses.append(torch.nn.functional.mse_loss(
                noise_pred.float(),
                velocity_target.float()
            ))

            # Denoise one step
            inputs["latents"] = self.step(
                self.scheduler,
                progress_id=progress_id,
                noise_pred=noise_pred,
                **inputs
            )

        # 4. Compute loss against clean latents
        denoised_latents = inputs["latents"]

        # Debug: Check denoised_latents
        debug_info = self._debug_tensor("denoised_latents", denoised_latents)
        if debug_info:
            print(f"[DEBUG] {debug_info}")

        # 4.1 Velocity loss across the K-step trajectory
        velocity_loss = torch.stack(velocity_losses).mean()

        gt_images = self._prepare_image_batch_tensor(inputs["gt_image"], device=actual_device, dtype=self.torch_dtype).clamp(-1, 1)
        denoised_images = self.vae.decode(denoised_latents).clamp(-1, 1)
        from .demoire_objective import endpoint_objective
        total_loss, terms = endpoint_objective(
            velocity_loss, denoised_images, gt_images, denoised_latents, clean_latents,
            self.lpips_loss_fn, self.ffl,
            velocity=self.velocity_weight, latent_mse=self.latent_mse_weight, l1=self.l1_weight,
            lpips_weight=self.lpips_weight, ffl_weight=self.ffl_weight,
        )
        self._print_loss("Demoire", {name: value.detach().item() for name, value in terms.items()}, inputs)
        return total_loss


    def training_loss(self, **inputs):
        timestep_id = torch.randint(0, self.scheduler.num_train_timesteps, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)

        noise = torch.randn_like(inputs["input_latents"])
        inputs["latents"] = self.scheduler.add_noise(inputs["input_latents"], noise, timestep)
        training_target = self.scheduler.training_target(inputs["input_latents"], noise, timestep)

        noise_pred = self.model_fn(**inputs, timestep=timestep)

        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.scheduler.training_weight(timestep)

        # Print loss
        loss_dict = {
            'total_loss': loss.item(),
            'mse_loss': loss.item(),
            'timestep': timestep.item() if isinstance(timestep, torch.Tensor) else timestep,
            'weight': self.scheduler.training_weight(timestep).item() if hasattr(self.scheduler, 'training_weight') else 1.0
        }
        self._print_loss("Training", loss_dict, inputs)

        return loss


class QwenImageBlockwiseMultiControlNet(torch.nn.Module):
    def __init__(self, models: list[QwenImageBlockWiseControlNet]):
        super().__init__()
        if not isinstance(models, list):
            models = [models]
        self.models = torch.nn.ModuleList(models)
        for model in models:
            if hasattr(model, "vram_management_enabled") and getattr(model, "vram_management_enabled"):
                self.vram_management_enabled = True

    def preprocess(self, controlnet_inputs: list[ControlNetInput], conditionings: list[torch.Tensor], **kwargs):
        processed_conditionings = []
        for controlnet_input, conditioning in zip(controlnet_inputs, conditionings):
            conditioning = rearrange(conditioning, "B C (H P) (W Q) -> B (H W) (C P Q)", P=2, Q=2)
            model_output = self.models[controlnet_input.controlnet_id].process_controlnet_conditioning(conditioning)
            processed_conditionings.append(model_output)
        return processed_conditionings

    def blockwise_forward(self, image, conditionings: list[torch.Tensor], controlnet_inputs: list[ControlNetInput], progress_id, num_inference_steps, block_id, **kwargs):
        res = 0
        for controlnet_input, conditioning in zip(controlnet_inputs, conditionings):
            progress = (num_inference_steps - 1 - progress_id) / max(num_inference_steps - 1, 1)
            if progress > controlnet_input.start + (1e-4) or progress < controlnet_input.end - (1e-4):
                continue
            model_output = self.models[controlnet_input.controlnet_id].blockwise_forward(image, conditioning, block_id)
            res = res + model_output * controlnet_input.scale
        return res


class QwenImageUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width"),
            output_params=("height", "width"),
        )

    def process(self, pipe: QwenImagePipeline, height, width):
        height, width = pipe.check_resize_height_width(height, width)
        return {"height": height, "width": width}



class QwenImageUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "seed", "rand_device", "layer_num", "input_image"),
            output_params=("noise",),
        )

    def process(self, pipe: QwenImagePipeline, height, width, seed, rand_device, layer_num, input_image):
        batch_size = len(input_image) if isinstance(input_image, list) else 1
        if layer_num is None:
            noise = pipe.generate_noise((batch_size, 16, height//8, width//8), seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype)
        else:
            noise = pipe.generate_noise((batch_size * (layer_num + 1), 16, height//8, width//8), seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype)
        return {"noise": noise}



class QwenImageUnit_InputImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "noise", "tiled", "tile_size", "tile_stride"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: QwenImagePipeline, input_image, noise, tiled, tile_size, tile_stride):
        if input_image is None:
            return {"latents": noise, "input_latents": None}
        pipe.load_models_to_device(['vae'])
        if isinstance(input_image, list):
            input_latents = []
            for image in input_image:
                image = pipe.preprocess_image(image).to(device=pipe.device, dtype=pipe.torch_dtype)
                input_latents.append(pipe.vae.encode(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride))
            input_latents = torch.concat(input_latents, dim=0)
        else:
            image = pipe.preprocess_image(input_image).to(device=pipe.device, dtype=pipe.torch_dtype)
            input_latents = pipe.vae.encode(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents, "input_latents": input_latents}


class QwenImageUnit_LayerInputImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("layer_input_image", "tiled", "tile_size", "tile_stride"),
            output_params=("layer_input_latents",),
            onload_model_names=("vae",)
        )

    def process(self, pipe: QwenImagePipeline, layer_input_image, tiled, tile_size, tile_stride):
        if layer_input_image is None:
            return {}
        pipe.load_models_to_device(['vae'])
        image = pipe.preprocess_image(layer_input_image).to(device=pipe.device, dtype=pipe.torch_dtype)
        latents = pipe.vae.encode(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return {"layer_input_latents": latents}


class QwenImageUnit_Inpaint(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("inpaint_mask", "height", "width", "inpaint_blur_size", "inpaint_blur_sigma"),
            output_params=("inpaint_mask",),
        )

    def process(self, pipe: QwenImagePipeline, inpaint_mask, height, width, inpaint_blur_size, inpaint_blur_sigma):
        if inpaint_mask is None:
            return {}
        inpaint_mask = pipe.preprocess_image(inpaint_mask.convert("RGB").resize((width // 8, height // 8)), min_value=0, max_value=1)
        inpaint_mask = inpaint_mask.mean(dim=1, keepdim=True)
        if inpaint_blur_size is not None and inpaint_blur_sigma is not None:
            from torchvision.transforms import GaussianBlur
            blur = GaussianBlur(kernel_size=inpaint_blur_size * 2 + 1, sigma=inpaint_blur_sigma)
            inpaint_mask = blur(inpaint_mask)
        return {"inpaint_mask": inpaint_mask}


class QwenImageUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            input_params=("edit_image",),
            output_params=("prompt_emb", "prompt_emb_mask"),
            onload_model_names=("text_encoder",)
        )
        
    def extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result
    
    def calculate_dimensions(self, target_area, ratio):
        width = math.sqrt(target_area * ratio)
        height = width / ratio
        width = round(width / 32) * 32
        height = round(height / 32) * 32
        return width, height
    
    def resize_image(self, image, target_area=384*384):
        width, height = self.calculate_dimensions(target_area, image.size[0] / image.size[1])
        return image.resize((width, height))
    
    def encode_prompt(self, pipe: QwenImagePipeline, prompt):
        template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        drop_idx = 34
        txt = [template.format(e) for e in prompt]
        model_inputs = pipe.tokenizer(txt, max_length=4096+drop_idx, padding=True, truncation=True, return_tensors="pt").to(pipe.device)
        if model_inputs.input_ids.shape[1] >= 1024:
            print(f"Warning!!! QwenImage model was trained on prompts up to 512 tokens. Current prompt requires {model_inputs['input_ids'].shape[1] - drop_idx} tokens, which may lead to unpredictable behavior.")
        hidden_states = pipe.text_encoder(input_ids=model_inputs.input_ids, attention_mask=model_inputs.attention_mask, output_hidden_states=True,)[-1]
        split_hidden_states = self.extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        return split_hidden_states
        
    def encode_prompt_edit(self, pipe: QwenImagePipeline, prompt, edit_image):
        template =  "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
        drop_idx = 64
        txt = [template.format(e) for e in prompt]
        model_inputs = pipe.processor(text=txt, images=edit_image, padding=True, return_tensors="pt").to(pipe.device)
        hidden_states = pipe.text_encoder(input_ids=model_inputs.input_ids, attention_mask=model_inputs.attention_mask, pixel_values=model_inputs.pixel_values, image_grid_thw=model_inputs.image_grid_thw, output_hidden_states=True,)[-1]
        split_hidden_states = self.extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        return split_hidden_states
    
    def encode_prompt_edit_multi(self, pipe: QwenImagePipeline, prompt, edit_image):
        template =  "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        drop_idx = 64
        img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
        base_img_prompt = "".join([img_prompt_template.format(i + 1) for i in range(len(edit_image))])
        txt = [template.format(base_img_prompt + e) for e in prompt]
        edit_image = [self.resize_image(image) for image in edit_image]
        model_inputs = pipe.processor(text=txt, images=edit_image, padding=True, return_tensors="pt").to(pipe.device)
        hidden_states = pipe.text_encoder(input_ids=model_inputs.input_ids, attention_mask=model_inputs.attention_mask, pixel_values=model_inputs.pixel_values, image_grid_thw=model_inputs.image_grid_thw, output_hidden_states=True,)[-1]
        split_hidden_states = self.extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        return split_hidden_states

    def process(self, pipe: QwenImagePipeline, prompt, edit_image=None) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        if pipe.text_encoder is not None:
            prompt = [prompt] if isinstance(prompt, str) else prompt
            if edit_image is None:
                split_hidden_states = self.encode_prompt(pipe, prompt)
            elif isinstance(edit_image, Image.Image) or (isinstance(edit_image, list) and len(prompt) == len(edit_image)):
                split_hidden_states = self.encode_prompt_edit(pipe, prompt, edit_image)
            else:
                split_hidden_states = self.encode_prompt_edit_multi(pipe, prompt, edit_image)
            attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
            max_seq_len = max([e.size(0) for e in split_hidden_states])
            prompt_embeds = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states])
            encoder_attention_mask = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list])
            prompt_embeds = prompt_embeds.to(dtype=pipe.torch_dtype, device=pipe.device)
            return {"prompt_emb": prompt_embeds, "prompt_emb_mask": encoder_attention_mask}
        else:
            return {}


class QwenImageUnit_EntityControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            input_params=("eligen_entity_prompts", "width", "height", "eligen_enable_on_negative", "cfg_scale"),
            output_params=("entity_prompt_emb", "entity_masks", "entity_prompt_emb_mask"),
            onload_model_names=("text_encoder",)
        )

    def extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result

    def get_prompt_emb(self, pipe: QwenImagePipeline, prompt) -> dict:
        if pipe.text_encoder is not None:
            prompt = [prompt]
            template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
            drop_idx = 34
            txt = [template.format(e) for e in prompt]
            txt_tokens = pipe.tokenizer(txt, max_length=1024+drop_idx, padding=True, truncation=True, return_tensors="pt").to(pipe.device)
            hidden_states = pipe.text_encoder(input_ids=txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask, output_hidden_states=True,)[-1]
            
            split_hidden_states = self.extract_masked_hidden(hidden_states, txt_tokens.attention_mask)
            split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
            attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
            max_seq_len = max([e.size(0) for e in split_hidden_states])
            prompt_embeds = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states])
            encoder_attention_mask = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list])
            prompt_embeds = prompt_embeds.to(dtype=pipe.torch_dtype, device=pipe.device)
            return {"prompt_emb": prompt_embeds, "prompt_emb_mask": encoder_attention_mask}
        else:
            return {}

    def preprocess_masks(self, pipe, masks, height, width, dim):
        out_masks = []
        for mask in masks:
            mask = pipe.preprocess_image(mask.resize((width, height), resample=Image.NEAREST)).mean(dim=1, keepdim=True) > 0
            mask = mask.repeat(1, dim, 1, 1).to(device=pipe.device, dtype=pipe.torch_dtype)
            out_masks.append(mask)
        return out_masks

    def prepare_entity_inputs(self, pipe, entity_prompts, entity_masks, width, height):
        entity_masks = self.preprocess_masks(pipe, entity_masks, height//8, width//8, 1)
        entity_masks = torch.cat(entity_masks, dim=0).unsqueeze(0) # b, n_mask, c, h, w
        prompt_embs, prompt_emb_masks = [], []
        for entity_prompt in entity_prompts:
            prompt_emb_dict = self.get_prompt_emb(pipe, entity_prompt)
            prompt_embs.append(prompt_emb_dict['prompt_emb'])
            prompt_emb_masks.append(prompt_emb_dict['prompt_emb_mask'])
        return prompt_embs, prompt_emb_masks, entity_masks

    def prepare_eligen(self, pipe, prompt_emb_nega, eligen_entity_prompts, eligen_entity_masks, width, height, enable_eligen_on_negative, cfg_scale):
        entity_prompt_emb_posi, entity_prompt_emb_posi_mask, entity_masks_posi = self.prepare_entity_inputs(pipe, eligen_entity_prompts, eligen_entity_masks, width, height)
        if enable_eligen_on_negative and cfg_scale != 1.0:
            entity_prompt_emb_nega = [prompt_emb_nega['prompt_emb']] * len(entity_prompt_emb_posi)
            entity_prompt_emb_nega_mask = [prompt_emb_nega['prompt_emb_mask']] * len(entity_prompt_emb_posi)
            entity_masks_nega = entity_masks_posi
        else:
            entity_prompt_emb_nega, entity_prompt_emb_nega_mask, entity_masks_nega = None, None, None
        eligen_kwargs_posi = {"entity_prompt_emb": entity_prompt_emb_posi, "entity_masks": entity_masks_posi, "entity_prompt_emb_mask": entity_prompt_emb_posi_mask}
        eligen_kwargs_nega = {"entity_prompt_emb": entity_prompt_emb_nega, "entity_masks": entity_masks_nega, "entity_prompt_emb_mask": entity_prompt_emb_nega_mask}
        return eligen_kwargs_posi, eligen_kwargs_nega

    def process(self, pipe: QwenImagePipeline, inputs_shared, inputs_posi, inputs_nega):
        eligen_entity_prompts, eligen_entity_masks = inputs_shared.get("eligen_entity_prompts", None), inputs_shared.get("eligen_entity_masks", None)
        if eligen_entity_prompts is None or eligen_entity_masks is None or len(eligen_entity_prompts) == 0 or len(eligen_entity_masks) == 0:
            return inputs_shared, inputs_posi, inputs_nega
        pipe.load_models_to_device(self.onload_model_names)
        eligen_enable_on_negative = inputs_shared.get("eligen_enable_on_negative", False)
        eligen_kwargs_posi, eligen_kwargs_nega = self.prepare_eligen(pipe, inputs_nega,
            eligen_entity_prompts, eligen_entity_masks, inputs_shared["width"], inputs_shared["height"],
            eligen_enable_on_negative, inputs_shared["cfg_scale"])
        inputs_posi.update(eligen_kwargs_posi)
        if inputs_shared.get("cfg_scale", 1.0) != 1.0:
            inputs_nega.update(eligen_kwargs_nega)
        return inputs_shared, inputs_posi, inputs_nega



class QwenImageUnit_BlockwiseControlNet(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("blockwise_controlnet_inputs", "tiled", "tile_size", "tile_stride"),
            output_params=("blockwise_controlnet_conditioning",),
            onload_model_names=("vae",)
        )

    def apply_controlnet_mask_on_latents(self, pipe, latents, mask):
        mask = (pipe.preprocess_image(mask) + 1) / 2
        mask = mask.mean(dim=1, keepdim=True)
        mask = 1 - torch.nn.functional.interpolate(mask, size=latents.shape[-2:])
        latents = torch.concat([latents, mask], dim=1)
        return latents

    def apply_controlnet_mask_on_image(self, pipe, image, mask):
        mask = mask.resize(image.size)
        mask = pipe.preprocess_image(mask).mean(dim=[0, 1]).cpu()
        image = np.array(image)
        image[mask > 0] = 0
        image = Image.fromarray(image)
        return image

    def process(self, pipe: QwenImagePipeline, blockwise_controlnet_inputs: list[ControlNetInput], tiled, tile_size, tile_stride):
        if blockwise_controlnet_inputs is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        conditionings = []
        for controlnet_input in blockwise_controlnet_inputs:
            image = controlnet_input.image
            if controlnet_input.inpaint_mask is not None:
                image = self.apply_controlnet_mask_on_image(pipe, image, controlnet_input.inpaint_mask)

            image = pipe.preprocess_image(image).to(device=pipe.device, dtype=pipe.torch_dtype)
            image = pipe.vae.encode(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)

            if controlnet_input.inpaint_mask is not None:
                image = self.apply_controlnet_mask_on_latents(pipe, image, controlnet_input.inpaint_mask)
            conditionings.append(image)
            
        return {"blockwise_controlnet_conditioning": conditionings}


class QwenImageUnit_EditImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("edit_image", "tiled", "tile_size", "tile_stride", "edit_image_auto_resize", "input_image"),
            output_params=("edit_latents", "edit_image"),
            onload_model_names=("vae",)
        )


    def calculate_dimensions(self, target_area, ratio):
        import math
        width = math.sqrt(target_area * ratio)
        height = width / ratio
        width = round(width / 32) * 32
        height = round(height / 32) * 32
        return width, height


    def edit_image_auto_resize(self, edit_image):
        calculated_width, calculated_height = self.calculate_dimensions(1024 * 1024, edit_image.size[0] / edit_image.size[1])
        return edit_image.resize((calculated_width, calculated_height))


    def process(self, pipe: QwenImagePipeline, edit_image, tiled, tile_size, tile_stride, edit_image_auto_resize=False, input_image=None):
        if edit_image is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        if isinstance(edit_image, Image.Image):
            resized_edit_image = self.edit_image_auto_resize(edit_image) if edit_image_auto_resize else edit_image
            edit_image = pipe.preprocess_image(resized_edit_image).to(device=pipe.device, dtype=pipe.torch_dtype)
            edit_latents = pipe.vae.encode(edit_image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        else:
            resized_edit_image, edit_latents = [], []
            for image in edit_image:
                if edit_image_auto_resize:
                    image = self.edit_image_auto_resize(image)
                resized_edit_image.append(image)
                image = pipe.preprocess_image(image).to(device=pipe.device, dtype=pipe.torch_dtype)
                latents = pipe.vae.encode(image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
                edit_latents.append(latents)
            if isinstance(input_image, list) and len(input_image) == len(resized_edit_image):
                edit_latents = torch.concat(edit_latents, dim=0)
        return {"edit_latents": edit_latents, "edit_image": resized_edit_image}


class QwenImageUnit_CleanImageEmbedder(PipelineUnit):
    """Encodes GT clean image to target_latents for demoire training"""
    def __init__(self):
        super().__init__(
            input_params=("gt_image", "tiled", "tile_size", "tile_stride"),
            output_params=("target_latents",),
            onload_model_names=("vae",)
        )

    def process(self, pipe: QwenImagePipeline, gt_image, tiled, tile_size, tile_stride):
        """Encode GT clean image to target latents"""
        if gt_image is None:
            return {}

        pipe.load_models_to_device(['vae'])

        if isinstance(gt_image, Image.Image):
            gt_tensor = pipe.preprocess_image(gt_image).to(
                device=pipe.device,
                dtype=pipe.torch_dtype
            )

            # Debug: Check preprocessed image
            if torch.isnan(gt_tensor).any() or torch.isinf(gt_tensor).any():
                print(f"[DEBUG CleanImageEmbedder] gt_tensor has NaN/Inf! shape={gt_tensor.shape}, device={gt_tensor.device}")
                print(f"  min={gt_tensor.min().item():.4f}, max={gt_tensor.max().item():.4f}")

            target_latents = pipe.vae.encode(gt_tensor, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)

            # Debug: Check VAE encode output
            if torch.isnan(target_latents).any() or torch.isinf(target_latents).any():
                print(f"[DEBUG CleanImageEmbedder] target_latents has NaN/Inf after VAE encode!")

            return {"target_latents": target_latents}
        elif isinstance(gt_image, list):
            target_latents = []
            for image in gt_image:
                gt_tensor = pipe.preprocess_image(image).to(
                    device=pipe.device,
                    dtype=pipe.torch_dtype
                )
                target_latents.append(pipe.vae.encode(gt_tensor, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride))
            return {"target_latents": torch.concat(target_latents, dim=0)}
        else:
            return {}


class QwenImageUnit_Image2LoRAEncode(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("image2lora_images",),
            output_params=("image2lora_x", "image2lora_residual", "image2lora_residual_highres"),
            onload_model_names=("siglip2_image_encoder", "dinov3_image_encoder", "text_encoder"),
        )
        from ..core.data.operators import ImageCropAndResize
        self.processor_lowres = ImageCropAndResize(height=28*8, width=28*8)
        self.processor_highres = ImageCropAndResize(height=1024, width=1024)

    def extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        return split_result

    def encode_prompt_edit(self, pipe: QwenImagePipeline, prompt, edit_image):
        prompt = [prompt]
        template =  "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
        drop_idx = 64
        txt = [template.format(e) for e in prompt]
        model_inputs = pipe.processor(text=txt, images=edit_image, padding=True, return_tensors="pt").to(pipe.device)
        hidden_states = pipe.text_encoder(input_ids=model_inputs.input_ids, attention_mask=model_inputs.attention_mask, pixel_values=model_inputs.pixel_values, image_grid_thw=model_inputs.image_grid_thw, output_hidden_states=True,)[-1]
        split_hidden_states = self.extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states])
        prompt_embeds = prompt_embeds.to(dtype=pipe.torch_dtype, device=pipe.device)
        return prompt_embeds.view(1, -1)
    
    def encode_images_using_siglip2(self, pipe: QwenImagePipeline, images: list[Image.Image]):
        pipe.load_models_to_device(["siglip2_image_encoder"])
        embs = []
        for image in images:
            image = self.processor_highres(image)
            embs.append(pipe.siglip2_image_encoder(image).to(pipe.torch_dtype))
        embs = torch.stack(embs)
        return embs
    
    def encode_images_using_dinov3(self, pipe: QwenImagePipeline, images: list[Image.Image]):
        pipe.load_models_to_device(["dinov3_image_encoder"])
        embs = []
        for image in images:
            image = self.processor_highres(image)
            embs.append(pipe.dinov3_image_encoder(image).to(pipe.torch_dtype))
        embs = torch.stack(embs)
        return embs
    
    def encode_images_using_qwenvl(self, pipe: QwenImagePipeline, images: list[Image.Image], highres=False):
        pipe.load_models_to_device(["text_encoder"])
        embs = []
        for image in images:
            image = self.processor_highres(image) if highres else self.processor_lowres(image)
            embs.append(self.encode_prompt_edit(pipe, prompt="", edit_image=image))
        embs = torch.stack(embs)
        return embs

    def encode_images(self, pipe: QwenImagePipeline, images: list[Image.Image]):
        if images is None:
            return {}
        if not isinstance(images, list):
            images = [images]
        embs_siglip2 = self.encode_images_using_siglip2(pipe, images)
        embs_dinov3 = self.encode_images_using_dinov3(pipe, images)
        x = torch.concat([embs_siglip2, embs_dinov3], dim=-1)
        residual = None
        residual_highres = None
        if pipe.image2lora_coarse is not None:
            residual = self.encode_images_using_qwenvl(pipe, images, highres=False)
        if pipe.image2lora_fine is not None:
            residual_highres = self.encode_images_using_qwenvl(pipe, images, highres=True)
        return x, residual, residual_highres

    def process(self, pipe: QwenImagePipeline, image2lora_images):
        if image2lora_images is None:
            return {}
        x, residual, residual_highres = self.encode_images(pipe, image2lora_images)
        return {"image2lora_x": x, "image2lora_residual": residual, "image2lora_residual_highres": residual_highres}


class QwenImageUnit_Image2LoRADecode(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("image2lora_x", "image2lora_residual", "image2lora_residual_highres"),
            output_params=("lora",),
            onload_model_names=("image2lora_coarse", "image2lora_fine", "image2lora_style"),
        )
    
    def process(self, pipe: QwenImagePipeline, image2lora_x, image2lora_residual, image2lora_residual_highres):
        if image2lora_x is None:
            return {}
        loras = []
        if pipe.image2lora_style is not None:
            pipe.load_models_to_device(["image2lora_style"])
            for x in image2lora_x:
                loras.append(pipe.image2lora_style(x=x, residual=None))
        if pipe.image2lora_coarse is not None:
            pipe.load_models_to_device(["image2lora_coarse"])
            for x, residual in zip(image2lora_x, image2lora_residual):
                loras.append(pipe.image2lora_coarse(x=x, residual=residual))
        if pipe.image2lora_fine is not None:
            pipe.load_models_to_device(["image2lora_fine"])
            for x, residual in zip(image2lora_x, image2lora_residual_highres):
                loras.append(pipe.image2lora_fine(x=x, residual=residual))
        lora = merge_lora(loras, alpha=1 / len(image2lora_x))
        return {"lora": lora}


class QwenImageUnit_ContextImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("context_image", "height", "width", "tiled", "tile_size", "tile_stride"),
            output_params=("context_latents",),
            onload_model_names=("vae",)
        )

    def process(self, pipe: QwenImagePipeline, context_image, height, width, tiled, tile_size, tile_stride):
        if context_image is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        context_image = pipe.preprocess_image(context_image.resize((width, height))).to(device=pipe.device, dtype=pipe.torch_dtype)
        context_latents = pipe.vae.encode(context_image, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return {"context_latents": context_latents}


def model_fn_qwen_image(
    dit: QwenImageDiT = None,
    blockwise_controlnet: QwenImageBlockwiseMultiControlNet = None,
    latents=None,
    timestep=None,
    prompt_emb=None,
    prompt_emb_mask=None,
    height=None,
    width=None,
    blockwise_controlnet_conditioning=None,
    blockwise_controlnet_inputs=None,
    progress_id=0,
    num_inference_steps=1,
    entity_prompt_emb=None,
    entity_prompt_emb_mask=None,
    entity_masks=None,
    edit_latents=None,
    layer_input_latents=None,
    layer_num=None,
    context_latents=None,
    enable_fp8_attention=False,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    edit_rope_interpolation=False,
    zero_cond_t=False,
    **kwargs
):
    batch_size = prompt_emb.shape[0] if prompt_emb is not None else latents.shape[0]
    if layer_num is None:
        layer_num = 1
        img_shapes = [(1, latents.shape[2]//2, latents.shape[3]//2)]
    else:
        layer_num = layer_num + 1
        batch_size = latents.shape[0] // layer_num
        img_shapes = [(1, latents.shape[2]//2, latents.shape[3]//2)] * layer_num
    txt_seq_lens = prompt_emb_mask.sum(dim=1).tolist()
    timestep = timestep / 1000
    
    image = rearrange(latents, "(B N) C (H P) (W Q) -> B (N H W) (C P Q)", H=height//16, W=width//16, P=2, Q=2, N=layer_num)
    image_seq_len = image.shape[1]

    if context_latents is not None:
        img_shapes += [(1, context_latents.shape[2]//2, context_latents.shape[3]//2)]
        context_image = rearrange(context_latents, "B C (H P) (W Q) -> B (H W) (C P Q)", H=context_latents.shape[2]//2, W=context_latents.shape[3]//2, P=2, Q=2)
        image = torch.cat([image, context_image], dim=1)
    if edit_latents is not None:
        edit_latents_list = edit_latents if isinstance(edit_latents, list) else [edit_latents]
        img_shapes += [(1, e.shape[2]//2, e.shape[3]//2) for e in edit_latents_list]
        edit_image = [rearrange(e, "B C (H P) (W Q) -> B (H W) (C P Q)", H=e.shape[2]//2, W=e.shape[3]//2, P=2, Q=2) for e in edit_latents_list]
        image = torch.cat([image] + edit_image, dim=1)
    if layer_input_latents is not None:
        layer_num = layer_num + 1
        img_shapes += [(1, layer_input_latents.shape[2]//2, layer_input_latents.shape[3]//2)]
        layer_input_latents = rearrange(layer_input_latents, "B C (H P) (W Q) -> B (H W) (C P Q)", P=2, Q=2)
        image = torch.cat([image, layer_input_latents], dim=1)

    image = dit.img_in(image)
    if zero_cond_t:
        timestep = torch.cat([timestep, timestep * 0], dim=0)
        modulate_index = torch.tensor(
            [[0] * prod(sample[0]) + [1] * sum([prod(s) for s in sample[1:]]) for sample in [img_shapes]],
            device=timestep.device,
            dtype=torch.int,
        )
    else:
        modulate_index = None
    conditioning = dit.time_text_embed(
        timestep,
        image.dtype,
        addition_t_cond=None if not dit.time_text_embed.use_additional_t_cond else torch.tensor([0]).to(device=image.device, dtype=torch.long)
    )

    if entity_prompt_emb is not None:
        text, image_rotary_emb, attention_mask = dit.process_entity_masks(
            latents, prompt_emb, prompt_emb_mask, entity_prompt_emb, entity_prompt_emb_mask,
            entity_masks, height, width, image, img_shapes,
        )
    else:
        text = dit.txt_in(dit.txt_norm(prompt_emb))
        if edit_rope_interpolation:
            image_rotary_emb = dit.pos_embed.forward_sampling(img_shapes, txt_seq_lens, device=latents.device)
        else:
            image_rotary_emb = dit.pos_embed(img_shapes, txt_seq_lens, device=latents.device)
        attention_mask = None
        
    if blockwise_controlnet_conditioning is not None:
        blockwise_controlnet_conditioning = blockwise_controlnet.preprocess(
            blockwise_controlnet_inputs, blockwise_controlnet_conditioning)

    for block_id, block in enumerate(dit.transformer_blocks):
        text, image = gradient_checkpoint_forward(
            block,
            use_gradient_checkpointing,
            use_gradient_checkpointing_offload,
            image=image,
            text=text,
            temb=conditioning,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
            enable_fp8_attention=enable_fp8_attention,
            modulate_index=modulate_index,
        )
        if blockwise_controlnet_conditioning is not None:
            image_slice = image[:, :image_seq_len].clone()
            controlnet_output = blockwise_controlnet.blockwise_forward(
                image=image_slice, conditionings=blockwise_controlnet_conditioning,
                controlnet_inputs=blockwise_controlnet_inputs, block_id=block_id,
                progress_id=progress_id, num_inference_steps=num_inference_steps,
            )
            image[:, :image_seq_len] = image_slice + controlnet_output
    
    if zero_cond_t:
        conditioning = conditioning.chunk(2, dim=0)[0]
    image = dit.norm_out(image, conditioning)
    image = dit.proj_out(image)
    image = image[:, :image_seq_len]
    
    latents = rearrange(image, "B (N H W) (C P Q) -> (B N) C (H P) (W Q)", H=height//16, W=width//16, P=2, Q=2, B=batch_size)
    return latents
