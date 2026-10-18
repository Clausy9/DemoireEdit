"""Selected historical dual-scale sampler, with portable model bindings."""
import gc
from dataclasses import dataclass
from pathlib import Path
from .recipe import condition_shift_steps
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from diffsynth.pipelines.qwen_image import ModelConfig, QwenImagePipeline
PROMPT = "Remove moiré patterns, color aliasing and color distoration from the image while preserving all details, colors, and textures of the original content."
@dataclass(frozen=True)
class StrategyConfig:
    name: str
    base_model_id: str
    lora_path: Path | None
    zero_cond_t: bool = False


def load_rgb(path: Path):
    with Image.open(path) as image:
        return image.convert("RGB")


def compute_inference_size(image: Image.Image):
    width, height = image.size
    height_4k = height // 8
    width_4k = width // 8
    infer_h = max((height_4k // 2) * 2, 32) * 8
    infer_w = max((width_4k // 2) * 2, 32) * 8
    return infer_w, infer_h


def resize_to_short_side(image: Image.Image, target_short_side: int):
    width, height = image.size
    if target_short_side <= 0:
        return image
    if min(width, height) == target_short_side and width % 16 == 0 and height % 16 == 0:
        return image
    scale = target_short_side / min(width, height)
    resized_w = max(int(round(width * scale)) // 16 * 16, 256)
    resized_h = max(int(round(height * scale)) // 16 * 16, 256)
    return image.resize((resized_w, resized_h), Image.LANCZOS)


def interpolate_latents(latents: torch.Tensor, target_hw: tuple[int, int]):
    target_h, target_w = target_hw
    target_h = max((target_h // 2) * 2, 2)
    target_w = max((target_w // 2) * 2, 2)
    if latents.shape[-2:] == (target_h, target_w):
        return latents
    dtype = latents.dtype
    return F.interpolate(
        latents.to(torch.float32),
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    ).to(dtype)


def inner_progress(debug):
    if debug:
        return tqdm
    return lambda iterable: iterable


def make_model_config(model_id: str, origin_file_pattern: str, device: str, vram_management: bool):
    kwargs = {}
    if vram_management:
        kwargs = {
            "offload_device": "cpu",
            "offload_dtype": torch.bfloat16,
            "onload_device": device,
            "onload_dtype": torch.bfloat16,
            "preparing_device": device,
            "preparing_dtype": torch.bfloat16,
            "computation_device": device,
            "computation_dtype": torch.bfloat16,
        }
    return ModelConfig(model_id=model_id, origin_file_pattern=origin_file_pattern, **kwargs)


def build_pipeline(strategy: StrategyConfig, device: str, vram_management: bool = False):
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            make_model_config(strategy.base_model_id, "transformer/diffusion_pytorch_model*.safetensors", device, vram_management),
            make_model_config("Qwen/Qwen-Image", "text_encoder/model*.safetensors", device, vram_management),
            make_model_config("Qwen/Qwen-Image", "vae/diffusion_pytorch_model.safetensors", device, vram_management),
        ],
        tokenizer_config=None,
        processor_config=ModelConfig(
            model_id="Qwen/Qwen-Image-Edit",
            origin_file_pattern="processor/",
        ),
    )
    # These loss modules are only used during training and should not occupy GPU memory in evaluation.
    pipe.lpips_loss_fn = None
    pipe.ffl = None
    if strategy.lora_path is not None:
        pipe.load_lora(pipe.dit, str(strategy.lora_path))
    if vram_management:
        pipe.load_models_to_device([])
    else:
        pipe.to(device=device, dtype=torch.bfloat16)
    return pipe


def run_mid_resolution_inference(pipe, low_lq: Image.Image, args, strategy: StrategyConfig):
    width, height = low_lq.size
    black = Image.new("RGB", (width, height), (0, 0, 0))
    stage1_ref_mode = getattr(args, "stage1_ref_mode", "black")
    stage1_ref = black if stage1_ref_mode == "black" else low_lq
    rand_device = "cuda" if args.device.startswith("cuda") else "cpu"
    stage1_cfg = args.cfg_scale if args.stage1_cfg_scale is None else args.stage1_cfg_scale
    prompt = getattr(args, "prompt", PROMPT)
    with torch.no_grad():
        output = pipe(
            prompt,
            negative_prompt="",
            cfg_scale=stage1_cfg,
            seed=args.seed,
            rand_device=rand_device,
            num_inference_steps=args.mid_steps,
            enable_fp8_attention=getattr(args, "enable_fp8_attention", False),
            height=height,
            width=width,
            edit_image=[low_lq, stage1_ref],
            edit_image_auto_resize=False,
            zero_cond_t=strategy.zero_cond_t,
            tiled=True,
            tile_size=args.tile_size,
            tile_stride=args.tile_stride,
            progress_bar_cmd=inner_progress(args.debug),
        )
    if output.size != low_lq.size:
        output = output.resize(low_lq.size, Image.LANCZOS)
    return output


def clone_edit_latents(edit_latents):
    if isinstance(edit_latents, list):
        return [lat.clone() for lat in edit_latents]
    if edit_latents is None:
        return None
    return edit_latents.clone()


def set_single_edit_latent(inputs_shared: dict, latent: torch.Tensor):
    original = inputs_shared.get("edit_latents")
    if isinstance(original, list):
        inputs_shared["edit_latents"] = [latent]
    else:
        inputs_shared["edit_latents"] = latent


@torch.no_grad()
def infer_one_progressive_2stage(pipe, image: Image.Image, args, strategy: StrategyConfig):
    orig_w, orig_h = image.size
    infer_w, infer_h = compute_inference_size(image)
    full_lq = image.resize((infer_w, infer_h), Image.LANCZOS) if (infer_w, infer_h) != (orig_w, orig_h) else image

    low_lq = resize_to_short_side(full_lq, args.target_short_side)
    rand_device = "cuda" if args.device.startswith("cuda") else "cpu"

    pipe.scheduler.set_timesteps(
        args.num_inference_steps,
        denoising_strength=1.0,
        dynamic_shift_len=(infer_h // 16) * (infer_w // 16),
    )
    inputs_posi = {"prompt": getattr(args, "prompt", PROMPT)}
    inputs_nega = {"negative_prompt": ""}
    inputs_shared = {
        "cfg_scale": args.cfg_scale,
        "input_image": full_lq,
        "denoising_strength": 1.0,
        "inpaint_mask": None,
        "inpaint_blur_size": None,
        "inpaint_blur_sigma": None,
        "height": infer_h,
        "width": infer_w,
        "seed": args.seed,
        "rand_device": rand_device,
        "num_inference_steps": args.num_inference_steps,
        "enable_fp8_attention": getattr(args, "enable_fp8_attention", False),
        "blockwise_controlnet_inputs": None,
        "tiled": True,
        "tile_size": args.tile_size,
        "tile_stride": args.tile_stride,
        "eligen_entity_prompts": None,
        "eligen_entity_masks": None,
        "eligen_enable_on_negative": False,
        # Match the historical Stage-2 prompt/image embedding path: use the
        # original full-resolution LQ image, while edit_latents are switched
        # manually in the denoising loop below.
        "edit_image": [full_lq],
        "edit_image_auto_resize": False,
        "edit_rope_interpolation": False,
        "context_image": None,
        "zero_cond_t": strategy.zero_cond_t,
        "layer_input_image": None,
        "layer_num": None,
    }
    manual_latent_unit_names = {
        "QwenImageUnit_NoiseInitializer",
        "QwenImageUnit_InputImageEmbedder",
        "QwenImageUnit_EditImageEmbedder",
        "QwenImageUnit_CleanImageEmbedder",
        "QwenImageUnit_LayerInputImageEmbedder",
        "QwenImageUnit_ContextImageEmbedder",
        "QwenImageUnit_BlockwiseControlNet",
    }
    for unit in pipe.units:
        if unit.__class__.__name__ in manual_latent_unit_names:
            continue
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(unit, pipe, inputs_shared, inputs_posi, inputs_nega)

    stage1 = run_mid_resolution_inference(pipe, low_lq, args, strategy)
    stage1_full = stage1.resize((infer_w, infer_h), Image.BICUBIC)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pipe.scheduler.set_timesteps(
        args.num_inference_steps,
        denoising_strength=1.0,
        dynamic_shift_len=(infer_h // 16) * (infer_w // 16),
    )

    infer_h = inputs_shared["height"]
    infer_w = inputs_shared["width"]
    if full_lq.size != (infer_w, infer_h):
        full_lq = full_lq.resize((infer_w, infer_h), Image.LANCZOS)
    if stage1_full.size != (infer_w, infer_h):
        stage1_full = stage1_full.resize((infer_w, infer_h), Image.BICUBIC)

    pipe.load_models_to_device(["vae"])
    full_lq_tensor = pipe.preprocess_image(full_lq).to(device=pipe.device, dtype=pipe.torch_dtype)
    input_latents = pipe.vae.encode(full_lq_tensor, tiled=True, tile_size=args.tile_size, tile_stride=args.tile_stride)
    del full_lq_tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    stage1_tensor = pipe.preprocess_image(stage1_full).to(device=pipe.device, dtype=pipe.torch_dtype)
    stage1_latents = pipe.vae.encode(stage1_tensor, tiled=True, tile_size=args.tile_size, tile_stride=args.tile_stride)
    del stage1_tensor
    stage1_latents = interpolate_latents(stage1_latents, input_latents.shape[-2:])

    # Stage 2 starts from the standard denoising noise. Only edit_latents are
    # progressively replaced by the upsampled Stage-1 result.
    stage2_latents = pipe.generate_noise(
        input_latents.shape, seed=args.seed, rand_device=rand_device,
        rand_torch_dtype=pipe.torch_dtype, device=pipe.device, torch_dtype=pipe.torch_dtype,
    )
    inputs_shared["input_latents"] = input_latents
    inputs_shared["latents"] = stage2_latents.clone()
    inputs_shared["edit_latents"] = [input_latents]
    original_edit_latents = clone_edit_latents(inputs_shared.get("edit_latents"))

    switch_steps = condition_shift_steps(args.progressive_edit_ratio, args.num_inference_steps)
    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for progress_id, timestep in enumerate(inner_progress(args.debug)(pipe.scheduler.timesteps)):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        if progress_id < switch_steps:
            set_single_edit_latent(inputs_shared, stage1_latents)
        else:
            inputs_shared["edit_latents"] = clone_edit_latents(original_edit_latents)
        noise_pred = pipe.cfg_guided_model_fn(
            pipe.model_fn,
            args.cfg_scale,
            inputs_shared,
            inputs_posi,
            inputs_nega,
            **models,
            timestep=timestep,
            progress_id=progress_id,
        )
        if not torch.isfinite(noise_pred).all():
            raise FloatingPointError("Nonfinite Stage 2 noise prediction")
        inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared)
        if not torch.isfinite(inputs_shared["latents"]).all():
            raise FloatingPointError("Nonfinite Stage 2 latent")

    pipe.load_models_to_device(["vae"])
    guidance_cache = pipe.prepare_vae_guidance_cache(full_lq, device=pipe.device, dtype=pipe.torch_dtype)
    output = pipe.vae.decode(
        inputs_shared["latents"],
        guidance_cache=guidance_cache,
        device=pipe.device,
        tiled=True,
        tile_size=args.tile_size,
        tile_stride=args.tile_stride,
    )
    if not torch.isfinite(output).all():
        raise FloatingPointError("Nonfinite Stage 2 VAE output")
    output = pipe.vae_output_to_image(output)
    pipe.load_models_to_device([])

    del inputs_shared, inputs_posi, inputs_nega, original_edit_latents, input_latents, stage1_latents
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if output.size != (orig_w, orig_h):
        output = output.resize((orig_w, orig_h), Image.LANCZOS)
    return output

