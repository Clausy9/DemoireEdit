# Inference

Complete [Installation](installation.md) and [Models](models.md) first. All example paths are relative to the repository root.

```bash
export DIFFSYNTH_MODEL_BASE_PATH=weights/base
export DIFFSYNTH_SKIP_DOWNLOAD=true
export DEMOIRE_EDIT_WEIGHTS_PATH=weights/demoireedit.safetensors

demoire-infer --input path/to/input.png --output outputs/restored.png --seed 0
```

The input is read as RGB after EXIF orientation correction. The output must differ from the input path. Use a PNG output path to avoid lossy re-encoding. Images above 40 megapixels are rejected; this guard is not a guarantee that smaller images fit every GPU.

| Option | Default | Purpose |
| --- | --- | --- |
| `--lora` | `DEMOIRE_EDIT_WEIGHTS_PATH` or `weights/demoireedit.safetensors` | Approved adapter path |
| `--device` | `cuda` | Inference device |
| `--seed` | `0` | Sampling seed |
| `--no-offload` | Off | Keep more of the model on the GPU; increases memory demand |
| `--debug` | Off | Enable backend debugging |

## Sampler

Stage 1 uses short-side 1024, a black auxiliary reference and ten denoising steps. Stage 2 begins from noise at the input resolution, using the Stage 1 image for three steps and the original input for seven. The switch is `ceil(0.3 × 10)`; CFG is 1.0. BF16, tiled VAE and CPU offload are the default runtime settings. Nonfinite predictions, latents or decoded tensors stop execution.

## Interactive demo

```bash
demoire-demo --port 7860
```

The interface binds to `127.0.0.1`, uses lossless PNG input/output, disables Gradio analytics and creates no public share. For a remote GPU:

```bash
ssh -N -L 7860:127.0.0.1:7860 YOUR_SSH_ALIAS
```

Then open `http://127.0.0.1:7860`. The default local upload cache is `outputs/.demo-cache`; override `GRADIO_TEMP_DIR` if needed.

## Expected behavior

The tested real-photo crop shows reduced fine stripes with residual broad color variation and softened detail. Editing may alter image content. Inspect both input and output; do not assume fidelity or universal artifact removal. Full 4K inference has not been verified.
