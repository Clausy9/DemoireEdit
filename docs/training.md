# Training

The training interface accepts paired images through a CSV manifest. Prepare licensed data separately; datasets and original private manifests are not bundled.

## Paired data

```text
data/
├── train.csv
├── lq/
└── gt/
```

Copy the schema in `recipes/train.example.csv` and use relative `lq,gt` paths below the data root. Each pair must share dimensions and support a 256-pixel crop. The loader applies the same crop, flips and right-angle rotation to both images. Paths escaping the data root are rejected.

Prepare `configs/local-model-files.json` as a JSON list of local model paths or nested lists of shard paths. Include the transformer, text encoder and VAE from the downloaded inventory. Point `--processor` at the processor component directory listed in that inventory.

## Launch

```bash
accelerate launch --num_processes 4 -m demoire_edit.train \
  --manifest data/train.csv --data-root data \
  --model-paths configs/local-model-files.json \
  --processor "$DEMOIRE_EDIT_PROCESSOR_PATH" \
  --output outputs/training \
  --micro-batch 1 --accumulation 24
```

Set `DEMOIRE_EDIT_PROCESSOR_PATH` to your local processor directory. The launch uses the maintained default schedule; use `demoire-train --help` to inspect the available options. `--steps` overrides the number of optimizer steps. `--initialize-lora` explicitly supplies an initialization adapter; no historical initialization adapter is loaded silently.

Four processes × micro-batch 1 × accumulation 24 gives an effective batch size of 96. The learning rate follows a cosine schedule from 1e-4 to 1e-7. Five denoising steps are unrolled for the maintained objective.

## Objective

| Component | Weight |
| --- | --- |
| Velocity | 1.0 |
| Latent MSE | 1.0 |
| Image L1 | 0.5 |
| LPIPS | 0.3 |
| Focal frequency loss | 0.2 |

LPIPS uses a VGG network internally; no additional VGG-feature, Lab or Chroma objective is applied. Missing or nonfinite components raise errors rather than silently contributing zero.

The maintained recipe includes the requested latent MSE extension. It does not establish the historical runtime recipe of the supplied weights. Consult [Recipe and checkpoint](recipe-and-checkpoint.md) before making training or reproduction claims. Objective weighting/backward checks passed with LPIPS/FFL test doubles; actual LPIPS initialization and a complete real training run remain unverified.
