# Models

## Layout

```text
weights/
├── base/                 # Downloaded using the pinned component manifest
└── demoireedit.safetensors   # Separately obtained restoration adapter
```

Download the adapter from [Hugging Face](https://huggingface.co/umu/DemoireEdit). The download helper pins the verified release revision and checks the source identity:

```bash
python scripts/download_adapter.py --output weights
```

The script checks the pinned size and SHA-256 after downloading. Set:

```bash
export DEMOIRE_EDIT_WEIGHTS_PATH=weights/demoireedit.safetensors
```

The filename is a local binding; renaming a file does not change its identity or training history. The accepted size is 472,047,184 bytes and the SHA-256 is:

```text
15ad05bfb1b8c9d975605b18195abcf8485f28a26bb9b07dc350740a465319fd
```

Both the CLI and demo verify this identity before loading. The adapter contains 1,440 BF16 tensors matching 720 linear adapter targets. For a structural check without loading the full pretrained model:

```bash
python scripts/validate_lora_structure.py \
  --lora "$DEMOIRE_EDIT_WEIGHTS_PATH" --report outputs/model-check.json
```

This check inspects actual tensors against a metadata-only model skeleton; it is not a complete GPU inference.

## Provenance and terms

The [model card](model-card.md) records the upstream components and historical checkpoint identity. The [recipe note](recipe-and-checkpoint.md) explains which training details are established and which remain unresolved. Model and example-image redistribution terms must be confirmed separately from the source-code license.
