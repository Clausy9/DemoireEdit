# Model card

## Intended use

The restoration adapter supports research on image demoiréing through dual-scale condition-shift editing. Results can alter fine detail and color; visual inspection is required where fidelity matters.

## Components and identity

The transformer is based on Qwen-Image-Edit-2509. The text encoder and VAE come from Qwen-Image, and the processor comes from Qwen-Image-Edit. Exact upstream repository identifiers, revisions and file hashes are pinned in `configs/base-models.json` for reproducible downloads and attribution.

The selected historical adapter is the step-3000 artifact, containing 1,440 BF16 tensors and occupying 472,047,184 bytes. Its SHA-256 is `15ad05bfb1b8c9d975605b18195abcf8485f28a26bb9b07dc350740a465319fd`. This identity is independent of its local filename. The original tensors have not been retrained or modified for this repository.

## Training provenance

The historical launcher initializes from an earlier step-9560 adapter and includes additional loss defaults. Actual runtime overrides have not been established. Therefore, the adapter is not described as trained from the base for exactly its checkpoint step count, or as trained under the maintained source recipe.

The maintained source uses velocity, latent MSE, L1, LPIPS and FFL, with separate VGG-feature, Lab and Chroma objectives excluded. The latent MSE extension and clean-target interpolation do not establish the training history of the existing tensors. See [Recipe and checkpoint](recipe-and-checkpoint.md).

## Validation

The actual adapter matches all 720 expected linear targets. Full base-model loading and one real 1536 × 1536 photo crop completed the two-stage GPU pipeline. The real Gradio API output matched the CLI pixels at the same seed. Hardware, timings and receipts are documented in [Reproducibility](reproducibility.md).

The observed crop has reduced fine stripes but residual broad color variation and softer detail. No aligned ground truth is available. Full 4K inference, complete training and paper-scale benchmarks remain unverified. No universal quality claim is made.

## Distribution terms

The source includes modified Apache-2.0 DiffSynth code with upstream notices. Base-model, adapter and image/data terms are separate. No public adapter distribution license is asserted here; redistribution review remains open. The project does not bundle private training manifests or claim to reproduce their ordering and data mixture.
