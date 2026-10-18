# Recipe and checkpoint binding

The maintained training recipe follows the manuscript with the owner-requested latent MSE addition: velocity, latent MSE, L1, LPIPS and focal frequency loss (FFL). VGG feature, Lab and Chroma losses are excluded. Velocity and latent MSE each have weight 1.0, with L1 0.5, LPIPS 0.3 and FFL 0.2. Latent MSE was explicitly added by the owner after the initial manuscript-alignment decision; this maintained recipe contains one more term than the current manuscript formula. The new clean-target interpolation follows the manuscript; historical training code interpolated the noisy input instead. Neither code modification establishes the historical runtime recipe.

The owner selected the existing step-3000 LoRA for this release. Its SHA-256 is `15ad05bfb1b8c9d975605b18195abcf8485f28a26bb9b07dc350740a465319fd`, and its size is 472,047,184 bytes. It contains 1,440 BF16 tensors. The source digest was independently confirmed after download.

The historical launcher includes additional loss defaults and initializes from an earlier step-9560 adapter. Actual runtime overrides have not been established. Aligning new code does not change the existing tensors or prove that this checkpoint was trained with the revised recipe. Preserve this distinction in model cards and reproduction instructions. Do not relabel the checkpoint as newly trained.

The selected inference route is the dual-scale condition-shift backend: short-side 1024 Stage 1 with a black auxiliary reference, 10 steps per stage, ratio 0.3, ceil-based switch, noise initialization for Stage 2, CFG 1.0. The older residual-subtraction server is not an equivalent backend. The manuscript companion adapter has unimplemented backend hooks and is not a runnable release by itself.

Existing outputs may be used as historical examples only after their source pairing is verified. One new real-photo GPU inference check has now completed on a 1536×1536 crop using the selected LoRA and complete dual-scale route. Full training and benchmark reproduction remain unverified.
