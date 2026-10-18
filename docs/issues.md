# Preparation issues and fixes

| Issue | Resolution | Validation limit |
| --- | --- | --- |
| Legacy service used residual subtraction instead of the selected dual-scale route | Extract actual progressive sampler and remove hardcoded experiment strategies | Full real GPU pipeline passed on one 1536px crop |
| Manuscript and historical training defaults differ | Separate maintained recipe from historical weight provenance | Runtime overrides unknown |
| Owner requested latent MSE | Add term with weight 1.0; retain velocity 1.0, L1 0.5, LPIPS 0.3, FFL 0.2 | Weighting and gradient checks passed with LPIPS/FFL test doubles |
| Historical training interpolated noisy LQ rather than the manuscript's clean target | Maintained trajectory uses clean-target interpolation | No new scientific results claimed |
| Width and height were reversed in the training preprocessor | Use PIL height for height and width for width | Correction made; full training integration remains unverified |
| Loss failures silently became zero | Missing or nonfinite LPIPS/FFL terms stop training | Actual LPIPS network initialization pending |
| System Python lacked Pillow for asset collection | Use existing image-capable interpreter, without installing on historical host | Collection retries required over intermittent SSH |
| Original-resolution sample transfer exceeded the connection window | Collect labeled downscaled previews and lossless matched crops in memory | Previews are not full-resolution evaluation assets |
| Offline mode left model paths unset | Resolve existing cache patterns even when downloads are disabled; fail early on missing files | Both dedicated regression checks passed in the fresh environment |
| Online installer hung while holding the isolated environment lock | Only matching project installers were stopped; fetch official PyPI wheels in bounded ranges, check SHA-256 and install offline | Fresh install, dependency check and all 11 CPU tests passed |
| UI defaults could re-encode results as WebP | Use explicit lossless PNG input and output | Real upload/inference/download passed; output pixels exactly matched CLI at the same seed |

No original checkpoint, experiment, manuscript, proxy job or service was changed by these code fixes.
