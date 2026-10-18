# Reproducibility

## Executed checks

| Check | Observed result |
| --- | --- |
| Fresh installation | Linux x86_64, Python 3.11; core and demo dependencies consistent |
| CPU checks | 11 passed, including condition scheduling and offline model lookup |
| Adapter validation | 1,440 finite BF16 tensors, 720 matching linear targets |
| GPU pipeline | One real 1536 × 1536 crop, 178.42 seconds including loading |
| Demo API | Real upload, inference and PNG download, 199.19 seconds |
| CLI/API agreement | Identical output pixels at the same seed |

The GPU was an RTX 6000 Ada 48 GB with CPU offload, BF16, ten steps per stage, short-side 1024 for the first stage, condition ratio 0.3, black auxiliary reference and CFG 1.0. Seed was 0. Durations are observations, not throughput guarantees.

Input crop and output hashes are in `provenance/a6000-preflight.json` and `provenance/a6000-gpu-check.json`. The API check is in `provenance/a6000-demo-check.json`. Base model revisions and hashes are in `configs/base-models.json`.

## Limits

The input is an unscaled center crop from a real photograph, without aligned ground truth. Fine stripes and colored rings were reduced, but residual broad color variation and detail softening remain. No paper-scale quality evaluation or full 4K run has been performed.

The objective unit checks use LPIPS/FFL test doubles. They establish weighting and error behavior, not real loss-network initialization or a complete training run. The source recipe and the historical weight's training history remain separate; see the model card.

## Paper examples and current package naming

The project page now shows historical qualitative pairs extracted from paper Figure 8, rather than the original integration crop. Source hashes and crop coordinates are recorded in `provenance/qualitative-examples.json`. These assets are not evidence of a new run. The original GPU and API receipts predate the package rename to `dsc_edit`; timings and pixel agreement describe those executions. Package, entry-point and CPU checks are repeated for the renamed source separately.

The public release is branded DemoiréEdit, with package `demoire_edit` and `demoire-*` commands. The release checks record the renamed source separately; historical GPU timings remain bound to their original run receipts.
