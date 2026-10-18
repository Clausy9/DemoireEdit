# Evaluation

## Real photographs

Run the inference entry point with the same adapter, seed and sampler settings for all images. Preserve the input, output and failure set. Review moiré reduction, color fidelity, detail retention and unintended content changes. A capture without aligned ground truth does not support PSNR or SSIM reporting.

The project page presents three historical input/result pairs from paper Figure 8. They are qualitative paper examples, not new repository inference runs or a benchmark dataset. The earlier one-crop integration run remains documented separately in the validation receipts.

## Paired evaluation

For licensed aligned pairs, record the dataset split, image alignment, crop policy, color space, dynamic range, metric implementation and aggregation. Report failures alongside successful outputs. Paired metric execution and paper-scale evaluation have not been run for this release candidate; an automated benchmark evaluator is not included.

## Reproduction records

Record the model hash, base-component revisions, sampler, seed, dimensions, precision, offload settings and hardware. Keep historical paper results distinct from new execution results. See [Reproducibility](reproducibility.md) for the current scoped checks.
