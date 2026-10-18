# Contributing

Use a separate environment and the documented dependency lock. Install the package editable before running `python -m unittest discover -s tests -v` so checks exercise the maintained package.

Describe the problem, the change and the validation performed. For inference changes, record the model and adapter hashes, seed, resolution, precision, sampler settings and hardware. Keep observed results distinct from historical paper results. Do not introduce silent loss fallbacks or substitute checkpoints without documenting the change.

Do not commit credentials, local machine paths, datasets, checkpoints, logs or raw experiment histories. Preserve upstream notices. Confirm redistribution rights before adding model or image assets. Model and data files are not covered automatically by the code license.

Full retraining is not required for a documentation or focused correctness change. State untested behavior and remaining limits explicitly.
