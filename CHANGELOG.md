# Changelog

## 0.1.0.dev0 — release candidate

- Provide dual-scale condition-shift inference and a loopback-only PNG demo.
- Expose paired-data training with the maintained five-component objective.
- Pin official base-model revisions and platform-specific dependency wheel hashes.
- Validate offline model discovery, condition scheduling, loss weighting and failure guards.
- Record one real-photo GPU run and a matching real Gradio API roundtrip.

The historical step-3000 adapter is unchanged. Full training and paper-scale evaluation are not claimed.

### Repository presentation

- Organize method, visual results and usage as a research project repository.
- Split installation, model preparation, inference, training, evaluation and reproducibility guides.
- Add a method illustration and local environment/data examples.
- Use a neutral runtime weight filename; the accepted digest and historical provenance are unchanged.

### Paper naming and examples

- Align project presentation with the supplied paper title, Dual-Scale Condition-Shift Editing for Robust Real-World Image Demoiréing.
- Rename the Python package to `dsc_edit`, commands to `dsc-infer`, `dsc-demo`, and `dsc-train`, and runtime bindings to `DSC_EDIT_*`. The package identifier is a code name, not a claimed paper acronym.
- Replace the homepage integration crop with three historical qualitative pairs extracted from paper Figure 8, with source hashes and crop coordinates. They are not new inference results.
- Keep the original model tensors and prior validation receipts unchanged.

## DemoiréEdit release preparation

- Align project branding, package `demoire_edit`, `demoire-*` commands and `DEMOIRE_EDIT_*` bindings.
- Prepare a Hugging Face adapter package with source identity checks; tensor values remain unchanged.
