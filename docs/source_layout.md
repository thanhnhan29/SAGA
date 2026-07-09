# Source Layout

`SAGA/backbones` contains code-only copies of the current local implementations:

- `self_forcing`: copied from `Self-Forcing`.
- `causal_forcing`: copied from `Causal-Forcing`.
- `causvid`: copied from `CausVid`.

The copy intentionally excludes checkpoints, model weights, generated videos,
evaluation outputs, VBench folders, Python caches, and local build artifacts.

The SAGA method code lives in the top-level `saga/` package:

- `saga/noise.py` contains Dual-AR and IID noise helpers.
- `saga/guidance.py` contains the inference-time guidance module used by all
  three backbones.
- The backbone folders are vendored inference targets and should only keep
  minimal glue code at the denoising call sites.

Wrapper scripts normalize paths, expose environment variables, and add both the
repo root and selected backbone to `PYTHONPATH`.
