from .model_configs import MODEL_CONFIGS
from .vram_management_module_maps import VRAM_MANAGEMENT_MODULE_MAPS

# Backward-compat shim for model loader implementations that expect
# VERSION_CHECKER_MAPS to exist in the configs package.
VERSION_CHECKER_MAPS = {}
