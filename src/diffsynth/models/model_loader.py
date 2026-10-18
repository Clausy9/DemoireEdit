import importlib
from typing import Any

from ..configs.model_configs import MODEL_CONFIGS
from ..configs.vram_management_module_maps import VRAM_MANAGEMENT_MODULE_MAPS
from ..core.loader import hash_model_file, load_model


def _import_symbol(dotted_path: str) -> Any:
    module_name, attr_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _build_state_dict_converter(converter_path: str | None):
    if converter_path is None:
        return None
    converter = _import_symbol(converter_path)
    if isinstance(converter, type):
        return converter()
    return converter


def _resolve_module_map(module_map: dict | None):
    if module_map is None:
        return None
    resolved = {}
    for source_module, target_module in module_map.items():
        source = _import_symbol(source_module) if isinstance(source_module, str) else source_module
        target = _import_symbol(target_module) if isinstance(target_module, str) else target_module
        resolved[source] = target
    return resolved


def _should_enable_vram_management(vram_config: dict | None) -> bool:
    if not vram_config:
        return False
    management_keys = (
        "offload_device",
        "offload_dtype",
        "onload_device",
        "onload_dtype",
        "preparing_device",
        "preparing_dtype",
    )
    return any(vram_config.get(key) is not None for key in management_keys)


class ModelPool:
    def __init__(self):
        self.model = []
        self.model_name = []
        self.model_path = []

    def _match_model_config(self, path):
        hash_with_shape = hash_model_file(path, with_shape=True)
        hash_without_shape = hash_model_file(path, with_shape=False)
        matches = [
            config for config in MODEL_CONFIGS
            if config["model_hash"] in {hash_with_shape, hash_without_shape}
        ]
        if not matches:
            raise ValueError(
                f"Unsupported model file: {path}. "
                f"hash_with_shape={hash_with_shape} hash_without_shape={hash_without_shape}"
            )
        return matches[0]

    def auto_load_model(self, path, vram_config=None, vram_limit=None, clear_parameters=False):
        model_config = self._match_model_config(path)
        model_class_path = model_config["model_class"]
        model_class = _import_symbol(model_class_path)
        state_dict_converter = _build_state_dict_converter(model_config.get("state_dict_converter"))
        module_map = None
        if _should_enable_vram_management(vram_config):
            module_map = _resolve_module_map(VRAM_MANAGEMENT_MODULE_MAPS.get(model_class_path))

        model = load_model(
            model_class,
            path,
            config=model_config.get("extra_kwargs"),
            state_dict_converter=state_dict_converter,
            module_map=module_map,
            vram_config=vram_config,
            vram_limit=vram_limit,
        )
        self.model.append(model)
        self.model_name.append(model_config["model_name"])
        self.model_path.append(path)
        return model

    def fetch_model(self, model_name: str, index=0):
        matched = [model for name, model in zip(self.model_name, self.model) if name == model_name]
        if not matched:
            return None
        if index == "all":
            return matched
        return matched[index]
