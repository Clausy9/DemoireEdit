import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HAS_DEPS = importlib.util.find_spec('torch') is not None and importlib.util.find_spec('huggingface_hub') is not None

@unittest.skipUnless(HAS_DEPS,'PyTorch and Hugging Face Hub are required')
class OfflineModelConfigChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).resolve().parents[1]/'src/diffsynth/core/loader/config.py'
        spec=importlib.util.spec_from_file_location('isolated_model_config',path)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        cls.ModelConfig=module.ModelConfig
    def test_skip_download_resolves_existing_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'Example/Model/transformer/shard.safetensors'
            target.parent.mkdir(parents=True);target.write_bytes(b'path-discovery-only')
            with patch.dict(os.environ,{'DIFFSYNTH_MODEL_BASE_PATH':tmp,'DIFFSYNTH_SKIP_DOWNLOAD':'true'}):
                config=self.ModelConfig(model_id='Example/Model',origin_file_pattern='transformer/*.safetensors')
                config.download_if_necessary()
            self.assertEqual(config.path,str(target))
    def test_missing_offline_files_fail_early(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ,{'DIFFSYNTH_MODEL_BASE_PATH':tmp,'DIFFSYNTH_SKIP_DOWNLOAD':'true'}):
                config=self.ModelConfig(model_id='Example/Model',origin_file_pattern='transformer/*.safetensors')
                with self.assertRaises(FileNotFoundError):config.download_if_necessary()
if __name__=='__main__':unittest.main()
