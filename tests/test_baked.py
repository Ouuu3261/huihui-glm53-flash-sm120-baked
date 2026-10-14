import hashlib
import importlib.util
import json
import os
import pathlib
import socket
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('verify', ROOT/'deploy/verify_baked_model.py')
verify = importlib.util.module_from_spec(spec); spec.loader.exec_module(verify)


class BakedStartupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)
        self.files = {f'model-{i:05d}-of-00090.safetensors': b'verified tiny fixture' for i in range(1, 91)}
        self.files['model.safetensors.index.json'] = json.dumps({'weight_map': {f'tensor{i}': name for i, name in enumerate(self.files)}}).encode()
        for name, data in self.files.items():
            (self.root/name).write_bytes(data); os.utime(self.root/name, ns=(0, 0))
        sums = ''.join(hashlib.sha256(data).hexdigest()+'  '+name+'\n' for name, data in self.files.items()).encode()
        (self.root/'SHA256SUMS').write_bytes(sums)
        self.receipt = {'revision': verify.REVISION, 'manifest_sha256': hashlib.sha256(sums).hexdigest(),
                        'files': {name: [len(data), 0] for name, data in self.files.items()}}
        self.write_receipt()

    def tearDown(self):
        self.temporary.cleanup()

    def write_receipt(self):
        (self.root/'.verified-vast.json').write_text(json.dumps(self.receipt))

    def test_complete_baked_model_has_no_network_and_opens_no_weight_contents(self):
        original_open = pathlib.Path.open
        def guarded_open(path, *args, **kwargs):
            if str(path).endswith('.safetensors'):
                self.fail('Boot must not rehash or copy baked weight contents')
            return original_open(path, *args, **kwargs)
        with patch('socket.socket', side_effect=AssertionError('No network at boot')), patch('pathlib.Path.open', guarded_open):
            verify.verify(self.root)

    def test_missing_weight_fails_without_network(self):
        (self.root/next(iter(self.files))).unlink()
        with patch('socket.socket', side_effect=AssertionError('No repair download')), self.assertRaises(RuntimeError):
            verify.verify(self.root)

    def test_changed_weight_fails_and_is_preserved(self):
        path = self.root/next(iter(self.files)); path.write_bytes(b'changed')
        with self.assertRaises(RuntimeError):
            verify.verify(self.root)
        self.assertEqual(path.read_bytes(), b'changed')

    def test_wrong_revision_rejected(self):
        self.receipt['revision'] = 'wrong'; self.write_receipt()
        with self.assertRaises(RuntimeError):
            verify.verify(self.root)

    def test_replaced_manifest_rejected(self):
        (self.root/'SHA256SUMS').write_bytes(b'new manifest')
        with self.assertRaises(RuntimeError):
            verify.verify(self.root)

    def test_tar_extract_preserves_epoch_mtime_and_boot_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary); path = root/'layer.tar'
            with tarfile.open(path, 'w') as archive:
                for source in self.root.iterdir():
                    info = archive.gettarinfo(source, arcname=source.name); info.mtime = 0
                    with source.open('rb') as stream:
                        archive.addfile(info, stream)
            target = root/'extract'; target.mkdir()
            with tarfile.open(path) as archive:
                try:
                    archive.extractall(target, filter='data')
                except TypeError:  # Local Python builds before the filter API.
                    archive.extractall(target)
            verify.verify(target)

    def test_launch_profile_and_baked_path_are_fixed(self):
        bootstrap = (ROOT/'deploy/bootstrap-vast.sh').read_text()
        serve = (ROOT/'deploy/serve-vast.sh').read_text()
        self.assertIn('export CONTEXT_LENGTH=786432 CLIENT_OUTPUT_BUDGET=65536', bootstrap)
        self.assertIn('export MODEL_DIR='+verify.MODEL_DIR, bootstrap)
        self.assertIn('verify_baked_model.py', bootstrap)
        self.assertNotIn('model_cache.py', bootstrap)
        self.assertIn('--max-running-requests 1', serve)
        self.assertIn('--context-length "$CONTEXT_LENGTH" --max-total-tokens "$CONTEXT_LENGTH"', serve)
        self.assertIn('--kv-cache-dtype fp8_e4m3', serve)

    def test_vast_ssh_overlay_layer_headroom_is_reserved(self):
        lock = json.loads((ROOT/'model-lock.json').read_text())
        shards = [row for row in lock['files'] if row['file'].endswith('.safetensors')]
        group_size = lock['weight_shards_per_layer']
        groups = [shards[index:index+group_size] for index in range(0, len(shards), group_size)]
        self.assertEqual(lock['release'], 'v2')
        self.assertEqual(len(groups), lock['weight_layer_count'])
        self.assertTrue(all(1 <= len(group) <= group_size for group in groups))
        self.assertLess(max(sum(row['bytes'] for row in group) for group in groups), 10_000_000_000)
        image_layers = lock['base_layer_count'] + lock['weight_layer_count'] + 1
        self.assertEqual(image_layers, 108)
        self.assertLessEqual(image_layers, lock['overlay2_safe_image_layer_limit'])
        self.assertLessEqual(image_layers + lock['vast_ssh_wrapper_layer_budget'], lock['overlay2_safe_image_layer_limit'])


if __name__ == '__main__':
    unittest.main()
