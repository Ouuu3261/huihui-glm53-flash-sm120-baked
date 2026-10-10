"""Verify the baked build receipt without networking, hashing, or copy-up."""
import hashlib
import json
import os
import pathlib

REVISION = 'fb12424e973ee12f7a13cf01e81128d5337bb407'
MODEL_DIR = '/opt/models/Huihui-GLM-5.3-Flash-TP2-W4A16'


def verify(root):
    root = pathlib.Path(root)
    receipt = json.loads((root/'.verified-vast.json').read_text())
    sums = (root/'SHA256SUMS').read_bytes()
    if receipt.get('revision') != REVISION or receipt.get('manifest_sha256') != hashlib.sha256(sums).hexdigest():
        raise RuntimeError('Baked checkpoint receipt does not match the pinned model')
    expected = {}
    for line in sums.decode().splitlines():
        digest, name = line.split('  ', 1)
        rel = pathlib.PurePosixPath(name)
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest) or rel.is_absolute() or '..' in rel.parts or name in expected:
            raise RuntimeError('Invalid baked checksum manifest')
        path = root/name
        if not path.is_file() or path.is_symlink():
            raise RuntimeError('A baked model file is missing; no download will be attempted')
        stat = path.stat()
        expected[name] = [stat.st_size, stat.st_mtime_ns]
    if receipt.get('files') != expected:
        raise RuntimeError('Baked model files changed; no download or copy will be attempted')
    index = json.loads((root/'model.safetensors.index.json').read_text())
    shards = set(index['weight_map'].values())
    if len(shards) != 90 or any(name not in expected for name in shards):
        raise RuntimeError('Baked checkpoint index is incomplete')
    print('BAKED_MODEL_READY 90 shards; build SHA256 receipt matched; no network, rehash, or copy', flush=True)


if __name__ == '__main__':
    verify(os.environ.get('MODEL_DIR', MODEL_DIR))
