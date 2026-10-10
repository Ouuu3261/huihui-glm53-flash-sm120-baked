"""Build bounded-size OCI layers; publish an assembled image without Docker."""
import argparse
import base64
import concurrent.futures
import copy
import hashlib
import io
import json
import os
import pathlib
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.parse

import requests

HERE = pathlib.Path(__file__).resolve().parents[1]
LOCK_PATH = HERE/'model-lock.json'
LOCK = json.loads(LOCK_PATH.read_text())
LOCK_SHA = hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest()
OCI_MANIFEST = 'application/vnd.oci.image.manifest.v1+json'
OCI_CONFIG = 'application/vnd.oci.image.config.v1+json'
OCI_LAYER = 'application/vnd.oci.image.layer.v1.tar+gzip'
SOURCE_LABEL = 'org.opencontainers.image.source'
MODEL_PREFIX = LOCK['model_dir'].lstrip('/')
DEPLOY_PREFIX = 'opt/huihui-deploy'
RECIPE = 'paired-safetensors-gzip-mtime0-v1'


def dump(path, value):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(path).write_text(json.dumps(value, indent=2)+'\n')


def json_bytes(value):
    return json.dumps(value, separators=(',', ':'), sort_keys=True).encode()


def digest(data):
    return 'sha256:'+hashlib.sha256(data).hexdigest()


def crane(*args, optional=False):
    result = subprocess.run(['crane', *args], capture_output=True)
    if result.returncode:
        if optional:
            return None
        # Registry errors can include signed redirects. Keep those out of logs.
        raise RuntimeError('crane '+args[0]+' failed, exit '+str(result.returncode))
    return result.stdout


def remote_image(ref, optional=False):
    raw = crane('manifest', ref, optional=optional)
    if raw is None:
        return None
    raw = raw.strip()
    manifest = json.loads(raw)
    config = json.loads(crane('config', ref))
    return raw, manifest, config


def download(row, target):
    url = 'https://huggingface.co/'+LOCK['repository']+'/resolve/'+LOCK['revision']+'/'+urllib.parse.quote(row['file'], safe='/')
    for attempt in range(5):
        sha = hashlib.sha256(); received = 0
        try:
            # A fresh resolve request on every retry avoids expired CDN redirects.
            with requests.get(url, stream=True, timeout=(30, 180)) as response:
                if response.status_code != 200:
                    raise RuntimeError('Model download HTTP '+str(response.status_code))
                with pathlib.Path(target).open('wb') as output:
                    for block in response.iter_content(4*1024*1024):
                        if block:
                            output.write(block); sha.update(block); received += len(block)
            if received != row['bytes'] or sha.hexdigest() != row['sha256']:
                raise RuntimeError('Model artifact size or SHA256 mismatch: '+row['file'])
            print('SOURCE_SHA256_OK', row['file'], received, flush=True)
            return
        except (requests.RequestException, RuntimeError):
            if attempt == 4:
                raise RuntimeError('Pinned source artifact could not be verified: '+row['file']) from None
            time.sleep(2**attempt)


class HashingWriter:
    def __init__(self, stream):
        self.stream = stream; self.sha = hashlib.sha256(); self.count = 0

    def write(self, data):
        self.sha.update(data); self.count += len(data)
        return self.stream.write(data)

    def flush(self):
        self.stream.flush()


class HashingReader:
    def __init__(self, stream):
        self.stream = stream; self.sha = hashlib.sha256()

    def read(self, size=-1):
        data = self.stream.read(size); self.sha.update(data); return data


def layer_info(name, size, mode=0o644, directory=False):
    item = tarfile.TarInfo(name)
    item.size = size; item.mode = mode; item.uid = item.gid = 0
    item.uname = item.gname = ''; item.mtime = 0
    if directory:
        item.type = tarfile.DIRTYPE
    return item


def create_layer(output, files):
    """Hash the uncompressed tar and every archived file during packing."""
    started = time.monotonic()
    with pathlib.Path(output).open('wb') as compressed:
        process = subprocess.Popen(['pigz', '-1', '-n', '-p', '4', '-c'], stdin=subprocess.PIPE, stdout=compressed)
        writer = HashingWriter(process.stdin)
        try:
            with tarfile.open(fileobj=writer, mode='w|', format=tarfile.USTAR_FORMAT) as archive:
                directories = sorted({str(parent) for name, _, _, _ in files for parent in pathlib.PurePosixPath(name).parents if str(parent) != '.'}, key=lambda x: (x.count('/'), x))
                for name in directories:
                    archive.addfile(layer_info(name, 0, 0o755, directory=True))
                for name, source, expected, mode in files:
                    if isinstance(source, bytes):
                        assert expected is None or hashlib.sha256(source).hexdigest() == expected
                        archive.addfile(layer_info(name, len(source), mode), io.BytesIO(source))
                    else:
                        with pathlib.Path(source).open('rb') as stream:
                            reader = HashingReader(stream)
                            archive.addfile(layer_info(name, pathlib.Path(source).stat().st_size, mode), reader)
                            if reader.sha.hexdigest() != expected:
                                raise RuntimeError('Archive input SHA256 mismatch: '+name)
            writer.flush()
        finally:
            process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError('Layer compression failed')
    sha = hashlib.sha256()
    with pathlib.Path(output).open('rb') as stream:
        for block in iter(lambda: stream.read(16*1024*1024), b''):
            sha.update(block)
    size = pathlib.Path(output).stat().st_size
    if size >= 10_000_000_000:
        raise RuntimeError('Layer exceeds GHCR layer limit')
    print('LAYER_PACKED', size, 'bytes', round(time.monotonic()-started, 1), 'seconds', flush=True)
    return {'mediaType': OCI_LAYER, 'digest': 'sha256:'+sha.hexdigest(), 'size': size}, 'sha256:'+writer.sha.hexdigest(), writer.count


def fragment(ref, layer_path, descriptor, diff_id, details):
    crane('append', '--oci-empty-base', '--new_layer', str(layer_path), '--new_tag', ref+'-stage')
    labels = {SOURCE_LABEL: LOCK['source_url'], 'ai.huihui.model-lock.sha256': LOCK_SHA,
              'ai.huihui.layer.recipe': RECIPE, 'ai.huihui.layer.record.b64': base64.urlsafe_b64encode(json_bytes(details)).decode()}
    args = ['mutate', ref+'-stage', '--tag', ref]
    for key, value in labels.items():
        args += ['--label', key+'='+value]
    crane(*args)
    raw, manifest, config = remote_image(ref)
    assert len(manifest['layers']) == 1
    assert manifest['layers'][0] == descriptor
    assert config['rootfs']['diff_ids'] == [diff_id]
    return {'manifest_digest': digest(raw), 'layer': descriptor, 'diff_id': diff_id, **details}


def weights(group, evidence):
    shards = [row for row in LOCK['files'] if row['file'].endswith('.safetensors')]
    rows = shards[group*2:group*2+2]
    assert 0 <= group < 45 and len(rows) == 2
    ref = LOCK['image']+':weights-'+LOCK['revision'][:8]+'-'+f'{group+1:02d}'
    existing = remote_image(ref, optional=True)
    if existing:
        raw, manifest, config = existing
        labels = config['config'].get('Labels', {})
        details = json.loads(base64.urlsafe_b64decode(labels.get('ai.huihui.layer.record.b64', 'e30=')))
        if labels.get('ai.huihui.model-lock.sha256') == LOCK_SHA and labels.get('ai.huihui.layer.recipe') == RECIPE and details.get('files') == rows:
            assert len(manifest['layers']) == len(config['rootfs']['diff_ids']) == 1
            record = {'manifest_digest': digest(raw), 'layer': manifest['layers'][0], 'diff_id': config['rootfs']['diff_ids'][0], **details}
            dump(evidence, record); print('PUBLISHED_LAYER_REUSED', group+1, flush=True); return
        raise RuntimeError('Existing fragment differs from the locked build')
    with tempfile.TemporaryDirectory(prefix='huihui-weight-') as temporary:
        root = pathlib.Path(temporary)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda row: download(row, root/row['file']), rows))
        output = root/'layer.tar.gz'
        files = [(MODEL_PREFIX+'/'+row['file'], root/row['file'], row['sha256'], 0o644) for row in rows]
        descriptor, diff_id, uncompressed = create_layer(output, files)
        # Remove the two verified sources before upload to bound peak disk use.
        for row in rows:
            (root/row['file']).unlink()
        details = {'group': group, 'files': rows, 'uncompressed_bytes': uncompressed,
                   'archived_file_sha256_verified': True, 'mtime': 0}
        record = fragment(ref, output, descriptor, diff_id, details)
        dump(evidence, record)
        print('WEIGHT_LAYER_PUBLISHED', group+1, descriptor['digest'], flush=True)


def base(evidence):
    _, original, original_config = remote_image(LOCK['base_image'])
    assert len(original['layers']) == len(original_config['rootfs']['diff_ids']) == LOCK['base_layer_count']
    assert original_config['os'] == 'linux' and original_config['architecture'] == 'amd64'
    ref = LOCK['image']+':runtime-base-v05'
    crane('mutate', LOCK['base_image'], '--label', SOURCE_LABEL+'='+LOCK['source_url'], '--tag', ref)
    raw, manifest, config = remote_image(ref)
    assert manifest['layers'] == original['layers']
    assert config['rootfs'] == original_config['rootfs']
    dump(evidence, {'ref': ref, 'manifest_digest': digest(raw), 'original_ref': LOCK['base_image'],
                    'manifest': manifest, 'config': config, 'layers_preserved': True})
    print('RUNTIME_BASE_PUBLISHED', len(manifest['layers']), flush=True)


def metadata(evidence):
    rows = [row for row in LOCK['files'] if not row['file'].endswith('.safetensors')]
    with tempfile.TemporaryDirectory(prefix='huihui-metadata-') as temporary:
        root = pathlib.Path(temporary)
        def fetch(row):
            path = root/row['file']; path.parent.mkdir(parents=True, exist_ok=True); download(row, path)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(fetch, rows))
        receipt = {'revision': LOCK['revision'], 'files': {row['file']: [row['bytes'], 0] for row in LOCK['files'] if row['file'] != 'SHA256SUMS'},
                   'manifest_sha256': next(row['sha256'] for row in rows if row['file'] == 'SHA256SUMS'),
                   'verification': 'All source and archived file SHA256 verified by the image build; Docker verifies OCI layer digests'}
        files = [(MODEL_PREFIX+'/'+row['file'], root/row['file'], row['sha256'], 0o644) for row in rows]
        files += [(MODEL_PREFIX+'/.verified-vast.json', json_bytes(receipt), None, 0o644),
                  (DEPLOY_PREFIX+'/model-lock.json', LOCK_PATH.read_bytes(), LOCK_SHA, 0o644)]
        for path in sorted((HERE/'deploy').glob('*')):
            if path.is_file():
                files.append((DEPLOY_PREFIX+'/'+path.name, path, hashlib.sha256(path.read_bytes()).hexdigest(), 0o755 if path.suffix == '.sh' else 0o644))
        descriptor, diff_id, uncompressed = create_layer(root/'metadata.tar.gz', files)
        # Verify real tar extraction preserves receipt mtimes on the Linux runner.
        extracted = root/'extracted'; extracted.mkdir()
        with tarfile.open(root/'metadata.tar.gz', 'r:gz') as archive:
            archive.extractall(extracted, filter='data')
        model = extracted/MODEL_PREFIX
        for row in LOCK['files']:
            if row['file'].endswith('.safetensors'):
                with (model/row['file']).open('wb') as stream:
                    stream.truncate(row['bytes'])  # Sparse placeholders: stat-only boot test.
                os.utime(model/row['file'], ns=(0, 0))
        subprocess.run(['python3', str(HERE/'deploy/verify_baked_model.py')], env={**os.environ, 'MODEL_DIR': str(model)}, check=True)
        details = {'files': rows, 'uncompressed_bytes': uncompressed, 'mtime': 0,
                   'archived_file_sha256_verified': True, 'extracted_receipt_stat_check_passed': True}
        record = fragment(LOCK['image']+':metadata-'+LOCK['revision'][:8]+'-v1', root/'metadata.tar.gz', descriptor, diff_id, details)
        dump(evidence, record)
    print('METADATA_AND_BOOT_RECEIPT_PUBLISHED', flush=True)


class Registry:
    """Only small config and manifest writes; large blobs use pinned crane."""
    def __init__(self):
        self.name = LOCK['image'].removeprefix('ghcr.io/')
        response = requests.get('https://ghcr.io/token', params={'service': 'ghcr.io', 'scope': 'repository:'+self.name+':pull,push'},
                                auth=(os.environ['GITHUB_ACTOR'], os.environ['GITHUB_TOKEN']), timeout=30)
        if response.status_code != 200:
            raise RuntimeError('Registry authentication HTTP '+str(response.status_code))
        self.headers = {'Authorization': 'Bearer '+response.json()['token']}
        self.base = 'https://ghcr.io/v2/'+self.name+'/'

    def request(self, method, path, allowed, **kwargs):
        url = path if path.startswith('https://') else urllib.parse.urljoin(self.base, path)
        if urllib.parse.urlsplit(url).netloc != 'ghcr.io':
            raise RuntimeError('Unexpected registry upload host')
        headers = {**self.headers, **kwargs.pop('headers', {})}
        for attempt in range(4):
            try:
                response = requests.request(method, url, headers=headers, timeout=(30, 180), **kwargs)
                if response.status_code in allowed:
                    return response
                if response.status_code not in [429, 500, 502, 503, 504]:
                    raise RuntimeError('Registry '+method+' HTTP '+str(response.status_code))
            except requests.RequestException:
                if attempt == 3:
                    raise RuntimeError('Registry '+method+' transport failed') from None
            time.sleep(2**attempt)
        raise RuntimeError('Registry '+method+' retry budget exhausted')

    def put_config(self, data):
        sha = digest(data)
        exists = self.request('HEAD', 'blobs/'+sha, [200, 404])
        if exists.status_code == 404:
            started = self.request('POST', 'blobs/uploads/', [202])
            location = urllib.parse.urljoin('https://ghcr.io', started.headers['Location'])
            parts = urllib.parse.urlsplit(location)
            query = urllib.parse.parse_qsl(parts.query)+[('digest', sha)]
            location = urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(query)))
            self.request('PUT', location, [201], data=data, headers={'Content-Type': 'application/octet-stream'})
        return {'mediaType': OCI_CONFIG, 'digest': sha, 'size': len(data)}

    def put_manifest(self, tag, data):
        response = self.request('PUT', 'manifests/'+tag, [201], data=data, headers={'Content-Type': OCI_MANIFEST})
        assert response.headers.get('Docker-Content-Digest') == digest(data)


def assemble(evidence_dir, output):
    root = pathlib.Path(evidence_dir)
    base_record = json.loads((root/'base.json').read_text())
    metadata_record = json.loads((root/'metadata.json').read_text())
    weights_records = [json.loads((root/f'weight-{i:02d}.json').read_text()) for i in range(45)]
    rows = [row for record in weights_records for row in record['files']]
    assert rows == [row for row in LOCK['files'] if row['file'].endswith('.safetensors')]
    assert sum(row['bytes'] for row in rows) == LOCK['shard_bytes']
    assert all(record['archived_file_sha256_verified'] and record['mtime'] == 0 for record in weights_records)
    assert metadata_record['extracted_receipt_stat_check_passed']
    records = weights_records+[metadata_record]
    config = copy.deepcopy(base_record['config'])
    config['rootfs']['diff_ids'] += [record['diff_id'] for record in records]
    config['history'] = config.get('history', [])+[{'created_by': 'Bake pinned Huihui model layer '+str(i), 'comment': RECIPE} for i in range(len(records))]
    runtime = config['config']
    overrides = {'MODEL_DIR': LOCK['model_dir'], 'HUIHUI_MODEL_BAKED': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                 'CONTEXT_LENGTH': '786432', 'CLIENT_OUTPUT_BUDGET': '65536', 'MTP': '0'}
    env = {entry.split('=', 1)[0]: entry.split('=', 1)[1] for entry in runtime.get('Env', [])}
    env.update(overrides); runtime['Env'] = [key+'='+value for key, value in env.items()]
    runtime['Labels'] = {**runtime.get('Labels', {}), SOURCE_LABEL: LOCK['source_url'],
                         'org.opencontainers.image.description': 'Huihui GLM-5.3-Flash TP2, all weights baked; 768K total context, 64K client output budget',
                         'ai.huihui.model.repository': LOCK['repository'], 'ai.huihui.model.revision': LOCK['revision'],
                         'ai.huihui.model-lock.sha256': LOCK_SHA, 'ai.huihui.model.baked': 'true'}
    runtime['Entrypoint'] = ['/opt/nvidia/nvidia_entrypoint.sh']
    runtime['Cmd'] = ['bash', '/opt/huihui-deploy/start-baked.sh']
    runtime['WorkingDir'] = '/workspace'
    # VOLUME declarations could hide immutable model files or allocate a second copy.
    assert not any(LOCK['model_dir'].startswith(path.rstrip('/')+'/') or path == LOCK['model_dir'] for path in runtime.get('Volumes', {}))
    layers = base_record['manifest']['layers']+[record['layer'] for record in records]
    assert len(layers) == len(config['rootfs']['diff_ids']) == 123 and len(layers) < 128
    assert max(layer['size'] for layer in layers) < 10_000_000_000
    registry = Registry()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        def verify_blob(layer):
            response = registry.request('HEAD', 'blobs/'+layer['digest'], [200])
            assert int(response.headers['Content-Length']) == layer['size']
        list(pool.map(verify_blob, layers))
    manifest = {'schemaVersion': 2, 'mediaType': OCI_MANIFEST, 'config': registry.put_config(json_bytes(config)), 'layers': layers,
                'annotations': {SOURCE_LABEL: LOCK['source_url'], 'org.opencontainers.image.base.name': LOCK['base_image']}}
    data = json_bytes(manifest)
    registry.put_manifest(LOCK['tag'], data)
    actual_raw, actual_manifest, actual_config = remote_image(LOCK['image']+':'+LOCK['tag'])
    assert digest(actual_raw) == digest(data) and actual_manifest == manifest and actual_config == config
    report = {'image': LOCK['image'], 'tag': LOCK['tag'], 'digest': digest(data), 'image_ref': LOCK['image']+'@'+digest(data),
              'layers': len(layers), 'compressed_bytes': sum(layer['size'] for layer in layers), 'model_bytes': LOCK['model_bytes'],
              'model_revision': LOCK['revision'], 'model_lock_sha256': LOCK_SHA, 'context_length': 786432, 'client_output_budget': 65536,
              'shards': 90, 'all_archived_artifact_sha256_verified': True, 'all_registry_blob_sizes_verified': True,
              'extracted_receipt_stat_check_passed': True, 'runtime_base_layers_preserved': True,
              'registry_manifest_and_config_readback_verified': True, 'gpu_cold_start_tested': False}
    dump(output, report); dump(pathlib.Path(output).with_name('final-manifest.json'), manifest); dump(pathlib.Path(output).with_name('final-config.json'), config)
    print('BAKED_IMAGE_PUBLISHED_AND_VERIFIED', json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['base', 'weights', 'metadata', 'assemble'])
    parser.add_argument('--group', type=int)
    parser.add_argument('--output', required=True)
    parser.add_argument('--evidence-dir')
    args = parser.parse_args()
    if args.command == 'base':
        base(args.output)
    elif args.command == 'weights':
        weights(args.group, args.output)
    elif args.command == 'metadata':
        metadata(args.output)
    else:
        assemble(args.evidence_dir, args.output)
