#!/usr/bin/env bash
set -euo pipefail
STATE_DIR=${STATE_DIR:-/workspace/huihui-glm53-tp2}
mkdir -p "$STATE_DIR"
if [[ "${BOOTSTRAP_LOCK_HELD:-0}" != 1 ]]; then
  exec 9>"$STATE_DIR/bootstrap.lock"
  flock -n 9 || { echo 'Existing deployment holds the startup lock'; exit 0; }
fi
export PATH=/opt/sglang/bin:/usr/local/cuda/bin:/usr/local/nvidia/bin:$PATH
export STATE_DIR
export MODEL_DIR=/opt/models/Huihui-GLM-5.3-Flash-TP2-W4A16
export HUIHUI_MODEL_BAKED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CONTEXT_LENGTH=786432 CLIENT_OUTPUT_BUDGET=65536
export CACHE_DIR=${CACHE_DIR:-/workspace/cache/huihui-glm53-sm120-v05}
export HF_XET_CACHE=${HF_XET_CACHE:-/workspace/cache/hf-xet}
export HF_HUB_CACHE=${HF_HUB_CACHE:-/workspace/cache/huggingface}
export MTP=${MTP:-0}
export P2P_MODE=${P2P_MODE:-auto}
mkdir -p "$CACHE_DIR"
trap 'rc=$?; echo "STARTUP_FAILED exit=$rc; model files and kernel cache retained" >&2; exit "$rc"' ERR
echo 'STAGE GPU and runtime preflight'
python - <<'PY'
import platform, torch
assert platform.machine() == 'x86_64', 'This image requires x86_64'
assert torch.cuda.is_available() and torch.cuda.device_count() == 2, 'Expected two assigned GPUs'
for i in range(2):
    assert torch.cuda.get_device_capability(i) == (12, 0), 'Expected SM120 Blackwell GPUs'
    prop = torch.cuda.get_device_properties(i)
    assert prop.total_memory >= 94_000_000_000, 'Expected the 96GB GPU class'
    print('GPU', i, prop.name, round(prop.total_memory/2**30, 2), 'GiB')
print('TORCH', torch.__version__, 'CUDA', torch.version.cuda)
PY
echo 'STAGE Runtime compatibility fix'
python "$STATE_DIR/apply_runtime_fixes.py"
echo 'STAGE Select GPU communication path'
python "$STATE_DIR/select_communication.py"
echo 'STAGE Verify baked model build receipt; no network or copying'
python "$STATE_DIR/verify_baked_model.py"
python - <<'PY'
import json, os, pathlib
root = pathlib.Path(os.environ['MODEL_DIR'])
index = json.loads((root/'model.safetensors.index.json').read_text())
shards = set(index['weight_map'].values())
assert len(shards) == 90 and all((root/name).is_file() for name in shards)
print('MODEL_READY', len(shards), 'shards; all artifact checksums passed')
PY
echo 'STAGE SGLang TP2 single request 768K, FP8 KV; MTP baseline disabled by default'
exec bash "$STATE_DIR/serve-vast.sh"
