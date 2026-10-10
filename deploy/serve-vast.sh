#!/usr/bin/env bash
# Candidate Vast container launcher; does not start nested Docker.
set -euo pipefail
MODEL_DIR=${MODEL_DIR:-/workspace/models/Huihui-GLM-5.3-Flash-TP2-W4A16}
CACHE_DIR=${CACHE_DIR:-/workspace/cache/huihui-glm53-sm120-v05}
STATE_DIR=${STATE_DIR:-/workspace/huihui-glm53-tp2}
PORT=${PORT:-8000}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-786432}
MEM_FRACTION=${MEM_FRACTION:-0.97}
MTP=${MTP:-0}
[[ -f "$MODEL_DIR/config.json" && -f "$MODEL_DIR/model.safetensors.index.json" ]] || {
  echo "MODEL_DIR is missing checkpoint metadata: $MODEL_DIR" >&2; exit 2;
}
mkdir -p "$CACHE_DIR"
export CUDA_VISIBLE_DEVICES=0,1
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:2:16:8
[[ -f "$STATE_DIR/communication.env" ]] || {
  echo 'Communication preflight receipt is missing' >&2; exit 2;
}
source "$STATE_DIR/communication.env"
export SGLANG_PCIE_IPC_MAX_NUMEL=786432
export SGLANG_EXPERIMENTAL_DSA_KPOOL_METADATA_FUSION=1
export SGLANG_LM_HEAD_FP8=1
export TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/torchinductor"
export TILELANG_CACHE_DIR="$CACHE_DIR/tilelang"
export TRITON_CACHE_DIR="$CACHE_DIR/triton"
export SGLANG_KDA_EXTEND_BLOCK_TOKENS=2048
export SGLANG_BCG_RAGGED_SHAPES=1
export SGLANG_BCG_RAGGED_MAX_BS=1
export SGLANG_BCG_SEPARATE_CAPTURE_SESSIONS=1
speculative_args=()
if [[ "$MTP" == 1 ]]; then
  speculative_args=(--speculative-algorithm EAGLE --speculative-num-steps 3
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4)
fi
[[ "$MTP" == 0 || "$MTP" == 1 ]] || exit 2
[[ "$CONTEXT_LENGTH" =~ ^[1-9][0-9]*$ ]] || exit 2
exec sglang serve \
  --model-path "$MODEL_DIR" \
  --served-model-name Huihui-GLM-5.3-Flash-TP2-W4A16 \
  --tp 2 --enable-multimodal --warmups serving_coverage \
  --image-processor-backend torchvision --mm-preprocessing-device cpu \
  --quantization modelopt_mixed --moe-runner-backend flashinfer_cutlass \
  --disable-shared-experts-fusion --disable-custom-all-reduce \
  --attention-backend dsa --linear-attn-backend triton \
  --context-length "$CONTEXT_LENGTH" --max-total-tokens "$CONTEXT_LENGTH" \
  --kv-cache-dtype fp8_e4m3 --mem-fraction-static "$MEM_FRACTION" \
  --chunked-prefill-size 4096 --max-prefill-tokens 4096 \
  --max-running-requests 1 --max-mamba-cache-size 12 --mamba-ssm-dtype bfloat16 \
  --cuda-graph-backend-prefill breakable --cuda-graph-bs-prefill 64 128 \
  --cuda-graph-backend-decode full --cuda-graph-bs-decode 1 \
  --dsa-prefill-backend flashinfer_sparse_mla --dsa-decode-backend flashinfer_sparse_mla \
  "${speculative_args[@]}" \
  --reasoning-parser glm45 --tool-call-parser glm47 \
  --enable-metrics --enable-cache-report --host 127.0.0.1 --port "$PORT"
