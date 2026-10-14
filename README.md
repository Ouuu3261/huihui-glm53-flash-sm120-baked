# Huihui GLM-5.3-Flash TP2 — complete weights baked into the image

For two RTX PRO 6000 Blackwell 96GB GPUs (SM120), using the pinned GLM-specific
SGLang v0.5.0 runtime. The complete public Huihui checkpoint is included in the
Docker image. A new Vast instance reads the model directly from the image after
image loading completes; there is no Hugging Face download or second model copy
at container startup.

Deployment image: `ghcr.io/ouuu3261/huihui-glm53-flash-sm120-baked:768k-fb12424e-v2-3d65d275639a`.

Immutable digest: `sha256:3d65d275639acaa5d77ffb5c91d2868ff26885cbadbdbc7e661f12cd217917e2`.

The complete compressed image is 190.7 GB. Use the digest-specific v2 tag above
in Vast. [Published build receipt](release.json) and
[public verification](public-verification.json) describe the final image.

The v1 image verified and downloaded correctly, but it had 123 layers. Vast SSH
adds setup layers before it starts the container, which triggered Docker's
`max depth exceeded` guard. v2 reduces the image to 108 layers without changing
the model, runtime, context, or output settings.

## Defaults

- TP=2, one running request, **786432 total context tokens (768K)**.
- **65536 (64K) output budget in the client/request**, including reasoning tokens.
  Input and output share the total window; allow roughly 704K input minus template
  overhead when reserving 64K output. EOS may finish generation earlier.
- FP8 E4M3 KV; BF16 recurrent state; 12 Mamba cache slots; 4096-token chunked prefill.
- `mem-fraction-static=0.97`; MTP disabled; no CPU weight offload.
- GPU topology selected at startup; SYS cross-NUMA pairs use the established NCCL
  fallback. The mixed FP8/BF16 shared-expert compatibility patch is idempotent.
- API: `127.0.0.1:8000`; model alias `Huihui-GLM-5.3-Flash-TP2-W4A16`.

## Vast startup

Select this image in SSH/direct mode. An onstart script should launch:

```bash
mkdir -p /workspace/huihui-glm53-tp2
nohup /opt/nvidia/nvidia_entrypoint.sh bash /opt/huihui-deploy/start-baked.sh \
  >> /workspace/huihui-glm53-tp2/server.log 2>&1 < /dev/null &
```

The launcher holds a startup lock, copies only small deployment scripts into the
writable state directory, validates the baked receipt, and starts SGLang.
Model files stay at `/opt/models/Huihui-GLM-5.3-Flash-TP2-W4A16`, outside a
`/workspace` bind mount. There is no model `VOLUME`, runtime `chown`, copying, or
paid re-download fallback. If the baked files are missing or changed, startup
fails while preserving them.

Kernel caches and logs are under `/workspace`. The existing 200GB Vast disk default
can be used for one model and caches; the host also needs enough free Docker storage
to download and unpack this large image. Do not copy the model into `/workspace`.
Full cold-start GPU validation on a newly rented host is separate from image build
verification; the build does not rent a GPU or change an existing instance.

Forward the API with:

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=3 -p YOUR_SSH_PORT root@YOUR_HOST \
  -L 127.0.0.1:18001:127.0.0.1:8000
```

Client Base URL: `http://127.0.0.1:18001/v1`; contextWindow: `786432`;
maxTokens / request max_tokens: `65536`.

## Reproducible packaging

- Model: [Chenwouu/Huihui-GLM-5.3-Flash-TP2-W4A16](https://huggingface.co/Chenwouu/Huihui-GLM-5.3-Flash-TP2-W4A16),
  pinned to `fb12424e973ee12f7a13cf01e81128d5337bb407`.
- Runtime: `ghcr.io/ormandj/sglang-glm53-flash-sm120@sha256:b4c2f4feb504dba6cb6358e480193a9cba1c9c6f5625dcefa926827b06c17c77`.
- `model-lock.json` fixes every artifact's size and SHA256. All 90 shards and
  published metadata are checked before and during packing. Tensor bytes and the
  Huihui conversion are unchanged.
- 77 preserved runtime layers + 30 three-shard weight layers + 1 metadata layer =
  108 layers. This reserves twenty layers for Vast's SSH setup image and the
  writable container layer, below overlay2's 128-layer limit. Each weight layer
  has at most 6.11 GB of uncompressed shard input, below GHCR's per-layer size cap.
  Fixed epoch mtimes allow cheap receipt checks at startup.
- The workflow publishes layers independently with bounded runner disk usage,
  then assembles an OCI image from their verified descriptors. Intermediate
  `weights-*`, `metadata-*`, and `runtime-base-v05` tags are build caches, not
  complete deployment images.
- The build verifies source/artifact SHA256, archive contents, all published blob
  sizes, layer/diff-ID count, image configuration, and extracted receipt behavior.
  It also tests missing/changed files and confirms boot verification opens no
  weight contents and makes no network calls. It does not claim a full 768K
  inference or 64K continuous generation benchmark.

Model licensing is included at the model path. The original runtime retains its
upstream packages and notices. Build scripts in this repository are MIT licensed.

Vast documents image downloads as the `Loading` phase, which is not charged as a
running instance. GPU model loading, first-time compilation, and warmup still run
after the container starts. This packaging does not promise zero network or disk
fees. See [Vast instance management](https://docs.vast.ai/guides/instances/manage-instances).
