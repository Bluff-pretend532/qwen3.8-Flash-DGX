# Qwen3.8-Flash-Next on a single DGX Spark (GB10)

Run **Qwen3.8-Flash-Next** — a ~176B-parameter model (125B main + 51B n-gram, 6B
active) — on **one NVIDIA DGX Spark / ASUS GX10** with **vLLM**, at full prefill
speed and with MTP speculative decoding.

The catch this repo solves: the NVFP4 checkpoint is **122 GiB**, which does not fit
next to a usable KV cache in the Spark's **128 GB unified pool**. 44 GiB of that is
the n-gram embedding ("PLE") table — a pure lookup that a token only touches 16 rows
of. This repo adds one patch to the official vLLM image that **serves that table from
NVMe via `mmap`** instead of keeping it resident. Weights drop to **~76 GiB**, the
rest of the pool goes to KV, and everything runs on stock GB10 kernels.

The result, versus the llama.cpp GGUF that was the only working option on a Spark
before: **~5× faster prefill, and MTP (which the GGUF cannot do).**

| | llama.cpp IQ4_XS | **this repo (vLLM NVFP4)** |
|---|---|---|
| Prefill | ~540 tok/s | **~2,400–2,660 tok/s** |
| Decode (no speculation) | ~22 tok/s | ~17 tok/s |
| Decode **+ MTP=2** | not supported | **~27 tok/s** (≈67% accept) |
| Resident GPU memory | ~94 GiB (GGUF) | ~76 GiB weights + KV |
| N-gram table | mmap (built in) | **mmap from NVMe (this patch)** |

*Measured on an ASUS GX10 (GB10, 128 GB), single request, ctx 32k. Prefill is the
headline: Flash-Next's whole point is its sparse attention (QSA), and llama.cpp has
no QSA kernel — it runs dense, so its prefill is its weakest axis. vLLM uses the real
kernels.*

---

## Requirements

- An **NVIDIA DGX Spark or compatible GB10 (sm_121)** box, 128 GB unified memory,
  aarch64, recent NVIDIA driver, Docker with the NVIDIA container runtime.
- **~130 GB free disk** for the checkpoint, on reasonably fast storage (the table is
  read from it at runtime — NVMe strongly recommended; the Spark's onboard NVMe is ideal).
- The base image is multi-arch, so `docker build` also works on x86 Blackwell
  (sm_120, e.g. RTX PRO 6000) for testing, though this is tuned for the Spark.

## Quickstart

```bash
git clone https://github.com/blazux/qwen3.8-Flash-DGX.git
cd qwen3.8-Flash-DGX

docker build -t qwen38-flash-dgx .        # ~1 min: official image + one patch
scripts/download-weights.sh               # ~122 GiB, resumable (one-time)
scripts/serve.sh                          # boots on :18300 (~8 min to load)
docker logs -f qwen38-flash               # wait for "Application startup complete"
scripts/smoke-test.sh                     # health + coherence + prefill/decode numbers
```

Then hit the OpenAI-compatible API:

```bash
curl http://localhost:18300/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-flash-next",
  "messages": [{"role":"user","content":"Write a haiku about a desktop supercomputer."}],
  "max_tokens": 512
}'
```

Turn on MTP speculative decoding and a longer context:

```bash
MTP=2 CTX=65536 scripts/serve.sh
```

## Tuning (env vars for `scripts/serve.sh`)

| Var | Default | Notes |
|---|---|---|
| `PORT` | `18300` | API port |
| `CTX` | `32768` | Max context (native 262144; KV grows with it) |
| `SEQS` | `2` | Max concurrent sequences |
| `GPU_MEM` | `0.78` | Fraction of the 128 GB pool for weights+KV. Keep headroom — on a Spark the OS and the GPU share this pool, and an OOM there can freeze the box. |
| `MTP` | `0` | Speculative tokens from the model's MTP head. `2`–`3` gives ~1.6× decode at ~67% accept. |
| `PREWARM` | `0` | `1` streams the 48 GiB table once at boot to warm the page cache — steadier first-request latency, ~10 s extra startup. |
| `WORKERS` | `32` | Threads used for the mmap gather. |

## How it fits — the one idea

A token's n-gram lookup reads **16 rows × 160 bytes ≈ 2.5 KB**. Over a 20k-token
prefill that's ~1.3 GB of small reads — under a second on NVMe, and the hot n-grams
stay in the page cache. So the 44 GiB table never needs to be in the unified pool:
we `mmap` the checkpoint's `model-plefp8-*.safetensors` shards and gather rows on
demand. Nothing else about the model changes — the hashing, dequant, and the sparse
attention all run stock.

Full details, including the three GB10-specific bugs this works around, are in
[docs/HOW-IT-WORKS.md](docs/HOW-IT-WORKS.md).

## What's in here

```
Dockerfile                 official vLLM Flash-Next image + the patch
src/vllm_ple_mmap.py       the patch (mmap PLE table; opaque splitting op)
src/test_ple_mmap_cpu.py   CPU unit test for the gather (no GPU needed)
scripts/download-weights.sh
scripts/serve.sh
scripts/smoke-test.sh
docs/HOW-IT-WORKS.md
```

Run the unit test (no GPU):

```bash
docker run --rm -v "$PWD/src:/t" -w /t --entrypoint python3 \
  qwen38-flash-dgx test_ple_mmap_cpu.py
```

## Limitations & notes

- **One big model at a time.** At `GPU_MEM=0.78` this uses most of the 128 GB pool;
  don't co-locate another large model.
- **`--no-enable-prefix-caching` is required** (a GB10 GDN kernel bug corrupts on the
  cached-block path) and **full `torch.compile` is off** (an Inductor int64-indexing
  assert on sm_121); the serve script sets both. See the doc for why.
- Decode is a touch slower than the GGUF without MTP, because the gather does one
  host↔device sync per step; MTP more than makes up for it. A pinned-buffer path to
  remove that sync is a natural next optimization (PRs welcome).
- **Weights are not included** and the checkpoint carries Qwen's license (with a
  MAU/revenue clause) — review it before production use.

## Credits

- Model: **Qwen team, Alibaba** — Qwen3.8-Flash-Next.
- NVFP4 checkpoint: **[RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)**.
- Serving engine and base image: **vLLM** (`vllm/vllm-openai:qwen38-flash-next`,
  the `release/qwen38next` recipe / PR #53896).
- The mmap-PLE patch and the GB10 serving recipe in this repo: see [LICENSE](LICENSE) (Apache-2.0).
