<div align="center">

# cuopus

**A bit-exact CUDA port of the libopus 1.5.2 decoder.**
Decode OGG/Opus entirely on the GPU — output byte-for-byte identical to the reference implementation.

Built by [Kaption AI](https://kaptionai.com).

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-12.x%2B-76b900.svg)](#build)
[![libopus](https://img.shields.io/badge/libopus-1.5.2-orange.svg)](https://opus-codec.org/)
[![Validation](https://img.shields.io/badge/bit--exactness-5.03M%20packets%2C%200%20mismatches-success.svg)](#validation)

</div>

---

cuopus ports the libopus 1.5.2 decoder — SILK, CELT, and hybrid modes, in its
`FIXED_POINT` configuration — to CUDA device code. A granule-exact C demuxer builds
segment "chains", one CUDA warp decodes one or two chains in parallel across thousands
of chains, and an on-device polyphase resampler emits float32 16 kHz PCM straight into a
caller-provided device buffer.

Because the port targets the fixed-point reference build, the output is not "close" — it
is **byte-for-byte identical to libopus**.

> [!IMPORTANT]
> **Read this before you benchmark it.** Entropy decoding is serial per stream, so one
> GPU warp decodes at a small fraction of one CPU core. cuopus only wins by running
> thousands of chains at once — the measured crossover against a 12-process CPU pool is
> around **200 concurrent chains** (roughly 3+ minutes of audio at 1 s segmentation).
> **Short single files are faster on the CPU.** This is a batch-shape win, not a
> per-stream speed win.

**Status: research-grade**, extracted from a production transcription stack. It works and
it is heavily validated, but see [Known limitations](#known-limitations) — particularly
the strict demuxer — before depending on it.

## Contents

- [Why](#why) · [Validation](#validation) · [Quick start](#quick-start) · [How it works](#how-it-works)
- [Performance](#performance) · [API](#api) · [Build](#build)
- [Known limitations](#known-limitations) · [Repo tour](#repo-tour) · [Contributing](#contributing) · [Licensing](#licensing)

## Why

As far as a sweep of GitHub and arXiv in August 2026 could find, no general-purpose GPU
audio decoder exists. NVDEC is video-only, and published GPU audio work is essentially all
encoding-adjacent DSP, not entropy-coded codec decoding.

That gap matters for GPU inference pipelines — speech recognition, audio LLMs — that
ingest hours of compressed audio. Decoding on the CPU and uploading raw PCM makes the CPU
pool and the PCIe copy the bottleneck while the model's GPU sits idle waiting. cuopus
decodes the compressed bytes on the device the model already occupies, so a 2-hour file
goes from upload-and-wait to a single kernel-side pipeline whose output tensor is already
resident.

## Validation

Bit-exactness is the whole point, so it is gated rather than asserted. Every decoder
change was re-run against a driver linked to a real `--enable-fixed-point` libopus 1.5.2
build, comparing sample counts, the final entropy-coder range, and **every PCM sample**:

| gate | compared | mismatches |
|---|---|---|
| Phases 1–3 (SILK, CELT 2.5/5/20 ms, hybrid, transition, embedded redundancy) | 1.62 M packet decodes / 1.07 B PCM samples | **0** |
| Warp-stride sweep, re-gated at strides 4 / 8 / 16 / 32 | 3.41 M packet decodes / 1.34 B PCM samples | **0** |
| On-device resampler vs the torch reference | 65.2 M samples | **max diff 0.0** |

Full methodology, corpora, and per-gate results: [PHASE2_RESULTS.md](PHASE2_RESULTS.md),
[PHASE3_RESULTS.md](PHASE3_RESULTS.md), [BENCHMARKS.md](BENCHMARKS.md).

## Quick start

```bash
git clone https://github.com/Kaption-AI/cuopus.git
cd cuopus

# Build out/libcuopus.so + out/cuopus_decode.
# Fetches vendor/opus-1.5.2 if absent; builds in a CUDA container by default.
./build_and_test.sh lib

# Decode a file to float32 16 kHz mono
./out/cuopus_decode --ogg input.ogg output.f32 --f16k
```

From Python, decoding straight into a torch CUDA tensor with no host round trip:

```python
from cuopus_native import decode_ogg_to_cuda_tensor, decode_ogg_multi

t  = decode_ogg_to_cuda_tensor("file.ogg")     # 1-D torch.cuda.FloatTensor @ 16 kHz
ts = decode_ogg_multi(["a.ogg", "b.ogg"])      # N files, one merged kernel launch
```

To verify bit-exactness on your own hardware, `./build_and_test.sh phase1` runs a
self-contained gate: it synthesizes a corpus, validates the OGG index against `ffprobe`,
then runs the same 144-symbol range-decoder schedule through real libopus on the CPU and
the CUDA port on the GPU, compared word-for-word on every packet. Needs a GPU and ffmpeg.

## How it works

- **Mechanical device port.** `device/gen/**` is generated from the vendored libopus
  sources by `device/port_silk.py` — regenerable, and diffable against upstream
  file-by-file. The transforms are deliberately mechanical: function definitions get
  `__device__` prefixes, file-scope const tables become `__device__` data, `ALLOC`/alloca
  becomes a per-thread arena in global memory, the two recursive call sites (CELT
  `quant_partition`, PLC frame splitting) become explicit iteration, and varargs `ctl`
  calls become typed helpers. Every deviation from upstream is listed in the phase docs.

- **Chain-per-warp parallelism.** Opus entropy decoding is control-flow divergent, so
  packing decoders into adjacent lanes serializes them. One decoder per warp is 6.7×
  faster than one per lane; the tuned default is stride 16 (2 chains per warp), proven
  bit-exact at strides 4/8/16/32.

- **Granule-exact C demuxer** (`cuopus_demux.h`): OGG page parse, packet reassembly, exact
  per-packet 48 kHz offsets derived from TOC bytes and cross-checked against every page
  granule. Preskip and RFC 7845 end-trim fold into per-chain trim/keep, so concatenated
  output is exactly the file's playable PCM. ~31 ms to demux a 2 h file.

- **250 ms preroll convergence.** Chains restart the decoder, so each chain starts 250 ms
  early and discards the preroll; measured −71.7 dBFS RMS (max 133 LSB) against a serial
  full-stream fixed-point decode.

- **On-device 49-tap Kaiser 48 k → 16 k polyphase resampler**, bit-identical to the
  reference torch `conv1d` implementation (max diff 0.0 over 65 M samples).

- **Two ways to run it:** a CLI/daemon (`decode_cli.cu`) with dual-stream copy/compute
  overlap and pinned D2H for standalone use, and an in-process library (`libcuopus.cu` →
  `libcuopus.so`) that runs on the caller's CUDA context and stream with zero D2H copies,
  spawns no threads or streams, and refuses to run during CUDA stream capture.

## Performance

RTX 5090 (sm_120), 24-core host, against a tuned 12-process CPU decode pool
(ffmpeg/libopus), warm daemon, median of 4:

| duration | CPU pool | cuopus (GPU) | GPU/CPU |
|---|---|---|---|
| 1 m | 63 ms | **59 ms** | 0.93× |
| 5 m | 60 ms | 86 ms | 1.44× — *CPU wins* |
| 15 m | 144 ms | **111 ms** | 0.77× |
| 30 m | 292 ms | **148 ms** | 0.51× |
| 1 h | 573 ms | **289 ms** | 0.50× |
| 2 h | 1,172 ms | **521 ms** | 0.44× |

Roughly 2–2.25× faster than the 12-process pool at 1–2 h files, about break-even at a few
minutes, and slower below the ~200-chain crossover. At a 1 h file the daemon wall splits
into demux 12 ms, decode kernel 176 ms, compact + resample 1.2 ms, D2H 10.6 ms
(overlapped).

**VRAM:** ~1.4 GB floor (CUDA context + the driver's per-resident-thread local-memory
pool) plus working buffers; budget ~2.4 GB peak for 2 h files in the f16k path.

[BENCHMARKS.md](BENCHMARKS.md) has the full component splits, the parameter sweeps that
were run and rejected, tunables, and two negative results worth not repeating.

## API

The in-process library. The full contract is in the `libcuopus.cu` header comment — read
it before integrating.

```c
int  cuopus_probe(const uint8_t *ogg, size_t n, int64_t *samples16k,
                  char *errbuf, size_t errcap);
     /* demux + validate only (host work, no GPU): exact output length */

int  cuopus_decode_f16k(const uint8_t *ogg, size_t n, float *out_dev,
                        size_t out_cap, int64_t *out_len,
                        char *errbuf, size_t errcap);
     /* decode + resample into the caller's DEVICE buffer, default stream */

int  cuopus_decode_f16k_on(..., void *cuda_stream, ...);
     /* same, on an explicit cudaStream_t (e.g. torch's current stream) */

int  cuopus_decode_f16k_multi(const char **oggs, const size_t *lens, int n,
                              void **out_devs, const size_t *caps,
                              int64_t *out_lens, void *cuda_stream,
                              char *errbuf, size_t errcap);
     /* N files in one merged kernel launch; per-file output bitwise
        identical to the single-file call */

int  cuopus_warmup(char *errbuf, size_t errcap);
     /* pre-size the driver's local-memory pool (~910 MB, once per context) */
```

Plus `cuopus_last_arena_overflow`, `cuopus_workspace_bytes`, and
`cuopus_release_workspaces`.

**Return codes:** `0` ok · `1` soft failure (bad file — fall back to CPU) · `2` CUDA error
(context suspect) · `3` caller error.

The library never creates a CUDA context, spawns no threads or streams, and must not be
called during stream capture — it refuses when it can detect it.

A Python ctypes wrapper is included at `python/cuopus_native.py`. Separately,
`decode_cli.cu` builds a standalone CLI (`--ogg in.ogg out.pcm` for 48 kHz s16, `--f16k`
for float32 16 kHz, `--serve <unix socket>` for a persistent daemon), and
`cuopus_pipeline.py` is a daemon client returning numpy arrays.

## Build

Requires CUDA 12.x+ `nvcc`. Only sm_120 (RTX 5090, CUDA 13 driver stack) has been tested;
other architectures compile but are unvalidated.

`./build_and_test.sh lib` is the one-command path — it builds inside a throwaway Docker
CUDA container (override the image with `CUOPUS_IMG`, or set `CUOPUS_NO_DOCKER=1` to use a
local `nvcc`, and `CUOPUS_ARCH` to retarget). To build by hand:

```bash
python3 device/port_silk.py    # optional: regenerate device/gen from vendor/

nvcc -O2 -arch=sm_120 -rdc=true -DCUOPUS_CUDA -DFIXED_POINT=1 -DOPUS_BUILD \
  -Idevice -Idevice/gen -Idevice/gen/silk -Idevice/gen/celt -Idevice/gen/include \
  -Xcompiler -fPIC -shared libcuopus.cu -o out/libcuopus.so

nvcc -O2 -arch=sm_120 -rdc=true -DCUOPUS_CUDA -DFIXED_POINT=1 -DOPUS_BUILD \
  -Idevice -Idevice/gen -Idevice/gen/silk -Idevice/gen/celt -Idevice/gen/include \
  decode_cli.cu -o out/cuopus_decode
```

> [!NOTE]
> The single-translation-unit build relies on the generated `__noinline__` discipline.
> Naive full inlining takes nvcc's `cicc` **over 25 minutes**; as generated, it builds in
> seconds.

## Known limitations

Read these before using it.

1. **The OGG demuxer's granule cross-check is strict and currently rejects files from some
   real-world encoders.** The decode path itself is bit-exact; container parsing is the
   open issue, and it is the best first contribution — relax the cross-check to a warning
   plus resync, keeping the per-chain sample-count validation.
2. **Output is focused on float32 16 kHz mono** (the `f16k` path, built for speech-model
   front-ends). A 48 kHz int16 path exists in the CLI. Mono only — stereo decode paths are
   ported but untested.
3. **Tested only on sm_120 with a CUDA 13 driver stack** (built inside a CUDA 12.9
   container). Bit-exactness elsewhere is expected, since it is integer arithmetic, but it
   is not validated.
4. **Fixed-point build only.** The port targets libopus's `FIXED_POINT` configuration —
   that is what makes bit-exactness achievable. There is no float decode path.
5. **Not exercised by the validation corpora:** CELT NB/MB configs 16–23, stereo, DTX/CNG
   under real loss, and non-zero OggOpus output gain.

## Repo tour

| path | what |
|---|---|
| `libcuopus.cu` | in-process library (runs on the caller's context/stream) |
| `decode_cli.cu` | CLI + persistent daemon (dual-stream, pinned D2H) |
| `cuopus_demux.h` | granule-exact OGG/Opus demuxer (C, header-only) |
| `cuopus_kernels.cuh`, `cuopus_entdec.cuh`, `cuopus_io.h` | kernels, range decoder, formats |
| `device/port_silk.py` | generates `device/gen/**` from `vendor/opus-1.5.2` |
| `device/gen/**` | the mechanically ported decoder (committed, regenerable) |
| `harness.cu`, `harness2.cu`, `harness_host.cpp` | bit-exactness gate harnesses |
| `python/cuopus_native.py` | ctypes wrapper (torch CUDA tensors) |
| `cuopus_pipeline.py`, `ogg_index.py`, `gpu_resample.py` | daemon client, Python demuxer, torch resampler reference |
| `BENCHMARKS.md`, `ROADMAP.md`, `PHASE2_RESULTS.md`, `PHASE3_RESULTS.md` | measurement history and gate results |
| `vendor/opus-1.5.2/` | upstream libopus, unmodified |

## Contributing

Issues and PRs are welcome. Two things to know first:

- **The demuxer is the known weak point.** Limitation 1 above is the highest-value fix and
  needs no GPU to work on.
- **Anything touching the decoder must stay bit-exact.** The gates in
  `PHASE2_RESULTS.md` / `PHASE3_RESULTS.md` are the standard. Note that those full gates
  used private audio corpora and reference dumps from a driver linked to a real
  fixed-point libopus, so they are **not one-command reproducible from this repo alone** —
  the harness sources and the full methodology are here, the corpora are not.
  `./build_and_test.sh phase1` is the self-contained gate that does run from a fresh
  clone.

## Licensing

cuopus is a derived work of libopus (BSD 3-clause, Xiph.Org Foundation and contributors).
The CUDA port is © 2026 the cuopus contributors under the same BSD 3-clause terms — see
[LICENSE](LICENSE).

The vendored sources under `vendor/opus-1.5.2/` are the unmodified upstream 1.5.2 release
(verified byte-identical to the official tarball, sha256 `65c1d2f7…`) and retain their
original `COPYING` file. Opus is additionally subject to the royalty-free patent licenses
listed there.
