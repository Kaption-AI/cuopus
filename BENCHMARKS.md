# cuopus — performance work and measurements

**Hardware:** RTX 5090 (sm_120), CUDA 13 driver stack, 24-core host, builds inside a
CUDA 12.9 container. **Date:** 2026-08-30.

All numbers below are measured, not modelled. The CPU baseline throughout is a tuned
12-process decode pool (ffmpeg/libopus) on the same 24-core host — a deliberately strong
baseline, not a single-threaded strawman. Every decoder-affecting change in this document
was gated on the phase-3 bit-exactness harness before being kept (see
[PHASE3_RESULTS.md](PHASE3_RESULTS.md)).

Read the [crossover caveat](#the-crossover-is-the-whole-story) before quoting any of it.

---

## Headline: GPU vs a 12-process CPU pool

Warm daemon, median of 4, CPU pool re-measured in the same session. "GPU before" is the
untuned pipeline at the start of the performance work; "GPU after" is the tuned one.

| duration | CPU pool | GPU before | **GPU after** | GPU/CPU |
|---|---|---|---|---|
| 1 m | 63 ms | 287 ms | **59 ms** | **0.93×** |
| 5 m | 60 ms | 365 ms | 86 ms | 1.44× — CPU wins |
| 15 m | 144 ms | 526 ms | **111 ms** | **0.77×** |
| 30 m | 292 ms | 760 ms | **148 ms** | **0.51×** |
| 1 h | 573 ms | ~710 ms | **289 ms** | **0.50×** |
| 2 h | 1,172 ms | 2,294 ms | **521 ms** | **0.44×** |

Cold call (daemon start + decode, 1 m file): 206 ms. Output length matches the CPU pool
at every duration.

### The crossover is the whole story

Entropy decoding is inherently serial per stream, so a single GPU warp decodes at a small
fraction of one CPU core. The GPU only wins by running thousands of chains at once. The
measured crossover against the 12-process pool is around **~200 concurrent chains**
(roughly 3+ minutes of audio at 1 s segmentation). Below that, decode on the CPU.

This is a batch-shape win, not a per-stream speed win. Do not read the table above as
"cuopus is 2× faster than libopus" — it is not, and cannot be.

---

## Where the time goes (1 h file, warm daemon)

| component | before | after |
|---|---|---|
| demux (C, in daemon) | ~14 ms | 12 ms |
| H2D | (in wall) | 0.5 ms |
| decode kernel | 392 ms (preroll 1000, stride 32) | **176 ms** (preroll 250, stride 16) |
| compact + resample | — (torch, client side) | 1.2 ms (on device) |
| D2H | (serialized) | 10.6 ms (overlapped, pinned) |
| write out | (client) | 48 ms (mostly overlapped) |
| daemon wall | 481 ms | **244 ms** |
| client Python | 260 ms | **48 ms** |
| **pipeline wall** | **741 ms** | **292 ms** |

At the start of this work the kernel was only ~11 % of wall time. The rest was Python
demux (~0.25 ms per audio-second), per-call CUDA context creation (~200–500 ms fixed),
and un-overlapped transfers. Four of the five improvements below target that overhead
rather than the kernel.

---

## 1. Demux in C

`decode_cli.cu` takes the `.ogg` path directly, parsing pages, reassembling packets,
granule cross-checking and building the chain table in C (`cuopus_demux.h`). The Python
indexer (`ogg_index.py`) is retained as the reference implementation and for the harness.

**Validation:** `--dump-blob` / `--dump-chains` output is `cmp`-identical to
`ogg_index.py` on 9/9 files — both phase-2 corpora (480 / 23,872 and 445 / 3,979
pages/packets), the 2 h file (358,064 packets), and the six sweep files.

**Speed:** 2 h file demuxes in **31 ms in C** vs ~1,800 ms in Python (~60×); a 477 s file
in 1.6 ms.

## 2. Persistent daemon

`--serve <unix socket>` holds the CUDA context, module, and grow-only device buffers
across calls, removing the ~200–500 ms per-call context cost. The protocol is line-based
(`PING` / `DECODE` / `DECODE16F` / `QUIT`). Soft errors (bad file, decode error) reply
`ERR …` and keep serving; CUDA errors exit so the client can restart it. A zero-batch warm
launch at startup sizes the driver's local-memory pool, so the first real request is
already warm. Idle exit (`CUOPUS_SERVE_IDLE_S`, default 600 s) returns the ~1.3 GB
context + pool when unused.

Warm-call cost with the still-untuned kernel, already enough to win at 2 h:

| file | daemon warm wall | kernel | CPU pool |
|---|---|---|---|
| 477 s | **85 ms** | 67 ms | 61–77 ms — near parity |
| 2 h | **~800 ms** | ~475 ms | 1,083–1,178 ms — **GPU wins** |

For comparison, the one-shot subprocess it replaced took 339 ms for the same 477 s decode.

## 3. Preroll 1000 → 250 ms

Chains restart the decoder, so each chain starts early and discards the preroll until the
decoder converges. Preroll is pure wasted decode work: at 1 s chains it was 50 % of the
kernel, now 20 %. Isolated at 1 h, the kernel went 373 → 217 ms (−42 %).

Convergence measured on a 477 s corpus, 1 s chains, 48 kHz PCM vs a full-stream
FIXED_POINT serial libopus reference:

| preroll | max diff | exact | ±1 LSB | RMS |
|---|---|---|---|---|
| 80 ms (RFC minimum) | 6,958 | 34.5 % | 50.9 % | −56.4 dBFS |
| **250 ms (default)** | **133** | **35.89 %** | **52.06 %** | **−71.7 dBFS** |
| 500 ms | 136 | 38.17 % | 53.77 % | −72.1 dBFS |
| 1000 ms (previous default) | 137 | 40.95 % | 55.86 % | −72.2 dBFS |

250 ms passes the ≤ −70 dBFS quality bar with a 4× cut in preroll work. The C-vs-Python
chain tables were re-validated byte-identical at 250 ms on the 477 s / 1 h / 2 h files.

## 4. Chain length — 1 s stays

At preroll 250, stride 32, 1 h corpus (kernel, median of 4): 1 s = 216 ms (3,600 chains),
2 s = 192 ms (1,800), 4 s = 221 ms (900). But at stride 16 and shorter files the ranking
flips hard:

| duration | 1 s chains | 2 s chains |
|---|---|---|
| 15 min | **81 ms** (900 chains) | 143 ms (450) |
| 30 min | **89 ms** | 149 ms |
| 1 h | 176 ms | **165 ms** |
| 2 h | 373 ms | **371 ms** |

**1 s stays the default:** it wins decisively at 15–30 min, where occupancy needs
N ≥ 900 chains, and is within 7 % at 1–2 h.

## 5. Warp stride — 16 (2 chains per warp)

Opus entropy decoding is control-flow divergent, so packing decoders into adjacent lanes
serializes them. 1 h, seg 1 s, preroll 250, kernel median:

| stride | kernel | note |
|---|---|---|
| 4 | 533 ms | 8 chains/warp — divergence dominates |
| 8 | 212 ms | 4 chains/warp |
| **16** | **176 ms** | **2 chains/warp — default, −19 % vs 32** |
| 32 | 218 ms | 1 chain/warp |

Two chains per warp tolerate the divergence; four break even; eight lose badly.

**Bit-exactness proven at every stride:** the full phase-3 protocol-A gate (7 corpora ×
2 segmentations) run at strides 16, 8, 4 and the 32 control — **56/56 runs PASS, 0
mismatches** (4 × 853,691 packet decodes / 4 × 335.9 M PCM samples).

## 6. Dual-stream copy/compute overlap

`cuopus_run` uses a compute stream (decode → device-side trim/keep compaction → resample)
and a copy stream (D2H into a pinned `cudaHostAlloc` buffer) with double-buffered staging
and events. Batch N's kernel overlaps batch N−1's D2H, and completed chunks are written
out while later batches still decode.

Output is byte-identical (`cmp`) to the pre-overlap binary on 477 s (1 batch), 1 h
(2 batches) and 2 h (4 batches), at both old and new defaults.

## 7. On-device 48 k → 16 k resampling

The 49-tap Kaiser polyphase filter from `gpu_resample.py` is ported into `decode_cli.cu`:
taps computed on the host in double precision with the identical formula and uploaded to
`__constant__` memory, then a decimation kernel over the compacted 48 kHz PCM with a
≤ 51-sample carry across batches. Exposed as daemon verb `DECODE16F` and CLI `--f16k`.

**Validation vs the torch reference path: max diff 0.0 — bit-identical** on 477 s and 1 h
(the tolerance was 1e-5), 65.2 M samples compared.

This removes torch from the client entirely: the pipeline becomes
demux → daemon → `np.fromfile`, with no second CUDA context in the client process. The
torch path is kept as a fallback (`CUOPUS_RESAMPLE=torch`).

---

## Two negative results, so nobody retries them

The ~1.4 GB VRAM floor (CUDA context + the driver's per-resident-thread local-memory pool)
is real and does not respond to the usual levers:

1. **Launching with 96 KB dynamic shared memory per block does not shrink the ~910 MB
   local pool.**
2. **`-maxrregcount=255` does not shrink it either.**

The driver sizes that pool for device-max resident threads regardless of occupancy limits.
It also does not shrink with fewer chains or smaller batches — batch 128 OOMs at exactly
the point batch 478 does. The floor stands until per-thread local memory itself shrinks.

## VRAM budget

~1.4 GB floor (context + pool). The f16k path adds ~0.5 GB of staging at 2 h shapes, so
budget **~2.4 GB peak for a 2 h file**. The daemon idle-exits and returns it.

## Compile time

The single-translation-unit device build initially took **>25 min in cicc** (nvcc's
inliner). The generated code therefore emits functions `__device__ __noinline__` by
default, with a `HOT_FNS` whitelist in `port_silk.py` marking ~25 hot leaf functions
`__device__ __forceinline__` instead — the whole range decoder (`ec_dec_*` / `ec_decode*`,
called per symbol), laplace, small mathops, and SILK `lin2log` / `log2lin` / `bwexpander`.
Only whitelisted leaves inline, so the explosion is avoided and the build takes **~6–12 s**.

---

## A note on comparing against ffmpeg

If you validate cuopus output against ffmpeg, check which Opus decoder ffmpeg used.

| comparison | RMS | max |
|---|---|---|
| cuopus 48 k vs ffmpeg **libopus** (float) | **−72.2 dBFS** | 137 LSB |
| cuopus 48 k vs libopus FIXED_POINT serial reference | −72.2 dBFS | 137 LSB (40.9 % exact) |
| ffmpeg **native** opus vs libopus | −39.8 dBFS | 25,975 LSB |

ffmpeg's built-in native Opus decoder sits ~30 dB further from the libopus reference than
cuopus does, because of its own SILK 16 k → 48 k upsampler — the difference is broadband
and proportional to signal. A cuopus-vs-ffmpeg-native comparison inherits that −39.8 dBFS
floor and says nothing about cuopus. Compare against `-c:a libopus`, or better, against an
`--enable-fixed-point` libopus build, where the decode path is bit-exact.

## Gate totals for this performance work

- Phase-3 protocol-A CUDA gate at strides 16/8/4/32: 56 runs, **3,414,764 packet decodes /
  1.34 B PCM samples compared, 0 mismatches**, 0 device errors, 0 arena overflows.
- s16 byte-identity, new vs old binary: 5 file/config combos, `cmp` IDENTICAL.
- Native resample vs torch: 65.2 M samples, max diff 0.0.
- Preroll 250 ms quality: −71.7 dBFS vs the FIXED_POINT serial reference (PASS, ≤ −70).
- C demux chain tables at 250 ms: byte-identical to Python on 477 s / 1 h / 2 h.

## Tunables

| env | default | meaning |
|---|---|---|
| `CUOPUS_SEG_S` | 1.0 | chain length in seconds (swept {1,2,4} — 1 s wins) |
| `CUOPUS_PREROLL_MS` | 250 | decoder convergence preroll per chain |
| `CUOPUS_STRIDE` | 16 | threads per decode slot; 16 = 2 chains/warp. Bit-exact at 4/8/16/32 |
| `CUOPUS_RESAMPLE` | native | `native` = on-device 48 k→16 k; `torch` = legacy client-side path |
| `CUOPUS_MAX_BATCH` | 2048 | chains per kernel launch (bounds arena/PCM buffers) |
| `CUOPUS_CHECK_CRC` | 0 | verify OGG page CRCs (costs ~1.3 s on a 2 h file in Python) |
| `CUOPUS_SERVE_IDLE_S` | 600 | daemon idle exit, returns ~1.3 GB VRAM |
| `CUOPUS_MIN_FREE_MB` | 2400 | free-VRAM pre-check floor; refuse rather than OOM |
| `CUOPUS_TIMEOUT_S` | 120 | client-side decode timeout |

## Reproducing

The sweep and gate numbers here were produced against private audio corpora and reference
dumps from a driver linked to a real `--enable-fixed-point` libopus build, neither of
which ships in this repo. The harness sources (`harness2.cu`, `harness_host.cpp`) and the
full methodology do — see [PHASE2_RESULTS.md](PHASE2_RESULTS.md) and
[PHASE3_RESULTS.md](PHASE3_RESULTS.md). `./build_and_test.sh phase1` is the self-contained
gate that runs end-to-end from this repo alone.
