# cuopus phase 2 — device-side SILK decode port, bit-exactness gate

**Date:** 2026-08-30 · **Hardware:** RTX 5090 (sm_120), CUDA 13 driver, container nvcc
**Scope:** mechanical CUDA port of the libopus **1.5.2 FIXED_POINT SILK decode path**
(97 % of packets in the target corpus are SILK-WB config 9/11). CELT and hybrid packets return
handled errors from the device path (phase 3). Top-level TOC dispatch
(`opus_packet_parse_impl`, mode/bandwidth selection) is ported now and final.

## What was built

```
device/port_silk.py        regenerates device/gen/** from vendor/opus-1.5.2 with three
                           mechanical transforms (all diffable: diff device/gen/silk/X.c
                           vendor/opus-1.5.2/silk/X.c):
                             1. col-0 function defs/prototypes -> `CUOPUS_DEVFN` prefix
                                (.c out-of-line defs -> CUOPUS_DEVFN_NOINL = __device__
                                __noinline__; headers/OPUS_INLINE stay inlinable)
                             2. file-scope const tables / extern const -> CUOPUS_DEVDATA
                                (= __device__)
                             3. `#include "cuopus_device.h"` prepended
                           plus 2 wrapped files (stack_alloc.h -> per-thread global-memory
                           arena; ecintrin.h -> __clz EC_ILOG) and 1 macro patch
                           (silk_memmove/OPUS_MOVE -> cuopus_memmove; no device memmove).
device/cuopus_device.h     DEVFN/DEVDATA macros + device memmove.
device/cuopus_stack.h      arena ALLOC/SAVE_STACK/RESTORE_STACK (VARDECL surface of
                           celt/stack_alloc.h), 48 KB per thread in global memory,
                           overflow flagged per chain (voids comparison, never silent).
device/cuopus_silk_all.cuh single-TU include of the 46 ported .c files (entdec/entcode/
                           entenc + 43 silk decode files: dec_API, decode_*, NLSF*, LPC*,
                           resampler*, CNG, PLC, gain/shell/sign coders, tables).
device/cuopus_opus.cuh     hand-written SILK-only specialization of opus_decoder.c
                           (opus_decode_frame / opus_decode_native / opus_decoder_init),
                           line-faithful, CELT branches -> CUOPUS_ERR_CELT / _REDUNDANCY /
                           _TRANSITION. Packet parse helpers extracted verbatim from
                           src/opus.c into gen/opus_parse_gen.inc by the port script.
harness2.cu                CUDA gate+bench harness: 1 thread = 1 segment chain, states in
                           global memory, compares nsamp + OPUS_GET_FINAL_RANGE + PCM
                           sample-for-sample against the reference dump.
harness_host.cpp           the SAME ported single TU compiled as plain C++
                           (-DCUOPUS_HOSTPORT, DEVICE macros empty) — separates "the port
                           is bit-exact" from "the CUDA execution is bit-exact". Also
                           `mkskip` (see below).
ref/ref_decode.c           oracle harness on real libopus 1.5.2 built with
                           ./configure --enable-fixed-point (vendor/build-fixed/):
                           fresh OpusDecoder per chain, per-packet nsamp + final range +
                           PCM dumps, plus 1-/24-thread CPU benchmarks.
ogg_index.py chains        packet-exact chain tables (N-second windows + 80 ms preroll).
build_and_test.sh phase2   the whole pipeline, reproducible.
```

**FIXED_POINT on both sides** (`--enable-fixed-point` reference, `-DFIXED_POINT=1`
device) — identical integer arithmetic is what makes bit-exactness achievable.
Arch dispatch pinned to the plain-C kernels (no OPUS_X86_MAY_HAVE_*, arch=0).

### Comparison protocols

- **Protocol A (true stream):** reference decodes every packet; device decodes until the
  first packet needing CELT, prefix compared. Honest but low-coverage on this corpus
  (CELT packets appear every ~30 packets on average).
- **Protocol B (primary gate):** packets that need CELT are skipped on BOTH sides, so the
  SILK decoder state evolution is identical everywhere and every other packet is compared.
  "Needs CELT" = non-SILK TOC modes **plus SILK packets carrying CELT transition
  redundancy** (see finding below), precomputed into `testdata/skip_*.mask` by
  `cuopus_host mkskip`. Every compared packet's output still comes from real libopus.

### Finding: SILK packets can need CELT (transition redundancy)

First gate run failed 8/16 chains with `CUOPUS_ERR_REDUNDANCY`: at SILK↔CELT mode-switch
boundaries, libopus embeds a 5 ms CELT redundancy frame *inside SILK-only packets*
(opus_decoder.c redundancy check: spare bits after SILK decode ⇒ decode CELT band 0–17
and smooth_fade over 2.5 ms of output). These packets cannot be reproduced without the
CELT decoder. They are rare — **20 of 23,061 SILK packets (0.09 %) in the 20 ms corpus,
20 of 3,848 in the 120 ms corpus** — and are detectable from the packet bits alone, so
`mkskip` marks them like CELT packets. The detection logic in the device path is kept
identical to upstream, so such packets are flagged, never mis-decoded. Phase 3 (CELT)
removes the exclusion.

## Gate 1 — host same-source port gate: **PASS, 0 mismatches**

The ported sources (the exact files the CUDA build compiles) vs real libopus
FIXED_POINT, protocol B, both corpora, both segmentations:

| corpus | chains | packets compared | final-range mismatches | PCM samples compared | PCM mismatches |
|---|---|---|---|---|---|
| 20 ms, 16×30 s | 16/16 full | 23,101 | 0 | 22,176,960 | **0** |
| 20 ms, 478×1 s | 478/478 full | 24,877 | 0 | 23,881,920 | **0** |
| 120 ms, 16×30 s | 16/16 full | 3,843 | 0 | 22,135,680 | **0** |
| 120 ms, 478×1 s | 478/478 full | 4,439 | 0 | 25,568,640 | **0** |

Totals: **56,260 packet decodes compared, 93.76 M PCM samples, 0 mismatches of any kind**
(sample counts, final entropy-coder range, PCM). Per-corpus SILK coverage: 23,041/23,061
= 99.91 % of all SILK packets in the 20 ms file (the 20 redundancy packets are the only
exclusions; 3,828/3,848 for 120 ms). The port also decodes 60 ms frames (config 11,
3 SILK frames/packet), code-1/2/3 multi-frame packets, and exercises the SILK-internal
16 kHz → 48 kHz resampler on every packet. CNG/DTX and PLC are ported (not stubbed) but
unexercised by this corpus (no DTX packets); PLC's >20 ms split is done iteratively in
`cu_opus_decode` instead of by recursion (see below).

## Gate 2 — CUDA execution gate: **PASS, 0 mismatches on device**

Run 2026-08-30 after a VRAM window opened (~3.4 GB free; the GPU is shared with
co-resident inference services). Same protocol-B comparisons as the host gate, decoded by
the CUDA kernel (one decoder per chain, states + arena in global memory):

| corpus | chains | packets compared | range mismatches | PCM samples | PCM mismatches |
|---|---|---|---|---|---|
| 20 ms, 16×30 s | 16/16 full | 23,101 | 0 | 22,176,960 | **0** |
| 20 ms, 478×1 s | 478/478 full | 24,877 | 0 | 23,881,920 | **0** |
| 120 ms, 16×30 s | 16/16 full | 3,843 | 0 | 22,135,680 | **0** |
| 120 ms, 478×1 s | 478/478 full | 4,439 | 0 | 25,568,640 | **0** |

Protocol A (true stream, device stops at the first CELT-needing packet, prefix
compared): 18,327–24,548 packets compared per corpus, **0 mismatches**, coverage
79.3 %/98.6 % (30 s/1 s chains, 20 ms corpus) — low 30 s-chain coverage is just CELT
packet spacing, not decoder failures. Grand total across all 8 device runs:
**146,767 packet decodes, 178 M PCM samples compared, 0 mismatches** of any kind
(sample counts, final entropy range, PCM), 0 device errors, 0 arena overflows.

### VRAM context floor (measured)

With <540 MiB free the harness cannot even create a CUDA context (`cudaSetDevice: out
of memory`) — a GB202 context costs ~500 MB. Measured overhead once running: **~260–290
MB total** (context + 80 MB buffers + local-memory pool). Two rules learned:
- Never `cudaDeviceSetLimit(cudaLimitStackSize, …)` here: an explicit 32 KB limit
  reserves stack for all ~261 k resident threads (~8 GB) and OOMs instantly. The ported
  call graph was made non-recursive (PLC split hoisted to an iterative loop in
  `cu_opus_decode`) so the driver sizes the stack statically.
- A co-resident inference server's caching allocator regrows over traffic (was holding
  9.2 GB); the gate needs ~600 MB free — reclaim VRAM from co-resident services first.

## Benchmark — GPU vs CPU wall time, 477 s file

All-chain decode including 80 ms prerolls. CPU = real libopus FIXED_POINT (best of
5–7); GPU = kernel best of 5 (end-to-end adds H2D packets ~1.3 MB + D2H PCM ~46 MB,
+4 ms). "chain/warp" = one decoder per warp (stride 32; no intra-warp divergence
serialization); "chain/lane" packs chains into adjacent lanes, which serialize on
every data-dependent branch.

| workload | CPU 1 core | CPU 24 threads | GPU chain/lane | GPU chain/warp | GPU e2e (warp) |
|---|---|---|---|---|---|
| 20 ms frames, **16×30 s** chains | 223 ms | 20.7 ms | 3,426 ms | 739 ms | 744 ms |
| 20 ms frames, **478×1 s** chains | 242 ms | 13.4 ms | 198 ms | **29.5 ms** | 33.3 ms |
| 120 ms frames, **16×30 s** chains | 214 ms | 20.4 ms | 3,378 ms | 730 ms | 734 ms |
| 120 ms frames, **478×1 s** chains | 256 ms | 14.2 ms | 105 ms | **32.0 ms** | 35.9 ms |

Reading:
- **The chain-per-warp mapping is mandatory** — 6.7× faster than chain-per-lane at
  N=478 (29.5 vs 198 ms). SILK's data-dependent control flow serializes adjacent lanes
  almost completely; paying 31/32 idle lanes is the better trade. It is the harness
  default in `build_and_test.sh` (last arg 32).
- At the **16-chain** segmentation the GPU loses ~36× to the 24-core CPU (739 vs
  20.7 ms) — exactly the feasibility study's prediction that N≈16 is hopeless.
- At **478 chains** the GPU is within **2.2×** of the 24-core CPU (29.5 vs 13.4 ms)
  and 8× faster than 1 core. One GPU warp decodes at ≈1/60 of a CPU core here
  (feasibility estimated 1/8.7 for hand-tuned code; the gap is ABI calls from the
  noinline port, global-memory state/arena, and fixed-point emulation — all tunable
  in a later pass, none needed for the phase gate).
- Extrapolating the warp-mapped numbers, GPU-vs-24-core crossover sits around
  N ≈ 1,000–2,000 chains — a 2 h file at 5 s segments (1,440 chains) is in range,
  the 477 s test file is not. This matches the roadmap's "multi-hour files only"
  premise and must be re-checked with real CELT+SILK mixed decode in phase 3.
- The 12-process CPU decode pool does the whole 477 s file (demux+decode+resample) in
  56–88 ms; the GPU SILK path at this file size is not competitive with it and is not
  meant to be.

## Compile-time note (cost a rebuild cycle)

The merged single-TU device build initially took **>25 min in cicc** (nvcc's inliner on
the whole decoder flattened into one kernel) — at any of -O1/-O2/-O3. Marking every
out-of-line ported function `__device__ __noinline__` (`CUOPUS_DEVFN_NOINL`, mirroring
libopus's real TU boundaries) brought the -O2 device build to **2.5 s** with no
correctness impact. Keep this if phase 3 adds CELT to the same TU.

## Deviations from a pure mechanical port (all documented in-source)

1. `stack_alloc.h` ALLOC → per-thread bump arena in global memory (48 KB/thread,
   overflow-flagged). Alloca/VLA are unusable on device without a per-resident-thread
   stack reservation measured in GB.
2. PLC >20 ms split: recursion → equivalent iteration (VRAM reason above). Loss-free
   decode paths are unaffected; concealment chunk sequence is identical.
3. `silk_memmove`/`OPUS_MOVE` → `cuopus_memmove` (device libc has no memmove).
4. `EC_ILOG` → `__clz` on device (same contract, validated by phase 1 on every packet).
5. CELT/hybrid/transition/redundancy branches → handled error codes (`CUOPUS_ERR_*`),
   phase-3 fill-ins. Non-SILK packets are rejected before any decoder-state update.
6. `QA` macro collision between LPC_inv_pred_gain.c (24) and NLSF2A.c (16) in the single
   TU → `#undef QA` between includes (values preserved per file).

## What remains for full coverage

1. **CELT decode (phase 3)** — removes the 3.4 % non-SILK packet exclusion AND the
   0.09 % SILK-transition-redundancy exclusion, and enables protocol-A full-stream
   parity. The dispatch structure in cuopus_opus.cuh is final; fill the
   `CUOPUS_ERR_CELT/_REDUNDANCY/_TRANSITION` branches.
2. **Hybrid packets** (configs 12–15) — none in the corpus TOC stats, same phase-3 path.
3. **DTX/CNG and PLC on-device validation** — code is ported, needs a corpus with DTX
   (`ffmpeg -application voip -dtx 1`) and a loss-injection harness.
4. **Stereo** — ported (dec_API stereo paths compile), untested; the target workload is mono.
5. **Per-warp performance pass** (optional, before the phase-3 NO-GO check): the port
   runs at ~1/60 CPU-core per warp; selective re-inlining of the hot decode_core/
   resampler path, state in registers/shared, and arena-in-local could close part of
   the gap to the 1/8.7 estimate.
6. Multi-TU split of the device build if phase 3 makes the single TU unwieldy.
