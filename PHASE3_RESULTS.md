# cuopus phase 3 — device-side CELT decode port, full-stream bit-exactness gate

**Date:** 2026-08-30 · **Hardware:** RTX 5090 (sm_120), CUDA 13 driver, container nvcc
**Scope:** mechanical CUDA port of the libopus **1.5.2 FIXED_POINT CELT decode path** and the
full `opus_decode_frame` dispatch — CELT-only, hybrid (configs 12–15), SILK↔CELT transition
frames, and the embedded CELT redundancy that phase 2 had to exclude (the 0.09 % "mkskip"
packets). With phase 2's SILK port this completes **full-stream decode coverage: every packet
of every corpus decodes on the device, protocol A (true stream), 0 mismatches.**

## What was built (delta over phase 2 — same machinery, extended)

```
device/port_silk.py        EXTENDED (same script, same diffable-prefix approach):
                             - CELT decode closure added to the emit lists (14 .c files:
                               celt_decoder, bands, vq, cwrs, kiss_fft, mdct, pitch, celt_lpc,
                               laplace, mathops, quant_bands, rate, modes, celt + 15 headers
                               incl. static_modes_fixed.h and include/opus_custom.h)
                             - GENERIC_FUNC_RE: celt files have column-0 functions without a
                               library prefix (comb_filter, haar1, quant_all_bands, _celt_lpc,
                               tf_decode, ...) and export-macro'd defs (OPUS_CUSTOM_NOSTATIC);
                               generic "type name(" matcher applied to celt/** only, silk
                               output unchanged
                             - STRUCT_PATCHES (pre-transform, still diffable):
                               * bands.c: recursive quant_partition wrapped under
                                 CUOPUS_QP_RECURSIVE (never defined) and replaced by an
                                 explicit-frame-stack version (see below)
                               * celt_decoder.c: cuopus_celt_reset_state inserted (the
                                 OPUS_RESET_STATE ctl case as a plain function); the varargs
                                 opus_custom_decoder_ctl compiled out under CUOPUS_CUDA
                             - PATCHES: celt/pitch.c celt_pitch_xcorr_c (return type inside
                               #ifdef on the lines above the name) marked explicitly
                             - DATA_RE hardened (init/terminator required) so column-0
                               function-parameter lines stop matching
                             - smooth_fade added to the opus_decoder.c extraction
device/cuopus_celt_all.cuh NEW: single-TU include of the 14 ported CELT .c files, after
                             cuopus_silk_all.cuh. Defines CELT_DECODER_C first (single-TU
                             artifact: opus_custom.h only exposes the decoder prototypes to
                             the TU that claims to be celt_decoder.c).
device/cuopus_opus.cuh     REWRITTEN: full line-faithful opus_decode_frame — CELT-only,
                             hybrid (SILK 0–16 kHz + CELT start_band 17, celt_accum on
                             FIXED), redundancy (both celt_to_silk directions, range XOR),
                             transition fades (smooth_fade), decode_gain. Typed ctl helpers
                             (cu_celt_set_start_band/end_band/channels/signalling,
                             get_final_range/get_mode) mirror the ctl cases field-for-field.
                             CUOPUS_ERR_* are dead (kept only so old harness code compiles).
harness2.cu / harness_host.cpp  arena 48→96 KB/slot (CELT quant_all_bands + synthesis + PLC
                             scratch; 0 overflows observed), "packet coverage" gate line,
                             include of cuopus_celt_all.cuh. File formats unchanged.
build_and_test.sh phase3   corpora → masks → protocol-A refs → host gate → GPU gate →
                             CPU baselines → the roadmap 256×30 s NO-GO measurement.
```

### No-recursion rule: quant_partition → explicit frame stack

`bands.c quant_partition` recurses on band splits (each split halves N and decrements LM;
no split at LM = −1 ⇒ provable depth ≤ 5 from LM ≤ 3). Device threads must not recurse
(phase-2 lesson: the driver must size the per-thread stack statically; forcing a stack limit
reserves it for every resident thread — GBs). The replacement keeps vendor control flow,
arithmetic and entropy-coder call order exactly: each recursive call becomes a pushed frame
(10 deep max, ~90 B each, in registers/local); the vendor's *post-call* rebalance
(`rebalance > 3<<BITRES && itheta != 0/16384`) runs in the matching resume phase, and the
`<<(B0>>1)` shift is applied to the same child it was applied to upstream. The vendor
recursive text stays in the generated file under `#ifdef CUOPUS_QP_RECURSIVE` for diffing.

### Varargs rule: celt_decoder_ctl → typed helpers

CUDA device code cannot compile varargs functions. `opus_custom_decoder_ctl` is compiled out
under CUOPUS_CUDA; the six requests `opus_decode_frame` actually uses became typed one-liner
helpers (validation included, copied from the ctl cases), and OPUS_RESET_STATE became
`cuopus_celt_reset_state` patched into celt_decoder.c itself (it needs the private struct
layout). Both host-port and CUDA builds use the same helpers — the host gate validates them.

### Other single-TU findings

- The big static mode tables (static_modes_fixed.h: window120, eBands, FFT twiddles/bitrev,
  PVQ cache, the CELTMode itself) went to `__device__ const` **global** memory via the same
  CUOPUS_DEVDATA mechanism as phase 2's silk tables — NOT `__constant__` (too big). Pointer
  members in static initializers (mode → tables, fft state → twiddles) link fine, as the
  phase-2 NLSF codebooks already proved.
- `celt_pitch_xcorr_c` has its return type on a separate `#ifdef FIXED_POINT` line — the only
  split-line definition in the closure; handled by an exact-string patch.
- The two recursive `opus_decode_frame(st, NULL, …)` uses (PLC >20 ms splitting, 5 ms
  transition frame) are served by `cu_opus_plc_frame`, a dedicated recursion-free copy of the
  data==NULL specialization; chunk sequence and state updates are vendor-identical.
- No new macro collisions beyond phase 2's QA (undef'd again at the celt TU boundary).
- nvcc -O2 device build: **~6 s** — the `__device__ __noinline__` per-.c-function discipline
  from phase 2 still holds with the CELT closure added (naive inlining was >25 min).

## Corpora (477 s real speech, host ffmpeg libopus)

| corpus | encode | packets | TOC configs (100 %) |
|---|---|---|---|
| 20 ms (phase 2) | 16 k, 24 kbps voip-ish default | 23,872 | 96.6 % SILK-WB 9/11 + 3.4 % CELT + transitions/redundancy |
| 120 ms (phase 2) | 24 kbps, 120 ms | 3,979 | SILK code-3 multiframe + CELT |
| **celt** | 48 k, 96 kbps, 20 ms, `-application lowdelay` | 23,872 | **config 31 = 100 % CELT-FB 20 ms** (MDCT 960) |
| **celt5** | 48 k, 96 kbps, 5 ms, lowdelay | 95,485 | config 29 = 100 % CELT-FB 5 ms (MDCT 240) |
| **celt25** | 48 k, 96 kbps, 2.5 ms, lowdelay | 190,969 | config 28 = 100 % CELT-FB 2.5 ms (MDCT 120) |
| **hybrid** | 48 k, 32 kbps, 20 ms, `-application audio` | 23,872 | **97.2 % hybrid: 22,948×cfg 15 (FB) + 262×cfg 13 (SWB)** + 2.8 % CELT |
| **hybrid10** | 48 k, 32 kbps, 10 ms | 47,743 | 45,929×cfg 14 + 517×cfg 12 + 1,297×cfg 30 (CELT-SWB 10 ms, MDCT 480) |

Coverage honesty: all four MDCT sizes (120/240/480/960 ⇒ LM 0–3), all four hybrid configs
12–15, CELT-FB/SWB, transitions and embedded redundancy (both fade directions decoded — the
20 ms corpus protocol-A refs contain the 20 redundancy packets phase 2 excluded, now compared).
NOT exercised: CELT NB/MB (configs 16–23; ffmpeg's libopus won't produce them at these rates),
stereo (ported, untested — the target workload is mono), DTX/CNG and real packet loss (unchanged from
phase 2), decode_gain ≠ 0 (OggOpus output-gain layer isn't part of the harness protocol).

## Gate 1 — host same-source port gate: **PASS, 0 mismatches, 100 % coverage**

Ported single TU as plain C++ vs real `--enable-fixed-point` libopus, **protocol A (true
stream, every packet)**, 7 corpora × both segmentations (16×30 s, 478×1 s):

| corpus | chains full | packets compared | range mism. | PCM samples | PCM mism. |
|---|---|---|---|---|---|
| 20 ms (30 s / 1 s) | 16+478 | 23,932 + 25,780 | 0 | 47.7 M | **0** |
| 120 ms (30 s / 1 s) | 16+478 | 3,994 + 4,615 | 0 | 49.6 M | **0** |
| celt (30 s / 1 s) | 16+478 | 23,932 + 25,780 | 0 | 47.7 M | **0** |
| celt5 (30 s / 1 s) | 16+478 | 95,725 + 103,117 | 0 | 47.7 M | **0** |
| celt25 (30 s / 1 s) | 16+478 | 191,449 + 206,233 | 0 | 47.7 M | **0** |
| hybrid (30 s / 1 s) | 16+478 | 23,932 + 25,780 | 0 | 47.7 M | **0** |
| hybrid10 (30 s / 1 s) | 16+478 | 47,863 + 51,559 | 0 | 47.7 M | **0** |

Totals: **853,691 packet decodes, 335.9 M PCM samples, 0 mismatches of any kind** (sample
counts, final entropy range incl. the redundancy XOR, PCM), **100.00 % packet coverage**.

## Gate 2 — CUDA execution gate: **PASS, 0 mismatches on device, 100 % coverage**

Same protocol-A comparisons decoded by the CUDA kernel (one decoder per chain-per-warp,
states + 96 KB arena in global memory). All 14 runs: **chains 100 % full, 0 device errors,
0 arena overflows, 0 nsamp/range/PCM mismatches** — identical counts to the host table above
(853,691 packets, 335.9 M samples). Protocol-B phase-2 regression runs (old masks, old refs)
still pass unchanged. Including the two 256×30 s NO-GO runs below (382,912 packet slots /
367.6 M samples each, also 0 mismatches), the phase-3 device total is **1.62 M packet decodes
and 1.07 B PCM samples compared, 0 mismatches**.

### VRAM footprint (grew vs phase 2 — plan runs accordingly)

Measured overhead at 478 chains: **~910 MB** (phase 2: ~260–290 MB) — the local-memory pool
grew with the deeper ABI call chains and the CELT scratch (this is per-resident-thread stack,
not per-chain), plus 46 MB arena (479×96 KB) + ~47 MB PCM + states. The 256×30 s shape peaks
at **~1,548 MB** (701 MB PCM span). Context still ~500 MB. Rule of thumb: **need ≥1.5 GB free
VRAM for the standard gate, ≥2.1 GB for the NO-GO shape** (this run: 2.26 GB free, 206 MB
left at peak). All device buffers stayed under 1 GB.

## Benchmark — GPU vs CPU wall time, 477 s file, full stream (protocol A)

CPU = real libopus FIXED_POINT (`ref_decode bench`, best of 5–7); GPU = chain-per-warp
kernel best of 5 (e2e adds H2D packets + D2H PCM ~47 MB).

| workload | CPU 1 core | CPU 24 thr | GPU warp kernel | GPU e2e | GPU vs 24-thr |
|---|---|---|---|---|---|
| 20 ms, 16×30 s | 223.0 ms | 20.6 ms | 775.4 ms | 779.3 ms | 37.7× slower |
| 20 ms, **478×1 s** | 231.9 ms | 13.5 ms | **37.2 ms** | 41.1 ms | 2.75× slower |
| 120 ms, 16×30 s | 213.9 ms | 20.7 ms | 763.3 ms | 766.7 ms | 36.9× slower |
| 120 ms, **478×1 s** | 247.0 ms | 14.2 ms | **38.3 ms** | 42.5 ms | 2.70× slower |
| celt, 16×30 s | 396.9 ms | 34.9 ms | 1,509.6 ms | 1,514.5 ms | 43.3× slower |
| celt, **478×1 s** | 442.6 ms | 23.4 ms | **61.3 ms** | 65.6 ms | 2.62× slower |
| celt5, 478×1 s | 501.9 ms | 26.2 ms | 112.3 ms | 116.9 ms | 4.29× slower |
| celt25, 478×1 s | 568.4 ms | 31.4 ms | 174.8 ms | 179.2 ms | 5.57× slower |
| hybrid, 16×30 s | 375.5 ms | 33.7 ms | 1,453.2 ms | 1,457.1 ms | 43.1× slower |
| hybrid, **478×1 s** | 405.3 ms | 22.1 ms | **56.4 ms** | 60.4 ms | 2.55× slower |
| hybrid10, 478×1 s | 431.8 ms | 23.1 ms | 74.3 ms | 78.3 ms | 3.21× slower |

Reading:
- The phase-2 shape holds with CELT in the stream: at 478 short chains the GPU sits
  **2.5–2.8× behind the 24-core CPU** (SILK-majority and 20 ms CELT alike) and ~6–7× ahead
  of one core. Extrapolated crossover vs 24 cores stays at **N ≈ 1,200–1,600 chains** — a
  2 h file at 5 s segments (1,440 chains) remains the first realistic winning shape.
- Short CELT frames are relatively worse on the GPU (5.6× behind at 2.5 ms): more packets ⇒
  more serial per-chain iterations and more entropy-decode divergence per sample.
- One CELT warp-chain decodes ~2.6× more slowly than a SILK one (61.3 vs 37.2 ms for the
  same audio) — PVQ/cwrs division-heavy integer work and the serial IMDCT dominate; the
  IMDCT is still plain per-chain serial code in this phase (cuFFT batching is the planned
  later optimization, correctness first).

### Roadmap NO-GO check (256 chains × 30 s must beat the 24-core CPU): **FAILS as written**

Measured on the honest shape (the 16 distinct 30 s chains tiled 16× = 256 independent
decoders; ref side identical, gate PASS 0 mismatches at this shape too):

| corpus, 256×30 s | CPU 1 core | CPU 24 thr | GPU warp kernel | verdict |
|---|---|---|---|---|
| celt | 6,581 ms | **350.2 ms** | 1,552.5 ms | GPU **4.4× slower** |
| 20 ms (SILK) | 3,570 ms | **197.6 ms** | 790.8 ms | GPU 4.0× slower |

At 30 s chains the wall is per-chain serial length, not chain count: 256 warps each grind
~1,500 packets sequentially while 24 cores each take ~11 chains. The GPU only closes when
chains are *short and numerous* (478×1 s is already within 2.6×). So the literal roadmap
condition fails on the un-tuned port; the port runs at ~1/55–1/60 of a CPU core per warp
(phase-2 estimate for hand-tuned code was 1/8.7). **Decision point**: the
sanctioned levers are the phase-2 backlog items — selective re-inlining of the hot decode
path, decoder state/arena locality, and cuFFT-batched IMDCT — before integration, or accept
the segment-length constraint (fine segmentation is also what integration planned: 5 s
segments at N ≥ 200).

## Deviations from a pure mechanical port (phase-3 additions; phase-2 list still applies)

1. `quant_partition` recursion → explicit frame stack (identical decode order; vendor text
   preserved under CUOPUS_QP_RECURSIVE).
2. `opus_custom_decoder_ctl` varargs → typed helpers + `cuopus_celt_reset_state`.
3. PLC/transition recursion of `opus_decode_frame` → `cu_opus_plc_frame` + caller loops.
4. Fixed-size CELT decoder blob in `cu_opus_decoder` (init-time size check) instead of
   variable-size allocation.
5. `celt_pitch_xcorr_c` split-line definition marked by exact-string patch.

## What remains

1. **Performance pass** (the only thing between this and integration): selective
   re-inlining, state/arena locality, cuFFT-batched IMDCT, and re-measuring the NO-GO shape.
   Target from the feasibility study: ~1/8.7 CPU-core per warp (currently ~1/55–60).
2. **Integration** into a host pipeline (see ROADMAP) once the perf pass lands, plus
   the phase-4 on-device 48 k→16 k resampler that makes PCM stay on-GPU for the mel stage.
3. Backlog (unchanged): DTX/CNG + loss-injection validation, stereo corpus, CELT NB/MB
   corpora if a non-ffmpeg encoder can produce them.
