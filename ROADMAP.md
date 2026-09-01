# cuopus — CUDA Opus decoder, phase roadmap

**Date:** 2026-08-30 · **Hardware:** RTX 5090 (sm_120), CUDA 13 driver, 24-core CPU
**Context:** a prior feasibility study measured that GPU decode wins only at
**N ≈ 200+ concurrent segment chains** (~100 minutes of audio decoded simultaneously) and
recommended against building it for general traffic. cuopus was built anyway, deliberately,
for the very-long-file workload. Every phase below therefore carries an explicit
go/no-go gate; passing a gate means "the next phase is technically justified", not "ship it".

## Phase 1 — DONE (this directory)

**Delivered:**
- `ogg_index.py` — CPU OGG Opus page/packet indexer: per page `(byte_offset,
  granule_position, packet_boundaries)`, per-packet exact 48 kHz sample offsets from the
  TOC byte, RFC 7845 end-trim handling, packet-blob extractor, exact-segmentation demo.
- `cuopus_entdec.cuh` — device-side range decoder (RFC 6716 §4.1), faithful port of
  libopus 1.5.2 `celt/entdec.c` + `entcode.c`: `ec_dec_init/ec_decode/ec_dec_update/
  ec_dec_bit_logp/ec_dec_icdf(+16)/ec_dec_uint/ec_dec_bits/ec_tell`. One thread = one
  packet stream.
- `harness.cu` + `schedule.inc` — the same fixed symbol schedule (64 bit_logp + 32 icdf +
  32 uint(2000) + 16 raw-bits(11) = 144 symbols + tell/rng/error state) run through real
  libopus entdec on CPU and the CUDA port on GPU, compared word-for-word on every packet.
- `build_and_test.sh` — full reproduction (test-file generation → index validation →
  vendor fetch → container nvcc build → bit-exactness + bench).

**Measured (477 s real speech, ffmpeg libopus 24 kbps):**

| Check | 20 ms frames | 120 ms frames |
|---|---|---|
| Pages / audio packets | 480 / 23,872 | 445 / 3,979 |
| Index vs ffprobe (packet count, durations, duration_ts) | MATCH (PASS) | MATCH (PASS) |
| Page granules vs TOC cumulative sum | 478/478 | 443/443 |
| Total samples | 22,916,055 @48 kHz = **477.4178125 s exactly** (= 7,638,685 source samples × 3) | same |
| **Bit-exactness vs libopus entdec** | **23,872 packets, 0 mismatches** | **3,979 packets, 0 mismatches** |

| Symbol throughput (144 sym/pkt schedule) | 20 ms (N=23,872 chains) | 120 ms (N=3,979 chains) |
|---|---|---|
| GPU kernel only | **~73,000 Msym/s** (0.046 ms) | ~20,000 Msym/s (0.029 ms) |
| GPU end-to-end (H2D + kernel + D2H of all symbols) | ~2,450 Msym/s (1.40 ms) | ~1,470 Msym/s |
| CPU 1 core (libopus entdec) | 208 Msym/s | 204 Msym/s |
| CPU 24 threads (OpenMP) | 2,300–2,500 Msym/s | 750–870 Msym/s |

**Honest reading of those numbers:**
- Kernel-only GPU is ~30× the 24-core CPU at N≈24k chains — but this schedule is
  *GPU-friendlier than real Opus decode*: every thread executes the same 144-symbol
  sequence (minimal warp divergence) and there is no DSP feedback into decoder state. The
  feasibility study's realistic chain simulacrum (data-dependent icdf walks + LPC feedback)
  measured one GPU thread at 1/8.7 of one CPU core; the truth for a full decoder lies
  between these two measurements.
- End-to-end GPU ≈ 24-core CPU here only because D2H returns *every decoded symbol*
  (14 MB); a real decoder returns PCM instead, and phase 2+ keeps symbols on-device.
- Exact segmentation from granule positions works: cut points land byte- and
  sample-exact on every 30 s boundary (`ogg_index.py segment`), replacing a fixed 1 s
  preroll guess (the 80 ms decoder preroll requirement remains).

> **2026-08-30 re-ordering:** TOC analysis of the target corpus shows **97 % of packets are
> SILK-mode** (config 9/11 WB speech, 3 % CELT). SILK was therefore pulled forward:
> **phase 2 = SILK decode (below, DONE — bit-exact on device)**, phase 3 = CELT.
> The CELT section below keeps its original design notes.

## Phase 2 — SILK decode (DONE, see PHASE2_RESULTS.md)

Mechanical DEVICE-macro port of the libopus 1.5.2 FIXED_POINT SILK decode path
(46 .c files: dec_API → decode_frame → parameters/pulses/core, NLSF→LPC, LTP,
resampler, CNG, PLC) + SILK-only specialization of opus_decode_frame with the final
TOC dispatch. One CUDA thread = one segment chain; per-thread global-memory arena
replaces ALLOC/alloca; no recursion (driver-sized static stack).

**Gate results (2026-08-30):**
- Host same-source gate (ported sources as plain C++ vs real `--enable-fixed-point`
  libopus): **56,260 packets, 93.76 M PCM samples, 0 mismatches** (nsamp, final range,
  PCM) across 20 ms/120 ms corpora × 16×30 s/478×1 s chains. 99.91 % SILK coverage —
  the only exclusions are 20 SILK packets carrying CELT *transition redundancy*
  (a phase-2 finding: SILK-only packets at mode-switch boundaries embed a 5 ms CELT
  frame; needs phase 3).
- CUDA execution gate: **PASS — 146,767 packet decodes / 178 M PCM samples compared
  across 8 runs, 0 mismatches, 0 errors** (nvcc -O2 builds in 2.5 s after marking
  ported .c functions __noinline__; the naive single-TU build spent >25 min in cicc).
  Needs ~600 MB free VRAM for the CUDA context; measured footprint ~260–290 MB.
- Benchmark (477 s file, all chains incl. prerolls): CPU libopus 214–256 ms 1-core /
  13–21 ms 24-thread. GPU with the mandatory **chain-per-warp** mapping (6.7× over
  chain-per-lane): 730–739 ms at 16×30 s (GPU loses 36×, as predicted for N=16);
  **29.5–32 ms at 478×1 s** — within 2.2× of the 24-core CPU, 8× over 1 core.
  Extrapolated crossover vs 24 cores: N ≈ 1,000–2,000 chains (a 2 h file at 5 s
  segments = 1,440 chains is in range). The 12-process CPU pool (demux+decode+resample):
  56–88 ms — untouched as the short-file path.

## Phase 3 — CELT decode (DONE, see PHASE3_RESULTS.md)

**2026-08-30: DONE — full-stream decode coverage, bit-exact on device.** Mechanical
DEVICE-macro port of the CELT decode closure (celt_decoder, bands, vq, cwrs, kiss_fft,
mdct, pitch, celt_lpc, laplace, mathops, quant_bands, rate, modes + static mode tables in
__device__ global memory) plus the full opus_decode_frame: CELT-only, hybrid configs 12–15,
SILK↔CELT transitions, and the embedded CELT redundancy packets phase 2 excluded.
quant_partition's recursion became an explicit frame stack (no device recursion);
the varargs celt_decoder_ctl became typed helpers. IMDCT is plain serial per-chain device
code this phase — cuFFT batching deliberately deferred to the perf pass.

**Gate (2026-08-30):** protocol A (true stream, EVERY packet) on 7 corpora × 2
segmentations — 20 ms/120 ms speech + CELT-only 20/5/2.5 ms (configs 31/29/28) + hybrid
20 ms (97 % cfgs 13/15) + hybrid 10 ms (cfgs 12/14 + CELT-SWB 30): host same-source gate
**853,691 packets / 335.9 M samples, 0 mismatches, 100 % coverage**; CUDA gate identical
counts, 0 errors, 0 arena overflows — device total incl. the NO-GO shape: **1.62 M packet
decodes / 1.07 B samples, 0 mismatches**. Phase-2 protocol-B regression still passes.

**Bench (477 s, full stream):** GPU chain-per-warp at 478×1 s chains: 37–75 ms/corpus —
2.5–2.8× behind the 24-core CPU (13–23 ms), 6–7× ahead of 1 core; crossover still
N ≈ 1,200–1,600 chains. **The roadmap NO-GO condition (256×30 s CELT beats 24-core CPU)
FAILS on the un-tuned port: 1,553 ms vs 350 ms (4.4×)** — per-chain serial length dominates
at 30 s; short-and-many chains is the winning shape. VRAM: gate needs ~1.5 GB free
(~910 MB overhead; local-memory pool grew with the CELT call graph). Next lever before
integration: the perf pass (selective re-inlining, state locality, cuFFT-batched IMDCT).

Original design notes kept below.

Scope: CELT frame parse (band energies via Laplace decode, tf/spread flags, bit
allocation), PVQ pulse decoding (`cwrs.c` — pure integer combinatorics, bit-exact
achievable), anti-collapse, denormalization, IMDCT, overlap-add, deemphasis.

- **Where cuFFT fits:** `clt_mdct_backward` is pre-rotation → N/4-point complex FFT →
  post-rotation → windowed overlap-add. CELT MDCT sizes are 120/240/480/960 (N/4 = 30/60/
  120/240, radix 2·3·5 — all cuFFT-supported). Use one `cufftPlanMany` batched C2C plan
  per size over all frames of all chains in flight (thousands of tiny FFTs per launch is
  exactly cuFFT's batched sweet spot); pre/post-rotation as separate elementwise kernels,
  or hand-rolled per-block FFTs if plan/launch overhead dominates at small batch.
- **Parallelism:** entropy stays serial per chain (phase-1 decoder), but PVQ per band and
  the whole IMDCT/OLA stage batch across frames × bands — this is the part of Opus that is
  genuinely GPU-shaped (feasibility: CELT IMDCT/FFT ≈ 27% of libopus decode time).
- **Bit-exactness policy:** target the libopus **fixed-point** build as reference
  (float IMDCT is not bit-reproducible across FFT implementations); float path, if ever,
  validates against opus test-vector tolerance instead.
- **GO gate (met for the re-ordered phase 2):** phase-1 device symbol throughput at N ≥ 1,000 chains must exceed 4×
  the 24-core CPU — measured ~26–30× at N = 3,979 and 23,872. **Proceed.**
- **NO-GO check at end of the CELT phase:** full CELT-frame decode (entropy + DSP) for N = 256
  chains of 30 s must beat the 24-core CPU running libopus on the same shape; if the
  realistic-divergence penalty eats the margin, stop the project before CELT.
  **Measured 2026-08-30: FAILS as written (GPU 1,552 ms vs CPU-24 350 ms at 256×30 s CELT;
  478×1 s is within 2.6×). Decision owed: run the perf pass and re-measure, or accept the
  fine-segmentation constraint (5 s segments were the integration plan anyway).**

(SILK — originally slotted here — moved up to phase 2 and is done; its design notes
now live in the phase-2 section and PHASE2_RESULTS.md. The phase-3 GO gate inherits the
old phase-2/3 conditions: the phase-2 CUDA benchmark must beat the 24-core CPU at
N = 256+ chains once the VRAM window allows the measurement, AND the multi-hour
finely-segmented workload must actually exist — a 2 h file at 5 s segments = 1,440
chains; at 30 s = 240 chains, both above the N ≈ 200 crossover. If the product never
ships multi-hour ingestion, stop before CELT.)

## Phase 4 — 48 k → 16 k polyphase resampler (DONE)

Scope: replace the libopus/SILK resampler with a GPU polyphase FIR (3:1 decimation to the
16 kHz input that speech models expect). Embarrassingly parallel across output samples;
per-chain state only at segment boundaries (history = filter length). Cheap to build, but
pointless standalone — CPU resampling of already-decoded PCM is not a bottleneck. It earns
its place only by saving the final D2H + H2D round trip, keeping PCM on-device for a mel
front-end that is already there.

**DONE:** 49-tap Kaiser-windowed polyphase, taps computed on the host in double precision
and uploaded to `__constant__` memory, decimation kernel over the compacted 48 kHz PCM
with a ≤ 51-sample carry across batches. Validated **bit-identical to the torch reference
implementation — max diff 0.0** over 65.2 M samples (the tolerance was 1e-5). Exposed as
`--f16k` on the CLI and `DECODE16F` on the daemon.

## Phase 5 — performance pass (DONE, see BENCHMARKS.md)

The bit-exact decoder of phases 1–3 was correct but slower than the CPU pool at every
duration: the kernel was only ~11 % of wall time, and the rest was Python demux,
per-call CUDA context creation, and un-overlapped transfers.

Five changes, each gated on the phase-3 bit-exactness harness before being kept:

1. **Demux in C** (`cuopus_demux.h`, `decode_cli.cu --ogg`) — page parse, packet
   reassembly, granule cross-check and chain building in the CLI. Byte-identical (`cmp`)
   to `ogg_index.py` on 9/9 files. 2 h demux: 31 ms, down from ~1.8 s of Python.
2. **Persistent daemon** (`--serve <socket>`) — removes the ~200–500 ms per-call CUDA
   context cost. This alone won the 2 h shape with the still-untuned kernel.
3. **Preroll 1000 → 250 ms** — preroll was 50 % of kernel work at 1 s chains; −71.7 dBFS
   vs the fixed-point serial reference, passing the ≤ −70 bar.
4. **Warp stride 32 → 16** (2 chains per warp) — −19 % kernel at 1 h. Bit-exact at
   strides 4/8/16/32, all gated.
5. **Dual-stream copy/compute overlap + on-device resampling** — output byte-identical to
   the pre-pass binary; the client no longer touches CUDA at all.

**Result: the GPU pipeline beats the 12-process CPU pool at every sweep duration except
~5 minutes.** GPU/CPU = 0.93 / 1.44 / 0.77 / 0.51 / 0.50 / 0.44× at 1 m / 5 m / 15 m /
30 m / 1 h / 2 h. Gate totals for the pass: 56/56 runs PASS, 3.41 M packet decodes /
1.34 B PCM samples compared, 0 mismatches. Full tables, component splits, the sweeps that
were run and rejected, and two negative results are in [BENCHMARKS.md](BENCHMARKS.md).

## Standing honesty check

Restated from the feasibility study, and still true after the performance pass: cuopus's
win is batch shape, not per-stream speed. One GPU warp decodes at a small fraction of one
CPU core, and short files lose. Its only justification is multi-hour, finely-segmented
ingestion at N ≥ 200 concurrent chains. Keep that workload test in front of every gate.
