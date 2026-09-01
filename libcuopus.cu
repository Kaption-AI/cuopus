/* libcuopus.so — in-process GPU OGG-Opus decoder.
 *
 * The perf-pass-2 decoder (stride-16 chain-per-2/warp, 1 s chains, 250 ms
 * preroll, on-device 49-tap Kaiser 48k->16k resample) packaged as a shared
 * library called from Python on the server's own GPU thread. Decoded audio
 * lands DIRECTLY in a caller-provided device buffer (a torch CUDA tensor):
 * zero D2H copies, no daemon, no second CUDA context.
 *
 * C API (all functions return 0 on success; errbuf is always NUL-terminated):
 *
 *   int  cuopus_probe(const uint8_t *ogg, size_t n, int64_t *samples16k,
 *                     char *errbuf, size_t errcap);
 *        Demux + validate (mono, granule cross-check) and return the EXACT
 *        output length: (playable_48k - 48) / 3 float32 samples at 16 kHz
 *        (the CPU path's never-flushed swresample tail contract). Pure host
 *        work (~12 ms/h of audio), no GPU touch.
 *
 *   int  cuopus_decode_f16k(const uint8_t *ogg, size_t n, float *out_dev,
 *                           size_t out_cap, int64_t *out_len,
 *                           char *errbuf, size_t errcap);
 *        Decode + resample into the caller's DEVICE pointer (capacity out_cap
 *        float samples) on the caller's current CUDA context, default stream.
 *        Synchronizes the stream before returning: on return 0 the buffer
 *        holds *out_len finished samples, bit-identical to the daemon's
 *        DECODE16F output (same kernels, same math, same batching).
 *
 *   int  cuopus_decode_f16k_on(const uint8_t *ogg, size_t n, float *out_dev,
 *                              size_t out_cap, int64_t *out_len,
 *                              void *cuda_stream, char *errbuf, size_t errcap);
 *        Same, on an explicit cudaStream_t (pass torch's current stream).
 *
 *   int  cuopus_warmup(char *errbuf, size_t errcap);
 *        Optional: launch the decode kernel with n_batch=0 so the driver
 *        sizes its per-resident-thread local-memory pool (~910 MB, one-time
 *        per context) at startup instead of on the first real decode.
 *
 *   int  cuopus_last_arena_overflow(void);
 *        1 if the most recent decode failed due to a device arena overflow.
 *
 *   long long cuopus_workspace_bytes(void);
 *        Bytes currently held in cudaMalloc'd grow-only workspaces (excludes
 *        the driver's local-memory pool, which belongs to the context).
 *
 *   void cuopus_release_workspaces(void);
 *        Free every internal device/host workspace (grow-only pools). Safe to
 *        call any time no decode is in flight on this thread.
 *
 * Return codes: 0 ok; 1 soft failure (bad file / decode validation — the
 * CUDA context is healthy, caller falls back to CPU decode); 2 CUDA error
 * (context suspect — treat like any other CUDA failure in the process);
 * 3 caller error (capacity too small, NULL pointer, stream capturing).
 *
 * THREADING / CAPTURE CONTRACT:
 *   - The library never creates a CUDA context and never calls cudaSetDevice:
 *     it runs on whatever context/device the calling thread has current
 *     (torch's primary context, in a torch host process).
 *   - It spawns NO threads and NO internal streams: every kernel and copy is
 *     issued on the caller-supplied stream (single-stream by design — with
 *     the output already on-device there is no D2H bulk to overlap, so the
 *     daemon's dual-stream machinery buys nothing and is deliberately gone;
 *     fewer stream objects also keeps coexistence with TRT-LLM's CUDA graph
 *     capture simple).
 *   - It must NEVER be called while stream capture is active anywhere on the
 *     device (cudaMalloc + kernel launches during capture are illegal). A
 *     host that captures graphs must guarantee this by serialization; a cheap
 *     cudaStreamIsCapturing check refuses the obvious misuse anyway.
 *   - Nothing is cached that breaks across capture epochs: no events, no
 *     graphs, no stream state — only device buffers and __constant__ taps.
 *
 * Build (inside nvcr.io/nvidia/tensorrt-llm/release:1.2.1):
 *   nvcc -O2 -arch=sm_120 -rdc=true -DCUOPUS_CUDA -DFIXED_POINT=1 -DOPUS_BUILD
 *     -Idevice -Idevice/gen -Idevice/gen/silk -Idevice/gen/celt
 *     -Idevice/gen/include -Xcompiler -fPIC -shared libcuopus.cu -o out/libcuopus.so
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cuopus_device.h"
#include "device/cuopus_silk_all.cuh"
#include "device/cuopus_celt_all.cuh"
#include "device/cuopus_opus.cuh"
#include "cuopus_io.h"
#include "cuopus_demux.h"
#include "cuopus_kernels.cuh"

#define RC_OK 0
#define RC_SOFT 1
#define RC_CUDA 2
#define RC_ARG 3

/* CUDA check for library code: fills err and jumps to fail with rc=RC_CUDA.
 * Never exits — the context belongs to the caller. */
#define LCHK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
    snprintf(err, errsz, "CUDA error %s at %s:%d: %s", #x, __FILE__, __LINE__, \
             cudaGetErrorString(_e)); rc = RC_CUDA; goto fail; } } while (0)

/* ---- grow-only workspaces (lazy, persistent, freed by release) ---- */
static struct {
    unsigned char *d_data;            size_t cap_data;
    unsigned *d_off;                  size_t cap_off;
    unsigned *d_len;                  size_t cap_len;
    unsigned *d_first;                size_t cap_first;
    unsigned *d_npkts;                size_t cap_npkts;
    unsigned *d_trim;                 size_t cap_trim;
    unsigned *d_keep;                 size_t cap_keep;
    unsigned long long *d_pcm_off;    size_t cap_pcm_off;
    unsigned long long *d_dst_off;    size_t cap_dst_off;
    cu_opus_decoder *d_states;        size_t cap_states;
    int16_t *d_pcm;                   size_t cap_pcm;
    int16_t *d_cat;                   size_t cap_cat;
    int16_t *d_carry;                 size_t cap_carry;
    cu_cli_result *d_res;             size_t cap_res;
    char *d_arena;                    size_t cap_arena;
    char **d_tops;                    size_t cap_tops;
    int *d_ovf;                       size_t cap_ovf;
    /* host (plain malloc — nothing pinned: no bulk D2H exists) */
    unsigned long long *h_off;        size_t hcap_off;   /* nb*max_bn pcm+dst pairs */
    uint64_t *h_caps;                 size_t hcap_caps;
    cu_cli_result *h_res;             size_t hcap_res;
    int taps_set;
} L;                                  /* zero-initialized */

static long long l_dev_bytes = 0;
static int l_last_ovf = 0;

static int l_gpu_grow(void **p, size_t *cap, size_t need, char *err, size_t errsz)
{
    if (need <= *cap) return 0;
    if (*p) { cudaFree(*p); l_dev_bytes -= (long long)*cap; *p = NULL; *cap = 0; }
    cudaError_t e = cudaMalloc(p, need);
    if (e != cudaSuccess) {
        snprintf(err, errsz, "cudaMalloc %zu bytes: %s", need, cudaGetErrorString(e));
        return RC_CUDA;
    }
    *cap = need; l_dev_bytes += (long long)need;
    return 0;
}

static int l_host_grow(void **p, size_t *cap, size_t need, char *err, size_t errsz)
{
    if (need <= *cap) return 0;
    void *np = realloc(*p, need);
    if (!np) { snprintf(err, errsz, "host OOM (%zu bytes)", need); return RC_SOFT; }
    *p = np; *cap = need;
    return 0;
}

#define GG(ptr, cap, need) do { int _r = l_gpu_grow((void **)&(ptr), &(cap), (need), err, errsz); \
    if (_r) { rc = _r; goto fail; } } while (0)
#define HG(ptr, cap, need) do { int _r = l_host_grow((void **)&(ptr), &(cap), (need), err, errsz); \
    if (_r) { rc = _r; goto fail; } } while (0)

static double l_env_f(const char *name, double dflt)
{
    const char *e = getenv(name);
    return (e && *e) ? atof(e) : dflt;
}

static unsigned l_env_u(const char *name, unsigned dflt)
{
    const char *e = getenv(name);
    return (e && atoi(e) > 0) ? (unsigned)atoi(e) : dflt;
}

/* Demux + chain-table from caller bytes (copies them: cu_demux_mem takes
 * ownership of a malloc'd buffer). chains==NULL -> demux only (probe). */
static int l_demux(const uint8_t *ogg, size_t n, double seg_s, double preroll_ms,
                   cu_demux_t *d, cu_chains2_t *c, char *err, size_t errsz)
{
    memset(d, 0, sizeof(*d));
    if (c) memset(c, 0, sizeof(*c));
    if (!ogg || !n) { snprintf(err, errsz, "empty input"); return RC_ARG; }
    unsigned char *copy = (unsigned char *)malloc(n);
    if (!copy) { snprintf(err, errsz, "host OOM (%zu bytes input copy)", n); return RC_SOFT; }
    memcpy(copy, ogg, n);
    if (!cu_demux_mem(copy, (uint64_t)n, d, err, errsz)) return RC_SOFT;
    if (d->channels != 1) {
        snprintf(err, errsz, "only mono supported (channels=%d)", d->channels);
        cu_demux_free(d); return RC_SOFT;
    }
    if (d->granule_match != d->granule_checked) {
        snprintf(err, errsz, "granule cross-check failed (%u/%u)",
                 d->granule_match, d->granule_checked);
        cu_demux_free(d); return RC_SOFT;
    }
    if (c && !cu_build_chains2(d, seg_s, preroll_ms, c, err, errsz)) {
        cu_demux_free(d); return RC_SOFT;
    }
    return 0;
}

static int64_t l_out16_of(int64_t play48)
{
    int64_t v = (play48 - SWR_TAIL_48K) / 3;
    return v < 0 ? 0 : v;
}

extern "C" int cuopus_last_arena_overflow(void) { return l_last_ovf; }

extern "C" long long cuopus_workspace_bytes(void) { return l_dev_bytes; }

extern "C" void cuopus_release_workspaces(void)
{
#define FREE_D(p, c) do { if (L.p) { cudaFree(L.p); L.p = NULL; } L.c = 0; } while (0)
    FREE_D(d_data, cap_data); FREE_D(d_off, cap_off); FREE_D(d_len, cap_len);
    FREE_D(d_first, cap_first); FREE_D(d_npkts, cap_npkts);
    FREE_D(d_trim, cap_trim); FREE_D(d_keep, cap_keep);
    FREE_D(d_pcm_off, cap_pcm_off); FREE_D(d_dst_off, cap_dst_off);
    FREE_D(d_states, cap_states); FREE_D(d_pcm, cap_pcm); FREE_D(d_cat, cap_cat);
    FREE_D(d_carry, cap_carry); FREE_D(d_res, cap_res);
    FREE_D(d_arena, cap_arena); FREE_D(d_tops, cap_tops); FREE_D(d_ovf, cap_ovf);
#undef FREE_D
    free(L.h_off); L.h_off = NULL; L.hcap_off = 0;
    free(L.h_caps); L.h_caps = NULL; L.hcap_caps = 0;
    free(L.h_res); L.h_res = NULL; L.hcap_res = 0;
    L.taps_set = 0;   /* __constant__ survives, but re-upload is 196 bytes */
    l_dev_bytes = 0;
}

extern "C" int cuopus_probe(const uint8_t *ogg, size_t n, int64_t *samples16k,
                            char *errbuf, size_t errcap)
{
    char *err = errbuf; size_t errsz = errcap ? errcap : 1;
    char scratch[8]; if (!err) { err = scratch; errsz = sizeof(scratch); }
    err[0] = 0;
    if (!samples16k) { snprintf(err, errsz, "samples16k is NULL"); return RC_ARG; }
    cu_demux_t d;
    int rc = l_demux(ogg, n, 1.0, 250.0, &d, NULL, err, errsz);
    if (rc) return rc;
    *samples16k = l_out16_of(d.pcm_samples_48k);
    cu_demux_free(&d);
    return RC_OK;
}

extern "C" int cuopus_decode_f16k_on(const uint8_t *ogg, size_t n, float *out_dev,
                                     size_t out_cap, int64_t *out_len,
                                     void *cuda_stream, char *errbuf, size_t errcap)
{
    char *err = errbuf; size_t errsz = errcap ? errcap : 1;
    char scratch[8]; if (!err) { err = scratch; errsz = sizeof(scratch); }
    err[0] = 0;
    l_last_ovf = 0;
    if (out_len) *out_len = 0;

    cudaStream_t st = (cudaStream_t)cuda_stream;
    int rc = RC_OK;
    cu_demux_t d; memset(&d, 0, sizeof(d));
    cu_chains2_t c; memset(&c, 0, sizeof(c));
    int have_dc = 0;

    double seg_s = l_env_f("CUOPUS_SEG_S", 1.0);
    double preroll_ms = l_env_f("CUOPUS_PREROLL_MS", DEF_PREROLL_MS);
    unsigned stride = l_env_u("CUOPUS_STRIDE", DEF_STRIDE);
    unsigned max_batch = l_env_u("CUOPUS_MAX_BATCH", CUOPUS_MAX_BATCH);

    {
        cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
        cudaError_t e = cudaStreamIsCapturing(st, &cs);
        if (e == cudaSuccess && cs != cudaStreamCaptureStatusNone) {
            snprintf(err, errsz, "refusing to decode during CUDA stream capture");
            return RC_ARG;
        }
        cudaGetLastError();   /* swallow a possible cudaErrorStreamCaptureImplicit */
    }

    rc = l_demux(ogg, n, seg_s, preroll_ms, &d, &c, err, errsz);
    if (rc) return rc;
    have_dc = 1;

    {
    const cu_blob *b = &d.blob;
    uint32_t nch = c.n_chains;
    if (nch == 0) { snprintf(err, errsz, "no chains"); rc = RC_SOFT; goto fail; }

    /* per-chain decode capacity from TOC durations, validated vs trim+keep */
    HG(L.h_caps, L.hcap_caps, 8ull * nch);
    uint64_t *cap = L.h_caps;
    int64_t play48 = 0;
    for (uint32_t i = 0; i < nch; i++) {
        uint64_t s = 0;
        if ((uint64_t)c.first[i] + c.npkts[i] > b->n_packets) {
            snprintf(err, errsz, "chain %u out of packet range", i); rc = RC_SOFT; goto fail;
        }
        for (uint32_t k = 0; k < c.npkts[i]; k++) {
            uint32_t p = c.first[i] + k;
            s += (uint64_t)cu_dmx_pkt_samples48(b->data + b->off[p], b->len[p]);
        }
        cap[i] = s;
        if (s < (uint64_t)c.trim[i] + c.keep[i]) {
            snprintf(err, errsz, "chain %u capacity %llu < trim %u + keep %u",
                     i, (unsigned long long)s, c.trim[i], c.keep[i]);
            rc = RC_SOFT; goto fail;
        }
        play48 += c.keep[i];
    }
    int64_t out16_total = l_out16_of(play48);
    if ((uint64_t)out16_total > out_cap) {
        snprintf(err, errsz, "output capacity %zu < required %lld samples",
                 out_cap, (long long)out16_total);
        rc = RC_ARG; goto fail;
    }
    if (out16_total > 0 && !out_dev) {
        snprintf(err, errsz, "out_dev is NULL"); rc = RC_ARG; goto fail;
    }
    if (out16_total == 0) {   /* nothing to emit; skip all GPU work */
        if (out_len) *out_len = 0;
        rc = RC_OK; goto fail;   /* shared cleanup, rc==0 */
    }

    uint32_t max_bn = nch < max_batch ? nch : max_batch;
    uint32_t nb = (nch + max_bn - 1) / max_bn;
    uint64_t max_bsamp = 0, max_bkeep = 0;
    for (uint32_t i0 = 0; i0 < nch; i0 += max_bn) {
        uint32_t bn = (nch - i0) < max_bn ? (nch - i0) : max_bn;
        uint64_t s = 0, k = 0;
        for (uint32_t i = 0; i < bn; i++) { s += cap[i0 + i]; k += c.keep[i0 + i]; }
        if (s > max_bsamp) max_bsamp = s;
        if (k > max_bkeep) max_bkeep = k;
    }

    GG(L.d_data, L.cap_data, b->data_bytes ? b->data_bytes : 1);
    GG(L.d_off, L.cap_off, 4ull * b->n_packets);
    GG(L.d_len, L.cap_len, 4ull * b->n_packets);
    GG(L.d_first, L.cap_first, 4ull * nch);
    GG(L.d_npkts, L.cap_npkts, 4ull * nch);
    GG(L.d_trim, L.cap_trim, 4ull * nch);
    GG(L.d_keep, L.cap_keep, 4ull * nch);
    GG(L.d_pcm_off, L.cap_pcm_off, 8ull * max_bn);
    GG(L.d_dst_off, L.cap_dst_off, 8ull * max_bn);
    GG(L.d_states, L.cap_states, sizeof(cu_opus_decoder) * (size_t)max_bn);
    GG(L.d_pcm, L.cap_pcm, 2ull * max_bsamp);
    GG(L.d_cat, L.cap_cat, 2ull * (max_bkeep + CARRY_MAX));
    GG(L.d_carry, L.cap_carry, 2ull * CARRY_MAX);
    GG(L.d_res, L.cap_res, sizeof(cu_cli_result) * (size_t)nch);
    HG(L.h_res, L.hcap_res, sizeof(cu_cli_result) * (size_t)nch);
    /* per-batch host offset slices — never reused before their copy lands */
    HG(L.h_off, L.hcap_off, 8ull * 2 * (uint64_t)nb * max_bn);

    unsigned n_launch_max = ((max_bn * stride + BLOCK - 1) / BLOCK) * BLOCK;
    unsigned n_slots = n_launch_max / stride + 1;
    GG(L.d_arena, L.cap_arena, (size_t)n_slots * ARENA_BYTES);
    GG(L.d_tops, L.cap_tops, sizeof(char *) * n_slots);
    GG(L.d_ovf, L.cap_ovf, sizeof(int) * n_slots);
    {
        unsigned long arena_per_thread = ARENA_BYTES, stride_ul = stride;
        LCHK(cudaMemcpyToSymbolAsync(cuopus_arena_stride, &stride_ul, sizeof(stride_ul), 0,
                                     cudaMemcpyHostToDevice, st));
        LCHK(cudaMemcpyToSymbolAsync(cuopus_arena_base, &L.d_arena, sizeof(L.d_arena), 0,
                                     cudaMemcpyHostToDevice, st));
        LCHK(cudaMemcpyToSymbolAsync(cuopus_arena_per_thread, &arena_per_thread,
                                     sizeof(arena_per_thread), 0, cudaMemcpyHostToDevice, st));
        LCHK(cudaMemcpyToSymbolAsync(cuopus_arena_tops, &L.d_tops, sizeof(L.d_tops), 0,
                                     cudaMemcpyHostToDevice, st));
        LCHK(cudaMemcpyToSymbolAsync(cuopus_arena_ovf, &L.d_ovf, sizeof(L.d_ovf), 0,
                                     cudaMemcpyHostToDevice, st));
    }
    if (!L.taps_set) {
        float taps[RS_TAPS];
        rs_make_taps(taps);
        LCHK(cudaMemcpyToSymbolAsync(cuopus_rs_taps, taps, sizeof(taps), 0,
                                     cudaMemcpyHostToDevice, st));
        L.taps_set = 1;
    }

    LCHK(cudaMemcpyAsync(L.d_data, b->data, b->data_bytes, cudaMemcpyHostToDevice, st));
    LCHK(cudaMemcpyAsync(L.d_off, b->off, 4ull * b->n_packets, cudaMemcpyHostToDevice, st));
    LCHK(cudaMemcpyAsync(L.d_len, b->len, 4ull * b->n_packets, cudaMemcpyHostToDevice, st));
    LCHK(cudaMemcpyAsync(L.d_first, c.first, 4ull * nch, cudaMemcpyHostToDevice, st));
    LCHK(cudaMemcpyAsync(L.d_npkts, c.npkts, 4ull * nch, cudaMemcpyHostToDevice, st));
    LCHK(cudaMemcpyAsync(L.d_trim, c.trim, 4ull * nch, cudaMemcpyHostToDevice, st));
    LCHK(cudaMemcpyAsync(L.d_keep, c.keep, 4ull * nch, cudaMemcpyHostToDevice, st));

    /* ---- batched decode -> compact -> resample straight into out_dev ----
     * Same math/order as the daemon's mode16 path (cuopus_run): the per-batch
     * out_hi split assigns outputs to launches but every output sample is a
     * pure 49-tap function of the same compacted window, so the result is
     * bit-identical. Single stream: with no D2H bulk there is nothing to
     * overlap (dual-stream removed by design — see header comment). */
    int64_t out_done = 0, cat_base = 0, carry = 0;
    uint32_t bi = 0;
    for (uint32_t i0 = 0; i0 < nch; i0 += max_bn, bi++) {
        uint32_t bn = (nch - i0) < max_bn ? (nch - i0) : max_bn;
        unsigned long long *h_pcm_off = L.h_off + 2ull * bi * max_bn;
        unsigned long long *h_dst_off = h_pcm_off + max_bn;
        uint64_t s = 0, k = 0;
        for (uint32_t i = 0; i < bn; i++) {
            h_pcm_off[i] = s; s += cap[i0 + i];
            h_dst_off[i] = (uint64_t)carry + k; k += c.keep[i0 + i];
        }
        LCHK(cudaMemcpyAsync(L.d_pcm_off, h_pcm_off, 8ull * bn, cudaMemcpyHostToDevice, st));
        LCHK(cudaMemcpyAsync(L.d_dst_off, h_dst_off, 8ull * bn, cudaMemcpyHostToDevice, st));

        unsigned n_launch = ((bn * stride + BLOCK - 1) / BLOCK) * BLOCK;
        cuopus_cli_decode<<<n_launch / BLOCK, BLOCK, 0, st>>>(
            L.d_data, L.d_off, L.d_len, L.d_first, L.d_npkts, (int)i0, (int)bn,
            L.d_states, L.d_pcm, L.d_pcm_off, L.d_res + i0);
        LCHK(cudaGetLastError());

        cuopus_compact<<<bn, 256, 0, st>>>(L.d_pcm, L.d_pcm_off,
            L.d_trim + i0, L.d_keep + i0, L.d_dst_off, L.d_cat, (int)bn);
        LCHK(cudaGetLastError());

        int64_t avail = carry + (int64_t)k;
        int64_t out_hi = (cat_base + avail - RS_PAD - 1) / 3 + 1;
        if (bi == nb - 1) out_hi = out16_total;   /* final: exact trim length */
        if (out_hi > out16_total) out_hi = out16_total;
        if (out_hi < out_done) out_hi = out_done;
        int64_t n_out = out_hi - out_done;
        if (n_out > 0) {
            unsigned rgrid = (unsigned)((n_out + 255) / 256);
            cuopus_resample48to16<<<rgrid, 256, 0, st>>>(
                L.d_cat, cat_base, avail, out_done, n_out, out_dev + out_done);
            LCHK(cudaGetLastError());
        }
        /* carry the resample halo into the next batch's window */
        int64_t new_base = 3 * out_hi - RS_PAD;
        if (new_base < cat_base) new_base = cat_base;
        int64_t new_carry = cat_base + avail - new_base;
        if (new_carry > CARRY_MAX) new_carry = CARRY_MAX;   /* only reachable on the final batch */
        if (bi + 1 < nb && new_carry > 0) {
            LCHK(cudaMemcpyAsync(L.d_carry, L.d_cat + (new_base - cat_base),
                                 2ull * new_carry, cudaMemcpyDeviceToDevice, st));
            LCHK(cudaMemcpyAsync(L.d_cat, L.d_carry, 2ull * new_carry,
                                 cudaMemcpyDeviceToDevice, st));
        }
        cat_base = new_base; carry = new_carry; out_done = out_hi;
    }

    LCHK(cudaMemcpyAsync(L.h_res, L.d_res, sizeof(cu_cli_result) * (size_t)nch,
                         cudaMemcpyDeviceToHost, st));
    LCHK(cudaStreamSynchronize(st));

    /* validate every chain BEFORE trusting the output */
    for (uint32_t ci = 0; ci < nch; ci++) {
        if (L.h_res[ci].ovf) {
            snprintf(err, errsz, "chain %u: arena overflow", ci);
            l_last_ovf = 1; rc = RC_SOFT; goto fail;
        }
        if (L.h_res[ci].stop_reason != 0) {
            snprintf(err, errsz, "chain %u: device decode error %d",
                     ci, L.h_res[ci].stop_reason);
            rc = RC_SOFT; goto fail;
        }
        if (L.h_res[ci].samples != cap[ci]) {
            snprintf(err, errsz, "chain %u: decoded %u samples, expected %llu",
                     ci, L.h_res[ci].samples, (unsigned long long)cap[ci]);
            rc = RC_SOFT; goto fail;
        }
    }
    if (out_len) *out_len = out16_total;
    rc = RC_OK;
    }

fail:
    if (have_dc) { cu_chains2_free(&c); cu_demux_free(&d); }
    return rc;
}

extern "C" int cuopus_decode_f16k(const uint8_t *ogg, size_t n, float *out_dev,
                                  size_t out_cap, int64_t *out_len,
                                  char *errbuf, size_t errcap)
{
    return cuopus_decode_f16k_on(ogg, n, out_dev, out_cap, out_len,
                                 (void *)0, errbuf, errcap);
}


/* ---- v22: cross-request merged decode ---------------------------------------
 * N small OGG-Opus files in (nearly) one GPU trip: demux each on host, merge
 * every file's packet blob + chain table into ONE cuopus_cli_decode launch and
 * ONE cuopus_compact launch (per-chain dst offsets place each file's kept PCM
 * in its own region of d_cat), then one tiny resample launch per file into its
 * own out_dev, one result D2H, one sync. Per-file output is BIT-IDENTICAL to a
 * cuopus_decode_f16k_on call for that file alone: decode is per-chain, compact
 * is per-chain, and each file's resample sees exactly the window a single-file
 * final batch would (cat_base=0, avail=file keep, out_hi=file out16).
 * Falls back to RC_ARG if the merged chain count exceeds one launch
 * (max_batch) -- the caller should then decode files individually. */
extern "C" int cuopus_decode_f16k_multi(
    const uint8_t *const *oggs, const size_t *ns, int nfiles,
    float *const *out_devs, const size_t *out_caps, int64_t *out_lens,
    void *cuda_stream, char *errbuf, size_t errcap)
{
    char *err = errbuf; size_t errsz = errcap ? errcap : 1;
    char scratch[8]; if (!err) { err = scratch; errsz = sizeof(scratch); }
    err[0] = 0;
    l_last_ovf = 0;
    if (out_lens) for (int f = 0; f < nfiles; f++) out_lens[f] = 0;
    if (nfiles <= 0) return RC_OK;
    if (nfiles == 1)
        return cuopus_decode_f16k_on(oggs[0], ns[0], out_devs[0], out_caps[0],
                                     out_lens ? &out_lens[0] : (int64_t *)0,
                                     cuda_stream, errbuf, errcap);

    cudaStream_t st = (cudaStream_t)cuda_stream;
    int rc = RC_OK;

    double seg_s = l_env_f("CUOPUS_SEG_S", 1.0);
    double preroll_ms = l_env_f("CUOPUS_PREROLL_MS", DEF_PREROLL_MS);
    unsigned stride = l_env_u("CUOPUS_STRIDE", DEF_STRIDE);
    unsigned max_batch = l_env_u("CUOPUS_MAX_BATCH", CUOPUS_MAX_BATCH);

    {
        cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
        cudaError_t e = cudaStreamIsCapturing(st, &cs);
        if (e == cudaSuccess && cs != cudaStreamCaptureStatusNone) {
            snprintf(err, errsz, "refusing to decode during CUDA stream capture");
            return RC_ARG;
        }
        cudaGetLastError();
    }

    cu_demux_t *dd = (cu_demux_t *)calloc((size_t)nfiles, sizeof(cu_demux_t));
    cu_chains2_t *cc = (cu_chains2_t *)calloc((size_t)nfiles, sizeof(cu_chains2_t));
    int n_demuxed = 0;
    uint8_t *m_data = NULL; uint32_t *m_off = NULL, *m_len = NULL;
    uint32_t *m_first = NULL, *m_npkts = NULL, *m_trim = NULL, *m_keep = NULL;
    unsigned long long *m_pcm_off = NULL, *m_dst_off = NULL;
    if (!dd || !cc) { snprintf(err, errsz, "host OOM (demux tables)"); rc = RC_SOFT; goto mfail; }

    {
    uint64_t tot_bytes = 0, tot_pkts = 0; uint32_t nch_all = 0;
    for (int f = 0; f < nfiles; f++) {
        char ferr[256]; ferr[0] = 0;
        rc = l_demux(oggs[f], ns[f], seg_s, preroll_ms, &dd[f], &cc[f], ferr, sizeof(ferr));
        if (rc) { snprintf(err, errsz, "file %d: %s", f, ferr); goto mfail; }
        n_demuxed = f + 1;
        tot_bytes += dd[f].blob.data_bytes;
        tot_pkts += dd[f].blob.n_packets;
        nch_all += cc[f].n_chains;
    }
    if (nch_all == 0) { snprintf(err, errsz, "no chains in any file"); rc = RC_SOFT; goto mfail; }
    if (nch_all > max_batch) {
        snprintf(err, errsz, "merged chains %u > max_batch %u (decode files individually)",
                 nch_all, max_batch);
        rc = RC_ARG; goto mfail;
    }

    /* merged host tables */
    m_data = (uint8_t *)malloc(tot_bytes ? tot_bytes : 1);
    m_off = (uint32_t *)malloc(4ull * tot_pkts);
    m_len = (uint32_t *)malloc(4ull * tot_pkts);
    m_first = (uint32_t *)malloc(4ull * nch_all);
    m_npkts = (uint32_t *)malloc(4ull * nch_all);
    m_trim = (uint32_t *)malloc(4ull * nch_all);
    m_keep = (uint32_t *)malloc(4ull * nch_all);
    m_pcm_off = (unsigned long long *)malloc(8ull * nch_all);
    m_dst_off = (unsigned long long *)malloc(8ull * nch_all);
    if (!m_data || !m_off || !m_len || !m_first || !m_npkts || !m_trim || !m_keep
        || !m_pcm_off || !m_dst_off) {
        snprintf(err, errsz, "host OOM (merged tables)"); rc = RC_SOFT; goto mfail;
    }
    { int _r = l_host_grow((void **)&L.h_caps, &L.hcap_caps, 8ull * nch_all, err, errsz);
      if (_r) { rc = _r; goto mfail; } }
    uint64_t *cap = L.h_caps;

    uint64_t data_base = 0, pkt_base = 0, pcm_cursor = 0, cat_cursor = 0;
    uint32_t ci = 0;
    int64_t out16_tot[/* VLA avoided */ 1]; (void)out16_tot;
    int64_t *f_out16 = (int64_t *)malloc(8ull * nfiles);
    uint64_t *f_cat_base = (uint64_t *)malloc(8ull * nfiles);
    uint64_t *f_keep = (uint64_t *)malloc(8ull * nfiles);
    if (!f_out16 || !f_cat_base || !f_keep) {
        free(f_out16); free(f_cat_base); free(f_keep);
        snprintf(err, errsz, "host OOM (per-file tables)"); rc = RC_SOFT; goto mfail;
    }

    for (int f = 0; f < nfiles; f++) {
        const cu_blob *b = &dd[f].blob;
        memcpy(m_data + data_base, b->data, b->data_bytes);
        for (uint32_t p = 0; p < b->n_packets; p++) {
            m_off[pkt_base + p] = b->off[p] + (uint32_t)data_base;
            m_len[pkt_base + p] = b->len[p];
        }
        int64_t play48 = 0;
        uint64_t file_keep = 0;
        f_cat_base[f] = cat_cursor;
        for (uint32_t i = 0; i < cc[f].n_chains; i++, ci++) {
            if ((uint64_t)cc[f].first[i] + cc[f].npkts[i] > b->n_packets) {
                snprintf(err, errsz, "file %d chain %u out of packet range", f, i);
                free(f_out16); free(f_cat_base); free(f_keep);
                rc = RC_SOFT; goto mfail;
            }
            uint64_t s = 0;
            for (uint32_t k = 0; k < cc[f].npkts[i]; k++) {
                uint32_t p = cc[f].first[i] + k;
                s += (uint64_t)cu_dmx_pkt_samples48(b->data + b->off[p], b->len[p]);
            }
            if (s < (uint64_t)cc[f].trim[i] + cc[f].keep[i]) {
                snprintf(err, errsz, "file %d chain %u capacity %llu < trim+keep",
                         f, i, (unsigned long long)s);
                free(f_out16); free(f_cat_base); free(f_keep);
                rc = RC_SOFT; goto mfail;
            }
            m_first[ci] = cc[f].first[i] + (uint32_t)pkt_base;
            m_npkts[ci] = cc[f].npkts[i];
            m_trim[ci] = cc[f].trim[i];
            m_keep[ci] = cc[f].keep[i];
            cap[ci] = s;
            m_pcm_off[ci] = pcm_cursor; pcm_cursor += s;
            m_dst_off[ci] = cat_cursor + file_keep; file_keep += cc[f].keep[i];
            play48 += cc[f].keep[i];
        }
        f_keep[f] = file_keep;
        cat_cursor += file_keep;
        f_out16[f] = l_out16_of(play48);
        if ((uint64_t)f_out16[f] > out_caps[f]) {
            snprintf(err, errsz, "file %d: output capacity %zu < required %lld",
                     f, out_caps[f], (long long)f_out16[f]);
            free(f_out16); free(f_cat_base); free(f_keep);
            rc = RC_ARG; goto mfail;
        }
        if (f_out16[f] > 0 && !out_devs[f]) {
            snprintf(err, errsz, "file %d: out_dev is NULL", f);
            free(f_out16); free(f_cat_base); free(f_keep);
            rc = RC_ARG; goto mfail;
        }
        data_base += b->data_bytes;
        pkt_base += b->n_packets;
    }

    /* device workspaces (grow-only, shared with the single-file path) */
    int mrc = RC_OK;
    do {
#define MG(ptr, capv, need) { int _r = l_gpu_grow((void **)&(ptr), &(capv), (need), err, errsz); \
        if (_r) { mrc = _r; break; } }
#define MH(ptr, capv, need) { int _r = l_host_grow((void **)&(ptr), &(capv), (need), err, errsz); \
        if (_r) { mrc = _r; break; } }
        MG(L.d_data, L.cap_data, tot_bytes ? tot_bytes : 1);
        MG(L.d_off, L.cap_off, 4ull * tot_pkts);
        MG(L.d_len, L.cap_len, 4ull * tot_pkts);
        MG(L.d_first, L.cap_first, 4ull * nch_all);
        MG(L.d_npkts, L.cap_npkts, 4ull * nch_all);
        MG(L.d_trim, L.cap_trim, 4ull * nch_all);
        MG(L.d_keep, L.cap_keep, 4ull * nch_all);
        MG(L.d_pcm_off, L.cap_pcm_off, 8ull * nch_all);
        MG(L.d_dst_off, L.cap_dst_off, 8ull * nch_all);
        MG(L.d_states, L.cap_states, sizeof(cu_opus_decoder) * (size_t)nch_all);
        MG(L.d_pcm, L.cap_pcm, 2ull * pcm_cursor);
        MG(L.d_cat, L.cap_cat, 2ull * (cat_cursor ? cat_cursor : 1));
        MG(L.d_res, L.cap_res, sizeof(cu_cli_result) * (size_t)nch_all);
        MH(L.h_res, L.hcap_res, sizeof(cu_cli_result) * (size_t)nch_all);
        unsigned n_launch_max = ((nch_all * stride + BLOCK - 1) / BLOCK) * BLOCK;
        unsigned n_slots = n_launch_max / stride + 1;
        MG(L.d_arena, L.cap_arena, (size_t)n_slots * ARENA_BYTES);
        MG(L.d_tops, L.cap_tops, sizeof(char *) * n_slots);
        MG(L.d_ovf, L.cap_ovf, sizeof(int) * n_slots);
#undef MG
#undef MH
    } while (0);
    if (mrc) { free(f_out16); free(f_cat_base); free(f_keep); rc = mrc; goto mfail; }

    {
        unsigned long arena_per_thread = ARENA_BYTES, stride_ul = stride;
        cudaError_t e;
#define MC(call) { e = (call); if (e) { snprintf(err, errsz, "%s", cudaGetErrorString(e)); \
        free(f_out16); free(f_cat_base); free(f_keep); rc = RC_CUDA; goto mfail; } }
        MC(cudaMemcpyToSymbolAsync(cuopus_arena_stride, &stride_ul, sizeof(stride_ul), 0,
                                   cudaMemcpyHostToDevice, st));
        MC(cudaMemcpyToSymbolAsync(cuopus_arena_base, &L.d_arena, sizeof(L.d_arena), 0,
                                   cudaMemcpyHostToDevice, st));
        MC(cudaMemcpyToSymbolAsync(cuopus_arena_per_thread, &arena_per_thread,
                                   sizeof(arena_per_thread), 0, cudaMemcpyHostToDevice, st));
        MC(cudaMemcpyToSymbolAsync(cuopus_arena_tops, &L.d_tops, sizeof(L.d_tops), 0,
                                   cudaMemcpyHostToDevice, st));
        MC(cudaMemcpyToSymbolAsync(cuopus_arena_ovf, &L.d_ovf, sizeof(L.d_ovf), 0,
                                   cudaMemcpyHostToDevice, st));
        if (!L.taps_set) {
            float taps[RS_TAPS];
            rs_make_taps(taps);
            MC(cudaMemcpyToSymbolAsync(cuopus_rs_taps, taps, sizeof(taps), 0,
                                       cudaMemcpyHostToDevice, st));
            L.taps_set = 1;
        }
        MC(cudaMemcpyAsync(L.d_data, m_data, tot_bytes ? tot_bytes : 1, cudaMemcpyHostToDevice, st));
        MC(cudaMemcpyAsync(L.d_off, m_off, 4ull * tot_pkts, cudaMemcpyHostToDevice, st));
        MC(cudaMemcpyAsync(L.d_len, m_len, 4ull * tot_pkts, cudaMemcpyHostToDevice, st));
        MC(cudaMemcpyAsync(L.d_first, m_first, 4ull * nch_all, cudaMemcpyHostToDevice, st));
        MC(cudaMemcpyAsync(L.d_npkts, m_npkts, 4ull * nch_all, cudaMemcpyHostToDevice, st));
        MC(cudaMemcpyAsync(L.d_trim, m_trim, 4ull * nch_all, cudaMemcpyHostToDevice, st));
        MC(cudaMemcpyAsync(L.d_keep, m_keep, 4ull * nch_all, cudaMemcpyHostToDevice, st));
        MC(cudaMemcpyAsync(L.d_pcm_off, m_pcm_off, 8ull * nch_all, cudaMemcpyHostToDevice, st));
        MC(cudaMemcpyAsync(L.d_dst_off, m_dst_off, 8ull * nch_all, cudaMemcpyHostToDevice, st));

        unsigned n_launch = ((nch_all * stride + BLOCK - 1) / BLOCK) * BLOCK;
        cuopus_cli_decode<<<n_launch / BLOCK, BLOCK, 0, st>>>(
            L.d_data, L.d_off, L.d_len, L.d_first, L.d_npkts, 0, (int)nch_all,
            L.d_states, L.d_pcm, L.d_pcm_off, L.d_res);
        MC(cudaGetLastError());
        cuopus_compact<<<nch_all, 256, 0, st>>>(L.d_pcm, L.d_pcm_off,
            L.d_trim, L.d_keep, L.d_dst_off, L.d_cat, (int)nch_all);
        MC(cudaGetLastError());
        for (int f = 0; f < nfiles; f++) {
            if (f_out16[f] <= 0) continue;
            unsigned rgrid = (unsigned)((f_out16[f] + 255) / 256);
            cuopus_resample48to16<<<rgrid, 256, 0, st>>>(
                L.d_cat + f_cat_base[f], 0, (long long)f_keep[f],
                0, f_out16[f], out_devs[f]);
            MC(cudaGetLastError());
        }
        MC(cudaMemcpyAsync(L.h_res, L.d_res, sizeof(cu_cli_result) * (size_t)nch_all,
                           cudaMemcpyDeviceToHost, st));
        MC(cudaStreamSynchronize(st));
#undef MC
    }

    for (uint32_t k = 0; k < nch_all; k++) {
        if (L.h_res[k].ovf) {
            snprintf(err, errsz, "merged chain %u: arena overflow", k);
            l_last_ovf = 1;
            free(f_out16); free(f_cat_base); free(f_keep);
            rc = RC_SOFT; goto mfail;
        }
        if (L.h_res[k].stop_reason != 0) {
            snprintf(err, errsz, "merged chain %u: device decode error %d",
                     k, L.h_res[k].stop_reason);
            free(f_out16); free(f_cat_base); free(f_keep);
            rc = RC_SOFT; goto mfail;
        }
        if (L.h_res[k].samples != cap[k]) {
            snprintf(err, errsz, "merged chain %u: decoded %u samples, expected %llu",
                     k, L.h_res[k].samples, (unsigned long long)cap[k]);
            free(f_out16); free(f_cat_base); free(f_keep);
            rc = RC_SOFT; goto mfail;
        }
    }
    if (out_lens) for (int f = 0; f < nfiles; f++) out_lens[f] = f_out16[f];
    free(f_out16); free(f_cat_base); free(f_keep);
    rc = RC_OK;
    }

mfail:
    free(m_data); free(m_off); free(m_len); free(m_first); free(m_npkts);
    free(m_trim); free(m_keep); free(m_pcm_off); free(m_dst_off);
    for (int f = 0; f < n_demuxed; f++) { cu_chains2_free(&cc[f]); cu_demux_free(&dd[f]); }
    free(dd); free(cc);
    return rc;
}

/* Warm launch: sizes the driver's per-resident-thread local-memory pool
 * (~910 MB, fixed regardless of batch) so the first real decode pays only
 * kernels. Runs on the caller's context, default stream, n_batch=0. */
extern "C" int cuopus_warmup(char *errbuf, size_t errcap)
{
    char *err = errbuf; size_t errsz = errcap ? errcap : 1;
    char scratch[8]; if (!err) { err = scratch; errsz = sizeof(scratch); }
    err[0] = 0;
    int rc = RC_OK;
    GG(L.d_arena, L.cap_arena, 4ull * ARENA_BYTES);
    GG(L.d_tops, L.cap_tops, sizeof(char *) * 4);
    GG(L.d_ovf, L.cap_ovf, sizeof(int) * 4);
    {
        unsigned long apt = ARENA_BYTES, stride_ul = DEF_STRIDE;
        LCHK(cudaMemcpyToSymbol(cuopus_arena_stride, &stride_ul, sizeof(stride_ul)));
        LCHK(cudaMemcpyToSymbol(cuopus_arena_base, &L.d_arena, sizeof(L.d_arena)));
        LCHK(cudaMemcpyToSymbol(cuopus_arena_per_thread, &apt, sizeof(apt)));
        LCHK(cudaMemcpyToSymbol(cuopus_arena_tops, &L.d_tops, sizeof(L.d_tops)));
        LCHK(cudaMemcpyToSymbol(cuopus_arena_ovf, &L.d_ovf, sizeof(L.d_ovf)));
    }
    cuopus_cli_decode<<<1, BLOCK>>>(NULL, NULL, NULL, NULL, NULL, 0, 0,
                                    NULL, NULL, NULL, NULL);
    LCHK(cudaGetLastError());
    LCHK(cudaDeviceSynchronize());
    if (!L.taps_set) {
        float taps[RS_TAPS];
        rs_make_taps(taps);
        LCHK(cudaMemcpyToSymbol(cuopus_rs_taps, taps, sizeof(taps)));
        L.taps_set = 1;
    }
    return RC_OK;
fail:
    return rc;
}
