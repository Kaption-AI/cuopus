/* cuopus integration: production decode CLI (perf-pass 2 edition).
 *
 * Modes:
 *   cuopus_decode <packets.blob> <chains.bin> <out.pcm> [stride]
 *       Legacy blob mode (harness path). chains.bin is the 'CUC2' format
 *       written by ogg_index.py chains2.
 *   cuopus_decode --ogg <in.ogg> <out.pcm> [--seg S] [--preroll MS]
 *                 [--stride N] [--f16k] [--dump-blob F] [--dump-chains F]
 *       Demux the OGG container in C (cuopus_demux.h — port of ogg_index.py,
 *       validated byte-identical), build the CUC2 chain table, decode. The
 *       optional dumps exist for the Python-vs-C validation harness.
 *       --f16k emits float32 16 kHz (on-device polyphase resample + the
 *       CPU-path tail trim) instead of int16 48 kHz.
 *   cuopus_decode --serve <unix_socket>
 *       Persistent daemon: holds the CUDA context + module + grow-only device
 *       buffers, accepts line requests over a unix stream socket:
 *           PING\n                                        -> PONG 2\n
 *           DECODE <seg_s> <preroll_ms> <stride> <ogg> <out.pcm>\n
 *               int16 48 kHz playable PCM (legacy mode)
 *           DECODE16F <seg_s> <preroll_ms> <stride> <ogg> <out.f32>\n
 *               float32 16 kHz, resampled ON DEVICE with the same 49-tap
 *               Kaiser polyphase as gpu_resample.py and tail-trimmed to the
 *               CPU path's length contract ((play48 - 48) // 3) — the client
 *               needs no torch and no CUDA context at all.
 *           Replies (both DECODE forms):
 *               OK <samples> <play48> <kernel_ms> <wall_ms> <demux_ms>
 *                  <h2d_ms> <d2h_ms> <resample_ms> <fwrite_ms>\n
 *               ERR <message>\n            (daemon keeps serving)
 *           QUIT\n                                        -> exits 0
 *       Idle exit after CUOPUS_SERVE_IDLE_S seconds (default 600) frees the
 *       ~1.4 GB VRAM (context + driver local-memory pool) when unused; the
 *       client restarts the daemon lazily. CUDA errors still exit(1) — the
 *       context is not trustworthy after one, the client restarts us.
 *
 * Perf-pass-2 changes over the pre-perf2 CLI (outputs byte-identical in the
 * int16 mode — validated by cmp against the pre-perf2 binary):
 *   - Dual-stream copy/compute overlap: batch N's decode kernel (compute
 *     stream) runs while batch N-1's D2H (copy stream) is in flight; output
 *     staging is double-buffered, host output buffer is pinned
 *     (cudaHostAlloc), and completed chunks are fwritten while later batches
 *     are still decoding.
 *   - Per-chain trim/keep extraction moved to a device compaction kernel, so
 *     the D2H transfers only the playable samples (contiguous), not preroll.
 *   - On-device 48 k -> 16 k resample (--f16k / DECODE16F): 49-tap Kaiser
 *     polyphase, coefficients computed on host in double with the exact
 *     gpu_resample.py formula, decimation kernel over the compacted PCM with
 *     a 26-sample carry across batches (bit-close to torch conv1d: validated
 *     max diff <= 1e-5 on the 477 s corpus).
 *   - Defaults: preroll 250 ms (was 1000 — quality-identical, -71.7 dBFS vs
 *     the FIXED serial reference, and preroll is pure wasted decode work) and
 *     stride 16 (2 chains per warp — bit-exact at strides 4/8/16/32 by the
 *     phase-3 gate, ~19 % faster than 32 at 1 h).
 *
 * Decode semantics are unchanged: one chain per stride slot, batches of
 * CUOPUS_MAX_BATCH chains per launch, per-chain trim/keep so concatenated
 * output is exactly the file's playable PCM. ANY device decode error, arena
 * overflow, or per-chain sample-count mismatch fails the request (exit 1
 * one-shot / ERR reply in daemon) so the caller falls back to the CPU decoder.
 *
 * Build (phase-3 recipe, inside nvcr.io/nvidia/tensorrt-llm/release:1.2.1):
 *   nvcc -O2 -arch=sm_120 -rdc=true -DCUOPUS_CUDA -DFIXED_POINT=1 -DOPUS_BUILD
 *     -Idevice -Idevice/gen -Idevice/gen/silk -Idevice/gen/celt
 *     -Idevice/gen/include decode_cli.cu -o out/cuopus_decode
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>

#include "cuopus_device.h"
#include "device/cuopus_silk_all.cuh"
#include "device/cuopus_celt_all.cuh"
#include "device/cuopus_opus.cuh"
#include "cuopus_io.h"
#include "cuopus_demux.h"

#define CUCHK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr, "cuopus_decode: CUDA error %s at %s:%d: %s\n", #x, __FILE__, \
            __LINE__, cudaGetErrorString(_e)); exit(1); } } while (0)

#include "cuopus_kernels.cuh"

static cu_chains2_t cu_load_chains2_file(const char *path)
{
    cu_chains2_t c; memset(&c, 0, sizeof(c));
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cuopus_decode: cannot open %s\n", path); exit(2); }
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "CUC2", 4)) {
        fprintf(stderr, "cuopus_decode: %s: bad magic (want CUC2)\n", path); exit(2);
    }
    if (fread(&c.n_chains, 4, 1, f) != 1) exit(2);
    c.first = (uint32_t *)malloc(4ull * c.n_chains);
    c.npkts = (uint32_t *)malloc(4ull * c.n_chains);
    c.trim  = (uint32_t *)malloc(4ull * c.n_chains);
    c.keep  = (uint32_t *)malloc(4ull * c.n_chains);
    for (uint32_t i = 0; i < c.n_chains; i++) {
        uint32_t rec[4];
        if (fread(rec, 4, 4, f) != 4) { fprintf(stderr, "cuopus_decode: %s: truncated\n", path); exit(2); }
        c.first[i] = rec[0]; c.npkts[i] = rec[1]; c.trim[i] = rec[2]; c.keep[i] = rec[3];
    }
    fclose(f);
    return c;
}

/* Grow-only device + host buffers, persistent across daemon requests. */
struct gpu_ctx {
    int inited;
    unsigned char *d_data;            size_t cap_data;
    unsigned *d_off, *d_len;          size_t cap_pkts;
    unsigned *d_first, *d_npkts;      size_t cap_chains;
    unsigned *d_trim, *d_keep;        size_t cap_tk;
    unsigned long long *d_pcm_off;    size_t cap_pcm_off;
    unsigned long long *d_dst_off;    size_t cap_dst_off;
    cu_opus_decoder *d_states;        size_t cap_states;
    int16_t *d_pcm;                   size_t cap_pcm;
    int16_t *d_stage[2];              size_t cap_stage[2];   /* s16 D2H staging */
    float *d_out16[2];                size_t cap_out16[2];   /* f16k D2H staging */
    int16_t *d_cat;                   size_t cap_cat;        /* resample input + carry */
    int16_t *d_carry;                                        /* CARRY_MAX bounce */
    cu_cli_result *d_res[2];          size_t cap_res[2];
    unsigned long long *h_pcm_off;    size_t hcap_pcm_off;
    unsigned long long *h_dst_off;    size_t hcap_dst_off;
    unsigned char *h_out;             size_t hcap_out;       /* pinned */
    cu_cli_result *h_res_all;         size_t hcap_res_all;   /* pinned */
    uint64_t *cap_samples;            size_t hcap_caps;
    int taps_set;
    cudaStream_t s_comp, s_copy;
    cudaEvent_t ev_ready[2], ev_copied[2];
};
static struct gpu_ctx G;   /* zero-initialized */

static void gpu_grow(void **p, size_t *cap, size_t need)
{
    if (need <= *cap) return;
    if (*p) CUCHK(cudaFree(*p));
    *p = NULL;
    CUCHK(cudaMalloc(p, need));
    *cap = need;
}

static void host_grow(void **p, size_t *cap, size_t need)
{
    if (need <= *cap) return;
    void *np = realloc(*p, need);
    if (!np) { fprintf(stderr, "cuopus_decode: OOM (%zu bytes host)\n", need); exit(1); }
    *p = np; *cap = need;
}

static void pinned_grow(void **p, size_t *cap, size_t need)
{
    if (need <= *cap) return;
    if (*p) CUCHK(cudaFreeHost(*p));
    *p = NULL;
    CUCHK(cudaHostAlloc(p, need, cudaHostAllocDefault));
    *cap = need;
}

static void gpu_init(void)
{
    if (G.inited) return;
    CUCHK(cudaSetDevice(0));
    CUCHK(cudaFree(0));   /* create the context now, not on first request */
    CUCHK(cudaStreamCreateWithFlags(&G.s_comp, cudaStreamNonBlocking));
    CUCHK(cudaStreamCreateWithFlags(&G.s_copy, cudaStreamNonBlocking));
    for (int i = 0; i < 2; i++) {
        CUCHK(cudaEventCreate(&G.ev_ready[i]));
        CUCHK(cudaEventCreate(&G.ev_copied[i]));
    }
    CUCHK(cudaMalloc(&G.d_carry, 2ull * CARRY_MAX));
    G.inited = 1;
}

static void rs_taps_init(void)
{
    if (G.taps_set) return;
    float taps[RS_TAPS];
    rs_make_taps(taps);
    CUCHK(cudaMemcpyToSymbol(cuopus_rs_taps, taps, sizeof(taps)));
    G.taps_set = 1;
}

/* Occupancy cap experiment (negative result, kept for reproducibility). */
static unsigned env_smem_cap(void)
{
    const char *e = getenv("CUOPUS_SMEM_CAP");
    return (e && atoi(e) > 0) ? (unsigned)atoi(e) : 0;
}

static void smem_optin(unsigned smem)
{
    if (smem > 48 * 1024)
        CUCHK(cudaFuncSetAttribute(cuopus_cli_decode,
              cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
}

/* Warm launch: sizes the driver's per-resident-thread local-memory pool
 * (~900 MB, fixed regardless of batch — measured) at daemon startup so the
 * first real request pays nothing. n_batch=0 -> every thread returns. */
static void gpu_warmup(void)
{
    gpu_init();
    rs_taps_init();
    char *d_arena_w = NULL; size_t cap_w = 0;
    gpu_grow((void **)&d_arena_w, &cap_w, 4 * ARENA_BYTES);
    char **d_tops; int *d_ovf;
    CUCHK(cudaMalloc(&d_tops, sizeof(char *) * 4));
    CUCHK(cudaMalloc(&d_ovf, sizeof(int) * 4));
    unsigned long apt = ARENA_BYTES, stride_ul = DEF_STRIDE;
    CUCHK(cudaMemcpyToSymbol(cuopus_arena_stride, &stride_ul, sizeof(stride_ul)));
    CUCHK(cudaMemcpyToSymbol(cuopus_arena_base, &d_arena_w, sizeof(d_arena_w)));
    CUCHK(cudaMemcpyToSymbol(cuopus_arena_per_thread, &apt, sizeof(apt)));
    CUCHK(cudaMemcpyToSymbol(cuopus_arena_tops, &d_tops, sizeof(d_tops)));
    CUCHK(cudaMemcpyToSymbol(cuopus_arena_ovf, &d_ovf, sizeof(d_ovf)));
    unsigned smem = env_smem_cap();
    smem_optin(smem);
    cuopus_cli_decode<<<1, BLOCK, smem>>>(NULL, NULL, NULL, NULL, NULL, 0, 0,
                                          NULL, NULL, NULL, NULL);
    CUCHK(cudaGetLastError());
    CUCHK(cudaDeviceSynchronize());
    CUCHK(cudaFree(d_tops)); CUCHK(cudaFree(d_ovf)); CUCHK(cudaFree(d_arena_w));
}

/* arena for the decode kernel — shared by all requests, grow-only */
static char *g_arena = NULL; static size_t g_cap_arena = 0;
static char **g_tops = NULL; static size_t g_cap_tops = 0;
static int *g_ovf = NULL; static size_t g_cap_ovf = 0;

struct run_stats {
    uint64_t out_samples;    /* samples written to the output file */
    double kernel_ms;        /* decode kernel (event-timed, sum over batches) */
    double h2d_ms;           /* bulk uploads (packets + tables) */
    double d2h_ms;           /* staged D2H copies (event-timed on copy stream) */
    double resample_ms;      /* compact + resample kernels (event-timed) */
    double fwrite_ms;        /* host fwrite of completed chunks */
};

/* Decode one blob+chain-table to out_path. mode16: 0 = int16 48 kHz playable
 * PCM (byte-identical to the pre-perf2 CLI), 1 = float32 16 kHz (on-device
 * resample + CPU-path tail trim). Returns 0 on success; on soft failure
 * returns 1 with err filled (daemon keeps serving). CUDA errors exit. */
static int cuopus_run(const cu_blob *b, const cu_chains2_t *c, const char *out_path,
                      unsigned stride, unsigned max_batch, int mode16,
                      struct run_stats *st, char *err, size_t errsz)
{
    uint32_t n = c->n_chains;
    memset(st, 0, sizeof(*st));
    if (n == 0) { snprintf(err, errsz, "no chains"); return 1; }
    gpu_init();
    if (mode16) rs_taps_init();

    /* per-chain decode capacity from TOC durations, validated vs trim+keep */
    host_grow((void **)&G.cap_samples, &G.hcap_caps, 8ull * n);
    uint64_t *cap = G.cap_samples;
    int64_t play48 = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint64_t s = 0;
        if ((uint64_t)c->first[i] + c->npkts[i] > b->n_packets) {
            snprintf(err, errsz, "chain %u out of packet range", i); return 1;
        }
        for (uint32_t k = 0; k < c->npkts[i]; k++) {
            uint32_t p = c->first[i] + k;
            s += (uint64_t)cu_dmx_pkt_samples48(b->data + b->off[p], b->len[p]);
        }
        cap[i] = s;
        if (s < (uint64_t)c->trim[i] + c->keep[i]) {
            snprintf(err, errsz, "chain %u capacity %llu < trim %u + keep %u",
                     i, (unsigned long long)s, c->trim[i], c->keep[i]);
            return 1;
        }
        play48 += c->keep[i];
    }
    int64_t out16_total = (play48 - SWR_TAIL_48K) / 3;
    if (out16_total < 0) out16_total = 0;

    uint32_t max_bn = n < max_batch ? n : max_batch;
    uint32_t nb = (n + max_bn - 1) / max_bn;
    uint64_t max_bsamp = 0, max_bkeep = 0;
    for (uint32_t i0 = 0; i0 < n; i0 += max_bn) {
        uint32_t bn = (n - i0) < max_bn ? (n - i0) : max_bn;
        uint64_t s = 0, k = 0;
        for (uint32_t i = 0; i < bn; i++) { s += cap[i0 + i]; k += c->keep[i0 + i]; }
        if (s > max_bsamp) max_bsamp = s;
        if (k > max_bkeep) max_bkeep = k;
    }

    static size_t cap_len = 0, cap_npk = 0, cap_keep = 0;
    gpu_grow((void **)&G.d_data, &G.cap_data, b->data_bytes ? b->data_bytes : 1);
    gpu_grow((void **)&G.d_off, &G.cap_pkts, 4ull * b->n_packets);
    gpu_grow((void **)&G.d_len, &cap_len, 4ull * b->n_packets);
    gpu_grow((void **)&G.d_first, &G.cap_chains, 4ull * n);
    gpu_grow((void **)&G.d_npkts, &cap_npk, 4ull * n);
    gpu_grow((void **)&G.d_trim, &G.cap_tk, 4ull * n);
    gpu_grow((void **)&G.d_keep, &cap_keep, 4ull * n);
    gpu_grow((void **)&G.d_pcm_off, &G.cap_pcm_off, 8ull * max_bn);
    gpu_grow((void **)&G.d_dst_off, &G.cap_dst_off, 8ull * max_bn);
    gpu_grow((void **)&G.d_states, &G.cap_states, sizeof(cu_opus_decoder) * (size_t)max_bn);
    gpu_grow((void **)&G.d_pcm, &G.cap_pcm, 2ull * max_bsamp);
    for (int i = 0; i < 2; i++) {
        gpu_grow((void **)&G.d_res[i], &G.cap_res[i], sizeof(cu_cli_result) * (size_t)max_bn);
        if (mode16)
            gpu_grow((void **)&G.d_out16[i], &G.cap_out16[i], 4ull * (max_bkeep / 3 + 64));
        else
            gpu_grow((void **)&G.d_stage[i], &G.cap_stage[i], 2ull * max_bkeep);
    }
    if (mode16) gpu_grow((void **)&G.d_cat, &G.cap_cat, 2ull * (max_bkeep + CARRY_MAX));

    unsigned n_launch_max = ((max_bn * stride + BLOCK - 1) / BLOCK) * BLOCK;
    unsigned n_slots = n_launch_max / stride + 1;
    gpu_grow((void **)&g_arena, &g_cap_arena, (size_t)n_slots * ARENA_BYTES);
    gpu_grow((void **)&g_tops, &g_cap_tops, sizeof(char *) * n_slots);
    gpu_grow((void **)&g_ovf, &g_cap_ovf, sizeof(int) * n_slots);
    unsigned long arena_per_thread = ARENA_BYTES, stride_ul = stride;
    CUCHK(cudaMemcpyToSymbol(cuopus_arena_stride, &stride_ul, sizeof(stride_ul)));
    CUCHK(cudaMemcpyToSymbol(cuopus_arena_base, &g_arena, sizeof(g_arena)));
    CUCHK(cudaMemcpyToSymbol(cuopus_arena_per_thread, &arena_per_thread, sizeof(arena_per_thread)));
    CUCHK(cudaMemcpyToSymbol(cuopus_arena_tops, &g_tops, sizeof(g_tops)));
    CUCHK(cudaMemcpyToSymbol(cuopus_arena_ovf, &g_ovf, sizeof(g_ovf)));

    double th0 = now_s();
    CUCHK(cudaMemcpy(G.d_data, b->data, b->data_bytes, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(G.d_off, b->off, 4ull * b->n_packets, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(G.d_len, b->len, 4ull * b->n_packets, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(G.d_first, c->first, 4ull * n, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(G.d_npkts, c->npkts, 4ull * n, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(G.d_trim, c->trim, 4ull * n, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(G.d_keep, c->keep, 4ull * n, cudaMemcpyHostToDevice));
    st->h2d_ms = (now_s() - th0) * 1e3;

    FILE *fo = fopen(out_path, "wb");
    if (!fo) { snprintf(err, errsz, "cannot write %s", out_path); return 1; }

    size_t out_bytes_total = mode16 ? 4ull * out16_total : 2ull * (uint64_t)play48;
    host_grow((void **)&G.h_pcm_off, &G.hcap_pcm_off, 8ull * max_bn);
    host_grow((void **)&G.h_dst_off, &G.hcap_dst_off, 8ull * max_bn);
    pinned_grow((void **)&G.h_out, &G.hcap_out, out_bytes_total ? out_bytes_total : 1);
    pinned_grow((void **)&G.h_res_all, &G.hcap_res_all, sizeof(cu_cli_result) * (size_t)n);

    /* per-batch timing events (nb is small: <= ceil(7200/2048) = 4 for 2 h) */
    cudaEvent_t *evK0 = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * nb * 6);
    cudaEvent_t *evK1 = evK0 + nb, *evR0 = evK0 + 2 * nb, *evR1 = evK0 + 3 * nb,
                *evC0 = evK0 + 4 * nb, *evC1 = evK0 + 5 * nb;
    for (uint32_t i = 0; i < nb * 6; i++) CUCHK(cudaEventCreate(&evK0[i]));

    /* per-batch chunk bookkeeping */
    uint64_t *chunk_off = (uint64_t *)malloc(8ull * nb);   /* bytes into h_out */
    uint64_t *chunk_len = (uint64_t *)malloc(8ull * nb);

    uint64_t kept_before = 0;          /* s16: samples written before this batch */
    int64_t out_done = 0;              /* f16k: 16 k samples emitted */
    int64_t cat_base = 0, carry = 0;   /* f16k: resample window state */
    uint32_t bi = 0;
    unsigned smem = env_smem_cap();
    smem_optin(smem);

    for (uint32_t i0 = 0; i0 < n; i0 += max_bn, bi++) {
        uint32_t bn = (n - i0) < max_bn ? (n - i0) : max_bn;
        int buf = bi & 1;

        /* before reusing this buffer set, its previous D2H must be done; that
         * also means chunk bi-2 is complete -> overlap its fwrite with GPU */
        if (bi >= 2) {
            CUCHK(cudaEventSynchronize(G.ev_copied[buf]));
            double tw0 = now_s();
            if (fwrite(G.h_out + chunk_off[bi - 2], 1, chunk_len[bi - 2], fo)
                    != chunk_len[bi - 2]) {
                snprintf(err, errsz, "short write to %s", out_path);
                fclose(fo); free(evK0); free(chunk_off); free(chunk_len); return 1;
            }
            st->fwrite_ms += (now_s() - tw0) * 1e3;
        }

        uint64_t s = 0, k = 0;
        for (uint32_t i = 0; i < bn; i++) {
            G.h_pcm_off[i] = s; s += cap[i0 + i];
            G.h_dst_off[i] = (mode16 ? (uint64_t)carry : 0) + k; k += c->keep[i0 + i];
        }
        CUCHK(cudaMemcpyAsync(G.d_pcm_off, G.h_pcm_off, 8ull * bn,
                              cudaMemcpyHostToDevice, G.s_comp));
        CUCHK(cudaMemcpyAsync(G.d_dst_off, G.h_dst_off, 8ull * bn,
                              cudaMemcpyHostToDevice, G.s_comp));

        unsigned n_launch = ((bn * stride + BLOCK - 1) / BLOCK) * BLOCK;
        CUCHK(cudaEventRecord(evK0[bi], G.s_comp));
        cuopus_cli_decode<<<n_launch / BLOCK, BLOCK, smem, G.s_comp>>>(
            G.d_data, G.d_off, G.d_len, G.d_first, G.d_npkts, (int)i0, (int)bn,
            G.d_states, G.d_pcm, G.d_pcm_off, G.d_res[buf]);
        CUCHK(cudaEventRecord(evK1[bi], G.s_comp));
        CUCHK(cudaGetLastError());

        void *d2h_src; size_t d2h_bytes; uint64_t out_off_bytes;
        CUCHK(cudaEventRecord(evR0[bi], G.s_comp));
        if (!mode16) {
            cuopus_compact<<<bn, 256, 0, G.s_comp>>>(G.d_pcm, G.d_pcm_off,
                G.d_trim + i0, G.d_keep + i0, G.d_dst_off, G.d_stage[buf], (int)bn);
            CUCHK(cudaGetLastError());
            d2h_src = G.d_stage[buf]; d2h_bytes = 2ull * k;
            out_off_bytes = 2ull * kept_before;
        } else {
            cuopus_compact<<<bn, 256, 0, G.s_comp>>>(G.d_pcm, G.d_pcm_off,
                G.d_trim + i0, G.d_keep + i0, G.d_dst_off, G.d_cat, (int)bn);
            CUCHK(cudaGetLastError());
            int64_t avail = carry + (int64_t)k;
            int64_t out_hi = (cat_base + avail - RS_PAD - 1) / 3 + 1;
            if (bi == nb - 1) out_hi = out16_total;   /* final: exact trim length */
            if (out_hi > out16_total) out_hi = out16_total;
            if (out_hi < out_done) out_hi = out_done;
            int64_t n_out = out_hi - out_done;
            if (n_out > 0) {
                unsigned rgrid = (unsigned)((n_out + 255) / 256);
                cuopus_resample48to16<<<rgrid, 256, 0, G.s_comp>>>(
                    G.d_cat, cat_base, avail, out_done, n_out, G.d_out16[buf]);
                CUCHK(cudaGetLastError());
            }
            /* carry the resample halo into the next batch's window */
            int64_t new_base = 3 * out_hi - RS_PAD;
            if (new_base < cat_base) new_base = cat_base;
            int64_t new_carry = cat_base + avail - new_base;
            if (new_carry > CARRY_MAX) new_carry = CARRY_MAX;   /* only reachable on the final batch */
            if (bi + 1 < nb && new_carry > 0) {
                CUCHK(cudaMemcpyAsync(G.d_carry, G.d_cat + (new_base - cat_base),
                                      2ull * new_carry, cudaMemcpyDeviceToDevice, G.s_comp));
                CUCHK(cudaMemcpyAsync(G.d_cat, G.d_carry, 2ull * new_carry,
                                      cudaMemcpyDeviceToDevice, G.s_comp));
            }
            d2h_src = G.d_out16[buf]; d2h_bytes = 4ull * (uint64_t)n_out;
            out_off_bytes = 4ull * (uint64_t)out_done;
            cat_base = new_base; carry = new_carry; out_done = out_hi;
        }
        CUCHK(cudaEventRecord(evR1[bi], G.s_comp));
        CUCHK(cudaEventRecord(G.ev_ready[buf], G.s_comp));

        /* copy stream: D2H of this batch's output + results, overlapping the
         * NEXT batch's decode kernel on the compute stream */
        CUCHK(cudaStreamWaitEvent(G.s_copy, G.ev_ready[buf], 0));
        CUCHK(cudaEventRecord(evC0[bi], G.s_copy));
        if (d2h_bytes)
            CUCHK(cudaMemcpyAsync(G.h_out + out_off_bytes, d2h_src, d2h_bytes,
                                  cudaMemcpyDeviceToHost, G.s_copy));
        CUCHK(cudaMemcpyAsync(G.h_res_all + i0, G.d_res[buf],
                              sizeof(cu_cli_result) * bn, cudaMemcpyDeviceToHost, G.s_copy));
        CUCHK(cudaEventRecord(evC1[bi], G.s_copy));
        CUCHK(cudaEventRecord(G.ev_copied[buf], G.s_copy));

        chunk_off[bi] = out_off_bytes;
        chunk_len[bi] = d2h_bytes;
        kept_before += k;
    }

    CUCHK(cudaStreamSynchronize(G.s_comp));
    CUCHK(cudaStreamSynchronize(G.s_copy));

    /* validate every chain BEFORE trusting the output */
    int bad = 0;
    for (uint32_t ci = 0; ci < n && !bad; ci++) {
        if (G.h_res_all[ci].ovf) {
            snprintf(err, errsz, "chain %u: arena overflow", ci); bad = 1;
        } else if (G.h_res_all[ci].stop_reason != 0) {
            snprintf(err, errsz, "chain %u: device decode error %d",
                     ci, G.h_res_all[ci].stop_reason); bad = 1;
        } else if (G.h_res_all[ci].samples != cap[ci]) {
            snprintf(err, errsz, "chain %u: decoded %u samples, expected %llu",
                     ci, G.h_res_all[ci].samples, (unsigned long long)cap[ci]);
            bad = 1;
        }
    }
    if (!bad) {
        /* flush remaining chunks (the loop wrote up to bi-2) */
        double tw0 = now_s();
        for (uint32_t bwr = nb >= 2 ? nb - 2 : 0; bwr < nb && !bad; bwr++) {
            if (fwrite(G.h_out + chunk_off[bwr], 1, chunk_len[bwr], fo) != chunk_len[bwr]) {
                snprintf(err, errsz, "short write to %s", out_path); bad = 1;
            }
        }
        st->fwrite_ms += (now_s() - tw0) * 1e3;
    }

    for (uint32_t i = 0; i < nb; i++) {
        float ms;
        CUCHK(cudaEventElapsedTime(&ms, evK0[i], evK1[i])); st->kernel_ms += ms;
        CUCHK(cudaEventElapsedTime(&ms, evR0[i], evR1[i])); st->resample_ms += ms;
        CUCHK(cudaEventElapsedTime(&ms, evC0[i], evC1[i])); st->d2h_ms += ms;
    }
    for (uint32_t i = 0; i < nb * 6; i++) CUCHK(cudaEventDestroy(evK0[i]));
    free(evK0); free(chunk_off); free(chunk_len);
    fclose(fo);
    if (bad) return 1;
    st->out_samples = mode16 ? (uint64_t)out16_total : (uint64_t)play48;
    return 0;
}

static unsigned env_max_batch(void)
{
    unsigned max_batch = CUOPUS_MAX_BATCH;
    const char *mb = getenv("CUOPUS_MAX_BATCH");
    if (mb && atoi(mb) > 0) max_batch = (unsigned)atoi(mb);
    return max_batch;
}

/* One-shot: demux an OGG file and decode. Returns process exit code. */
static int run_ogg_once(const char *ogg, const char *out, double seg_s,
                        double preroll_ms, unsigned stride, int mode16,
                        const char *dump_blob, const char *dump_chains)
{
    char err[512];
    double t0 = now_s();
    cu_demux_t d;
    if (!cu_demux(ogg, &d, err, sizeof(err))) {
        fprintf(stderr, "cuopus_decode: demux: %s\n", err); return 1;
    }
    if (d.channels != 1) {
        fprintf(stderr, "cuopus_decode: only mono supported (channels=%d)\n", d.channels);
        cu_demux_free(&d); return 1;
    }
    if (d.granule_match != d.granule_checked) {
        fprintf(stderr, "cuopus_decode: granule cross-check failed (%u/%u)\n",
                d.granule_match, d.granule_checked);
        cu_demux_free(&d); return 1;
    }
    cu_chains2_t c;
    if (!cu_build_chains2(&d, seg_s, preroll_ms, &c, err, sizeof(err))) {
        fprintf(stderr, "cuopus_decode: chains: %s\n", err);
        cu_demux_free(&d); return 1;
    }
    double t_demux = now_s() - t0;
    if (dump_blob && !cu_dump_blob(&d, dump_blob)) {
        fprintf(stderr, "cuopus_decode: cannot write %s\n", dump_blob); return 1;
    }
    if (dump_chains && !cu_dump_chains2(&c, dump_chains)) {
        fprintf(stderr, "cuopus_decode: cannot write %s\n", dump_chains); return 1;
    }
    if (!out || !strcmp(out, "-")) {   /* demux-only (validation harness) */
        printf("cuopus_decode: demux %s: %u packets, %u chains, %lld playable samples, %.1f ms\n",
               ogg, d.blob.n_packets, c.n_chains, (long long)d.pcm_samples_48k, t_demux * 1e3);
        cu_chains2_free(&c); cu_demux_free(&d);
        return 0;
    }
    struct run_stats rs;
    int rc = cuopus_run(&d.blob, &c, out, stride, env_max_batch(), mode16,
                        &rs, err, sizeof(err));
    double t_all = now_s() - t0;
    if (rc) fprintf(stderr, "cuopus_decode: %s\n", err);
    else printf("cuopus_decode: %u chains (stride %u), %u packets, %llu samples out (%s)\n"
                "cuopus_decode: demux %.1f ms, h2d %.1f, kernel %.1f, resample %.1f, "
                "d2h %.1f, fwrite %.1f, wall %.1f ms\n",
                c.n_chains, stride, d.blob.n_packets,
                (unsigned long long)rs.out_samples,
                mode16 ? "float32 @16k" : "int16 @48k",
                t_demux * 1e3, rs.h2d_ms, rs.kernel_ms, rs.resample_ms,
                rs.d2h_ms, rs.fwrite_ms, t_all * 1e3);
    cu_chains2_free(&c); cu_demux_free(&d);
    return rc;
}

/* ---- daemon mode ---- */

static void serve_reply(int fd, const char *msg)
{
    size_t len = strlen(msg);
    ssize_t w = write(fd, msg, len);
    (void)w;
}

static int run_serve(const char *sock_path)
{
    signal(SIGPIPE, SIG_IGN);
    int idle_s = 600;
    const char *e = getenv("CUOPUS_SERVE_IDLE_S");
    if (e && atoi(e) > 0) idle_s = atoi(e);

    unlink(sock_path);
    int srv = socket(AF_UNIX, SOCK_STREAM, 0);
    if (srv < 0) { perror("socket"); return 2; }
    struct sockaddr_un addr; memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, sock_path, sizeof(addr.sun_path) - 1);
    if (bind(srv, (struct sockaddr *)&addr, sizeof(addr)) < 0) { perror("bind"); return 2; }
    if (listen(srv, 4) < 0) { perror("listen"); return 2; }

    double t0 = now_s();
    gpu_warmup();
    fprintf(stderr, "cuopus_decode: serving on %s (warmup %.0f ms, idle exit %ds)\n",
            sock_path, (now_s() - t0) * 1e3, idle_s);

    unsigned max_batch = env_max_batch();
    char line[8192];
    for (;;) {
        struct pollfd pfd = { srv, POLLIN, 0 };
        int pr = poll(&pfd, 1, idle_s * 1000);
        if (pr == 0) {
            fprintf(stderr, "cuopus_decode: idle %ds, exiting to free VRAM\n", idle_s);
            unlink(sock_path);
            return 0;
        }
        if (pr < 0) { if (errno == EINTR) continue; perror("poll"); return 2; }
        int fd = accept(srv, NULL, NULL);
        if (fd < 0) continue;
        /* read one line */
        size_t got = 0;
        while (got < sizeof(line) - 1) {
            ssize_t r = read(fd, line + got, sizeof(line) - 1 - got);
            if (r <= 0) break;
            got += (size_t)r;
            if (memchr(line, '\n', got)) break;
        }
        line[got] = 0;
        char *nl = strchr(line, '\n');
        if (nl) *nl = 0;

        int mode16 = -1;
        const char *args = NULL;
        if (!strncmp(line, "DECODE16F ", 10)) { mode16 = 1; args = line + 10; }
        else if (!strncmp(line, "DECODE ", 7)) { mode16 = 0; args = line + 7; }

        if (!strncmp(line, "PING", 4)) {
            serve_reply(fd, "PONG 2\n");
        } else if (!strncmp(line, "QUIT", 4)) {
            serve_reply(fd, "BYE\n");
            close(fd); unlink(sock_path);
            return 0;
        } else if (mode16 >= 0) {
            double seg_s = 0, preroll_ms = 0;
            unsigned stride = DEF_STRIDE;
            char ogg[4096], out[4096];
            if (sscanf(args, "%lf %lf %u %4095s %4095s",
                       &seg_s, &preroll_ms, &stride, ogg, out) != 5) {
                serve_reply(fd, "ERR bad request\n");
            } else {
                char err[512];
                double rt0 = now_s();
                cu_demux_t d; cu_chains2_t c; c.n_chains = 0;
                int ok = cu_demux(ogg, &d, err, sizeof(err));
                double demux_ms = (now_s() - rt0) * 1e3;
                if (ok && d.channels != 1) {
                    snprintf(err, sizeof(err), "only mono supported (channels=%d)", d.channels);
                    ok = 0;
                }
                if (ok && d.granule_match != d.granule_checked) {
                    snprintf(err, sizeof(err), "granule cross-check failed (%u/%u)",
                             d.granule_match, d.granule_checked);
                    ok = 0;
                }
                if (ok) ok = cu_build_chains2(&d, seg_s, preroll_ms, &c, err, sizeof(err));
                struct run_stats rs; memset(&rs, 0, sizeof(rs));
                if (ok) ok = cuopus_run(&d.blob, &c, out, stride, max_batch, mode16,
                                        &rs, err, sizeof(err)) == 0;
                double wall_ms = (now_s() - rt0) * 1e3;
                if (ok) {
                    char rep[320];
                    snprintf(rep, sizeof(rep),
                             "OK %llu %lld %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n",
                             (unsigned long long)rs.out_samples,
                             (long long)(d.data ? d.pcm_samples_48k : 0),
                             rs.kernel_ms, wall_ms, demux_ms, rs.h2d_ms,
                             rs.d2h_ms, rs.resample_ms, rs.fwrite_ms);
                    serve_reply(fd, rep);
                } else {
                    char rep[600];
                    snprintf(rep, sizeof(rep), "ERR %s\n", err);
                    serve_reply(fd, rep);
                }
                if (c.n_chains) cu_chains2_free(&c);
                if (d.data) cu_demux_free(&d);
            }
        } else {
            serve_reply(fd, "ERR unknown command\n");
        }
        close(fd);
    }
}

int main(int argc, char **argv)
{
    if (argc >= 3 && !strcmp(argv[1], "--serve"))
        return run_serve(argv[2]);

    if (argc >= 3 && !strcmp(argv[1], "--ogg")) {
        const char *ogg = argv[2];
        const char *out = NULL, *dump_blob = NULL, *dump_chains = NULL;
        double seg_s = 1.0, preroll_ms = DEF_PREROLL_MS;
        unsigned stride = DEF_STRIDE;
        int mode16 = 0;
        int ai = 3;
        if (ai < argc && (argv[ai][0] != '-' || !strcmp(argv[ai], "-"))) out = argv[ai++];
        for (; ai < argc; ai++) {
            if (!strcmp(argv[ai], "--seg") && ai + 1 < argc) seg_s = atof(argv[++ai]);
            else if (!strcmp(argv[ai], "--preroll") && ai + 1 < argc) preroll_ms = atof(argv[++ai]);
            else if (!strcmp(argv[ai], "--stride") && ai + 1 < argc) stride = (unsigned)atoi(argv[++ai]);
            else if (!strcmp(argv[ai], "--f16k")) mode16 = 1;
            else if (!strcmp(argv[ai], "--dump-blob") && ai + 1 < argc) dump_blob = argv[++ai];
            else if (!strcmp(argv[ai], "--dump-chains") && ai + 1 < argc) dump_chains = argv[++ai];
            else { fprintf(stderr, "cuopus_decode: unknown option %s\n", argv[ai]); return 2; }
        }
        if (stride < 1) stride = 1;
        return run_ogg_once(ogg, out, seg_s, preroll_ms, stride, mode16, dump_blob, dump_chains);
    }

    if (argc < 4) {
        fprintf(stderr,
            "usage: %s <packets.blob> <chains.bin> <out.pcm> [stride]\n"
            "       %s --ogg <in.ogg> [out.pcm|-] [--seg S] [--preroll MS] [--stride N] [--f16k]\n"
            "                 [--dump-blob F] [--dump-chains F]\n"
            "       %s --serve <unix_socket>\n", argv[0], argv[0], argv[0]);
        return 2;
    }
    /* legacy blob mode */
    unsigned stride = argc > 4 ? (unsigned)atoi(argv[4]) : DEF_STRIDE;
    if (stride < 1) stride = 1;
    double t0 = now_s();
    cu_blob b = cu_load_blob(argv[1]);
    cu_chains2_t c = cu_load_chains2_file(argv[2]);
    char err[512];
    struct run_stats rs;
    int rc = cuopus_run(&b, &c, argv[3], stride, env_max_batch(), 0,
                        &rs, err, sizeof(err));
    double t_all = now_s() - t0;
    if (rc) { fprintf(stderr, "cuopus_decode: %s\n", err); return 1; }
    size_t free1, total1;
    CUCHK(cudaMemGetInfo(&free1, &total1));
    printf("cuopus_decode: %u chains (stride %u), %u packets, %llu samples out (%.3f s @48k)\n",
           c.n_chains, stride, b.n_packets, (unsigned long long)rs.out_samples,
           rs.out_samples / 48000.0);
    printf("cuopus_decode: kernel %.1f ms, wall %.1f ms, vram %zu MB free\n",
           rs.kernel_ms, t_all * 1e3, free1 >> 20);
    return 0;
}
