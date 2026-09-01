/* cuopus phase 2: CUDA harness — device SILK decode of segment chains,
 * bit-exactness gate vs the FIXED_POINT libopus reference dump, and
 * benchmark.
 *
 * One CUDA thread = one segment chain: fresh cu_opus_decoder, packets decoded
 * sequentially, states in global memory, scratch in a per-thread global-memory
 * arena (cuopus_stack.h). No intra-frame parallelism.
 *
 * Usage:
 *   cuopus_phase2 <packets.blob> <chains.bin> <ref.dump> [reps]
 *
 * The protocol (A = stop chain at first non-SILK packet, B = skip non-SILK
 * packets on both sides) is taken from the ref dump header, so the device
 * always mirrors exactly what the reference did.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "cuopus_device.h"
#include "device/cuopus_silk_all.cuh"
#include "device/cuopus_celt_all.cuh"   /* phase 3: CELT decode closure */
#include "device/cuopus_opus.cuh"
#include "cuopus_io.h"

#define CUCHK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, \
            cudaGetErrorString(_e)); exit(1); } } while (0)

#define ARENA_BYTES   (96u * 1024u)   /* per-thread scratch (SILK peak ~8 KB; CELT
                                         quant_all_bands + synthesis + PLC adds ~25 KB
                                         nested worst-case; overflow is flagged) */
#define BLOCK         64
#define MAX_PKT_SAMPLES 5760

/* arena globals declared extern in cuopus_stack.h */
__device__ char        *cuopus_arena_base;
__device__ unsigned long cuopus_arena_per_thread;
__device__ char       **cuopus_arena_tops;
__device__ int         *cuopus_arena_ovf;
__device__ unsigned long cuopus_arena_stride;

struct cu_chain_result {
    int slots_done;      /* packets consumed (decoded or skipped) */
    int stop_reason;     /* 0 = ran to end, else CUOPUS_ERR_* / OPUS_* */
    unsigned samples;    /* PCM samples written for this chain */
    int ovf;             /* arena overflow flag (voids the chain) */
};

__global__ void cuopus_decode_chains(
    const unsigned char *__restrict__ data,
    const unsigned *__restrict__ off, const unsigned *__restrict__ len,
    const unsigned *__restrict__ first, const unsigned *__restrict__ npkts,
    int n_chains, cu_opus_decoder *states, int16_t *pcm,
    const unsigned long long *__restrict__ chain_pcm_off,
    unsigned *pkt_nsamp, unsigned *pkt_range,
    const unsigned long long *__restrict__ chain_slot_off,
    cu_chain_result *res, int skip_nonsilk, const unsigned char *__restrict__ skip_mask)
{
    unsigned gt = blockIdx.x * blockDim.x + threadIdx.x;
    if (gt % (unsigned)cuopus_arena_stride != 0) return;   /* one decoder per slot */
    int t = (int)(gt / (unsigned)cuopus_arena_stride);     /* chain index */
    if (t >= n_chains) return;
    cuopus_stack_reset();
    cu_opus_decoder *st = &states[t];
    cu_opus_decoder_init(st, 48000, 1);
    unsigned long long slot0 = chain_slot_off[t];
    int16_t *out = pcm + chain_pcm_off[t];
    unsigned samples = 0;
    int stop = 0, k = 0, n = (int)npkts[t];
    for (; k < n; k++) {
        unsigned p = first[t] + (unsigned)k;
        const unsigned char *pkt = data + off[p];
        int L = (int)len[p];
        if (skip_nonsilk && skip_mask[p]) {
            pkt_nsamp[slot0 + k] = 0xFFFFFFFFu;
            pkt_range[slot0 + k] = 0;
            continue;
        }
        int ret = cu_opus_decode(st, pkt, L, out + samples, MAX_PKT_SAMPLES);
        if (ret < 0) { stop = ret; break; }
        pkt_nsamp[slot0 + k] = (unsigned)ret;
        pkt_range[slot0 + k] = st->rangeFinal;
        samples += (unsigned)ret;
    }
    res[t].slots_done = k;
    res[t].stop_reason = stop;
    res[t].samples = samples;
    res[t].ovf = cuopus_arena_ovf[cuopus_tid()];
}

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

int main(int argc, char **argv)
{
    if (argc < 5) {
        fprintf(stderr, "usage: %s <packets.blob> <chains.bin> <ref.dump> <skip.mask> [reps]\n", argv[0]);
        return 2;
    }
    int reps = argc > 5 ? atoi(argv[5]) : 10;
    unsigned stride = argc > 6 ? (unsigned)atoi(argv[6]) : 1;  /* 1=chain/lane, 32=chain/warp */
    if (stride < 1) stride = 1;
    cu_blob b = cu_load_blob(argv[1]);
    cu_chains c = cu_load_chains(argv[2]);
    cu_refdump ref = cu_load_refdump(argv[3]);
    unsigned char *h_mask = cu_load_mask(argv[4], b.n_packets);
    if (ref.n_chains != c.n_chains) { fprintf(stderr, "chain count mismatch ref=%u chains=%u\n", ref.n_chains, c.n_chains); return 1; }
    int skip_nonsilk = ref.protocol == 1;
    uint32_t n = c.n_chains;

    size_t free0, total0, free1, total1;
    CUCHK(cudaSetDevice(0));
    /* NOTE: no cudaDeviceSetLimit(cudaLimitStackSize): the ported call graph
       is non-recursive, so the driver sizes the per-thread stack statically.
       Forcing 32 KB here reserved stack for every resident thread on the
       device (~GBs on GB202) and OOM'd next to the production models. */
    CUCHK(cudaMemGetInfo(&free0, &total0));

    /* device buffers */
    unsigned char *d_data;  CUCHK(cudaMalloc(&d_data, b.data_bytes));
    unsigned *d_off, *d_len, *d_first, *d_npkts, *d_nsamp, *d_range;
    CUCHK(cudaMalloc(&d_off, 4ull * b.n_packets));
    CUCHK(cudaMalloc(&d_len, 4ull * b.n_packets));
    CUCHK(cudaMalloc(&d_first, 4ull * n));
    CUCHK(cudaMalloc(&d_npkts, 4ull * n));
    CUCHK(cudaMalloc(&d_nsamp, 4ull * ref.total_slots));
    CUCHK(cudaMalloc(&d_range, 4ull * ref.total_slots));
    unsigned long long *d_pcm_off, *d_slot_off;
    CUCHK(cudaMalloc(&d_pcm_off, 8ull * n));
    CUCHK(cudaMalloc(&d_slot_off, 8ull * n));
    unsigned char *d_mask; CUCHK(cudaMalloc(&d_mask, b.n_packets));
    cu_opus_decoder *d_states; CUCHK(cudaMalloc(&d_states, sizeof(cu_opus_decoder) * (size_t)n));
    int16_t *d_pcm; CUCHK(cudaMalloc(&d_pcm, 2ull * ref.total_samples));
    cu_chain_result *d_res; CUCHK(cudaMalloc(&d_res, sizeof(cu_chain_result) * (size_t)n));

    unsigned n_launch = ((n * stride + BLOCK - 1) / BLOCK) * BLOCK;
    unsigned n_threads = n_launch / stride + 1;   /* decode slots (arena entries) */
    char *d_arena; CUCHK(cudaMalloc(&d_arena, (size_t)n_threads * ARENA_BYTES));
    char **d_tops; CUCHK(cudaMalloc(&d_tops, sizeof(char *) * n_threads));
    int *d_ovf;    CUCHK(cudaMalloc(&d_ovf, sizeof(int) * n_threads));
    unsigned long arena_per_thread = ARENA_BYTES;
    unsigned long stride_ul = stride;
    CUCHK(cudaMemcpyToSymbol(cuopus_arena_stride, &stride_ul, sizeof(stride_ul)));
    CUCHK(cudaMemcpyToSymbol(cuopus_arena_base, &d_arena, sizeof(d_arena)));
    CUCHK(cudaMemcpyToSymbol(cuopus_arena_per_thread, &arena_per_thread, sizeof(arena_per_thread)));
    CUCHK(cudaMemcpyToSymbol(cuopus_arena_tops, &d_tops, sizeof(d_tops)));
    CUCHK(cudaMemcpyToSymbol(cuopus_arena_ovf, &d_ovf, sizeof(d_ovf)));

    /* host-side offsets from the ref dump layout (device writes a prefix of
       the reference PCM span for each chain) */
    unsigned long long *h_pcm_off = (unsigned long long *)malloc(8ull * n);
    unsigned long long *h_slot_off = (unsigned long long *)malloc(8ull * n);
    for (uint32_t i = 0; i < n; i++) {
        h_pcm_off[i] = ref.chain[i].pcm_off;
        h_slot_off[i] = ref.chain[i].pkt_slot_off;
    }

    double t_h2d0 = now_s();
    CUCHK(cudaMemcpy(d_data, b.data, b.data_bytes, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(d_off, b.off, 4ull * b.n_packets, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(d_len, b.len, 4ull * b.n_packets, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(d_first, c.first, 4ull * n, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(d_npkts, c.npkts, 4ull * n, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(d_mask, h_mask, b.n_packets, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(d_pcm_off, h_pcm_off, 8ull * n, cudaMemcpyHostToDevice));
    CUCHK(cudaMemcpy(d_slot_off, h_slot_off, 8ull * n, cudaMemcpyHostToDevice));
    CUCHK(cudaMemset(d_nsamp, 0xFE, 4ull * ref.total_slots));
    CUCHK(cudaMemset(d_range, 0xFE, 4ull * ref.total_slots));
    CUCHK(cudaDeviceSynchronize());
    double t_h2d = now_s() - t_h2d0;

    dim3 grid(n_launch / BLOCK), block(BLOCK);
    printf("mapping: stride %u (%s), %u launched threads, %u decode slots\n",
           stride, stride >= 32 ? "chain-per-warp" : "chain-per-lane", n_launch, n_threads);
    /* correctness launch (also warm-up) */
    cuopus_decode_chains<<<grid, block>>>(d_data, d_off, d_len, d_first, d_npkts,
        (int)n, d_states, d_pcm, d_pcm_off, d_nsamp, d_range, d_slot_off, d_res, skip_nonsilk, d_mask);
    CUCHK(cudaGetLastError());
    CUCHK(cudaDeviceSynchronize());
    CUCHK(cudaMemGetInfo(&free1, &total1));

    /* pull results */
    unsigned *h_nsamp = (unsigned *)malloc(4ull * ref.total_slots);
    unsigned *h_range = (unsigned *)malloc(4ull * ref.total_slots);
    int16_t *h_pcm = (int16_t *)malloc(2ull * ref.total_samples);
    cu_chain_result *h_res = (cu_chain_result *)malloc(sizeof(cu_chain_result) * n);
    double t_d2h0 = now_s();
    CUCHK(cudaMemcpy(h_nsamp, d_nsamp, 4ull * ref.total_slots, cudaMemcpyDeviceToHost));
    CUCHK(cudaMemcpy(h_range, d_range, 4ull * ref.total_slots, cudaMemcpyDeviceToHost));
    CUCHK(cudaMemcpy(h_pcm, d_pcm, 2ull * ref.total_samples, cudaMemcpyDeviceToHost));
    CUCHK(cudaMemcpy(h_res, d_res, sizeof(cu_chain_result) * n, cudaMemcpyDeviceToHost));
    double t_d2h = now_s() - t_d2h0;

    /* ---- bit-exactness gate ---- */
    unsigned long long pkts_total = 0, pkts_silk_total = 0, pkts_compared = 0;
    unsigned long long samples_compared = 0, sample_mismatch = 0;
    unsigned long long nsamp_mismatch = 0, range_mismatch = 0, skipped = 0;
    unsigned chains_full = 0, chains_stopped_celt = 0, chains_err = 0, chains_ovf = 0;
    for (uint32_t i = 0; i < n; i++) {
        cu_chain_result *r = &h_res[i];
        uint64_t s0 = ref.chain[i].pkt_slot_off;
        pkts_total += c.npkts[i];
        for (uint32_t k = 0; k < c.npkts[i]; k++) {
            uint32_t p = c.first[i] + k;
            if (cu_is_silk(b.data + b.off[p], b.len[p])) pkts_silk_total++;
        }
        if (r->ovf) { chains_ovf++; continue; }
        if (r->stop_reason == CUOPUS_ERR_CELT || r->stop_reason == CUOPUS_ERR_REDUNDANCY
            || r->stop_reason == CUOPUS_ERR_TRANSITION) chains_stopped_celt++;
        else if (r->stop_reason != 0) {
            chains_err++;
            fprintf(stderr, "chain %u: device error %d at packet %d\n", i, r->stop_reason, r->slots_done);
        } else chains_full++;
        /* compare the processed prefix */
        uint64_t dev_samples = 0;
        for (int k = 0; k < r->slots_done; k++) {
            uint32_t dn = h_nsamp[s0 + k], rn = ref.pkt_nsamp[s0 + k];
            if (dn == CUOPUS_SKIPPED && rn == CUOPUS_SKIPPED) { skipped++; continue; }
            pkts_compared++;
            if (dn != rn) { nsamp_mismatch++; continue; }
            if (h_range[s0 + k] != ref.pkt_range[s0 + k]) range_mismatch++;
            dev_samples += dn;
        }
        /* PCM prefix compare */
        uint64_t off0 = ref.chain[i].pcm_off;
        uint64_t cmp = (dev_samples < r->samples) ? dev_samples : r->samples;
        for (uint64_t s = 0; s < cmp; s++) {
            if (h_pcm[off0 + s] != ref.pcm[off0 + s]) sample_mismatch++;
        }
        samples_compared += cmp;
    }

    printf("=== cuopus phase 2/3: bit-exactness gate (protocol %s) ===\n", skip_nonsilk ? "B skip-nonSILK" : "A true-stream");
    printf("chains:            %u total | %u full | %u stopped-at-CELT/redund | %u device-error | %u arena-ovf\n",
           n, chains_full, chains_stopped_celt, chains_err, chains_ovf);
    printf("packets:           %llu total in chains (%llu SILK) | %llu decoded+compared | %llu skipped non-SILK\n",
           pkts_total, pkts_silk_total, pkts_compared, skipped);
    printf("per-packet fields: %llu nsamp mismatches | %llu final-range mismatches\n",
           nsamp_mismatch, range_mismatch);
    printf("PCM:               %llu samples compared | %llu mismatches\n", samples_compared, sample_mismatch);
    printf("SILK coverage:     %.2f%% of SILK packets in corpus chains\n",
           pkts_silk_total ? 100.0 * (double)pkts_compared / (double)pkts_silk_total : 0.0);
    printf("packet coverage:   %.2f%% of ALL packets in corpus chains (phase-3 full-stream gate)\n",
           pkts_total ? 100.0 * (double)pkts_compared / (double)pkts_total : 0.0);
    int pass = (nsamp_mismatch == 0 && range_mismatch == 0 && sample_mismatch == 0
                && chains_err == 0 && chains_ovf == 0);
    printf("GATE: %s\n", pass ? "PASS" : "FAIL");

    /* ---- benchmark ---- */
    cudaEvent_t ev0, ev1;
    CUCHK(cudaEventCreate(&ev0)); CUCHK(cudaEventCreate(&ev1));
    float best_ms = 1e9f;
    for (int r = 0; r < reps; r++) {
        CUCHK(cudaEventRecord(ev0));
        cuopus_decode_chains<<<grid, block>>>(d_data, d_off, d_len, d_first, d_npkts,
            (int)n, d_states, d_pcm, d_pcm_off, d_nsamp, d_range, d_slot_off, d_res, skip_nonsilk, d_mask);
        CUCHK(cudaEventRecord(ev1));
        CUCHK(cudaEventSynchronize(ev1));
        float ms; CUCHK(cudaEventElapsedTime(&ms, ev0, ev1));
        if (ms < best_ms) best_ms = ms;
    }
    /* end-to-end: H2D packet blob + kernel + D2H PCM */
    double e2e_best = 1e9;
    for (int r = 0; r < 3; r++) {
        double t0 = now_s();
        CUCHK(cudaMemcpy(d_data, b.data, b.data_bytes, cudaMemcpyHostToDevice));
        cuopus_decode_chains<<<grid, block>>>(d_data, d_off, d_len, d_first, d_npkts,
            (int)n, d_states, d_pcm, d_pcm_off, d_nsamp, d_range, d_slot_off, d_res, skip_nonsilk, d_mask);
        CUCHK(cudaMemcpy(h_pcm, d_pcm, 2ull * ref.total_samples, cudaMemcpyDeviceToHost));
        CUCHK(cudaDeviceSynchronize());
        double t = now_s() - t0;
        if (t < e2e_best) e2e_best = t;
    }
    printf("=== benchmark (%u chains, %llu samples) ===\n", n, (unsigned long long)ref.total_samples);
    printf("gpu kernel only:   %.3f ms (best of %d)\n", best_ms, reps);
    printf("gpu end-to-end:    %.3f ms (H2D %.0f KB + kernel + D2H %.1f MB; first-run H2D was %.3f ms, D2H %.3f ms)\n",
           e2e_best * 1e3, b.data_bytes / 1024.0, 2.0 * ref.total_samples / 1048576.0, t_h2d * 1e3, t_d2h * 1e3);
    printf("gpu mem overhead:  %.1f MB (buffers+local pool; free %zu -> %zu MB)\n",
           (double)(free0 - free1) / 1048576.0, free0 >> 20, free1 >> 20);
    return pass ? 0 : 1;
}
