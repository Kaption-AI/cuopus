/* cuopus phase 2: same-source host validation of the DEVICE-macro port.
 *
 * Compiles the IDENTICAL ported single TU (device/gen/** via
 * cuopus_silk_all.cuh + cuopus_opus.cuh) as plain C++ with the DEVICE macros
 * empty (-DCUOPUS_HOSTPORT, no nvcc), runs the same chain protocol as the
 * CUDA kernel, and compares against the FIXED_POINT libopus reference dump.
 *
 * This validates that the mechanical port (DEVFN prefixes, arena-free host
 * path, SILK-only opus_decode specialization, no-recursion PLC split) is
 * bit-exact libopus — separating "the port is right" from "the CUDA
 * execution is right" (the latter needs harness2 on the GPU).
 *
 *   g++ -O2 -DCUOPUS_HOSTPORT -DFIXED_POINT=1 -DOPUS_BUILD -DUSE_ALLOCA \
 *       -DHAVE_ALLOCA_H -I... harness_host.cpp -o out/cuopus_host
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cuopus_device.h"
#include "device/cuopus_silk_all.cuh"
#include "device/cuopus_celt_all.cuh"   /* phase 3: CELT decode closure */
#include "device/cuopus_opus.cuh"
#include "cuopus_io.h"

#define MAX_PKT_SAMPLES 5760

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* mkskip: mark every packet that needs CELT — non-SILK TOC modes, plus SILK
 * packets whose bitstream carries transition redundancy (detected by running
 * the ported decoder over the whole stream; the redundancy flag is intrinsic
 * to the packet bits, RFC 6716 sec 4.4/opus_decoder.c redundancy check). */
static int run_mkskip(cu_blob *b, const char *out)
{
    cu_opus_decoder *st = (cu_opus_decoder *)malloc(sizeof(cu_opus_decoder));
    int16_t *buf = (int16_t *)malloc(2 * MAX_PKT_SAMPLES);
    unsigned char *mask = (unsigned char *)calloc(b->n_packets, 1);
    uint32_t n_nonsilk = 0, n_redund = 0;
    cu_opus_decoder_init(st, 48000, 1);
    for (uint32_t p = 0; p < b->n_packets; p++) {
        const unsigned char *pkt = b->data + b->off[p];
        int L = (int)b->len[p];
        if (!cu_is_silk(pkt, L)) { mask[p] = 1; n_nonsilk++; continue; }
        int ret = cu_opus_decode(st, pkt, L, buf, MAX_PKT_SAMPLES);
        if (ret == CUOPUS_ERR_REDUNDANCY) { mask[p] = 1; n_redund++; }
        else if (ret < 0) { fprintf(stderr, "mkskip: pkt %u unexpected err %d\n", p, ret); free(mask); return 1; }
    }
    cu_write_mask(out, mask, b->n_packets);
    printf("mkskip: %u packets, %u non-SILK, %u SILK-with-CELT-redundancy -> %s\n",
           b->n_packets, n_nonsilk, n_redund, out);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc >= 4 && !strcmp(argv[1], "mkskip")) {
        cu_blob b = cu_load_blob(argv[2]);
        return run_mkskip(&b, argv[3]);
    }
    if (argc < 5) {
        fprintf(stderr, "usage: %s <packets.blob> <chains.bin> <ref.dump> <skip.mask>\n"
                        "       %s mkskip <packets.blob> <out.mask>\n", argv[0], argv[0]);
        return 2;
    }
    cu_blob b = cu_load_blob(argv[1]);
    cu_chains c = cu_load_chains(argv[2]);
    cu_refdump ref = cu_load_refdump(argv[3]);
    if (ref.n_chains != c.n_chains) { fprintf(stderr, "chain count mismatch\n"); return 1; }
    unsigned char *mask = cu_load_mask(argv[4], b.n_packets);
    int skip_nonsilk = ref.protocol == 1;
    uint32_t n = c.n_chains;

    cu_opus_decoder *st = (cu_opus_decoder *)malloc(sizeof(cu_opus_decoder));
    int16_t *pcm = (int16_t *)malloc(2ull * ref.total_samples);
    uint32_t *nsamp = (uint32_t *)malloc(4ull * ref.total_slots);
    uint32_t *range = (uint32_t *)malloc(4ull * ref.total_slots);
    memset(nsamp, 0xFE, 4ull * ref.total_slots);

    unsigned long long pkts_total = 0, pkts_silk_total = 0, pkts_compared = 0;
    unsigned long long samples_compared = 0, sample_mismatch = 0;
    unsigned long long nsamp_mismatch = 0, range_mismatch = 0, skipped = 0;
    unsigned chains_full = 0, chains_stopped_celt = 0, chains_err = 0;

    double t0 = now_s();
    for (uint32_t i = 0; i < n; i++) {
        cu_opus_decoder_init(st, 48000, 1);
        uint64_t s0 = ref.chain[i].pkt_slot_off;
        int16_t *out = pcm + ref.chain[i].pcm_off;
        unsigned samples = 0;
        int stop = 0;
        uint32_t k = 0;
        for (; k < c.npkts[i]; k++) {
            uint32_t p = c.first[i] + k;
            const unsigned char *pkt = b.data + b.off[p];
            int L = (int)b.len[p];
            if (cu_is_silk(pkt, L)) pkts_silk_total++;
            if (skip_nonsilk && mask[p]) {
                nsamp[s0 + k] = CUOPUS_SKIPPED; range[s0 + k] = 0;
                continue;
            }
            int ret = cu_opus_decode(st, pkt, L, out + samples, MAX_PKT_SAMPLES);
            if (ret < 0) { stop = ret; break; }
            nsamp[s0 + k] = (uint32_t)ret;
            range[s0 + k] = st->rangeFinal;
            samples += (unsigned)ret;
        }
        pkts_total += c.npkts[i];
        if (stop == CUOPUS_ERR_CELT || stop == CUOPUS_ERR_REDUNDANCY || stop == CUOPUS_ERR_TRANSITION) chains_stopped_celt++;
        else if (stop != 0) { chains_err++; fprintf(stderr, "chain %u: port error %d at pkt %u\n", i, stop, k); }
        else chains_full++;
        uint64_t port_samples = 0;
        for (uint32_t kk = 0; kk < k; kk++) {
            uint32_t dn = nsamp[s0 + kk], rn = ref.pkt_nsamp[s0 + kk];
            if (dn == CUOPUS_SKIPPED && rn == CUOPUS_SKIPPED) { skipped++; continue; }
            pkts_compared++;
            if (dn != rn) { nsamp_mismatch++; continue; }
            if (range[s0 + kk] != ref.pkt_range[s0 + kk]) range_mismatch++;
            port_samples += dn;
        }
        uint64_t off0 = ref.chain[i].pcm_off;
        uint64_t cmp = (port_samples < samples) ? port_samples : samples;
        for (uint64_t s = 0; s < cmp; s++)
            if (pcm[off0 + s] != ref.pcm[off0 + s]) sample_mismatch++;
        samples_compared += cmp;
    }
    double dt = now_s() - t0;

    printf("=== cuopus phase 2/3: HOST same-source port gate (protocol %s) ===\n",
           skip_nonsilk ? "B skip-nonSILK" : "A true-stream");
    printf("chains:            %u total | %u full | %u stopped-at-CELT/redundancy | %u port-error\n",
           n, chains_full, chains_stopped_celt, chains_err);
    printf("packets:           %llu total (%llu SILK) | %llu decoded+compared | %llu skipped non-SILK\n",
           pkts_total, pkts_silk_total, pkts_compared, skipped);
    printf("per-packet fields: %llu nsamp mismatches | %llu final-range mismatches\n",
           nsamp_mismatch, range_mismatch);
    printf("PCM:               %llu samples compared | %llu mismatches\n",
           samples_compared, sample_mismatch);
    printf("packet coverage:   %.2f%% of ALL packets (phase-3 full-stream gate)\n",
           pkts_total ? 100.0 * (double)pkts_compared / (double)pkts_total : 0.0);
    printf("wall (1 core, ported source): %.1f ms\n", dt * 1e3);
    int pass = (nsamp_mismatch == 0 && range_mismatch == 0 && sample_mismatch == 0 && chains_err == 0);
    printf("HOST-PORT GATE: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
