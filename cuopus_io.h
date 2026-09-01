/* cuopus phase 2: shared file formats between the host reference harness
 * (ref/ref_decode.c, real libopus FIXED_POINT) and the CUDA harness
 * (harness2.cu). Plain C, little-endian, mmap-free.
 *
 *   packet blob  'CUOP' (from ogg_index.py extract):
 *       u32 magic, u32 n_packets, u32 len[n], bytes...
 *   chain table  'CUCH' (from ogg_index.py chains):
 *       u32 magic, u32 n_chains, { u32 first_pkt, u32 n_pkts }[n]
 *   ref dump     'CUR2' (written by ref_decode dump, read by harness2):
 *       header below, then u32 pkt_nsamp[slots], u32 pkt_range[slots],
 *       i16 pcm[total_samples]
 *       pkt_nsamp = 0xFFFFFFFFu marks a packet skipped by protocol B.
 */
#ifndef CUOPUS_IO_H
#define CUOPUS_IO_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUOPUS_SKIPPED 0xFFFFFFFFu

typedef struct {
    uint32_t n_packets;
    uint32_t *len;
    uint32_t *off;          /* byte offset of each packet within data */
    unsigned char *data;    /* concatenated packet bytes */
    uint64_t data_bytes;
} cu_blob;

typedef struct {
    uint32_t n_chains;
    uint32_t *first;
    uint32_t *npkts;
} cu_chains;

typedef struct {
    uint32_t n_pkts;
    uint32_t n_samples;
    uint64_t pkt_slot_off;
    uint64_t pcm_off;
} cu_refchain;

typedef struct {
    uint32_t protocol;      /* 0 = A (full stream), 1 = B (skip non-SILK) */
    uint32_t n_chains;
    uint64_t total_slots;
    uint64_t total_samples;
    cu_refchain *chain;
    uint32_t *pkt_nsamp;
    uint32_t *pkt_range;
    int16_t *pcm;
} cu_refdump;

static int cu_is_silk(const unsigned char *pkt, uint32_t len)
{
    return len > 0 && (pkt[0] >> 3) < 12;   /* configs 0-11 = SILK-only */
}

static cu_blob cu_load_blob(const char *path)
{
    cu_blob b; memset(&b, 0, sizeof(b));
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "CUOP", 4)) { fprintf(stderr, "%s: bad magic\n", path); exit(1); }
    if (fread(&b.n_packets, 4, 1, f) != 1) exit(1);
    b.len = (uint32_t *)malloc(4ull * b.n_packets);
    b.off = (uint32_t *)malloc(4ull * b.n_packets);
    if (fread(b.len, 4, b.n_packets, f) != b.n_packets) exit(1);
    uint64_t tot = 0;
    for (uint32_t i = 0; i < b.n_packets; i++) { b.off[i] = (uint32_t)tot; tot += b.len[i]; }
    b.data = (unsigned char *)malloc(tot ? tot : 1);
    b.data_bytes = tot;
    if (fread(b.data, 1, tot, f) != tot) exit(1);
    fclose(f);
    return b;
}

static cu_chains cu_load_chains(const char *path)
{
    cu_chains c; memset(&c, 0, sizeof(c));
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "CUCH", 4)) { fprintf(stderr, "%s: bad magic\n", path); exit(1); }
    if (fread(&c.n_chains, 4, 1, f) != 1) exit(1);
    c.first = (uint32_t *)malloc(4ull * c.n_chains);
    c.npkts = (uint32_t *)malloc(4ull * c.n_chains);
    for (uint32_t i = 0; i < c.n_chains; i++) {
        if (fread(&c.first[i], 4, 1, f) != 1 || fread(&c.npkts[i], 4, 1, f) != 1) exit(1);
    }
    fclose(f);
    return c;
}

static void cu_write_refdump(const char *path, const cu_refdump *d)
{
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); exit(1); }
    fwrite("CUR2", 1, 4, f);
    fwrite(&d->protocol, 4, 1, f);
    fwrite(&d->n_chains, 4, 1, f);
    uint32_t pad = 0; fwrite(&pad, 4, 1, f);
    fwrite(&d->total_slots, 8, 1, f);
    fwrite(&d->total_samples, 8, 1, f);
    fwrite(d->chain, sizeof(cu_refchain), d->n_chains, f);
    fwrite(d->pkt_nsamp, 4, d->total_slots, f);
    fwrite(d->pkt_range, 4, d->total_slots, f);
    fwrite(d->pcm, 2, d->total_samples, f);
    fclose(f);
}

static cu_refdump cu_load_refdump(const char *path)
{
    cu_refdump d; memset(&d, 0, sizeof(d));
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "CUR2", 4)) { fprintf(stderr, "%s: bad magic\n", path); exit(1); }
    uint32_t pad;
    if (fread(&d.protocol, 4, 1, f) != 1) exit(1);
    if (fread(&d.n_chains, 4, 1, f) != 1) exit(1);
    if (fread(&pad, 4, 1, f) != 1) exit(1);
    if (fread(&d.total_slots, 8, 1, f) != 1) exit(1);
    if (fread(&d.total_samples, 8, 1, f) != 1) exit(1);
    d.chain = (cu_refchain *)malloc(sizeof(cu_refchain) * d.n_chains);
    d.pkt_nsamp = (uint32_t *)malloc(4 * d.total_slots);
    d.pkt_range = (uint32_t *)malloc(4 * d.total_slots);
    d.pcm = (int16_t *)malloc(2 * d.total_samples);
    if (fread(d.chain, sizeof(cu_refchain), d.n_chains, f) != d.n_chains) exit(1);
    if (fread(d.pkt_nsamp, 4, d.total_slots, f) != d.total_slots) exit(1);
    if (fread(d.pkt_range, 4, d.total_slots, f) != d.total_slots) exit(1);
    if (fread(d.pcm, 2, d.total_samples, f) != d.total_samples) exit(1);
    fclose(f);
    return d;
}

/* skip mask 'CUSK': u32 magic, u32 n_packets, u8 flags[n] (1 = packet needs
 * CELT — CELT/hybrid mode or SILK transition-redundancy — skip in protocol B) */
static unsigned char *cu_load_mask(const char *path, uint32_t expect_n)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    char magic[4]; uint32_t n;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "CUSK", 4)) { fprintf(stderr, "%s: bad magic\n", path); exit(1); }
    if (fread(&n, 4, 1, f) != 1 || n != expect_n) { fprintf(stderr, "%s: packet count mismatch\n", path); exit(1); }
    unsigned char *m = (unsigned char *)malloc(n);
    if (fread(m, 1, n, f) != n) exit(1);
    fclose(f);
    return m;
}

static void cu_write_mask(const char *path, const unsigned char *m, uint32_t n)
{
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); exit(1); }
    fwrite("CUSK", 1, 4, f);
    fwrite(&n, 4, 1, f);
    fwrite(m, 1, n, f);
    fclose(f);
}

#endif /* CUOPUS_IO_H */
