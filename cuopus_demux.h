/* cuopus perf pass: OGG-Opus demux + CUC2 chain building in C, inside the CLI.
 *
 * Port of ogg_index.py (build_index + write_blob + build_chains2). The Python
 * implementation is the reference; validation is byte-identical blob + chain
 * files on the phase-2/3 corpora, the 2 h file and the sweep files
 * (see build_and_test.sh perf).
 *
 * Everything is host-side plain C (compiled as C++ by nvcc). Errors do NOT
 * exit: they fill err[] and return 0 so the daemon can keep serving.
 */
#ifndef CUOPUS_DEMUX_H
#define CUOPUS_DEMUX_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuopus_io.h"   /* cu_blob */

typedef struct {
    uint32_t n_chains;
    uint32_t *first, *npkts, *trim, *keep;
} cu_chains2_t;

typedef struct {
    /* file */
    unsigned char *data;      /* whole file (owned) */
    uint64_t file_bytes;
    /* packets (audio only, headers stripped) */
    cu_blob blob;             /* off/len into blob.data (owned, contiguous copy) */
    uint32_t *pkt_page;       /* page index on which each audio packet completes */
    int64_t *sample_off;      /* exact cumulative 48 kHz offset per audio packet */
    int64_t toc_total;        /* sum of TOC durations */
    /* stream metadata */
    int channels;
    int preskip;
    int64_t last_granule;
    int64_t pcm_samples_48k;  /* last_granule - preskip (playable) */
    int64_t end_trim;
    uint32_t n_pages;
    uint32_t granule_checked, granule_match;
} cu_demux_t;

/* RFC 6716 sec 3.1 frame duration per TOC config in quarter-ms (== ogg_index.py) */
static const int cu_dmx_qms[32] = {
    40, 80, 160, 240,  40, 80, 160, 240,  40, 80, 160, 240,
    40, 80,  40, 80,
    10, 20, 40, 80,  10, 20, 40, 80,  10, 20, 40, 80,  10, 20, 40, 80
};

static int cu_dmx_pkt_samples48(const unsigned char *p, uint32_t len)
{
    if (!len) return 0;
    int config = p[0] >> 3, code = p[0] & 0x3, nf;
    if (code == 0) nf = 1;
    else if (code < 3) nf = 2;
    else { if (len < 2) return 0; nf = p[1] & 0x3F; }
    return cu_dmx_qms[config] * 12 * nf;
}

#define CU_DMX_ERR(...) do { snprintf(err, errsz, __VA_ARGS__); goto fail; } while (0)

static void cu_demux_free(cu_demux_t *d)
{
    free(d->data); free(d->blob.len); free(d->blob.off); free(d->blob.data);
    free(d->pkt_page); free(d->sample_off);
    memset(d, 0, sizeof(*d));
}

/* Parse an in-memory OGG-Opus stream into blob + index. TAKES OWNERSHIP of
 * `data` (a malloc'd buffer of fsz bytes): it becomes d->data on success and
 * is freed via cu_demux_free on failure. Returns 1 on success, 0 on error
 * (err filled, *d zeroed). Mirrors ogg_index.build_index(check_crc=False):
 * page CRC is skipped (granule cross-check + decoder validation remain).
 * Extracted from cu_demux (perf pass) so libcuopus can demux caller-provided
 * bytes without touching the filesystem — the parse below is byte-for-byte
 * the validated original. */
static int cu_demux_mem(unsigned char *data, uint64_t fsz, cu_demux_t *d,
                        char *err, size_t errsz)
{
    memset(d, 0, sizeof(*d));
    /* span walk state (grow-only arrays) */
    uint64_t *span_off = NULL; uint32_t *span_len = NULL; uint8_t *span_done = NULL;
    uint32_t *span_page = NULL;
    int64_t *page_granule = NULL; int64_t *page_pkt_end = NULL;
    uint32_t n_spans = 0, cap_spans = 0, n_pages = 0, cap_pages = 0;

    if (fsz == 0 || !data) { free(data); snprintf(err, errsz, "empty stream"); return 0; }
    d->data = data;
    d->file_bytes = (uint64_t)fsz;

    /* ---- parse pages into spans (== ogg_index.parse_pages) ---- */
    uint64_t pos = 0, n = (uint64_t)fsz;
    while (pos < n) {
        /* find "OggS" from pos */
        unsigned char *hit = (unsigned char *)memmem(data + pos, n - pos, "OggS", 4);
        if (!hit) break;
        uint64_t idx = (uint64_t)(hit - data);
        if (idx + 27 > n) break;
        int64_t granule; memcpy(&granule, data + idx + 6, 8);   /* LE host assumed */
        uint8_t nsegs = data[idx + 26];
        const unsigned char *seg_table = data + idx + 27;
        uint64_t body_start = idx + 27 + nsegs;
        uint32_t body_len = 0;
        for (int i = 0; i < nsegs; i++) body_len += seg_table[i];
        if (body_start + body_len > n) break;

        if (n_pages == cap_pages) {
            cap_pages = cap_pages ? cap_pages * 2 : 1024;
            page_granule = (int64_t *)realloc(page_granule, 8ull * cap_pages);
            page_pkt_end = (int64_t *)realloc(page_pkt_end, 8ull * cap_pages);
        }
        page_granule[n_pages] = granule;
        page_pkt_end[n_pages] = -1;      /* filled during packet reassembly */

        uint64_t off = body_start, run_start = off;
        uint32_t run_len = 0;
        int last_lacing = 0;
        for (int i = 0; i < nsegs; i++) {
            uint8_t lac = seg_table[i];
            run_len += lac;
            last_lacing = lac;
            if (lac < 255) {
                if (n_spans == cap_spans) {
                    cap_spans = cap_spans ? cap_spans * 2 : 4096;
                    span_off = (uint64_t *)realloc(span_off, 8ull * cap_spans);
                    span_len = (uint32_t *)realloc(span_len, 4ull * cap_spans);
                    span_done = (uint8_t *)realloc(span_done, cap_spans);
                    span_page = (uint32_t *)realloc(span_page, 4ull * cap_spans);
                }
                span_off[n_spans] = run_start; span_len[n_spans] = run_len;
                span_done[n_spans] = 1; span_page[n_spans] = n_pages;
                n_spans++;
                run_start += run_len;
                run_len = 0;
            }
        }
        if (run_len > 0 || (nsegs > 0 && last_lacing == 255)) {
            if (n_spans == cap_spans) {
                cap_spans = cap_spans ? cap_spans * 2 : 4096;
                span_off = (uint64_t *)realloc(span_off, 8ull * cap_spans);
                span_len = (uint32_t *)realloc(span_len, 4ull * cap_spans);
                span_done = (uint8_t *)realloc(span_done, cap_spans);
                span_page = (uint32_t *)realloc(span_page, 4ull * cap_spans);
            }
            span_off[n_spans] = run_start; span_len[n_spans] = run_len;
            span_done[n_spans] = 0; span_page[n_spans] = n_pages;
            n_spans++;
        }
        n_pages++;
        pos = body_start + body_len;
    }
    if (n_pages == 0) CU_DMX_ERR("no OGG pages found");
    d->n_pages = n_pages;

    /* ---- reassemble packets: pass 1 counts, pass 2 copies ---- */
    {
        uint32_t n_packets = 0;
        uint64_t cur = 0;
        for (uint32_t i = 0; i < n_spans; i++) {
            cur += span_len[i];
            if (span_done[i]) { n_packets++; cur = 0; }
        }
        /* trailing partial (cur > 0) is dropped, matching the Python WARNING path */
        if (n_packets < 2) CU_DMX_ERR("not an OGG Opus stream (fewer than 2 packets)");

        uint32_t *plen = (uint32_t *)malloc(4ull * n_packets);
        uint32_t *ppage = (uint32_t *)malloc(4ull * n_packets);
        uint32_t pi = 0;
        uint64_t acc = 0;
        for (uint32_t i = 0; i < n_spans; i++) {
            acc += span_len[i];
            if (span_done[i]) { plen[pi] = (uint32_t)acc; ppage[pi] = span_page[i]; pi++; acc = 0; }
        }

        /* header packet 0 must be OpusHead (need first span bytes) */
        if (plen[0] < 19 || memcmp(data + span_off[0], "OpusHead", 8) != 0) {
            free(plen); free(ppage);
            CU_DMX_ERR("not an OGG Opus stream (no OpusHead)");
        }
        d->channels = data[span_off[0] + 9];
        d->preskip = data[span_off[0] + 10] | (data[span_off[0] + 11] << 8);

        uint32_t n_audio = n_packets - 2;
        uint64_t audio_bytes = 0;
        for (uint32_t i = 2; i < n_packets; i++) audio_bytes += plen[i];

        d->blob.n_packets = n_audio;
        d->blob.len = (uint32_t *)malloc(4ull * (n_audio ? n_audio : 1));
        d->blob.off = (uint32_t *)malloc(4ull * (n_audio ? n_audio : 1));
        d->blob.data = (unsigned char *)malloc(audio_bytes ? audio_bytes : 1);
        d->blob.data_bytes = audio_bytes;
        d->pkt_page = (uint32_t *)malloc(4ull * (n_audio ? n_audio : 1));
        d->sample_off = (int64_t *)malloc(8ull * (n_audio ? n_audio : 1));

        /* pass 2: copy audio packet bytes contiguously */
        uint64_t w = 0;
        pi = 0;
        uint64_t pkt_start_w = 0;
        for (uint32_t i = 0; i < n_spans && pi < n_packets; i++) {
            if (pi >= 2) {
                memcpy(d->blob.data + w, data + span_off[i], span_len[i]);
                w += span_len[i];
            }
            if (span_done[i]) {
                if (pi >= 2) {
                    uint32_t ai = pi - 2;
                    d->blob.off[ai] = (uint32_t)pkt_start_w;
                    d->blob.len[ai] = plen[pi];
                    d->pkt_page[ai] = ppage[pi];
                    pkt_start_w = w;
                }
                pi++;
            }
        }
        if (w != audio_bytes) { free(plen); free(ppage); CU_DMX_ERR("internal: blob size mismatch"); }
        free(plen); free(ppage);
    }

    /* ---- per-packet 48 kHz offsets + granule cross-check ---- */
    {
        uint32_t n_audio = d->blob.n_packets;
        int64_t cum = 0;
        for (uint32_t k = 0; k < n_audio; k++) {
            d->sample_off[k] = cum;
            cum += cu_dmx_pkt_samples48(d->blob.data + d->blob.off[k], d->blob.len[k]);
        }
        d->toc_total = cum;

        int64_t final_data_page = -1;
        for (uint32_t k = 0; k < n_audio; k++) {
            uint32_t pg = d->pkt_page[k];
            page_pkt_end[pg] = d->sample_off[k]
                + cu_dmx_pkt_samples48(d->blob.data + d->blob.off[k], d->blob.len[k]);
            if ((int64_t)pg > final_data_page) final_data_page = (int64_t)pg;
        }
        int last_dur = n_audio ? cu_dmx_pkt_samples48(
            d->blob.data + d->blob.off[n_audio - 1], d->blob.len[n_audio - 1]) : 0;
        uint32_t checked = 0, match = 0;
        int64_t last_granule = 0;
        for (uint32_t pg = 0; pg < n_pages; pg++) {
            if (page_granule[pg] != -1 && page_granule[pg] > last_granule)
                last_granule = page_granule[pg];
            if (page_granule[pg] == -1 || page_pkt_end[pg] < 0) continue;
            checked++;
            if (page_granule[pg] == page_pkt_end[pg]) match++;
            else if ((int64_t)pg == final_data_page
                     && page_pkt_end[pg] - page_granule[pg] >= 0
                     && page_pkt_end[pg] - page_granule[pg] < last_dur)
                match++;   /* RFC 7845 end-trim on the final page */
        }
        d->granule_checked = checked;
        d->granule_match = match;
        d->last_granule = last_granule;
        d->end_trim = d->toc_total - last_granule;
        d->pcm_samples_48k = last_granule - d->preskip;
    }

    free(span_off); free(span_len); free(span_done); free(span_page);
    free(page_granule); free(page_pkt_end);
    return 1;

fail:
    free(span_off); free(span_len); free(span_done); free(span_page);
    free(page_granule); free(page_pkt_end);
    cu_demux_free(d);
    return 0;
}

/* Parse an OGG-Opus file into blob + index (reads the file, then the
 * in-memory parse above). Same contract as before the perf-pass split. */
static int cu_demux(const char *path, cu_demux_t *d, char *err, size_t errsz)
{
    memset(d, 0, sizeof(*d));
    FILE *f = fopen(path, "rb");
    if (!f) { snprintf(err, errsz, "cannot open %s", path); return 0; }
    fseek(f, 0, SEEK_END);
    long fsz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (fsz <= 0) { fclose(f); snprintf(err, errsz, "empty file %s", path); return 0; }
    unsigned char *data = (unsigned char *)malloc((size_t)fsz);
    if (!data || fread(data, 1, (size_t)fsz, f) != (size_t)fsz) {
        fclose(f); free(data); snprintf(err, errsz, "short read on %s", path); return 0;
    }
    fclose(f);
    return cu_demux_mem(data, (uint64_t)fsz, d, err, errsz);
}

/* Build the CUC2 chain table (== ogg_index.build_chains2): playable-sample-
 * space segmentation, preskip + RFC 7845 end-trim folded into trim/keep.
 * Returns 1 on success (caller frees c->first .. c->keep), 0 on error. */
static int cu_build_chains2(const cu_demux_t *d, double seconds, double preroll_ms,
                            cu_chains2_t *c, char *err, size_t errsz)
{
    memset(c, 0, sizeof(*c));
    int64_t step = (int64_t)(seconds * 48000);
    int64_t preroll = (int64_t)(preroll_ms * 48);
    int64_t preskip = d->preskip;
    int64_t total_play = d->pcm_samples_48k;
    uint32_t n_pk = d->blob.n_packets;
    if (total_play <= 0) { snprintf(err, errsz, "no playable samples"); return 0; }
    if (step <= 0) { snprintf(err, errsz, "bad segment length"); return 0; }

    /* prefix sums of packet durations for O(1) capacity checks */
    int64_t *cum = (int64_t *)malloc(8ull * (n_pk + 1));
    cum[0] = 0;
    for (uint32_t k = 0; k < n_pk; k++)
        cum[k + 1] = cum[k] + cu_dmx_pkt_samples48(d->blob.data + d->blob.off[k], d->blob.len[k]);

    uint32_t cap = (uint32_t)(total_play / step) + 2;
    c->first = (uint32_t *)malloc(4ull * cap);
    c->npkts = (uint32_t *)malloc(4ull * cap);
    c->trim  = (uint32_t *)malloc(4ull * cap);
    c->keep  = (uint32_t *)malloc(4ull * cap);

    const int64_t *so = d->sample_off;
    int64_t kept_total = 0;
    uint32_t nc = 0;
    for (int64_t k = 0; k * step < total_play; k++) {
        int64_t play_start = k * step;
        int64_t play_end = (k + 1) * step < total_play ? (k + 1) * step : total_play;
        int64_t toc_start = play_start + preskip;
        int64_t toc_end = play_end + preskip;
        int64_t start_target = toc_start - preroll;
        if (start_target < 0) start_target = 0;

        /* first = bisect_right(so, start_target) - 1, then the two Python fixups */
        int64_t lo = 0, hi = n_pk;
        while (lo < hi) {                 /* bisect_right */
            int64_t mid = (lo + hi) / 2;
            if (so[mid] <= start_target) lo = mid + 1; else hi = mid;
        }
        int64_t first = lo - 1;
        if (first < 0) first = 0;
        if (cum[first + 1] <= start_target) first++;     /* so[first]+dur <= target */

        /* last = bisect_left(so, toc_end) - 1 */
        lo = 0; hi = n_pk;
        while (lo < hi) {                 /* bisect_left */
            int64_t mid = (lo + hi) / 2;
            if (so[mid] < toc_end) lo = mid + 1; else hi = mid;
        }
        int64_t last = lo - 1;
        if (last < first) last = first;

        int64_t trim = toc_start - so[first];
        int64_t keep = play_end - play_start;
        if (trim < 0) { snprintf(err, errsz, "chain %lld: negative trim %lld",
                                 (long long)k, (long long)trim); goto fail; }
        int64_t capn = cum[last + 1] - cum[first];
        if (capn < trim + keep) {
            snprintf(err, errsz, "chain %lld: capacity %lld < trim %lld + keep %lld",
                     (long long)k, (long long)capn, (long long)trim, (long long)keep);
            goto fail;
        }
        if (nc == cap) { snprintf(err, errsz, "internal: chain overflow"); goto fail; }
        c->first[nc] = (uint32_t)first;
        c->npkts[nc] = (uint32_t)(last - first + 1);
        c->trim[nc] = (uint32_t)trim;
        c->keep[nc] = (uint32_t)keep;
        kept_total += keep;
        nc++;
    }
    if (kept_total != total_play) {
        snprintf(err, errsz, "internal: sum(keep) %lld != playable %lld",
                 (long long)kept_total, (long long)total_play);
        goto fail;
    }
    c->n_chains = nc;
    free(cum);
    return 1;
fail:
    free(cum);
    free(c->first); free(c->npkts); free(c->trim); free(c->keep);
    memset(c, 0, sizeof(*c));
    return 0;
}

static void cu_chains2_free(cu_chains2_t *c)
{
    free(c->first); free(c->npkts); free(c->trim); free(c->keep);
    memset(c, 0, sizeof(*c));
}

/* Debug/validation dumps: byte-identical to ogg_index.py write_blob/write_chains2 */
static int cu_dump_blob(const cu_demux_t *d, const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    fwrite("CUOP", 1, 4, f);
    fwrite(&d->blob.n_packets, 4, 1, f);
    fwrite(d->blob.len, 4, d->blob.n_packets, f);
    fwrite(d->blob.data, 1, d->blob.data_bytes, f);
    fclose(f);
    return 1;
}

static int cu_dump_chains2(const cu_chains2_t *c, const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    fwrite("CUC2", 1, 4, f);
    fwrite(&c->n_chains, 4, 1, f);
    for (uint32_t i = 0; i < c->n_chains; i++) {
        uint32_t rec[4] = { c->first[i], c->npkts[i], c->trim[i], c->keep[i] };
        fwrite(rec, 4, 4, f);
    }
    fclose(f);
    return 1;
}

#endif /* CUOPUS_DEMUX_H */
