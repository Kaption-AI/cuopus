#!/usr/bin/env python3
"""cuopus phase 1: OGG Opus page/packet indexer.

Parses an OGG Opus file into a per-page index:
    (byte_offset, granule_position, packet_boundaries)
Granule positions are EXACT 48 kHz sample offsets (RFC 7845), which replaces
v14's 1-second preroll guess with exact per-page segmentation (an 80 ms
decoder preroll is still required, but the trim points become exact).

Also derives per-packet durations from the Opus TOC byte (RFC 6716 sec 3.1)
and cross-checks the cumulative sample count against every page granule.

Commands:
    index    <file.ogg>              print index summary (add --json for full page table)
    extract  <file.ogg> <out.blob>   dump audio packets to a binary blob for the harness
    validate <file.ogg>              cross-check against ffprobe -show_packets
    segment  <file.ogg> <seconds>    demo: exact byte/sample cut points every N seconds\n    chains2  <file.ogg> <seconds> <out.bin>  integration chain table (CUC2: first,npkts,trim,keep)

Blob format (little-endian):
    magic 'CUOP', u32 n_packets, n*u32 packet lengths, concatenated packet bytes.
"""
import json
import struct
import subprocess
import sys

CRC_TABLE = []
def _init_crc():
    for i in range(256):
        r = i << 24
        for _ in range(8):
            r = ((r << 1) ^ 0x04C11DB7) & 0xFFFFFFFF if r & 0x80000000 else (r << 1) & 0xFFFFFFFF
        CRC_TABLE.append(r)
_init_crc()

def _crc(data):
    r = 0
    for b in data:
        r = ((r << 8) & 0xFFFFFFFF) ^ CRC_TABLE[((r >> 24) & 0xFF) ^ b]
    return r

# RFC 6716 sec 3.1: frame duration in ms*4 (to keep 2.5 ms integral) per config number.
_FRAME_QMS = []
for _fam, _sizes in (("silk", (10, 20, 40, 60)), ("silk", (10, 20, 40, 60)),
                     ("silk", (10, 20, 40, 60)), ("hybrid", (10, 20)),
                     ("hybrid", (10, 20)),
                     ("celt", (2.5, 5, 10, 20)), ("celt", (2.5, 5, 10, 20)),
                     ("celt", (2.5, 5, 10, 20)), ("celt", (2.5, 5, 10, 20))):
    _FRAME_QMS += [int(s * 4) for s in _sizes]
assert len(_FRAME_QMS) == 32

def packet_samples_48k(pkt):
    """Exact duration of one Opus packet in 48 kHz samples, from the TOC byte."""
    if not pkt:
        return 0
    toc = pkt[0]
    config = toc >> 3
    code = toc & 0x3
    if code == 0:
        nframes = 1
    elif code in (1, 2):
        nframes = 2
    else:
        if len(pkt) < 2:
            return 0
        nframes = pkt[1] & 0x3F
    # samples = ms*4 * 48000 / 4000 = qms * 12
    return _FRAME_QMS[config] * 12 * nframes


class Page:
    __slots__ = ("offset", "size", "granule", "serial", "seq", "header_type",
                 "packet_spans", "continued_in", "continued_out", "crc_ok")

def parse_pages(data, check_crc=True):
    """Yield Page objects. packet_spans: list of (start, end) byte spans within
    the file for each lacing-segment run; a run ending with lacing 255 continues
    onto the next page."""
    pages = []
    pos = 0
    n = len(data)
    while pos < n:
        idx = data.find(b"OggS", pos)
        if idx < 0:
            break
        if idx + 27 > n:
            break
        (version, header_type, granule, serial, seq, crc, nsegs) = struct.unpack_from(
            "<BBqIIIB", data, idx + 4)
        seg_table = data[idx + 27: idx + 27 + nsegs]
        body_start = idx + 27 + nsegs
        body_len = sum(seg_table)
        if body_start + body_len > n:
            break
        page = Page()
        page.offset = idx
        page.size = 27 + nsegs + body_len
        page.granule = granule
        page.serial = serial
        page.seq = seq
        page.header_type = header_type
        page.continued_in = bool(header_type & 0x1)
        # CRC check (field zeroed during computation). Optional: the pure-
        # Python CRC costs ~1.3 s on a 2 h file, so the GPU decode pipeline
        # skips it (granule cross-check + per-chain decode validation remain).
        if check_crc:
            raw = bytearray(data[idx: idx + page.size])
            raw[22:26] = b"\0\0\0\0"
            page.crc_ok = (_crc(raw) == crc)
        else:
            page.crc_ok = True
        spans = []
        off = body_start
        run_start = off
        run_len = 0
        last_lacing = 0
        for lac in seg_table:
            run_len += lac
            last_lacing = lac
            if lac < 255:
                spans.append((run_start, run_start + run_len, True))   # packet completes
                run_start += run_len
                run_len = 0
        if run_len > 0 or (nsegs > 0 and last_lacing == 255):
            spans.append((run_start, run_start + run_len, False))      # continues on next page
        page.continued_out = bool(spans) and not spans[-1][2]
        page.packet_spans = spans
        pages.append(page)
        pos = idx + page.size
    return pages


def build_index(path, check_crc=True):
    with open(path, "rb") as f:
        data = f.read()
    pages = parse_pages(data, check_crc)
    if not pages:
        raise SystemExit("no OGG pages found")

    # Reassemble packets across pages.
    packets = []          # list of bytes
    packet_pages = []     # index of page on which each packet COMPLETES
    partial = None
    for pi, page in enumerate(pages):
        for (s, e, complete) in page.packet_spans:
            chunk = data[s:e]
            if partial is not None:
                chunk = partial + chunk
                partial = None
            if complete:
                packets.append(chunk)
                packet_pages.append(pi)
            else:
                partial = chunk
    if partial:
        print("WARNING: file ends mid-packet (%d bytes dropped)" % len(partial), file=sys.stderr)

    # Identify Opus headers.
    if len(packets) < 2 or not packets[0].startswith(b"OpusHead"):
        raise SystemExit("not an OGG Opus stream (no OpusHead)")
    head = packets[0]
    channels = head[9]
    preskip = struct.unpack_from("<H", head, 10)[0]
    n_header_packets = 2  # OpusHead + OpusTags
    audio_packets = packets[n_header_packets:]
    audio_packet_pages = packet_pages[n_header_packets:]

    # Exact per-packet 48 kHz sample offsets from TOC durations.
    sample_off = []
    cum = 0
    for p in audio_packets:
        sample_off.append(cum)
        cum += packet_samples_48k(p)
    toc_total = cum

    # Cross-check cumulative TOC durations against page granules.
    granule_checked = granule_match = 0
    last_pkt_end = {}
    for k, pg in enumerate(audio_packet_pages):
        last_pkt_end[pg] = sample_off[k] + packet_samples_48k(audio_packets[k])
    final_data_page = max(last_pkt_end) if last_pkt_end else -1
    for pi, page in enumerate(pages):
        if page.granule == -1 or pi not in last_pkt_end:
            continue
        granule_checked += 1
        if page.granule == last_pkt_end[pi]:
            granule_match += 1
        elif pi == final_data_page and 0 <= last_pkt_end[pi] - page.granule < packet_samples_48k(audio_packets[-1]):
            granule_match += 1  # RFC 7845 end-trim: final page granule may be short of TOC sum
    last_granule = max((p.granule for p in pages if p.granule != -1), default=0)

    return {
        "path": path,
        "file_bytes": len(data),
        "pages": pages,
        "packets": packets,
        "audio_packets": audio_packets,
        "audio_packet_pages": audio_packet_pages,
        "sample_off": sample_off,
        "channels": channels,
        "preskip": preskip,
        "toc_total_samples": toc_total,
        "end_trim": toc_total - last_granule,
        "last_granule": last_granule,
        "pcm_samples_48k": last_granule - preskip,
        "granule_checked": granule_checked,
        "granule_match": granule_match,
        "crc_bad": sum(1 for p in pages if not p.crc_ok),
    }


def cmd_index(path, as_json=False):
    ix = build_index(path)
    pages = ix["pages"]
    print("file:                  %s (%d bytes)" % (path, ix["file_bytes"]))
    print("pages:                 %d  (bad CRC: %d)" % (len(pages), ix["crc_bad"]))
    print("packets:               %d total, %d audio (+2 header)" % (len(ix["packets"]), len(ix["audio_packets"])))
    print("channels / preskip:    %d / %d" % (ix["channels"], ix["preskip"]))
    print("last granule:          %d (48 kHz)" % ix["last_granule"])
    print("TOC-summed samples:    %d (end-trim %d)" % (ix["toc_total_samples"], ix["end_trim"]))
    print("PCM samples (granule-preskip): %d  = %.4f s" % (ix["pcm_samples_48k"], ix["pcm_samples_48k"] / 48000.0))
    print("granule cross-check:   %d/%d pages match TOC cumulative sum" % (ix["granule_match"], ix["granule_checked"]))
    if as_json:
        table = [{"byte_offset": p.offset, "granule_position": p.granule,
                  "packet_boundaries": [(s, e) for (s, e, c) in p.packet_spans]}
                 for p in pages]
        print(json.dumps(table))
    return ix


def cmd_extract(path, out):
    ix = build_index(path)
    pkts = ix["audio_packets"]
    with open(out, "wb") as f:
        f.write(b"CUOP")
        f.write(struct.pack("<I", len(pkts)))
        for p in pkts:
            f.write(struct.pack("<I", len(p)))
        for p in pkts:
            f.write(p)
    total = sum(len(p) for p in pkts)
    print("wrote %s: %d packets, %d payload bytes, min/max/avg len %d/%d/%.1f" %
          (out, len(pkts), total, min(map(len, pkts)), max(map(len, pkts)), total / len(pkts)))


def cmd_validate(path):
    ix = cmd_index(path)
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "packet=duration", "-show_entries", "stream=duration_ts,start_time",
         "-of", "json", path],
        capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    fp_packets = j["packets"]
    fp_count = len(fp_packets)
    fp_dur_sum = sum(int(p["duration"]) for p in fp_packets)
    fp_duration_ts = int(j["streams"][0]["duration_ts"])
    my_count = len(ix["audio_packets"])
    my_samples = ix["pcm_samples_48k"]
    print("--- ffprobe cross-check ---")
    print("audio packet count:    ours=%d  ffprobe=%d  %s" %
          (my_count, fp_count, "MATCH" if my_count == fp_count else "MISMATCH"))
    print("sum(packet durations): ours(TOC,end-trimmed)=%d  ffprobe=%d  %s" %
          (ix["last_granule"], fp_dur_sum, "MATCH" if ix["last_granule"] == fp_dur_sum else "MISMATCH"))
    print("total samples @48k:    ours(last granule)=%d  ffprobe duration_ts=%d  %s" %
          (ix["last_granule"], fp_duration_ts, "MATCH" if ix["last_granule"] == fp_duration_ts else "MISMATCH"))
    print("playable PCM @48k:     ours(granule-preskip)=%d = %.7f s" % (my_samples, my_samples / 48000.0))
    ok = (my_count == fp_count and ix["last_granule"] == fp_dur_sum
          and ix["last_granule"] == fp_duration_ts and ix["granule_match"] == ix["granule_checked"]
          and ix["crc_bad"] == 0)
    print("VALIDATION: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def cmd_chains(path, seconds, out, preroll_ms=80):
    """Phase 2: write a binary chain table for the decode harnesses.

    One chain per `seconds` window, starting `preroll_ms` early (decoder
    convergence preroll; RFC 6716 recommends 80 ms). Packet-exact: chain k
    spans the first packet still covering (k*step - preroll) through the last
    packet starting before (k+1)*step.
    Format (LE): magic 'CUCH', u32 n_chains, then n * { u32 first_pkt, u32 n_pkts }.
    """
    ix = build_index(path)
    step = int(float(seconds) * 48000)
    preroll = int(float(preroll_ms) * 48)
    sample_off = ix["sample_off"]
    pkts = ix["audio_packets"]
    total = ix["toc_total_samples"]
    n_pk = len(pkts)
    chains = []
    k = 0
    while k * step < total:
        start_target = max(0, k * step - preroll)
        end_target = min((k + 1) * step, total)
        # first packet whose end > start_target
        import bisect
        first = bisect.bisect_right(sample_off, start_target) - 1
        if first < 0:
            first = 0
        if sample_off[first] + packet_samples_48k(pkts[first]) <= start_target:
            first += 1
        # last packet whose start < end_target
        last = bisect.bisect_left(sample_off, end_target) - 1
        if last < first:
            last = first
        chains.append((first, last - first + 1))
        k += 1
    with open(out, "wb") as f:
        f.write(b"CUCH")
        f.write(struct.pack("<I", len(chains)))
        for first, n in chains:
            f.write(struct.pack("<II", first, n))
    tot_pkts = sum(n for _, n in chains)
    print("wrote %s: %d chains of %ss (+%d ms preroll), %d packet slots (%.2fx overlap)"
          % (out, len(chains), seconds, preroll_ms, tot_pkts, tot_pkts / float(n_pk)))


def build_chains2(ix, seconds, preroll_ms=80):
    """Integration: chain table with per-chain trim/keep sample counts.

    Playable-sample-space segmentation (preskip and RFC 7845 end-trim aware):
    chain k covers playable samples [k*step, min((k+1)*step, total_play)) at
    48 kHz. Each chain starts preroll_ms early for decoder convergence; `trim`
    is the exact count of decoded samples to drop from the chain front
    (preroll + preskip alignment), `keep` the exact playable samples to emit.
    Concatenating keep-windows of all chains reproduces the file's playable
    PCM exactly (sum(keep) == pcm_samples_48k).
    Returns list of (first_pkt, n_pkts, trim, keep).
    """
    import bisect
    step = int(float(seconds) * 48000)
    preroll = int(float(preroll_ms) * 48)
    preskip = ix["preskip"]
    total_play = ix["pcm_samples_48k"]
    if total_play <= 0:
        raise ValueError("no playable samples")
    sample_off = ix["sample_off"]
    pkts = ix["audio_packets"]
    chains = []
    k = 0
    while k * step < total_play:
        play_start = k * step
        play_end = min((k + 1) * step, total_play)
        toc_start = play_start + preskip
        toc_end = play_end + preskip
        start_target = max(0, toc_start - preroll)
        first = bisect.bisect_right(sample_off, start_target) - 1
        if first < 0:
            first = 0
        if sample_off[first] + packet_samples_48k(pkts[first]) <= start_target:
            first += 1
        last = bisect.bisect_left(sample_off, toc_end) - 1
        if last < first:
            last = first
        trim = toc_start - sample_off[first]
        keep = play_end - play_start
        if trim < 0:
            raise ValueError("chain %d: negative trim %d" % (k, trim))
        cap = sum(packet_samples_48k(pkts[p]) for p in range(first, last + 1))
        if cap < trim + keep:
            raise ValueError("chain %d: capacity %d < trim %d + keep %d"
                             % (k, cap, trim, keep))
        chains.append((first, last - first + 1, trim, keep))
        k += 1
    assert sum(c[3] for c in chains) == total_play
    return chains


def write_chains2(ix, seconds, out, preroll_ms=80):
    chains = build_chains2(ix, seconds, preroll_ms)
    with open(out, "wb") as f:
        f.write(b"CUC2")
        f.write(struct.pack("<I", len(chains)))
        for first, n, trim, keep in chains:
            f.write(struct.pack("<IIII", first, n, trim, keep))
    return chains


def write_blob(ix, out):
    pkts = ix["audio_packets"]
    with open(out, "wb") as f:
        f.write(b"CUOP")
        f.write(struct.pack("<I", len(pkts)))
        for p in pkts:
            f.write(struct.pack("<I", len(p)))
        for p in pkts:
            f.write(p)


def cmd_chains2(path, seconds, out, preroll_ms=80):
    ix = build_index(path)
    chains = write_chains2(ix, seconds, out, preroll_ms)
    tot = sum(n for _, n, _, _ in chains)
    print("wrote %s: %d chains2 of %ss (+%s ms preroll, preskip %d), "
          "%d packet slots, %d playable samples" %
          (out, len(chains), seconds, preroll_ms, ix["preskip"], tot,
           sum(c[3] for c in chains)))


def cmd_segment(path, seconds):
    ix = build_index(path)
    step = int(float(seconds) * 48000)
    pages = ix["pages"]
    gp = [(p.granule, p.offset) for p in pages if p.granule != -1]
    print("exact cut points every %s s (page byte offset, granule):" % seconds)
    target = step
    shown = 0
    for granule, offset in gp:
        if granule >= target:
            print("  >= %8.2f s : byte %9d granule %10d (%.4f s)" %
                  (target / 48000.0, offset, granule, granule / 48000.0))
            target += step
            shown += 1
            if shown >= 12:
                print("  ...")
                break


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    cmd = sys.argv[1]
    if cmd == "index":
        cmd_index(sys.argv[2], as_json="--json" in sys.argv)
    elif cmd == "extract":
        cmd_extract(sys.argv[2], sys.argv[3])
    elif cmd == "validate":
        sys.exit(cmd_validate(sys.argv[2]))
    elif cmd == "chains":
        cmd_chains(sys.argv[2], sys.argv[3], sys.argv[4],
                   float(sys.argv[5]) if len(sys.argv) > 5 else 80)
    elif cmd == "chains2":
        cmd_chains2(sys.argv[2], sys.argv[3], sys.argv[4],
                    float(sys.argv[5]) if len(sys.argv) > 5 else 80)
    elif cmd == "segment":
        cmd_segment(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
        sys.exit(2)
