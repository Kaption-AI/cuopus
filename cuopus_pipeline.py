"""cuopus: GPU OGG-Opus decode pipeline (daemon client).

decode_ogg_gpu(path, sr=16000) -> np.float32 mono audio at 16 kHz.

Perf-pass edition (2026-08-30). Differences from the original subprocess
pipeline (see BENCHMARKS.md for design + measurements):

  1. Demux moved into the CLI in C (cuopus_demux.h — port of ogg_index.py,
     validated byte-identical on all corpora + the 2 h file + the sweep set).
     Python no longer parses OGG pages or builds packet blobs: the CLI takes
     the .ogg path directly. (~0.25 ms/audio-s of Python time -> ~0.004.)
  2. Persistent daemon: `cuopus_decode --serve <unix socket>` holds the CUDA
     context + module + grow-only device buffers. Started lazily, reused,
     restarted on crash, exits by itself after CUOPUS_SERVE_IDLE_S (default
     600 s) to give the ~1.4 GB (context + driver local-memory pool) back.
     This removes the ~200-500 ms per-call CUDA context creation.
  3. Output length contract: a typical CPU path (PyAV -> swresample,
     never flushed) emits exactly (play48 - 48) // 3 samples at 16 kHz — the
     resampler keeps 48 input samples buffered at EOF. Measured constant on
     every corpus (9 files, 60 s .. 2 h). The GPU path trims its tail to
     match, so downstream length checks see identical shapes.

Steps (perf pass 2, CUOPUS_RESAMPLE=native default):  daemon DECODE16F
(demux + chain table + CUDA decode + ON-DEVICE 48 k -> 16 k Kaiser polyphase +
CPU-path tail trim, dual-stream copy/compute overlap, pinned output) -> numpy
fromfile of float32 16 k from /dev/shm. No torch, no second CUDA context.
CUOPUS_RESAMPLE=torch keeps the old path: daemon DECODE (48 k s16) -> torch GPU
polyphase (chunked, bit-identical to a full-array call) -> tail trim.
Perf-pass-2 defaults (measured + gated 2026-08-30): preroll 250 ms (was 1000;
-71.7 dBFS vs the FIXED serial reference, quality-identical, preroll is pure
wasted decode) and stride 16 (2 chains/warp; bit-exact at 4/8/16/32, ~19 %
faster than 32 at 1 h). Chain length stays 1 s (sweep winner at 15-30 min,
within 7 % of 2 s at 1-2 h).

EVERY failure raises — the caller catches and
falls back to the CPU decode paths. A cheap free-VRAM pre-check
(CUOPUS_MIN_FREE_MB, default 1800) runs before SPAWNING a daemon; once the
daemon is resident its VRAM is already accounted for and warm calls skip the
check. Concurrent calls are serialized by a module lock.
"""
import os
import socket
import subprocess
import threading
import time
import numpy as np

from .gpu_resample import _TAPS

_BIN = os.environ.get("CUOPUS_BIN") or next(
    (p for p in (os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuopus_decode"),
                 os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "cuopus_decode"))
     if os.path.exists(p)),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "cuopus_decode"))
_MIN_FREE_MB = int(os.environ.get("CUOPUS_MIN_FREE_MB", "2400"))  # perf pass 2: 2 h f16k working set ~2.4 GB (context+pool 1.4 + batch/stage/resample buffers ~1.0)
_SEG_S = float(os.environ.get("CUOPUS_SEG_S", "1.0"))
_PREROLL_MS = float(os.environ.get("CUOPUS_PREROLL_MS", "250"))
_STRIDE = int(os.environ.get("CUOPUS_STRIDE", "16"))
_TIMEOUT_S = float(os.environ.get("CUOPUS_TIMEOUT_S", "120"))
_IDLE_S = int(os.environ.get("CUOPUS_SERVE_IDLE_S", "600"))
_SOCK = os.environ.get("CUOPUS_SOCK", "/dev/shm/cuopus_decode_%d.sock" % os.getuid())
_TMPDIR = os.environ.get("CUOPUS_TMPDIR",
                         "/dev/shm" if os.access("/dev/shm", os.W_OK) else "/tmp")
_RESAMPLE_CHUNK = 32 * 1024 * 1024 * 3  # 48k samples per chunk (multiple of 3)
# swresample tail the CPU path never flushes: 48 samples @48 k == 16 @16 k
_SWR_TAIL_48K = 48
_RESAMPLE = os.environ.get("CUOPUS_RESAMPLE", "native")  # native = on-device (daemon DECODE16F); torch = legacy

_gpu_lock = threading.Lock()
_daemon = None          # Popen of the daemon we spawned (may exit on idle)
# Bounded daemon lifetime (2026-08-30, user decision: C host code stays ONLY because it is
# short-lived). Besides the 600 s idle-exit, the client recycles the daemon after
# CUOPUS_DAEMON_MAX_CALLS decodes or CUOPUS_DAEMON_MAX_AGE_S seconds even under continuous
# traffic, so any hypothetical leak is bounded by one recycle window. Recycle cost is one
# warm start (~0.5 s) per window -- negligible against the long files this path serves.
_DAEMON_MAX_CALLS = int(os.getenv("CUOPUS_DAEMON_MAX_CALLS", "200"))
_DAEMON_MAX_AGE_S = float(os.getenv("CUOPUS_DAEMON_MAX_AGE_S", "1800"))
_daemon_calls = 0
_daemon_born = 0.0


def _free_vram_mb():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10)
    if out.returncode != 0:
        raise RuntimeError("nvidia-smi failed: %s" % out.stderr.strip()[:200])
    return int(out.stdout.split("\n")[0].strip())


def _request(line, timeout):
    """One request/reply on the daemon socket. Raises on any transport error."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(_SOCK)
        s.sendall(line.encode())
        buf = b""
        while not buf.endswith(b"\n"):
            d = s.recv(4096)
            if not d:
                break
            buf += d
    finally:
        s.close()
    if not buf:
        raise RuntimeError("empty reply from cuopus daemon")
    return buf.decode().strip()


def _daemon_alive():
    try:
        return _request("PING\n", timeout=5).startswith("PONG")
    except OSError:
        return False
    except RuntimeError:
        return False


def _start_daemon():
    global _daemon_calls, _daemon_born
    _daemon_calls = 0
    _daemon_born = time.time()
    """Spawn the daemon and wait for it to accept. Caller holds _gpu_lock."""
    global _daemon
    if not os.path.exists(_BIN):
        raise RuntimeError("cuopus_decode binary missing at %s" % _BIN)
    free = _free_vram_mb()
    if free < _MIN_FREE_MB:
        raise RuntimeError("free VRAM %d MB < %d MB required" % (free, _MIN_FREE_MB))
    if _daemon is not None and _daemon.poll() is None:
        _daemon.kill()
        _daemon.wait()
    env = dict(os.environ)
    env.setdefault("CUOPUS_SERVE_IDLE_S", str(_IDLE_S))
    _daemon = subprocess.Popen([_BIN, "--serve", _SOCK], env=env,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if _daemon.poll() is not None:
            raise RuntimeError("cuopus daemon exited with %d during startup"
                               % _daemon.returncode)
        if os.path.exists(_SOCK) and _daemon_alive():
            return
        time.sleep(0.05)
    raise RuntimeError("cuopus daemon did not come up within 30 s")


def _recycle_if_due():
    global _daemon, _daemon_calls
    if _daemon is None:
        return
    if _daemon_calls >= _DAEMON_MAX_CALLS or (time.time() - _daemon_born) >= _DAEMON_MAX_AGE_S:
        try:
            shutdown_daemon()
        except Exception:
            pass


def _decode_via_daemon(path, out_pcm):
    """DECODE request with one restart-on-crash retry. Caller holds _gpu_lock.
    Returns (samples_written, play48)."""
    global _daemon_calls
    _recycle_if_due()
    verb = "DECODE16F" if _RESAMPLE == "native" else "DECODE"
    line = "%s %g %g %d %s %s\n" % (verb, _SEG_S, _PREROLL_MS, _STRIDE, path, out_pcm)
    for attempt in (0, 1):
        try:
            if not os.path.exists(_SOCK) or (attempt == 1):
                _start_daemon()
            rep = _request(line, timeout=_TIMEOUT_S)
        except (OSError, RuntimeError):
            if attempt == 1:
                raise
            continue   # stale socket / crashed or idle-exited daemon: restart once
        if rep.startswith("OK "):
            _daemon_calls += 1
            parts = rep.split()
            return int(parts[1]), int(parts[2])
        if rep.startswith("ERR "):
            # daemon is healthy; the file failed validation/decode -> caller falls back
            raise RuntimeError("cuopus daemon: %s" % rep[4:])
        raise RuntimeError("cuopus daemon: bad reply %r" % rep[:120])


def _resample_chunked(pcm_i16: np.ndarray, device: str):
    """48 k int16 -> 16 k float32 via gpu_resample, in chunks that are
    bit-identical to one full-array call (explicit halo, zero-padded edges)."""
    import torch
    pad = (_TAPS - 1) // 2
    n = pcm_i16.shape[0]
    outs = []
    for s0 in range(0, n - (n % 3), _RESAMPLE_CHUNK):
        s1 = min(s0 + _RESAMPLE_CHUNK, n - (n % 3))
        a, b = s0 - pad, s1 + pad
        left = max(0, -a)
        right = max(0, b - n)
        x = torch.from_numpy(np.ascontiguousarray(pcm_i16[max(0, a):min(n, b)]))
        x = x.to(device=device, dtype=torch.float32) / 32768.0
        if left or right:
            x = torch.nn.functional.pad(x.view(1, 1, -1), (left, right)).view(-1)
        # full-array conv1d(padding=pad) output i is centered at x[3i]; with the
        # halo included, a padding=0 conv over this slice yields exactly
        # y_full[s0//3 : s1//3] (same summation order -> bit-identical).
        from .gpu_resample import _KERNEL_CACHE, _kernel
        k = _KERNEL_CACHE.get(x.device)
        if k is None:
            k = _kernel(x.device)
            _KERNEL_CACHE[x.device] = k
        y = torch.nn.functional.conv1d(x.view(1, 1, -1), k, stride=3, padding=0)
        outs.append(y.view(-1)[: (s1 - s0) // 3].cpu().numpy())
        del x, y
    return np.concatenate(outs) if outs else np.zeros(0, np.float32)


def decode_ogg_gpu(path: str, sr: int = 16000) -> np.ndarray:
    """Decode an OGG-Opus file via the cuopus CUDA decoder daemon. Raises on
    any failure; the caller is expected to fall back to CPU decode."""
    if sr != 16000:
        raise ValueError("decode_ogg_gpu only supports sr=16000 (got %s)" % sr)

    out_pcm = os.path.join(_TMPDIR, "cuopus_%d_%d.pcm" % (os.getpid(), threading.get_ident()))
    with _gpu_lock:
        try:
            samples, play48 = _decode_via_daemon(path, out_pcm)
            if play48 <= 0:
                raise RuntimeError("no playable samples")
            n_cpu = max(0, (play48 - _SWR_TAIL_48K) // 3)
            if _RESAMPLE == "native":
                # DECODE16F: float32 16 kHz, resampled on device with the same
                # 49-tap Kaiser polyphase (validated <= 1e-5 vs the torch
                # path) and already trimmed to the CPU-path length contract.
                audio = np.fromfile(out_pcm, dtype=np.float32)
                if audio.shape[0] != samples or samples != n_cpu:
                    raise RuntimeError("native 16k: %d samples, daemon said %d, "
                                       "contract %d" % (audio.shape[0], samples, n_cpu))
                return audio
            pcm = np.fromfile(out_pcm, dtype=np.int16)
            if pcm.shape[0] != samples or samples != play48:
                raise RuntimeError("decoded %d samples, expected %d (play48 %d)"
                                   % (pcm.shape[0], samples, play48))
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            audio = _resample_chunked(pcm, device)
            # match the CPU path's length contract exactly (PyAV ->
            # swresample never flushed: 48 input samples held back at EOF)
            return audio[:n_cpu]
        finally:
            try:
                os.remove(out_pcm)
            except OSError:
                pass


def shutdown_daemon():
    """Best-effort daemon stop (tests / explicit VRAM handback)."""
    with _gpu_lock:
        try:
            _request("QUIT\n", timeout=5)
        except (OSError, RuntimeError):
            pass
