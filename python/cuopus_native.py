"""cuopus_native — ctypes wrapper for libcuopus.so (in-process GPU OGG-Opus
decode).

decode_ogg_to_cuda_tensor(path_or_bytes) -> torch.cuda.FloatTensor (1-D,
float32 @16 kHz, the exact (play48-48)//3 CPU-path length contract). The
decode runs on the CALLER'S current CUDA device/stream: the library creates
no context, no threads, no streams — every kernel is issued on torch's
current stream and the stream is synchronized before the call returns, so the
tensor is finished (and bit-identical to the daemon's DECODE16F output).

THREADING CONTRACT (see the libcuopus.cu header comment): in a server that
also captures CUDA graphs this MUST only be called on the one thread allowed
to issue GPU work — the same serialization that keeps graph capture safe.
Never call it while stream capture is in flight (the
library refuses if it can detect it). A single-threaded test process
qualifies trivially.

Every failure raises CuopusError; the caller is expected to fall back to the
CPU decode paths. rc==2 (CUDA error) means the shared context is suspect —
the server should treat it like any other CUDA failure, not retry.
"""
import ctypes
import os

_ERRCAP = 512

_RC_NAMES = {1: "soft", 2: "cuda", 3: "caller"}


class CuopusError(RuntimeError):
    def __init__(self, rc, msg):
        super().__init__("cuopus[%s]: %s" % (_RC_NAMES.get(rc, rc), msg))
        self.rc = rc


def _find_lib():
    env = os.environ.get("CUOPUS_LIB")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, "libcuopus.so"),
                 os.path.join(here, "..", "out", "libcuopus.so")):
        if os.path.exists(cand):
            return cand
    raise OSError("libcuopus.so not found (set CUOPUS_LIB)")


_lib = None


def _load():
    global _lib
    if _lib is not None:
        return _lib
    lib = ctypes.CDLL(_find_lib())
    lib.cuopus_probe.restype = ctypes.c_int
    lib.cuopus_probe.argtypes = [ctypes.c_char_p, ctypes.c_size_t,
                                 ctypes.POINTER(ctypes.c_int64),
                                 ctypes.c_char_p, ctypes.c_size_t]
    lib.cuopus_decode_f16k_on.restype = ctypes.c_int
    lib.cuopus_decode_f16k_on.argtypes = [ctypes.c_char_p, ctypes.c_size_t,
                                          ctypes.c_void_p, ctypes.c_size_t,
                                          ctypes.POINTER(ctypes.c_int64),
                                          ctypes.c_void_p,
                                          ctypes.c_char_p, ctypes.c_size_t]
    lib.cuopus_decode_f16k_multi.restype = ctypes.c_int
    lib.cuopus_decode_f16k_multi.argtypes = [
        ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_size_t), ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_int64), ctypes.c_void_p,
        ctypes.c_char_p, ctypes.c_size_t]
    lib.cuopus_warmup.restype = ctypes.c_int
    lib.cuopus_warmup.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    lib.cuopus_last_arena_overflow.restype = ctypes.c_int
    lib.cuopus_last_arena_overflow.argtypes = []
    lib.cuopus_workspace_bytes.restype = ctypes.c_longlong
    lib.cuopus_workspace_bytes.argtypes = []
    lib.cuopus_release_workspaces.restype = None
    lib.cuopus_release_workspaces.argtypes = []
    _lib = lib
    return lib


def _as_bytes(path_or_bytes):
    if isinstance(path_or_bytes, (bytes, bytearray, memoryview)):
        return bytes(path_or_bytes)
    with open(path_or_bytes, "rb") as f:
        return f.read()


def probe(path_or_bytes) -> int:
    """Exact output length in float32 @16 kHz samples (host-only, no GPU)."""
    lib = _load()
    data = _as_bytes(path_or_bytes)
    n16 = ctypes.c_int64(0)
    err = ctypes.create_string_buffer(_ERRCAP)
    rc = lib.cuopus_probe(data, len(data), ctypes.byref(n16), err, _ERRCAP)
    if rc:
        raise CuopusError(rc, err.value.decode(errors="replace"))
    return n16.value


def decode_ogg_to_cuda_tensor(path_or_bytes):
    """Decode an OGG-Opus file/bytes into a fresh torch CUDA float32 tensor
    at 16 kHz on the current device/stream. Raises CuopusError on failure."""
    import torch
    lib = _load()
    data = _as_bytes(path_or_bytes)
    n16 = probe(data)
    out = torch.empty(n16, dtype=torch.float32, device="cuda")
    if n16 == 0:
        return out
    stream = torch.cuda.current_stream().cuda_stream
    out_len = ctypes.c_int64(0)
    err = ctypes.create_string_buffer(_ERRCAP)
    rc = lib.cuopus_decode_f16k_on(data, len(data),
                                   ctypes.c_void_p(out.data_ptr()), n16,
                                   ctypes.byref(out_len),
                                   ctypes.c_void_p(stream), err, _ERRCAP)
    if rc:
        raise CuopusError(rc, err.value.decode(errors="replace"))
    if out_len.value != n16:
        raise CuopusError(1, "decode wrote %d samples, probe said %d"
                          % (out_len.value, n16))
    return out


def decode_ogg_multi(items):
    """v22: decode N OGG-Opus files/byte-blobs in (nearly) one GPU trip -- one
    merged cuopus_cli_decode launch + one compact + N tiny resamples + one sync.
    Per-file output is BITWISE identical to decode_ogg_to_cuda_tensor (gated in
    cuopus/test_multi.py). Returns a list of 1-D CUDA float32 tensors @16 kHz.
    Same threading contract as the single call: batch-worker GPU thread only.
    Raises CuopusError; rc==3 with a "max_batch" message means the merged chain
    count exceeds one launch -- callers should split or fall back per-file."""
    import torch
    lib = _load()
    blobs = [_as_bytes(x) for x in items]
    n = len(blobs)
    if n == 0:
        return []
    n16 = [probe(b) for b in blobs]
    outs = [torch.empty(k, dtype=torch.float32, device="cuda") for k in n16]
    stream = torch.cuda.current_stream().cuda_stream
    ba = (ctypes.c_char_p * n)(*blobs)
    la = (ctypes.c_size_t * n)(*[len(b) for b in blobs])
    oa = (ctypes.c_void_p * n)(*[o.data_ptr() for o in outs])
    ca = (ctypes.c_size_t * n)(*n16)
    ol = (ctypes.c_int64 * n)()
    err = ctypes.create_string_buffer(_ERRCAP)
    rc = lib.cuopus_decode_f16k_multi(ba, la, n, oa, ca, ol,
                                      ctypes.c_void_p(stream), err, _ERRCAP)
    if rc:
        raise CuopusError(rc, err.value.decode(errors="replace"))
    for i in range(n):
        if ol[i] != n16[i]:
            raise CuopusError(1, "file %d: decode wrote %d samples, probe said %d"
                              % (i, ol[i], n16[i]))
    return outs


def warmup():
    """Optional: pre-size the driver's ~910 MB local-memory pool at startup."""
    lib = _load()
    err = ctypes.create_string_buffer(_ERRCAP)
    rc = lib.cuopus_warmup(err, _ERRCAP)
    if rc:
        raise CuopusError(rc, err.value.decode(errors="replace"))


def workspace_bytes() -> int:
    return int(_load().cuopus_workspace_bytes())


def last_arena_overflow() -> bool:
    return bool(_load().cuopus_last_arena_overflow())


def release_workspaces():
    _load().cuopus_release_workspaces()
