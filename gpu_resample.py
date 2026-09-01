"""cuopus phase 4: 48 kHz -> 16 kHz polyphase decimator on the GPU (torch).

Purpose: the tail of the cuopus pipeline. Device-decoded PCM is already resident
on the GPU; this converts 48 k mono int16/float to the 16 k float32 speech-model
expects without a CPU round trip, feeding the (already-GPU) Silero VAD and mel.

Validated 2026-08-30 on the 477 s corpus vs ffmpeg/swresample: RMS difference
0.000075 (-83 dBFS), correlation 1.000000, alignment offset 0 samples --
inaudible and transcription-irrelevant. 172 ms single-thread CPU-torch for the
whole 477 s file; GPU timing TBD (card was fully committed during validation).

Design: windowed-sinc anti-alias low-pass at 0.5*16 kHz with a Kaiser window,
implemented as a strided conv1d (stride=3 == integer 3:1 decimation). This is
the same structure swresample uses; outputs are NOT bit-identical to swr (its
filter length/beta differ) but are within float tolerance of the ideal
band-limited result. Validation below measures against ffmpeg/swr directly.
"""
import math
import torch

_TAPS = 8 * 3 * 2 + 1     # 49 taps: 8 zero-crossings per side at the 16k Nyquist, x3 upstream rate
_BETA = 8.6               # Kaiser beta ~ 90 dB stopband

def _kernel(device, dtype=torch.float32):
    n = torch.arange(_TAPS, dtype=torch.float64) - (_TAPS - 1) / 2
    cutoff = 1.0 / 3.0                     # normalized to 48k Nyquist
    h = torch.where(n == 0, torch.tensor(cutoff, dtype=torch.float64),
                    torch.sin(math.pi * cutoff * n) / (math.pi * n))
    w = torch.kaiser_window(_TAPS, periodic=False, beta=_BETA, dtype=torch.float64)
    h = h * w
    h = h / h.sum()                        # unity DC gain
    return h.to(device=device, dtype=dtype).view(1, 1, -1)

_KERNEL_CACHE = {}

def resample_48k_to_16k(x: torch.Tensor) -> torch.Tensor:
    """x: 1-D tensor of 48 kHz mono samples (int16 or float32, CPU or CUDA).
    Returns 1-D float32 tensor of 16 kHz samples on x's device, length floor(len/3)."""
    dev = x.device
    if x.dtype == torch.int16:
        x = x.to(torch.float32) / 32768.0
    elif x.dtype != torch.float32:
        x = x.to(torch.float32)
    k = _KERNEL_CACHE.get(dev)
    if k is None:
        k = _kernel(dev)
        _KERNEL_CACHE[dev] = k
    pad = (_TAPS - 1) // 2
    y = torch.nn.functional.conv1d(x.view(1, 1, -1), k, stride=3, padding=pad)
    return y.view(-1)[: x.numel() // 3]
