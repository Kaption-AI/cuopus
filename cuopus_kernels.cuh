/* cuopus_kernels.cuh — device kernels + arena symbol definitions shared by the
 * decode CLI/daemon (decode_cli.cu) and the in-process library (libcuopus.cu).
 * Extracted VERBATIM from decode_cli.cu (perf-pass 2) on 2026-08-30: the two
 * translation units build into SEPARATE binaries (.so and CLI), so each carries
 * its own copy of the __global__/__device__/__constant__ symbols.
 *
 * The including .cu MUST first include: cuopus_device.h, device/cuopus_silk_all.cuh,
 * device/cuopus_celt_all.cuh, device/cuopus_opus.cuh (for cu_opus_decoder,
 * cuopus_stack_reset, cu_opus_decode, cuopus_tid).
 */
#ifndef CUOPUS_KERNELS_CUH
#define CUOPUS_KERNELS_CUH
#define ARENA_BYTES     (96u * 1024u)   /* same as the phase-3 gate harness */
#define BLOCK           64
#define MAX_PKT_SAMPLES 5760
#define CUOPUS_MAX_BATCH 2048           /* chains per kernel launch */
#define DEF_PREROLL_MS  250.0           /* perf pass 2 (was 1000) */
#define DEF_STRIDE      16              /* perf pass 2 (was 32): 2 chains/warp */

#define RS_TAPS 49                       /* == gpu_resample.py _TAPS */
#define RS_PAD  ((RS_TAPS - 1) / 2)      /* 24 */
#define RS_BETA 8.6                      /* == gpu_resample.py _BETA */
#define SWR_TAIL_48K 48                  /* CPU path's un-flushed swresample tail */
#define CARRY_MAX 64                     /* cross-batch resample carry bound (<=51) */

/* arena globals declared extern in cuopus_stack.h */
__device__ char        *cuopus_arena_base;
__device__ unsigned long cuopus_arena_per_thread;
__device__ char       **cuopus_arena_tops;
__device__ int         *cuopus_arena_ovf;
__device__ unsigned long cuopus_arena_stride;

__constant__ float cuopus_rs_taps[RS_TAPS];

struct cu_cli_result {
    int stop_reason;     /* 0 = decoded every packet, else OPUS_* error */
    int ovf;             /* arena overflow flag (voids the chain -> fail) */
    unsigned samples;    /* PCM samples written for this chain */
};

__global__ void cuopus_cli_decode(
    const unsigned char *__restrict__ data,
    const unsigned *__restrict__ off, const unsigned *__restrict__ len,
    const unsigned *__restrict__ first, const unsigned *__restrict__ npkts,
    int chain0, int n_batch, cu_opus_decoder *states, int16_t *pcm,
    const unsigned long long *__restrict__ pcm_off, cu_cli_result *res)
{
    unsigned gt = blockIdx.x * blockDim.x + threadIdx.x;
    if (gt % (unsigned)cuopus_arena_stride != 0) return;   /* one decoder per slot */
    int t = (int)(gt / (unsigned)cuopus_arena_stride);     /* batch-local chain idx */
    if (t >= n_batch) return;
    cuopus_stack_reset();
    cu_opus_decoder *st = &states[t];
    cu_opus_decoder_init(st, 48000, 1);
    int c = chain0 + t;
    int16_t *out = pcm + pcm_off[t];
    unsigned samples = 0;
    int stop = 0, n = (int)npkts[c];
    for (int k = 0; k < n; k++) {
        unsigned p = first[c] + (unsigned)k;
        int ret = cu_opus_decode(st, data + off[p], (int)len[p],
                                 out + samples, MAX_PKT_SAMPLES);
        if (ret < 0) { stop = ret; break; }
        samples += (unsigned)ret;
    }
    res[t].stop_reason = stop;
    res[t].samples = samples;
    res[t].ovf = cuopus_arena_ovf[cuopus_tid()];
}

/* Per-chain trim/keep extraction on device: copies each chain's playable
 * samples (skipping the preroll+preskip trim) into a contiguous staging
 * buffer, replicating exactly the host-side `src = pcm + off + trim,
 * len = keep` extraction of the pre-perf2 CLI. One block per chain. */
__global__ void cuopus_compact(
    const int16_t *__restrict__ pcm,
    const unsigned long long *__restrict__ pcm_off,
    const unsigned *__restrict__ trim, const unsigned *__restrict__ keep,
    const unsigned long long *__restrict__ dst_off,
    int16_t *__restrict__ dst, int n_batch)
{
    int c = blockIdx.x;
    if (c >= n_batch) return;
    const int16_t *src = pcm + pcm_off[c] + trim[c];
    int16_t *d = dst + dst_off[c];
    unsigned k = keep[c];
    for (unsigned i = threadIdx.x; i < k; i += blockDim.x) d[i] = src[i];
}

/* 48 k -> 16 k polyphase decimation (gpu_resample.py math): output sample oi
 * (global) is the 49-tap dot product centered at global 48 k index 3*oi,
 * zero-padded to the left of the stream start. `cat` holds global indices
 * [cat_base, cat_base + avail). Right-side reads past the stream cannot
 * happen: the emitted length is trimmed to (play48-48)//3, whose last output
 * reads at most index play48 - 27. */
__global__ void cuopus_resample48to16(
    const int16_t *__restrict__ cat, long long cat_base, long long avail,
    long long out0, long long n_out, float *__restrict__ out)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_out) return;
    long long center = 3 * (out0 + i);
    float acc = 0.f;
#pragma unroll
    for (int t = 0; t < RS_TAPS; t++) {
        long long li = center - RS_PAD + t - cat_base;
        float v = (li >= 0 && li < avail) ? (float)cat[li] * (1.0f / 32768.0f) : 0.f;
        acc += cuopus_rs_taps[t] * v;
    }
    out[i] = acc;
}

/* ---- host side ---- */

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* Kaiser I0 (double, series) — enough precision for 49 float taps. */
static double rs_bessel_i0(double x)
{
    double s = 1.0, t = 1.0, x2 = x * x / 4.0;
    for (int k = 1; k < 64; k++) {
        t *= x2 / ((double)k * k);
        s += t;
        if (t < 1e-18 * s) break;
    }
    return s;
}

/* Exact gpu_resample.py _kernel(): windowed sinc at cutoff 1/3, Kaiser beta
 * 8.6, unity DC gain — computed in double, cast to float (same as torch). */
static void rs_make_taps(float *out)
{
    double h[RS_TAPS], sum = 0.0;
    const double cutoff = 1.0 / 3.0;
    for (int i = 0; i < RS_TAPS; i++) {
        double n = i - (RS_TAPS - 1) / 2.0;
        double v = (n == 0.0) ? cutoff : sin(M_PI * cutoff * n) / (M_PI * n);
        double r = 2.0 * i / (RS_TAPS - 1) - 1.0;
        double w = rs_bessel_i0(RS_BETA * sqrt(1.0 - r * r)) / rs_bessel_i0(RS_BETA);
        h[i] = v * w;
        sum += h[i];
    }
    for (int i = 0; i < RS_TAPS; i++) out[i] = (float)(h[i] / sum);
}

#endif /* CUOPUS_KERNELS_CUH */
