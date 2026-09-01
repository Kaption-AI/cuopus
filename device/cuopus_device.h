/* cuopus phase 2: DEVICE-macro shim for the mechanical libopus port.
 *
 * Every ported function is prefixed CUOPUS_DEVFN and every ported file-scope
 * table CUOPUS_DEVDATA by device/port_silk.py. Compiling with -DCUOPUS_CUDA
 * under nvcc turns those into __device__; compiling without (plain host cc)
 * leaves the sources byte-equivalent to a normal libopus build of the same
 * translation units.
 */
#ifndef CUOPUS_DEVICE_H
#define CUOPUS_DEVICE_H

#include <stddef.h>

#if defined(CUOPUS_CUDA) && defined(__CUDACC__)
#define CUOPUS_DEVFN __device__
/* .c-file (out-of-line) functions: noinline preserves libopus's TU
 * boundaries and keeps the device inliner from exploding (see port_silk.py) */
#define CUOPUS_DEVFN_NOINL __device__ __noinline__
/* perf pass: hot leaf functions (HOT_FNS in port_silk.py) are force-inlined
 * into their callers — per-symbol range-decoder calls dominated warp time.
 * Overridable: build with -DCUOPUS_DEVFN_HOT='__device__ __noinline__' to get
 * device code identical to the phase-3-gated build (e.g. while the
 * bit-exactness gate for the inlined build has not run yet). */
#ifndef CUOPUS_DEVFN_HOT
#define CUOPUS_DEVFN_HOT __device__ __forceinline__
#endif
#define CUOPUS_DEVDATA __device__

/* CUDA device code has memcpy/memset builtins but no memmove; the ported
 * sources' silk_memmove/OPUS_MOVE are redirected here (see port_silk.py). */
__device__ static inline void *cuopus_memmove(void *dst, const void *src, size_t n)
{
    char *d = (char *)dst;
    const char *s = (const char *)src;
    size_t i;
    if (d == s || n == 0) return dst;
    if (d < s) {
        for (i = 0; i < n; i++) d[i] = s[i];
    } else {
        for (i = n; i-- > 0;) d[i] = s[i];
    }
    return dst;
}
#else
#define CUOPUS_DEVFN
#define CUOPUS_DEVFN_NOINL
#define CUOPUS_DEVFN_HOT
#define CUOPUS_DEVDATA
#endif

#endif /* CUOPUS_DEVICE_H */
