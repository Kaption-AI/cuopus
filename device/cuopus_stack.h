/* cuopus phase 2: device replacement for celt/stack_alloc.h.
 *
 * libopus's ALLOC() (alloca / VLA / pseudostack) cannot be used as-is on the
 * device: alloca would force a large cudaLimitStackSize, and CUDA reserves
 * stack for every *resident* thread on the GPU (SMs x max threads), which
 * costs gigabytes. Instead each decode thread gets a fixed-size bump arena in
 * global memory (caller-provided workspace, cudaMalloc'd by the harness).
 * SAVE_STACK/RESTORE_STACK give the same scoped-free semantics as upstream.
 *
 * Overflow behaviour: the allocation wraps to the arena base (stays in-bounds,
 * so no memory corruption) and sets a per-thread flag the harness checks —
 * a flagged chain is reported and its comparison is voided, never silently
 * accepted.
 */
#ifndef CUOPUS_STACK_H
#define CUOPUS_STACK_H

#include <stddef.h>

/* Set by the harness before launch (cudaMemcpyToSymbol). */
extern __device__ char        *cuopus_arena_base;
extern __device__ unsigned long cuopus_arena_per_thread;
extern __device__ char       **cuopus_arena_tops;   /* per decode slot */
extern __device__ int         *cuopus_arena_ovf;    /* per decode slot */
extern __device__ unsigned long cuopus_arena_stride; /* threads per decode slot:
    1 = chain-per-lane (adjacent lanes diverge on data-dependent branches and
    serialize within the warp); 32 = chain-per-warp (only lane 0 of each warp
    decodes — no divergence serialization, 31/32 lanes idle). */

__device__ static inline unsigned cuopus_tid(void)
{
    return (blockIdx.x * blockDim.x + threadIdx.x) / (unsigned)cuopus_arena_stride;
}

/* Called once at the top of each kernel thread. */
__device__ static inline void cuopus_stack_reset(void)
{
    unsigned t = cuopus_tid();
    cuopus_arena_tops[t] = cuopus_arena_base + (unsigned long)t * cuopus_arena_per_thread;
    cuopus_arena_ovf[t] = 0;
}

__device__ static inline char **cuopus_cur_stack(void)
{
    return &cuopus_arena_tops[cuopus_tid()];
}

__device__ static inline void *cuopus_stack_alloc(size_t n)
{
    unsigned t = cuopus_tid();
    char *base = cuopus_arena_base + (unsigned long)t * cuopus_arena_per_thread;
    char *lim  = base + cuopus_arena_per_thread;
    char *p = cuopus_arena_tops[t];
    n = (n + 15) & ~(size_t)15;
    if (p + n > lim) {           /* overflow: flag it, wrap (in-bounds) */
        cuopus_arena_ovf[t] = 1;
        p = base;
    }
    cuopus_arena_tops[t] = p + n;
    return p;
}

/* Same macro surface as celt/stack_alloc.h. */
#define VARDECL(type, var) type *var
#define ALLOC(var, size, type) ((var) = (type*)cuopus_stack_alloc(sizeof(type)*(size)))
#define SAVE_STACK char *_cuopus_saved_stack = *cuopus_cur_stack()
#define RESTORE_STACK (*cuopus_cur_stack() = _cuopus_saved_stack)
#define ALLOC_STACK SAVE_STACK
#define ALLOC_NONE 1

#endif /* CUOPUS_STACK_H */
