/* cuopus phase 1: bit-exactness harness + symbol-throughput benchmark.
 *
 * CPU side: the REAL libopus 1.5.2 range decoder (vendor/opus-1.5.2/celt/
 * entdec.c + entcode.c, linked in) runs the fixed symbol schedule over every
 * packet.  GPU side: cuopus_entdec.cuh (one thread = one packet) runs the
 * SAME schedule (schedule.inc included twice).  Every output word of every
 * packet is compared; then both sides are timed.
 *
 * Usage: cuopus_harness <packets.blob> [reps]
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <omp.h>
#include <cuda_runtime.h>

extern "C" {
#include "entdec.h"   /* libopus: ec_dec_* reference */
#include "entcode.h"  /* ec_tell */
}
#include "cuopus_entdec.cuh"

#define WORDS_PER_PKT 147
#define SYMS_PER_PKT  144

/* ---- CPU reference schedule (libopus entdec) ---- */
#define SCHED_NAME  run_schedule_cpu
#define SCHED_QUAL  static
#define SCHED_DEC   ec_dec
#define SCHED_EC(f) f
#include "schedule.inc"
#undef SCHED_NAME
#undef SCHED_QUAL
#undef SCHED_DEC
#undef SCHED_EC

/* ---- GPU schedule (cuopus port) ---- */
#define SCHED_NAME  run_schedule_gpu
#define SCHED_QUAL  __device__ static
#define SCHED_DEC   cu_ec_dec
#define SCHED_EC(f) cu_##f
#include "schedule.inc"
#undef SCHED_NAME
#undef SCHED_QUAL
#undef SCHED_DEC
#undef SCHED_EC

__global__ void decode_kernel(const unsigned char *blob, const unsigned int *offs,
                              const unsigned int *lens, unsigned int n,
                              unsigned int *out)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    run_schedule_gpu(blob + offs[i], lens[i], out + (size_t)i * WORDS_PER_PKT);
}

#define CUDA_CHECK(x) do { cudaError_t e = (x); if(e != cudaSuccess){ \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    exit(1);} } while(0)

static double now_s(void){
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv)
{
    if(argc < 2){ fprintf(stderr, "usage: %s packets.blob [reps]\n", argv[0]); return 2; }
    int reps = argc > 2 ? atoi(argv[2]) : 10;

    /* ---- load blob: 'CUOP', u32 n, n*u32 lens, payload ---- */
    FILE *f = fopen(argv[1], "rb");
    if(!f){ perror(argv[1]); return 1; }
    char magic[4]; unsigned int n;
    if(fread(magic, 1, 4, f) != 4 || memcmp(magic, "CUOP", 4) ||
       fread(&n, 4, 1, f) != 1){ fprintf(stderr, "bad blob\n"); return 1; }
    std::vector<unsigned int> lens(n), offs(n);
    if(fread(lens.data(), 4, n, f) != n){ fprintf(stderr, "bad blob lens\n"); return 1; }
    size_t total = 0;
    for(unsigned i = 0; i < n; i++){ offs[i] = (unsigned)total; total += lens[i]; }
    std::vector<unsigned char> blob(total);
    if(fread(blob.data(), 1, total, f) != total){ fprintf(stderr, "bad blob data\n"); return 1; }
    fclose(f);
    printf("packets: %u, payload %zu bytes, %d symbols/packet\n", n, total, SYMS_PER_PKT);

    /* ---- CPU reference (libopus entdec), correctness pass ---- */
    std::vector<unsigned int> cpu_out((size_t)n * WORDS_PER_PKT);
    for(unsigned i = 0; i < n; i++)
        run_schedule_cpu(blob.data() + offs[i], lens[i], cpu_out.data() + (size_t)i * WORDS_PER_PKT);

    /* ---- GPU pass ---- */
    unsigned char *d_blob; unsigned int *d_offs, *d_lens, *d_out;
    size_t out_bytes = (size_t)n * WORDS_PER_PKT * 4;
    CUDA_CHECK(cudaMalloc(&d_blob, total));
    CUDA_CHECK(cudaMalloc(&d_offs, n * 4));
    CUDA_CHECK(cudaMalloc(&d_lens, n * 4));
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
    printf("device buffers: %.1f MB\n", (total + 2.0*n*4 + out_bytes) / 1e6);

    double h2d0 = now_s();
    CUDA_CHECK(cudaMemcpy(d_blob, blob.data(), total, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offs, offs.data(), n * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lens, lens.data(), n * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    double h2d_ms = (now_s() - h2d0) * 1e3;

    int block = 128, grid = (int)((n + block - 1) / block);
    decode_kernel<<<grid, block>>>(d_blob, d_offs, d_lens, n, d_out);  /* warmup */
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<unsigned int> gpu_out((size_t)n * WORDS_PER_PKT);
    double d2h0 = now_s();
    CUDA_CHECK(cudaMemcpy(gpu_out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
    double d2h_ms = (now_s() - d2h0) * 1e3;

    /* ---- bit-exactness: compare EVERY word of EVERY packet ---- */
    unsigned long long bad_words = 0; unsigned bad_pkts = 0; int first_bad = -1;
    for(unsigned i = 0; i < n; i++){
        bool pkt_bad = false;
        for(int w = 0; w < WORDS_PER_PKT; w++){
            size_t idx = (size_t)i * WORDS_PER_PKT + w;
            if(cpu_out[idx] != gpu_out[idx]){ bad_words++; pkt_bad = true; }
        }
        if(pkt_bad){ bad_pkts++; if(first_bad < 0) first_bad = (int)i; }
    }
    printf("BIT-EXACTNESS: %u packets tested, %u mismatching packets, %llu mismatching words -> %s\n",
           n, bad_pkts, bad_words, bad_pkts == 0 ? "PASS" : "FAIL");
    if(first_bad >= 0){
        printf("first bad packet %d (len %u):\n", first_bad, lens[first_bad]);
        for(int w = 0; w < WORDS_PER_PKT; w++){
            size_t idx = (size_t)first_bad * WORDS_PER_PKT + w;
            if(cpu_out[idx] != gpu_out[idx])
                printf("  word %3d: cpu=%u gpu=%u\n", w, cpu_out[idx], gpu_out[idx]);
        }
    }

    /* ---- throughput: GPU kernel ---- */
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0)); CUDA_CHECK(cudaEventCreate(&ev1));
    float best_ms = 1e30f, sum_ms = 0;
    for(int r = 0; r < reps; r++){
        CUDA_CHECK(cudaEventRecord(ev0));
        decode_kernel<<<grid, block>>>(d_blob, d_offs, d_lens, n, d_out);
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        if(ms < best_ms) best_ms = ms;
        sum_ms += ms;
    }
    double syms = (double)n * SYMS_PER_PKT;
    printf("GPU kernel (all %u packets in parallel, 1 thread each): best %.3f ms, avg %.3f ms -> %.1f Msym/s\n",
           n, best_ms, sum_ms / reps, syms / (best_ms / 1e3) / 1e6);
    printf("GPU end-to-end incl. H2D %.3f ms + D2H %.3f ms: %.3f ms -> %.1f Msym/s\n",
           h2d_ms, d2h_ms, best_ms + h2d_ms + d2h_ms,
           syms / ((best_ms + h2d_ms + d2h_ms) / 1e3) / 1e6);

    /* ---- throughput: CPU 1 core ---- */
    std::vector<unsigned int> scratch((size_t)n * WORDS_PER_PKT);
    double best_cpu1 = 1e30;
    for(int r = 0; r < 3; r++){
        double t0 = now_s();
        for(unsigned i = 0; i < n; i++)
            run_schedule_cpu(blob.data() + offs[i], lens[i], scratch.data() + (size_t)i * WORDS_PER_PKT);
        double dt = now_s() - t0;
        if(dt < best_cpu1) best_cpu1 = dt;
    }
    printf("CPU 1 core  (libopus entdec): best %.3f ms -> %.1f Msym/s\n",
           best_cpu1 * 1e3, syms / best_cpu1 / 1e6);

    /* ---- throughput: CPU all cores (OpenMP) ---- */
    int ncores = omp_get_max_threads();
    double best_cpuN = 1e30;
    for(int r = 0; r < 3; r++){
        double t0 = now_s();
        #pragma omp parallel for schedule(static)
        for(int i = 0; i < (int)n; i++)
            run_schedule_cpu(blob.data() + offs[i], lens[i], scratch.data() + (size_t)i * WORDS_PER_PKT);
        double dt = now_s() - t0;
        if(dt < best_cpuN) best_cpuN = dt;
    }
    printf("CPU %d threads (libopus entdec, OpenMP): best %.3f ms -> %.1f Msym/s\n",
           ncores, best_cpuN * 1e3, syms / best_cpuN / 1e6);

    cudaFree(d_blob); cudaFree(d_offs); cudaFree(d_lens); cudaFree(d_out);
    return bad_pkts == 0 ? 0 : 1;
}
