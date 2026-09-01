#!/usr/bin/env python3
"""cuopus phases 2+3: mechanical DEVICE-macro port of the libopus 1.5.2 SILK
+ CELT decode paths to CUDA device code.

Reads vendor/opus-1.5.2 sources and regenerates device/gen/** with three
mechanical, diffable transformations (diff device/gen/silk/X.c
vendor/opus-1.5.2/silk/X.c shows exactly what changed):

  1. Function definitions/prototypes at column 0 whose name matches
     (silk_|ec_|opus_|celt_)* get a `CUOPUS_DEVFN ` prefix
     (= __device__ under CUOPUS_CUDA, empty otherwise).
  2. File-scope `const`/`static const` data definitions and `extern const`
     declarations get a `CUOPUS_DEVDATA ` prefix (= __device__).
  3. Every file gets `#include "cuopus_device.h"` prepended.

Two files get a wrapper instead of a line rewrite:
  - celt/stack_alloc.h: under CUOPUS_CUDA the ALLOC/SAVE_STACK machinery is
    replaced by a per-thread bump arena in global memory (cuopus_stack.h) —
    device alloca would multiply the per-thread stack reservation across all
    resident threads (GBs on GB202); a caller-provided workspace does not.
  - celt/ecintrin.h: EC_ILOG uses __clz on the device.

Additionally extracts (verbatim, plus the same DEVFN prefix) the packet-parse
helpers from src/opus.c and src/opus_decoder.c into gen/opus_parse_gen.inc,
because compiling those files whole would drag in the CELT decoder.

No DSP math is re-derived anywhere. Run from anywhere:
    python3 device/port_silk.py
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
VENDOR = os.path.join(ROOT, "vendor", "opus-1.5.2")
GEN = os.path.join(HERE, "gen")

# Decode-path closure of silk_Decode (dec_API.c) + the range decoder.
SILK_C = """
    dec_API decode_core decode_frame decode_indices decode_parameters
    decode_pitch decode_pulses decoder_set_fs init_decoder gain_quant
    shell_coder code_signs CNG PLC NLSF2A NLSF_decode NLSF_stabilize
    NLSF_unpack LPC_analysis_filter LPC_fit LPC_inv_pred_gain
    bwexpander bwexpander_32 lin2log log2lin pitch_est_tables
    resampler resampler_down2 resampler_down2_3 resampler_private_AR2
    resampler_private_down_FIR resampler_private_IIR_FIR
    resampler_private_up2_HQ resampler_rom sort stereo_decode_pred
    stereo_MS_to_LR sum_sqr_shift table_LSF_cos tables_gain tables_LTP
    tables_NLSF_CB_NB_MB tables_NLSF_CB_WB tables_other tables_pitch_lag
    tables_pulses_per_block
""".split()
CELT_C = ["entdec", "entcode", "entenc"]  # entenc: only referenced by dead
# encoder halves of shared silk files (shell_coder, stereo_decode_pred);
# compiled so the single TU links.

# Phase 3: decode-path closure of celt_decode_with_ec (celt_decoder.c) +
# everything opus_decode_frame needs from the CELT side. Order matters only
# in cuopus_celt_all.cuh, not here.
CELT_DEC_C = """
    mathops laplace cwrs vq pitch celt_lpc kiss_fft mdct modes
    quant_bands rate bands celt celt_decoder
""".split()

CELT_H = """arch.h entcode.h entdec.h entenc.h ecintrin.h os_support.h
    cpu_support.h stack_alloc.h fixed_generic.h mfrngcod.h celt_lpc.h
    bands.h celt.h cwrs.h float_cast.h kiss_fft.h _kiss_fft_guts.h
    laplace.h mathops.h mdct.h modes.h pitch.h quant_bands.h rate.h
    static_modes_fixed.h vq.h""".split()
INCLUDE_H = ["opus_types.h", "opus_defines.h", "opus_custom.h"]

TYPES = (r"(?:void|int|char|short|long|unsigned|float|double|size_t"
         r"|opus_int|opus_int8|opus_int16|opus_int32|opus_int64"
         r"|opus_uint|opus_uint8|opus_uint16|opus_uint32"
         r"|opus_val16|opus_val32|ec_uint32)")
FUNC_RE = re.compile(
    r"^(static\s+)?(OPUS_INLINE\s+)?(const\s+)?(unsigned\s+|signed\s+)?"
    + TYPES + r"(\s+\**|\s*\*+\s*|\s+)((silk|ec|opus|celt)_\w+)\s*\(")
# static helpers without a library prefix (decode_split, encode_split,
# LPC_inverse_pred_gain_QA_c, combine_pulses, ...)
STATIC_FUNC_RE = re.compile(
    r"^static\s+(OPUS_INLINE\s+)?(const\s+)?(unsigned\s+|signed\s+)?"
    + TYPES + r"(\s+\**|\s*\*+\s*|\s+)(\w+)\s*\(")
# Phase 3: celt decode files have many column-0 functions WITHOUT a library
# prefix (comb_filter, haar1, quant_all_bands, _celt_lpc, tf_decode, ...) and
# defs that start with export macros (OPUS_CUSTOM_NOSTATIC int
# opus_custom_decoder_init(...)). Generic column-0 "type name(" matcher,
# applied to celt/** and include/opus_custom.h only; the silk transform is
# unchanged (gen/silk output stays byte-identical to phase 2).
GENERIC_FUNC_RE = re.compile(
    r"^(?!typedef\b|return\b|else\b|goto\b|case\b|extern\b)"
    r"(static\s+)?"
    r"(OPUS_CUSTOM_NOSTATIC\s+|OPUS_CUSTOM_EXPORT_STATIC\s+|OPUS_CUSTOM_EXPORT\s+)?"
    r"(OPUS_WARN_UNUSED_RESULT\s+)?(OPUS_INLINE\s+)?(const\s+)?(struct\s+)?"
    r"(unsigned\s+|signed\s+)?[A-Za-z_]\w*(\s+\**|\s*\*+\s*|\s+)[A-Za-z_]\w*\s*\(")

EXTERN_RE = re.compile(r"^extern\s+const\b")
DATA_RE = re.compile(r"^(static\s+)?(silk_DWORD_ALIGN\s+)?const\s+\w")

# Perf pass (2026-08-30): selective re-inlining of hot leaf functions.
# The default for .c-file functions stays __noinline__ (mirrors libopus TU
# boundaries; keeps cicc from exploding — the naive full-inline build ran
# >25 min). The functions below are the hottest small leaves: the range
# decoder (called per SYMBOL, thousands of times per packet) and small
# fixed-point helpers on per-sample paths. Marking only these
# __forceinline__ keeps the compile at ~seconds while removing ABI
# call/spill overhead from the innermost loops. Gate: phase-3 bit-exactness
# must stay 0 mismatches (inlining cannot change fixed-point arithmetic).
HOT_FNS = {
    # celt/entdec.c — range decoder, per-symbol
    "ec_read_byte", "ec_read_byte_from_end", "ec_dec_normalize",
    "ec_decode", "ec_decode_bin", "ec_dec_update", "ec_dec_bit_logp",
    "ec_dec_icdf", "ec_dec_icdf16", "ec_dec_bits", "ec_dec_uint",
    # celt/entcode.c — tell/ilog helpers used by allocation + laplace
    "ec_tell_frac", "ec_ilog",
    # celt/laplace.c — per-band energy decode
    "ec_laplace_decode", "ec_laplace_get_freq1",
    # celt/mathops.c + celt_lpc small leaves
    "celt_sqrt", "celt_rcp", "celt_cos_norm", "_celt_cos_pi_2",
    # silk small per-sample/per-coef leaves
    "silk_lin2log", "silk_log2lin", "silk_bwexpander", "silk_bwexpander_32",
}
HOT_NAME_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")

# Exact-string patches applied after the line rewrite (device has no libc
# memmove; cuopus_memmove is defined in cuopus_device.h).
PATCHES = {
    "celt/pitch.c": [(
        """/* Pure C implementation. */
#ifdef FIXED_POINT
opus_val32
#else
void
#endif
celt_pitch_xcorr_c(""",
        """/* Pure C implementation. */
#ifdef FIXED_POINT
CUOPUS_DEVFN_NOINL opus_val32
#else
CUOPUS_DEVFN_NOINL void
#endif
celt_pitch_xcorr_c(""",
    )],
    "silk/SigProc_FIX.h": [(
        "#define silk_memmove(dest, src, size)       memmove((dest), (src), (size))",
        "#ifdef CUOPUS_CUDA\n"
        "#define silk_memmove(dest, src, size)       cuopus_memmove((dest), (src), (size))\n"
        "#else\n"
        "#define silk_memmove(dest, src, size)       memmove((dest), (src), (size))\n"
        "#endif",
    )],
    "celt/os_support.h": [(
        "#define OPUS_MOVE(dst, src, n) (memmove((dst), (src), (n)*sizeof(*(dst)) + 0*((dst)-(src)) ))",
        "#ifdef CUOPUS_CUDA\n"
        "#define OPUS_MOVE(dst, src, n) (cuopus_memmove((dst), (src), (n)*sizeof(*(dst)) + 0*((dst)-(src)) ))\n"
        "#else\n"
        "#define OPUS_MOVE(dst, src, n) (memmove((dst), (src), (n)*sizeof(*(dst)) + 0*((dst)-(src)) ))\n"
        "#endif",
    )],
}


# ---------------------------------------------------------------------------
# Phase 3 structural patches (applied to the RAW vendor text, before the line
# transform, so the inserted code gets the same mechanical treatment).
# ---------------------------------------------------------------------------

# bands.c quant_partition recurses on band splits (depth bounded by LM <= 3:
# every split halves N and decrements LM, no split at LM == -1 => depth <= 5).
# Device threads must not recurse (the driver cannot size the per-thread stack
# statically; an explicit cudaLimitStackSize reserves GBs across all resident
# threads -- the phase-2 lesson). The vendor function is kept under
# CUOPUS_QP_RECURSIVE (never defined) and replaced by an explicit-frame-stack
# version with identical control flow, arithmetic and entropy-coder call
# order: each recursive call becomes a pushed frame; the rebalance code that
# runs after each vendor call runs in the matching phase.
QP_ITERATIVE = r"""
#define CUOPUS_QP_MAX_DEPTH 10  /* provable depth <= 5 (LM 3,2,1,0,-1) */
struct cuopus_qp_frame {
   celt_norm *X;
   celt_norm *lowband;
   celt_norm *Y;
   celt_norm *next_lowband2;
   int N, b, B, B0, LM, fill;
   opus_val16 gain, mid, side;
   int mbits, sbits, itheta;
   opus_int32 rebalance;
   int order;    /* 0: X (mid) first, 1: Y (side) first  [vendor mbits>=sbits] */
   int phase;    /* 0 enter | 1 first child done | 2 second child done */
   unsigned cm;
};
static unsigned quant_partition(struct band_ctx *ctx, celt_norm *X,
      int N, int b, int B, celt_norm *lowband,
      int LM,
      opus_val16 gain, int fill)
{
   struct cuopus_qp_frame stk[CUOPUS_QP_MAX_DEPTH];
   int sp;
   unsigned retcm = 0;
   int encode;
   const CELTMode *m;
   int i;
   int spread;
   ec_ctx *ec;

   encode = ctx->encode;
   m = ctx->m;
   i = ctx->i;
   spread = ctx->spread;
   ec = ctx->ec;

   stk[0].X = X; stk[0].lowband = lowband;
   stk[0].N = N; stk[0].b = b; stk[0].B = B; stk[0].LM = LM;
   stk[0].gain = gain; stk[0].fill = fill;
   stk[0].phase = 0; stk[0].cm = 0;
   sp = 1;

   while (sp > 0)
   {
      struct cuopus_qp_frame *f = &stk[sp-1];
      if (f->phase == 0)
      {
         const unsigned char *cache;
         /* If we need 1.5 more bit than we can produce, split the band in two. */
         cache = m->cache.bits + m->cache.index[(f->LM+1)*m->nbEBands+i];
         if (f->LM != -1 && f->b > cache[cache[0]]+12 && f->N>2)
         {
            int mbits, sbits, delta;
            int itheta;
            int qalloc;
            struct split_ctx sctx;
            celt_norm *next_lowband2=NULL;
            struct cuopus_qp_frame *g;

            f->B0 = f->B;                      /* vendor: int B0=B at entry */
            f->N >>= 1;
            f->Y = f->X+f->N;
            f->LM -= 1;
            if (f->B==1)
               f->fill = (f->fill&1)|(f->fill<<1);
            f->B = (f->B+1)>>1;

            compute_theta(ctx, &sctx, f->X, f->Y, f->N, &f->b, f->B, f->B0,
                  f->LM, 0, &f->fill);
            itheta = sctx.itheta;
            delta = sctx.delta;
            qalloc = sctx.qalloc;
            f->itheta = itheta;
#ifdef FIXED_POINT
            f->mid = sctx.imid;
            f->side = sctx.iside;
#else
            f->mid = (1.f/32768)*sctx.imid;
            f->side = (1.f/32768)*sctx.iside;
#endif

            /* Give more bits to low-energy MDCTs than they would otherwise deserve */
            if (f->B0>1 && (itheta&0x3fff))
            {
               if (itheta > 8192)
                  /* Rough approximation for pre-echo masking */
                  delta -= delta>>(4-f->LM);
               else
                  /* Corresponds to a forward-masking slope of 1.5 dB per 10 ms */
                  delta = IMIN(0, delta + (f->N<<BITRES>>(5-f->LM)));
            }
            mbits = IMAX(0, IMIN(f->b, (f->b-delta)/2));
            sbits = f->b-mbits;
            f->mbits = mbits;
            f->sbits = sbits;
            ctx->remaining_bits -= qalloc;

            if (f->lowband)
               next_lowband2 = f->lowband+f->N; /* >32-bit split case */
            f->next_lowband2 = next_lowband2;

            f->rebalance = ctx->remaining_bits;
            f->order = (mbits >= sbits) ? 0 : 1;
            f->phase = 1;
            g = &stk[sp++];
            if (f->order == 0)
            {  /* vendor: cm = quant_partition(ctx, X, N, mbits, B, lowband, LM,
                                MULT16_16_P15(gain,mid), fill); */
               g->X = f->X; g->lowband = f->lowband;
               g->N = f->N; g->b = f->mbits; g->B = f->B; g->LM = f->LM;
               g->gain = MULT16_16_P15(f->gain, f->mid); g->fill = f->fill;
            } else {
               /* vendor: cm = quant_partition(ctx, Y, N, sbits, B, next_lowband2,
                                LM, MULT16_16_P15(gain,side), fill>>B)<<(B0>>1); */
               g->X = f->Y; g->lowband = f->next_lowband2;
               g->N = f->N; g->b = f->sbits; g->B = f->B; g->LM = f->LM;
               g->gain = MULT16_16_P15(f->gain, f->side); g->fill = f->fill>>f->B;
            }
            g->phase = 0; g->cm = 0;
            continue;
         } else {
            /* This is the basic no-split case (verbatim, f-> locals) */
            int q;
            int curr_bits;
            unsigned cm=0;
            q = bits2pulses(m, i, f->LM, f->b);
            curr_bits = pulses2bits(m, i, f->LM, q);
            ctx->remaining_bits -= curr_bits;

            /* Ensures we can never bust the budget */
            while (ctx->remaining_bits < 0 && q > 0)
            {
               ctx->remaining_bits += curr_bits;
               q--;
               curr_bits = pulses2bits(m, i, f->LM, q);
               ctx->remaining_bits -= curr_bits;
            }

            if (q!=0)
            {
               int K = get_pulses(q);

               /* Finally do the actual quantization */
               if (encode)
               {
                  cm = alg_quant(f->X, f->N, K, spread, f->B, ec, f->gain,
                        ctx->resynth, ctx->arch);
               } else {
                  cm = alg_unquant(f->X, f->N, K, spread, f->B, ec, f->gain);
               }
            } else {
               /* If there's no pulse, fill the band anyway */
               int j;
               if (ctx->resynth)
               {
                  unsigned cm_mask;
                  /* B can be as large as 16, so this shift might overflow an int on a
                     16-bit platform; use a long to get defined behavior.*/
                  cm_mask = (unsigned)(1UL<<f->B)-1;
                  f->fill &= cm_mask;
                  if (!f->fill)
                  {
                     OPUS_CLEAR(f->X, f->N);
                  } else {
                     if (f->lowband == NULL)
                     {
                        /* Noise */
                        for (j=0;j<f->N;j++)
                        {
                           ctx->seed = celt_lcg_rand(ctx->seed);
                           f->X[j] = (celt_norm)((opus_int32)ctx->seed>>20);
                        }
                        cm = cm_mask;
                     } else {
                        /* Folded spectrum */
                        for (j=0;j<f->N;j++)
                        {
                           opus_val16 tmp;
                           ctx->seed = celt_lcg_rand(ctx->seed);
                           /* About 48 dB below the "normal" folding level */
                           tmp = QCONST16(1.0f/256, 10);
                           tmp = (ctx->seed)&0x8000 ? tmp : -tmp;
                           f->X[j] = f->lowband[j]+tmp;
                        }
                        cm = f->fill;
                     }
                     renormalise_vector(f->X, f->N, f->gain, ctx->arch);
                  }
               }
            }
            retcm = cm;
            sp--;
            continue;
         }
      }
      else if (f->phase == 1)
      {
         opus_int32 rebalance;
         struct cuopus_qp_frame *g;
         g = &stk[sp++];
         if (f->order == 0)
         {
            f->cm = retcm;
            rebalance = f->mbits - (f->rebalance-ctx->remaining_bits);
            if (rebalance > 3<<BITRES && f->itheta!=0)
               f->sbits += rebalance - (3<<BITRES);
            /* vendor: cm |= quant_partition(ctx, Y, N, sbits, B, next_lowband2,
                             LM, MULT16_16_P15(gain,side), fill>>B)<<(B0>>1); */
            g->X = f->Y; g->lowband = f->next_lowband2;
            g->N = f->N; g->b = f->sbits; g->B = f->B; g->LM = f->LM;
            g->gain = MULT16_16_P15(f->gain, f->side); g->fill = f->fill>>f->B;
         } else {
            f->cm = retcm<<(f->B0>>1);
            rebalance = f->sbits - (f->rebalance-ctx->remaining_bits);
            if (rebalance > 3<<BITRES && f->itheta!=16384)
               f->mbits += rebalance - (3<<BITRES);
            /* vendor: cm |= quant_partition(ctx, X, N, mbits, B, lowband, LM,
                             MULT16_16_P15(gain,mid), fill); */
            g->X = f->X; g->lowband = f->lowband;
            g->N = f->N; g->b = f->mbits; g->B = f->B; g->LM = f->LM;
            g->gain = MULT16_16_P15(f->gain, f->mid); g->fill = f->fill;
         }
         f->phase = 2;
         g->phase = 0; g->cm = 0;
         continue;
      }
      else /* phase == 2: second child returned */
      {
         if (f->order == 0)
            f->cm |= retcm<<(f->B0>>1);
         else
            f->cm |= retcm;
         retcm = f->cm;
         sp--;
      }
   }
   return retcm;
}
"""


def bands_hoist_recursion(text):
    """Wrap the vendor recursive quant_partition under CUOPUS_QP_RECURSIVE
    (never defined) and insert the explicit-stack version in its place."""
    marker = "static unsigned quant_partition(struct band_ctx *ctx, celt_norm *X,"
    i0 = text.index(marker)
    j = text.index("{", i0)
    depth = 0
    k = j
    while True:
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                break
        k += 1
    end = k + 1
    original = text[i0:end]
    repl = ("#ifdef CUOPUS_QP_RECURSIVE  /* vendor original (recursion "
            "forbidden on device, see port_silk.py) */\n"
            + original
            + "\n#else  /* cuopus phase 3: explicit-frame-stack version */\n"
            + QP_ITERATIVE
            + "#endif /* CUOPUS_QP_RECURSIVE */")
    return text[:i0] + repl + text[end:]


CELT_RESET_HELPER = r"""
/* cuopus phase 3: non-vararg replacement for
   opus_custom_decoder_ctl(st, OPUS_RESET_STATE) -- CUDA device code cannot
   compile varargs functions. Body identical to the ctl case below (which is
   compiled out under CUOPUS_CUDA). Typed helpers for the other ctl requests
   used by opus_decode_frame live in cuopus_opus.cuh. */
int cuopus_celt_reset_state(CELTDecoder * OPUS_RESTRICT st)
{
   int i;
   opus_val16 *lpc, *oldBandE, *oldLogE, *oldLogE2;
   lpc = (opus_val16*)(st->_decode_mem+(DECODE_BUFFER_SIZE+st->overlap)*st->channels);
   oldBandE = lpc+st->channels*CELT_LPC_ORDER;
   oldLogE = oldBandE + 2*st->mode->nbEBands;
   oldLogE2 = oldLogE + 2*st->mode->nbEBands;
   OPUS_CLEAR((char*)&st->DECODER_RESET_START,
         opus_custom_decoder_get_size(st->mode, st->channels)-
         ((char*)&st->DECODER_RESET_START - (char*)st));
   for (i=0;i<2*st->mode->nbEBands;i++)
      oldLogE[i]=oldLogE2[i]=-QCONST16(28.f,DB_SHIFT);
   st->skip_plc = 1;
   return OPUS_OK;
}

"""


def celt_decoder_patches(text):
    """Insert the reset helper, redirect the one internal ctl call, and
    compile the varargs ctl out under CUOPUS_CUDA."""
    a = "OPUS_CUSTOM_NOSTATIC int opus_custom_decoder_init(CELTDecoder *st, const CELTMode *mode, int channels)"
    assert a in text
    text = text.replace(a, CELT_RESET_HELPER + a, 1)
    b = "   opus_custom_decoder_ctl(st, OPUS_RESET_STATE);"
    assert b in text
    text = text.replace(b, "   cuopus_celt_reset_state(st);   /* cuopus: non-vararg, see helper above */", 1)
    c = "int opus_custom_decoder_ctl(CELTDecoder * OPUS_RESTRICT st, int request, ...)"
    assert c in text
    text = text.replace(c, "#ifndef CUOPUS_CUDA /* varargs cannot compile as device code; typed helpers in cuopus_opus.cuh */\n" + c, 1)
    d = "  return OPUS_UNIMPLEMENTED;\n}"
    assert d in text
    text = text.replace(d, d + "\n#endif /* !CUOPUS_CUDA (cuopus) */", 1)
    return text


STRUCT_PATCHES = {
    "celt/bands.c": bands_hoist_recursion,
    "celt/celt_decoder.c": celt_decoder_patches,
}

HEADER = '#include "cuopus_device.h"\n'

STACK_WRAP_TOP = """#ifdef CUOPUS_CUDA
#include "cuopus_stack.h"  /* per-thread global-memory arena replaces alloca */
#else
"""
ECINTRIN_WRAP_TOP = """#if defined(CUOPUS_CUDA) && defined(__CUDACC__)
#ifndef _ecintrin_H
#define _ecintrin_H (1)
#define EC_ILOG(_x) (32 - __clz((int)(_x)))
#define EC_MINI(_a,_b)      ((_a)+(((_b)-(_a))&-((_b)<(_a))))
#endif
#else
"""
WRAP_BOT = "#endif /* !CUOPUS_CUDA (vendor original above) */\n"


def is_celt_generic(fname):
    f = fname.replace(os.sep, "/")
    return f.startswith("celt/") or f == "include/opus_custom.h"


def transform(text, fname):
    # Out-of-line functions in .c files are marked __noinline__: it mirrors
    # libopus's real translation-unit boundaries and keeps nvcc's inliner from
    # exploding on the merged call graph (cicc ran >25 min at -O2/-O3 without
    # it). OPUS_INLINE helpers and header inlines stay inlinable.
    is_c = fname.endswith(".c")
    generic = is_celt_generic(fname)
    out = []
    for line in text.split("\n"):
        if FUNC_RE.match(line) or STATIC_FUNC_RE.match(line) \
           or (generic and GENERIC_FUNC_RE.match(line)):
            noinl = (is_c and "OPUS_INLINE" not in line
                     and "OPUS_CUSTOM_NOSTATIC" not in line
                     and not line.rstrip().endswith(";"))
            m = HOT_NAME_RE.search(line)
            hot = (noinl and not line.rstrip().endswith(";")
                   and m is not None and m.group(1) in HOT_FNS)
            pfx = ("CUOPUS_DEVFN_HOT " if hot
                   else "CUOPUS_DEVFN_NOINL " if noinl else "CUOPUS_DEVFN ")
            out.append(pfx + line)
            continue
        if EXTERN_RE.match(line):
            out.append("extern CUOPUS_DEVDATA" + line[len("extern"):])
            continue
        if DATA_RE.match(line) and "(" not in line.split("=")[0].split("[")[0] \
           and ("=" in line or line.rstrip().endswith(";")):
            out.append("CUOPUS_DEVDATA " + line)
            continue
        out.append(line)
    return HEADER + "\n".join(out)


def emit(rel, wrap=None):
    src = os.path.join(VENDOR, rel)
    dst = os.path.join(GEN, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(src) as f:
        text = f.read()
    sp = STRUCT_PATCHES.get(rel.replace(os.sep, "/"))
    if sp is not None:
        text = sp(text)
    if wrap is not None:
        body = wrap + text + ("\n" if not text.endswith("\n") else "") + WRAP_BOT
        body = HEADER + body
    else:
        body = transform(text, rel)
    for old, new in PATCHES.get(rel.replace(os.sep, "/"), []):
        assert old in body, "patch anchor missing in %s" % rel
        body = body.replace(old, new)
    with open(dst, "w") as f:
        f.write(body)
    return dst


def extract_functions(path, names):
    """Extract complete top-level functions by their definition-line regex."""
    with open(path) as f:
        lines = f.read().split("\n")
    chunks = []
    i = 0
    while i < len(lines):
        line = lines[i]
        hit = None
        for nm in names:
            if re.match(r"^(static\s+)?(OPUS_INLINE\s+)?\w[\w\s\*]*\b%s\s*\(" % nm, line) \
               and not line.rstrip().endswith(";"):
                hit = nm
                break
        if hit:
            start = i
            depth = 0
            seen = False
            while i < len(lines):
                depth += lines[i].count("{") - lines[i].count("}")
                if "{" in lines[i]:
                    seen = True
                if seen and depth == 0:
                    break
                i += 1
            chunk = "\n".join(lines[start:i + 1])
            chunks.append((hit, chunk))
        i += 1
    return chunks


def main():
    for rel in INCLUDE_H:
        emit(os.path.join("include", rel))
    for rel in CELT_H:
        if rel == "stack_alloc.h":
            emit(os.path.join("celt", rel), wrap=STACK_WRAP_TOP)
        elif rel == "ecintrin.h":
            emit(os.path.join("celt", rel), wrap=ECINTRIN_WRAP_TOP)
        else:
            emit(os.path.join("celt", rel))
    import glob
    for src in sorted(glob.glob(os.path.join(VENDOR, "silk", "*.h"))):
        emit(os.path.join("silk", os.path.basename(src)))
    for base in CELT_C:
        emit(os.path.join("celt", base + ".c"))
    for base in CELT_DEC_C:
        emit(os.path.join("celt", base + ".c"))
    for base in SILK_C:
        emit(os.path.join("silk", base + ".c"))

    # Verbatim extraction of the packet-parse helpers (mode/bandwidth/frame
    # dispatch is final here per the phase-2 scope boundary).
    parts = []
    parts.append("/* GENERATED by port_silk.py — verbatim functions from "
                 "libopus 1.5.2\n   src/opus.c and src/opus_decoder.c, "
                 "prefixed for device compilation. */\n")
    for path, names in (
            (os.path.join(VENDOR, "src", "opus.c"),
             ["parse_size", "opus_packet_get_samples_per_frame",
              "opus_packet_parse_impl"]),
            (os.path.join(VENDOR, "src", "opus_decoder.c"),
             ["opus_packet_get_mode", "opus_packet_get_bandwidth",
              "opus_packet_get_nb_channels", "smooth_fade"])):
        for nm, chunk in extract_functions(path, names):
            # make everything static __device__ in our single TU
            chunk = re.sub(r"^(static\s+)?", "CUOPUS_DEVFN static ", chunk, count=1)
            parts.append("/* from %s */\n%s\n" % (os.path.relpath(path, VENDOR), chunk))
    with open(os.path.join(GEN, "opus_parse_gen.inc"), "w") as f:
        f.write("\n".join(parts))

    n = sum(len(files) for _, _, files in os.walk(GEN))
    print("port_silk.py: generated %d files under %s" % (n, GEN))


if __name__ == "__main__":
    sys.exit(main())
