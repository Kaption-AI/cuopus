/* cuopus phase 1: device-side Opus range decoder.
 *
 * Faithful CUDA __device__ port of the libopus 1.5.2 range decoder
 * (celt/entdec.c + celt/entcode.c; RFC 6716 section 4.1). One thread
 * decodes one packet stream. Must remain bit-exact with libopus —
 * validated by harness.cu against the vendored celt/entdec.c on every
 * packet of the test corpus.
 *
 * Port notes (vs upstream):
 *   - functions/types carry a cu_ prefix so a single translation unit can
 *     also link the real libopus entdec for the CPU reference path;
 *   - EC_ILOG uses __clz (defined only for v!=0, same contract as upstream);
 *   - celt_udiv / IMUL32 are plain 32-bit udiv/mul (exact, same results).
 */
#ifndef CUOPUS_ENTDEC_CUH
#define CUOPUS_ENTDEC_CUH

#include <stdint.h>

typedef uint32_t cu_ec_window;   /* EC_WINDOW_SIZE = 32 */

#define CU_EC_WINDOW_SIZE (32)
#define CU_EC_UINT_BITS   (8)
#define CU_EC_SYM_BITS    (8)
#define CU_EC_CODE_BITS   (32)
#define CU_EC_SYM_MAX     ((1U<<CU_EC_SYM_BITS)-1)
#define CU_EC_CODE_TOP    (((uint32_t)1U)<<(CU_EC_CODE_BITS-1))
#define CU_EC_CODE_BOT    (CU_EC_CODE_TOP>>CU_EC_SYM_BITS)
#define CU_EC_CODE_EXTRA  ((CU_EC_CODE_BITS-2)%CU_EC_SYM_BITS+1)

typedef struct {
    const unsigned char *buf;
    uint32_t     storage;
    uint32_t     end_offs;
    cu_ec_window end_window;
    int          nend_bits;
    int          nbits_total;
    uint32_t     offs;
    uint32_t     rng;
    uint32_t     val;
    uint32_t     ext;
    int          rem;
    int          error;
} cu_ec_dec;

__device__ __forceinline__ int cu_ec_ilog(uint32_t v){
    /* upstream contract: undefined for 0 via EC_CLZ; branchless ec_ilog(0)==0 */
    return v ? 32 - __clz((int)v) : 0;
}

__device__ __forceinline__ uint32_t cu_ec_min32(uint32_t a, uint32_t b){
    return a < b ? a : b;
}

__device__ __forceinline__ int cu_ec_read_byte(cu_ec_dec *_this){
    return _this->offs < _this->storage ? _this->buf[_this->offs++] : 0;
}

__device__ __forceinline__ int cu_ec_read_byte_from_end(cu_ec_dec *_this){
    return _this->end_offs < _this->storage ?
        _this->buf[_this->storage - ++(_this->end_offs)] : 0;
}

__device__ __forceinline__ void cu_ec_dec_normalize(cu_ec_dec *_this){
    while(_this->rng <= CU_EC_CODE_BOT){
        int sym;
        _this->nbits_total += CU_EC_SYM_BITS;
        _this->rng <<= CU_EC_SYM_BITS;
        sym = _this->rem;
        _this->rem = cu_ec_read_byte(_this);
        sym = (sym<<CU_EC_SYM_BITS | _this->rem) >> (CU_EC_SYM_BITS-CU_EC_CODE_EXTRA);
        _this->val = ((_this->val<<CU_EC_SYM_BITS) + (CU_EC_SYM_MAX&~sym)) & (CU_EC_CODE_TOP-1);
    }
}

__device__ void cu_ec_dec_init(cu_ec_dec *_this, const unsigned char *_buf, uint32_t _storage){
    _this->buf = _buf;
    _this->storage = _storage;
    _this->end_offs = 0;
    _this->end_window = 0;
    _this->nend_bits = 0;
    _this->nbits_total = CU_EC_CODE_BITS + 1
        - ((CU_EC_CODE_BITS-CU_EC_CODE_EXTRA)/CU_EC_SYM_BITS)*CU_EC_SYM_BITS;
    _this->offs = 0;
    _this->rng = 1U << CU_EC_CODE_EXTRA;
    _this->rem = cu_ec_read_byte(_this);
    _this->val = _this->rng - 1 - (uint32_t)(_this->rem >> (CU_EC_SYM_BITS-CU_EC_CODE_EXTRA));
    _this->error = 0;
    cu_ec_dec_normalize(_this);
}

__device__ unsigned cu_ec_decode(cu_ec_dec *_this, unsigned _ft){
    unsigned s;
    _this->ext = _this->rng / _ft;               /* celt_udiv */
    s = (unsigned)(_this->val / _this->ext);
    return _ft - cu_ec_min32(s+1, _ft);
}

__device__ unsigned cu_ec_decode_bin(cu_ec_dec *_this, unsigned _bits){
    unsigned s;
    _this->ext = _this->rng >> _bits;
    s = (unsigned)(_this->val / _this->ext);
    return (1U<<_bits) - cu_ec_min32(s+1U, 1U<<_bits);
}

__device__ void cu_ec_dec_update(cu_ec_dec *_this, unsigned _fl, unsigned _fh, unsigned _ft){
    uint32_t s;
    s = _this->ext * (_ft - _fh);                /* IMUL32 */
    _this->val -= s;
    _this->rng = _fl > 0 ? _this->ext * (_fh - _fl) : _this->rng - s;
    cu_ec_dec_normalize(_this);
}

/* The probability of having a "one" is 1/(1<<_logp). */
__device__ int cu_ec_dec_bit_logp(cu_ec_dec *_this, unsigned _logp){
    uint32_t r, d, s;
    int ret;
    r = _this->rng;
    d = _this->val;
    s = r >> _logp;
    ret = d < s;
    if(!ret) _this->val = d - s;
    _this->rng = ret ? s : r - s;
    cu_ec_dec_normalize(_this);
    return ret;
}

__device__ int cu_ec_dec_icdf(cu_ec_dec *_this, const unsigned char *_icdf, unsigned _ftb){
    uint32_t r, d, s, t;
    int ret;
    s = _this->rng;
    d = _this->val;
    r = s >> _ftb;
    ret = -1;
    do{
        t = s;
        s = r * (uint32_t)_icdf[++ret];
    } while(d < s);
    _this->val = d - s;
    _this->rng = t - s;
    cu_ec_dec_normalize(_this);
    return ret;
}

__device__ int cu_ec_dec_icdf16(cu_ec_dec *_this, const uint16_t *_icdf, unsigned _ftb){
    uint32_t r, d, s, t;
    int ret;
    s = _this->rng;
    d = _this->val;
    r = s >> _ftb;
    ret = -1;
    do{
        t = s;
        s = r * (uint32_t)_icdf[++ret];
    } while(d < s);
    _this->val = d - s;
    _this->rng = t - s;
    cu_ec_dec_normalize(_this);
    return ret;
}

/* Raw bits, read backwards from the end of the frame. */
__device__ uint32_t cu_ec_dec_bits(cu_ec_dec *_this, unsigned _bits){
    cu_ec_window window;
    int available;
    uint32_t ret;
    window = _this->end_window;
    available = _this->nend_bits;
    if((unsigned)available < _bits){
        do{
            window |= (cu_ec_window)cu_ec_read_byte_from_end(_this) << available;
            available += CU_EC_SYM_BITS;
        } while(available <= CU_EC_WINDOW_SIZE - CU_EC_SYM_BITS);
    }
    ret = (uint32_t)window & (((uint32_t)1<<_bits) - 1U);
    window >>= _bits;
    available -= _bits;
    _this->end_window = window;
    _this->nend_bits = available;
    _this->nbits_total += _bits;
    return ret;
}

__device__ uint32_t cu_ec_dec_uint(cu_ec_dec *_this, uint32_t _ft){
    unsigned ft, s;
    int ftb;
    /* upstream: celt_assert(_ft>1) */
    _ft--;
    ftb = cu_ec_ilog(_ft);
    if(ftb > CU_EC_UINT_BITS){
        uint32_t t;
        ftb -= CU_EC_UINT_BITS;
        ft = (unsigned)(_ft>>ftb) + 1;
        s = cu_ec_decode(_this, ft);
        cu_ec_dec_update(_this, s, s+1, ft);
        t = (uint32_t)s<<ftb | cu_ec_dec_bits(_this, ftb);
        if(t <= _ft) return t;
        _this->error = 1;
        return _ft;
    } else {
        _ft++;
        s = cu_ec_decode(_this, (unsigned)_ft);
        cu_ec_dec_update(_this, s, s+1, (unsigned)_ft);
        return s;
    }
}

__device__ __forceinline__ int cu_ec_tell(cu_ec_dec *_this){
    return _this->nbits_total - cu_ec_ilog(_this->rng);
}

#endif /* CUOPUS_ENTDEC_CUH */
