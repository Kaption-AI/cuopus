/* cuopus phase 1: the fixed symbol schedule, shared verbatim between the
 * CPU (libopus celt/entdec.c) and GPU (cuopus_entdec.cuh) sides.
 *
 * The schedule is not a semantically meaningful Opus parse; it exists to
 * exercise every decoder entry point identically on both sides:
 *   64 x ec_dec_bit_logp(1)      binary symbols
 *   32 x ec_dec_icdf(tab, 7)     icdf table walk (data-dependent length)
 *   32 x ec_dec_uint(2000)       >8-bit path: ec_decode + ec_dec_update + raw bits
 *   16 x ec_dec_bits(11)         raw bits read backwards from the frame end
 * plus final state: ec_tell, rng, error  ->  147 u32 outputs per packet.
 *
 * Include with macros defined:
 *   SCHED_NAME  function name
 *   SCHED_QUAL  qualifiers (__device__ or static)
 *   SCHED_DEC   decoder struct type (ec_dec or cu_ec_dec)
 *   SCHED_EC(f) maps to the real function name (f or cu_##f)
 */
SCHED_QUAL void SCHED_NAME(const unsigned char *buf, unsigned int len,
                           unsigned int *out)
{
    const unsigned char icdf_tab[8] = {120, 112, 96, 64, 32, 16, 8, 0}; /* ftb=7 */
    SCHED_DEC dec;
    int i, k = 0;
    SCHED_EC(ec_dec_init)(&dec, (unsigned char *)buf, len);
    for(i = 0; i < 64; i++)
        out[k++] = (unsigned int)SCHED_EC(ec_dec_bit_logp)(&dec, 1);
    for(i = 0; i < 32; i++)
        out[k++] = (unsigned int)SCHED_EC(ec_dec_icdf)(&dec, icdf_tab, 7);
    for(i = 0; i < 32; i++)
        out[k++] = (unsigned int)SCHED_EC(ec_dec_uint)(&dec, 2000);
    for(i = 0; i < 16; i++)
        out[k++] = (unsigned int)SCHED_EC(ec_dec_bits)(&dec, 11);
    out[k++] = (unsigned int)SCHED_EC(ec_tell)(&dec);
    out[k++] = dec.rng;
    out[k++] = (unsigned int)dec.error;
}
