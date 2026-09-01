/* cuopus phase 3: device-side top-level Opus packet dispatch — FULL decode
 * (SILK + CELT + hybrid + transition + redundancy).
 *
 * Line-faithful specialization of libopus 1.5.2 src/opus_decoder.c
 * (opus_decoder_init / opus_decode_frame / opus_decode_native) for the
 * FIXED_POINT build. Phase 2 shipped this file with every CELT-needing
 * branch replaced by a handled error; phase 3 fills them all in with the
 * ported CELT decoder (cuopus_celt_all.cuh). The CUOPUS_ERR_* codes are kept
 * only so the harnesses still compile (nothing returns them any more).
 *
 * Deviations from a pure copy (each marked "cuopus:" in place):
 *  1. The decoder object embeds silk + a fixed-size CELT blob instead of the
 *     offset-based variable-size allocation (device has no opus_alloc);
 *     an init-time check validates the blob against
 *     opus_custom_decoder_get_size.
 *  2. celt_decoder_ctl is varargs and cannot compile as device code. The
 *     typed helpers below mirror the ctl cases field-for-field;
 *     cuopus_celt_reset_state (patched into gen/celt/celt_decoder.c) is the
 *     OPUS_RESET_STATE case.
 *  3. No recursion (device rule, see PHASE2_RESULTS.md): the two recursive
 *     opus_decode_frame(st, NULL, ...) uses — PLC >20 ms splitting and the
 *     5 ms transition frame — are served by cu_opus_plc_frame, a dedicated
 *     copy of the data==NULL specialization of opus_decode_frame (itself
 *     recursion-free). Chunking order and state updates are identical.
 *
 * Must be included AFTER cuopus_silk_all.cuh AND cuopus_celt_all.cuh.
 */
#ifndef CUOPUS_OPUS_CUH
#define CUOPUS_OPUS_CUH

#define MODE_SILK_ONLY 1000   /* [opus_private.h] */
#define MODE_HYBRID    1001
#define MODE_CELT_ONLY 1002

/* phase-2 compatibility only — no longer returned anywhere */
#define CUOPUS_ERR_CELT        (-101)
#define CUOPUS_ERR_REDUNDANCY  (-102)
#define CUOPUS_ERR_TRANSITION  (-103)

#include "opus_parse_gen.inc"

/* ------------------------------------------------------------------ */
/* cuopus: typed replacements for the celt_decoder_ctl requests used by
 * opus_decode_frame (varargs cannot compile as device code). Each body is
 * the corresponding case of opus_custom_decoder_ctl [celt_decoder.c],
 * including its validation. OPUS_RESET_STATE is cuopus_celt_reset_state,
 * patched into gen/celt/celt_decoder.c next to the ctl it mirrors.        */

CUOPUS_DEVFN static int cu_celt_set_start_band(CELTDecoder *cst, opus_int32 value)
{
   if (value < 0 || value >= cst->mode->nbEBands)
      return OPUS_BAD_ARG;
   cst->start = value;
   return OPUS_OK;
}

CUOPUS_DEVFN static int cu_celt_set_end_band(CELTDecoder *cst, opus_int32 value)
{
   if (value < 1 || value > cst->mode->nbEBands)
      return OPUS_BAD_ARG;
   cst->end = value;
   return OPUS_OK;
}

CUOPUS_DEVFN static int cu_celt_set_channels(CELTDecoder *cst, opus_int32 value)
{
   if (value < 1 || value > 2)
      return OPUS_BAD_ARG;
   cst->stream_channels = value;
   return OPUS_OK;
}

CUOPUS_DEVFN static int cu_celt_set_signalling(CELTDecoder *cst, opus_int32 value)
{
   cst->signalling = value;
   return OPUS_OK;
}

CUOPUS_DEVFN static opus_uint32 cu_celt_get_final_range(const CELTDecoder *cst)
{
   return cst->rng;
}

CUOPUS_DEVFN static const CELTMode *cu_celt_get_mode(const CELTDecoder *cst)
{
   return cst->mode;
}

/* ------------------------------------------------------------------ */

/* cuopus: fixed-size CELT decoder blob — worst case (stereo) of
 * opus_custom_decoder_get_size(mode48000_960_120, 2); overlap of the static
 * 48 kHz mode is 120. Checked against the real get_size at init. */
#define CUOPUS_CELT_OVERLAP 120
#define CUOPUS_CELT_NBEBANDS 21
#define CUOPUS_CELT_DEC_BLOB (sizeof(struct OpusCustomDecoder) \
      + (2*(DECODE_BUFFER_SIZE+CUOPUS_CELT_OVERLAP)-1)*sizeof(celt_sig) \
      + 2*CELT_LPC_ORDER*sizeof(opus_val16) \
      + 4*2*CUOPUS_CELT_NBEBANDS*sizeof(opus_val16))

typedef struct {
    silk_decoder          silk;        /* embedded, replaces silk_dec_offset */
    silk_DecControlStruct DecControl;
    int          channels;
    opus_int32   Fs;
    int          decode_gain;
    int          complexity;
    int          arch;
    int          stream_channels;
    int          bandwidth;
    int          mode;
    int          prev_mode;
    int          frame_size;
    int          prev_redundancy;
    int          last_packet_duration;
    opus_uint32  rangeFinal;
    /* embedded CELT decoder (variable-tail vendor struct in a fixed blob) */
    union { double align8; char blob[CUOPUS_CELT_DEC_BLOB]; } celt;
} cu_opus_decoder;

/* [opus_decoder.c opus_decoder_init] */
CUOPUS_DEVFN_NOINL static int cu_opus_decoder_init(cu_opus_decoder *st, opus_int32 Fs, int channels)
{
    if ((Fs != 48000 && Fs != 24000 && Fs != 16000 && Fs != 12000 && Fs != 8000)
        || (channels != 1 && channels != 2))
        return OPUS_BAD_ARG;
    memset(st, 0, sizeof(*st));
    st->stream_channels = st->channels = channels;
    st->complexity = 0;
    st->Fs = Fs;
    st->DecControl.API_sampleRate = st->Fs;
    st->DecControl.nChannelsAPI = st->channels;
    if (silk_InitDecoder(&st->silk))
        return OPUS_INTERNAL_ERROR;
    /* [opus_decoder.c] celt_decoder_init(celt_dec, Fs, channels) + SET_SIGNALLING(0) */
    {
        const CELTMode *mode = opus_custom_mode_create(48000, 960, NULL);
        CELTDecoder *celt_dec = (CELTDecoder *)st->celt.blob;
        if (mode == NULL
            || opus_custom_decoder_get_size(mode, channels) > (int)sizeof(st->celt))
            return OPUS_INTERNAL_ERROR;    /* cuopus: blob undersized (cannot happen for 48k/960) */
        if (celt_decoder_init(celt_dec, Fs, channels) != OPUS_OK)
            return OPUS_INTERNAL_ERROR;
        cu_celt_set_signalling(celt_dec, 0);
    }
    st->prev_mode = 0;
    st->frame_size = Fs / 400;
    st->arch = 0;               /* cuopus: pinned to the plain-C kernels */
    return OPUS_OK;
}

/* cuopus: the data==NULL specialization of [opus_decoder.c opus_decode_frame]
 * (PLC / transition frame), used where the vendor recurses. frame_size is
 * <= F20 by construction at every call site (the >F20 split loops live in
 * the callers), so this function never needs to recurse. Every branch below
 * mirrors the vendor function with data==NULL, len=0, decode_fec=0 constant-
 * folded; redundancy/transition cannot occur (both require data != NULL). */
CUOPUS_DEVFN_NOINL static int cu_opus_plc_frame(cu_opus_decoder *st, opus_val16 *pcm, int frame_size)
{
    void *silk_dec;
    CELTDecoder *celt_dec;
    int i, silk_ret = 0, celt_ret = 0;
    ec_dec dec;                 /* untouched on the PLC path, as in vendor */
    opus_int32 silk_frame_size;
    int pcm_silk_size;
    VARDECL(opus_int16, pcm_silk);
    int audiosize;
    int mode;
    int start_band;
    int redundancy = 0;
    int celt_to_silk = 0;
    int F2_5, F5, F10, F20;
    int celt_accum;
    ALLOC_STACK;

    silk_dec = &st->silk;
    celt_dec = (CELTDecoder *)st->celt.blob;
    F20 = st->Fs / 50;
    F10 = F20 >> 1;
    F5 = F10 >> 1;
    F2_5 = F5 >> 1;
    if (frame_size < F2_5)
    {
        RESTORE_STACK;
        return OPUS_BUFFER_TOO_SMALL;
    }

    audiosize = frame_size;
    /* Run PLC using last used mode (CELT if we ended with CELT redundancy) */
    mode = st->prev_redundancy ? MODE_CELT_ONLY : st->prev_mode;

    if (mode == 0)
    {
        /* If we haven't got any packet yet, all we can do is return zeros */
        for (i = 0; i < audiosize * st->channels; i++)
            pcm[i] = 0;
        RESTORE_STACK;
        return audiosize;
    }

    /* Avoids trying to run the PLC on sizes other than 2.5 (CELT), 5 (CELT),
       10, or 20 (e.g. 12.5 or 30 ms). Caller guarantees audiosize <= F20
       (the vendor recursion for >F20 is the callers' split loop). */
    if (audiosize > F20)
    {
        RESTORE_STACK;
        return OPUS_INTERNAL_ERROR;   /* cuopus: unreachable, callers split */
    } else if (audiosize < F20)
    {
        if (audiosize > F10)
            audiosize = F10;
        else if (mode != MODE_SILK_ONLY && audiosize > F5 && audiosize < F10)
            audiosize = F5;
    }

    /* FIXED_POINT: CELT accumulates on top of the SILK PCM buffer */
    celt_accum = (mode != MODE_CELT_ONLY) && (frame_size >= F10);

    /* transition impossible: requires data != NULL */
    frame_size = audiosize;   /* vendor: audiosize <= frame_size always holds here */

    /* Don't allocate any memory when in CELT-only mode */
    pcm_silk_size = (mode != MODE_CELT_ONLY && !celt_accum) ? IMAX(F10, frame_size) * st->channels : ALLOC_NONE;
    ALLOC(pcm_silk, pcm_silk_size, opus_int16);

    /* SILK processing */
    if (mode != MODE_CELT_ONLY)
    {
        int lost_flag, decoded_samples;
        opus_int16 *pcm_ptr;
        if (celt_accum)
            pcm_ptr = pcm;
        else
            pcm_ptr = pcm_silk;

        if (st->prev_mode == MODE_CELT_ONLY)
            silk_ResetDecoder(silk_dec);

        /* The SILK PLC cannot produce frames of less than 10 ms */
        st->DecControl.payloadSize_ms = IMAX(10, 1000 * audiosize / st->Fs);

        /* (data == NULL: no internalSampleRate/nChannelsInternal update) */
        st->DecControl.enable_deep_plc = st->complexity >= 5;

        lost_flag = 1;   /* data == NULL */
        decoded_samples = 0;
        do {
            int first_frame = decoded_samples == 0;
            silk_ret = silk_Decode(silk_dec, &st->DecControl,
                                   lost_flag, first_frame, &dec, pcm_ptr, &silk_frame_size,
                                   st->arch);
            if (silk_ret) {
                if (lost_flag) {
                    /* PLC failure should not be fatal */
                    silk_frame_size = frame_size;
                    for (i = 0; i < frame_size * st->channels; i++)
                        pcm_ptr[i] = 0;
                } else {
                    RESTORE_STACK;
                    return OPUS_INTERNAL_ERROR;
                }
            }
            pcm_ptr += silk_frame_size * st->channels;
            decoded_samples += silk_frame_size;
        } while (decoded_samples < frame_size);
    }

    start_band = 0;
    /* redundancy parse skipped: requires data != NULL */
    if (mode != MODE_CELT_ONLY)
        start_band = 17;

    /* bandwidth == 0 on the PLC path: no CELT_SET_END_BAND */
    cu_celt_set_channels(celt_dec, st->stream_channels);

    /* MUST be after PLC */
    cu_celt_set_start_band(celt_dec, start_band);

    if (mode != MODE_SILK_ONLY)
    {
        int celt_frame_size = IMIN(F20, frame_size);
        /* Make sure to discard any previous CELT state */
        if (mode != st->prev_mode && st->prev_mode > 0 && !st->prev_redundancy)
            cuopus_celt_reset_state(celt_dec);
        /* Decode CELT (data == NULL: concealment) */
        celt_ret = celt_decode_with_ec(celt_dec, NULL, 0, pcm, celt_frame_size,
                                       &dec, celt_accum);
    } else {
        unsigned char silence[2] = {0xFF, 0xFF};
        if (!celt_accum)
        {
            for (i = 0; i < frame_size * st->channels; i++)
                pcm[i] = 0;
        }
        /* For hybrid -> SILK transitions, we let the CELT MDCT
           do a fade-out by decoding a silence frame */
        if (st->prev_mode == MODE_HYBRID && !(redundancy && celt_to_silk && st->prev_redundancy))
        {
            cu_celt_set_start_band(celt_dec, 0);
            celt_decode_with_ec(celt_dec, silence, 2, pcm, F2_5, NULL, celt_accum);
        }
    }

    if (mode != MODE_CELT_ONLY && !celt_accum)
    {
        for (i = 0; i < frame_size * st->channels; i++)
            pcm[i] = SAT16(ADD32(pcm[i], pcm_silk[i]));
    }

    /* redundancy / transition fades impossible here (data == NULL) */

    if (st->decode_gain)
    {
        opus_val32 gain;
        gain = celt_exp2(MULT16_16_P15(QCONST16(6.48814081e-4f, 25), st->decode_gain));
        for (i = 0; i < frame_size * st->channels; i++)
        {
            opus_val32 x;
            x = MULT16_32_P16(pcm[i], gain);
            pcm[i] = SATURATE(x, 32767);
        }
    }

    st->rangeFinal = 0;              /* len <= 1 */
    st->prev_mode = mode;
    st->prev_redundancy = 0;         /* redundancy && !celt_to_silk */

    RESTORE_STACK;
    return celt_ret < 0 ? celt_ret : audiosize;
}

/* [opus_decoder.c opus_decode_frame], FIXED_POINT, full modes. */
CUOPUS_DEVFN_NOINL static int cu_opus_decode_frame(cu_opus_decoder *st, const unsigned char *data,
        opus_int32 len, opus_val16 *pcm, int frame_size, int decode_fec)
{
    void *silk_dec;
    CELTDecoder *celt_dec;
    int i, silk_ret = 0, celt_ret = 0;
    ec_dec dec;
    opus_int32 silk_frame_size;
    int pcm_silk_size;
    VARDECL(opus_int16, pcm_silk);
    int pcm_transition_silk_size;
    VARDECL(opus_val16, pcm_transition_silk);
    int pcm_transition_celt_size;
    VARDECL(opus_val16, pcm_transition_celt);
    opus_val16 *pcm_transition = NULL;
    int redundant_audio_size;
    VARDECL(opus_val16, redundant_audio);

    int audiosize;
    int mode;
    int bandwidth;
    int transition = 0;
    int start_band;
    int redundancy = 0;
    int redundancy_bytes = 0;
    int celt_to_silk = 0;
    int c;
    int F2_5, F5, F10, F20;
    const opus_val16 *window;
    opus_uint32 redundant_rng = 0;
    int celt_accum;
    ALLOC_STACK;

    silk_dec = &st->silk;
    celt_dec = (CELTDecoder *)st->celt.blob;
    F20 = st->Fs / 50;
    F10 = F20 >> 1;
    F5 = F10 >> 1;
    F2_5 = F5 >> 1;
    if (frame_size < F2_5)
    {
        RESTORE_STACK;
        return OPUS_BUFFER_TOO_SMALL;
    }
    /* Limit frame_size to avoid excessive stack allocations. */
    frame_size = IMIN(frame_size, st->Fs / 25 * 3);
    /* Payloads of 1 (2 including ToC) or 0 trigger the PLC/DTX */
    if (len <= 1)
    {
        data = NULL;
        /* In that case, don't conceal more than what the ToC says */
        frame_size = IMIN(frame_size, st->frame_size);
    }
    if (data == NULL)
    {
        /* cuopus: whole data==NULL branch delegated to cu_opus_plc_frame;
           the >F20 case is the vendor's recursive split, hoisted into this
           loop with the identical IMIN(audiosize, F20) chunk sequence. */
        audiosize = frame_size;
        if (audiosize > F20)
        {
            do {
                int ret = cu_opus_plc_frame(st, pcm, IMIN(audiosize, F20));
                if (ret < 0)
                {
                    RESTORE_STACK;
                    return ret;
                }
                pcm += ret * st->channels;
                audiosize -= ret;
            } while (audiosize > 0);
            RESTORE_STACK;
            return frame_size;
        }
        {
            int ret = cu_opus_plc_frame(st, pcm, audiosize);
            RESTORE_STACK;
            return ret;
        }
    }

    audiosize = st->frame_size;
    mode = st->mode;
    bandwidth = st->bandwidth;
    ec_dec_init(&dec, (unsigned char *)data, len);

    /* In fixed-point, we can tell CELT to do the accumulation on top of the
       SILK PCM buffer. This saves some stack space. */
    celt_accum = (mode != MODE_CELT_ONLY) && (frame_size >= F10);

    pcm_transition_silk_size = ALLOC_NONE;
    pcm_transition_celt_size = ALLOC_NONE;
    if (data != NULL && st->prev_mode > 0 && (
        (mode == MODE_CELT_ONLY && st->prev_mode != MODE_CELT_ONLY && !st->prev_redundancy)
     || (mode != MODE_CELT_ONLY && st->prev_mode == MODE_CELT_ONLY))
       )
    {
        transition = 1;
        /* Decide where to allocate the stack memory for pcm_transition */
        if (mode == MODE_CELT_ONLY)
            pcm_transition_celt_size = F5 * st->channels;
        else
            pcm_transition_silk_size = F5 * st->channels;
    }
    ALLOC(pcm_transition_celt, pcm_transition_celt_size, opus_val16);
    if (transition && mode == MODE_CELT_ONLY)
    {
        pcm_transition = pcm_transition_celt;
        cu_opus_plc_frame(st, pcm_transition, IMIN(F5, audiosize));   /* vendor: opus_decode_frame(st, NULL, 0, ...) */
    }
    if (audiosize > frame_size)
    {
        RESTORE_STACK;
        return OPUS_BAD_ARG;
    } else {
        frame_size = audiosize;
    }

    /* Don't allocate any memory when in CELT-only mode */
    pcm_silk_size = (mode != MODE_CELT_ONLY && !celt_accum) ? IMAX(F10, frame_size) * st->channels : ALLOC_NONE;
    ALLOC(pcm_silk, pcm_silk_size, opus_int16);

    /* SILK processing */
    if (mode != MODE_CELT_ONLY)
    {
        int lost_flag, decoded_samples;
        opus_int16 *pcm_ptr;
        if (celt_accum)
            pcm_ptr = pcm;
        else
            pcm_ptr = pcm_silk;

        if (st->prev_mode == MODE_CELT_ONLY)
            silk_ResetDecoder(silk_dec);

        /* The SILK PLC cannot produce frames of less than 10 ms */
        st->DecControl.payloadSize_ms = IMAX(10, 1000 * audiosize / st->Fs);

        if (data != NULL)
        {
            st->DecControl.nChannelsInternal = st->stream_channels;
            if (mode == MODE_SILK_ONLY) {
                if (bandwidth == OPUS_BANDWIDTH_NARROWBAND) {
                    st->DecControl.internalSampleRate = 8000;
                } else if (bandwidth == OPUS_BANDWIDTH_MEDIUMBAND) {
                    st->DecControl.internalSampleRate = 12000;
                } else if (bandwidth == OPUS_BANDWIDTH_WIDEBAND) {
                    st->DecControl.internalSampleRate = 16000;
                } else {
                    st->DecControl.internalSampleRate = 16000;
                }
            } else {
                /* Hybrid mode */
                st->DecControl.internalSampleRate = 16000;
            }
        }
        st->DecControl.enable_deep_plc = st->complexity >= 5;

        lost_flag = data == NULL ? 1 : 2 * !!decode_fec;
        decoded_samples = 0;
        do {
            /* Call SILK decoder */
            int first_frame = decoded_samples == 0;
            silk_ret = silk_Decode(silk_dec, &st->DecControl,
                                   lost_flag, first_frame, &dec, pcm_ptr, &silk_frame_size,
                                   st->arch);
            if (silk_ret) {
                if (lost_flag) {
                    /* PLC failure should not be fatal */
                    silk_frame_size = frame_size;
                    for (i = 0; i < frame_size * st->channels; i++)
                        pcm_ptr[i] = 0;
                } else {
                    RESTORE_STACK;
                    return OPUS_INTERNAL_ERROR;
                }
            }
            pcm_ptr += silk_frame_size * st->channels;
            decoded_samples += silk_frame_size;
        } while (decoded_samples < frame_size);
    }

    start_band = 0;
    if (!decode_fec && mode != MODE_CELT_ONLY && data != NULL
        && ec_tell(&dec) + 17 + 20 * (mode == MODE_HYBRID) <= 8 * len)
    {
        /* Check if we have a redundant 0-8 kHz band */
        if (mode == MODE_HYBRID)
            redundancy = ec_dec_bit_logp(&dec, 12);
        else
            redundancy = 1;
        if (redundancy)
        {
            celt_to_silk = ec_dec_bit_logp(&dec, 1);
            /* redundancy_bytes will be at least two, in the non-hybrid
               case due to the ec_tell() check above */
            redundancy_bytes = mode == MODE_HYBRID ?
                (opus_int32)ec_dec_uint(&dec, 256) + 2 :
                len - ((ec_tell(&dec) + 7) >> 3);
            len -= redundancy_bytes;
            /* This is a sanity check. It should never happen for a valid
               packet, so the exact behaviour is not normative. */
            if (len * 8 < ec_tell(&dec))
            {
                len = 0;
                redundancy_bytes = 0;
                redundancy = 0;
            }
            /* Shrink decoder because of raw bits */
            dec.storage -= redundancy_bytes;
        }
    }
    if (mode != MODE_CELT_ONLY)
        start_band = 17;

    if (redundancy)
    {
        transition = 0;
        pcm_transition_silk_size = ALLOC_NONE;
    }

    ALLOC(pcm_transition_silk, pcm_transition_silk_size, opus_val16);

    if (transition && mode != MODE_CELT_ONLY)
    {
        pcm_transition = pcm_transition_silk;
        cu_opus_plc_frame(st, pcm_transition, IMIN(F5, audiosize));   /* vendor: opus_decode_frame(st, NULL, 0, ...) */
    }

    if (bandwidth)
    {
        int endband = 21;

        switch (bandwidth)
        {
        case OPUS_BANDWIDTH_NARROWBAND:
            endband = 13;
            break;
        case OPUS_BANDWIDTH_MEDIUMBAND:
        case OPUS_BANDWIDTH_WIDEBAND:
            endband = 17;
            break;
        case OPUS_BANDWIDTH_SUPERWIDEBAND:
            endband = 19;
            break;
        case OPUS_BANDWIDTH_FULLBAND:
            endband = 21;
            break;
        default:
            break;
        }
        cu_celt_set_end_band(celt_dec, endband);
    }
    cu_celt_set_channels(celt_dec, st->stream_channels);

    /* Only allocation memory for redundancy if/when needed */
    redundant_audio_size = redundancy ? F5 * st->channels : ALLOC_NONE;
    ALLOC(redundant_audio, redundant_audio_size, opus_val16);

    /* 5 ms redundant frame for CELT->SILK */
    if (redundancy && celt_to_silk)
    {
        /* If the previous frame did not use CELT (the first redundancy frame in
           a transition from SILK may have been lost) then the CELT decoder is
           stale at this point and the redundancy audio is not useful, however
           the final range is still needed (for testing), so the redundancy is
           always decoded but the decoded audio may not be used */
        cu_celt_set_start_band(celt_dec, 0);
        celt_decode_with_ec(celt_dec, data + len, redundancy_bytes,
                            redundant_audio, F5, NULL, 0);
        redundant_rng = cu_celt_get_final_range(celt_dec);
    }

    /* MUST be after PLC */
    cu_celt_set_start_band(celt_dec, start_band);

    if (mode != MODE_SILK_ONLY)
    {
        int celt_frame_size = IMIN(F20, frame_size);
        /* Make sure to discard any previous CELT state */
        if (mode != st->prev_mode && st->prev_mode > 0 && !st->prev_redundancy)
            cuopus_celt_reset_state(celt_dec);
        /* Decode CELT */
        celt_ret = celt_decode_with_ec(celt_dec, decode_fec ? NULL : data,
                                       len, pcm, celt_frame_size, &dec, celt_accum);
    } else {
        unsigned char silence[2] = {0xFF, 0xFF};
        if (!celt_accum)
        {
            for (i = 0; i < frame_size * st->channels; i++)
                pcm[i] = 0;
        }
        /* For hybrid -> SILK transitions, we let the CELT MDCT
           do a fade-out by decoding a silence frame */
        if (st->prev_mode == MODE_HYBRID && !(redundancy && celt_to_silk && st->prev_redundancy))
        {
            cu_celt_set_start_band(celt_dec, 0);
            celt_decode_with_ec(celt_dec, silence, 2, pcm, F2_5, NULL, celt_accum);
        }
    }

    if (mode != MODE_CELT_ONLY && !celt_accum)
    {
        for (i = 0; i < frame_size * st->channels; i++)
            pcm[i] = SAT16(ADD32(pcm[i], pcm_silk[i]));
    }

    {
        const CELTMode *celt_mode;
        celt_mode = cu_celt_get_mode(celt_dec);
        window = celt_mode->window;
    }

    /* 5 ms redundant frame for SILK->CELT */
    if (redundancy && !celt_to_silk)
    {
        cuopus_celt_reset_state(celt_dec);
        cu_celt_set_start_band(celt_dec, 0);

        celt_decode_with_ec(celt_dec, data + len, redundancy_bytes, redundant_audio, F5, NULL, 0);
        redundant_rng = cu_celt_get_final_range(celt_dec);
        smooth_fade(pcm + st->channels * (frame_size - F2_5), redundant_audio + st->channels * F2_5,
                    pcm + st->channels * (frame_size - F2_5), F2_5, st->channels, window, st->Fs);
    }
    /* 5ms redundant frame for CELT->SILK; ignore if the previous frame did not
       use CELT (the first redundancy frame in a transition from SILK may have
       been lost) */
    if (redundancy && celt_to_silk && (st->prev_mode != MODE_SILK_ONLY || st->prev_redundancy))
    {
        for (c = 0; c < st->channels; c++)
        {
            for (i = 0; i < F2_5; i++)
                pcm[st->channels * i + c] = redundant_audio[st->channels * i + c];
        }
        smooth_fade(redundant_audio + st->channels * F2_5, pcm + st->channels * F2_5,
                    pcm + st->channels * F2_5, F2_5, st->channels, window, st->Fs);
    }
    if (transition)
    {
        if (audiosize >= F5)
        {
            for (i = 0; i < st->channels * F2_5; i++)
                pcm[i] = pcm_transition[i];
            smooth_fade(pcm_transition + st->channels * F2_5, pcm + st->channels * F2_5,
                        pcm + st->channels * F2_5, F2_5,
                        st->channels, window, st->Fs);
        } else {
            /* Not enough time to do a clean transition, but we do it anyway
               This will not preserve amplitude perfectly and may introduce
               a bit of temporal aliasing, but it shouldn't be too bad and
               that's pretty much the best we can do. In any case, generating this
               transition it pretty silly in the first place */
            smooth_fade(pcm_transition, pcm,
                        pcm, F2_5,
                        st->channels, window, st->Fs);
        }
    }

    if (st->decode_gain)
    {
        opus_val32 gain;
        gain = celt_exp2(MULT16_16_P15(QCONST16(6.48814081e-4f, 25), st->decode_gain));
        for (i = 0; i < frame_size * st->channels; i++)
        {
            opus_val32 x;
            x = MULT16_32_P16(pcm[i], gain);
            pcm[i] = SATURATE(x, 32767);
        }
    }

    if (len <= 1)
        st->rangeFinal = 0;
    else
        st->rangeFinal = dec.rng ^ redundant_rng;

    st->prev_mode = mode;
    st->prev_redundancy = redundancy && !celt_to_silk;

    RESTORE_STACK;
    return celt_ret < 0 ? celt_ret : audiosize;
}

/* [opus_decoder.c opus_decode_native] with decode_fec==0, self_delimited==0 */
CUOPUS_DEVFN_NOINL static int cu_opus_decode(cu_opus_decoder *st, const unsigned char *data,
        opus_int32 len, opus_int16 *pcm, int frame_size)
{
    int i, nb_samples;
    int count, offset;
    unsigned char toc;
    int packet_frame_size, packet_bandwidth, packet_mode, packet_stream_channels;
    opus_int16 size[48];

    if (len == 0 || data == NULL)
    {
        int pcm_count = 0;
        int F20 = st->Fs / 50;
        if (frame_size % (st->Fs / 400) != 0)
            return OPUS_BAD_ARG;
        do {
            /* cuopus: IMIN(.., F20) + cu_opus_plc_frame hoists opus_decode_frame's
               recursive PLC split into this loop — identical 20 ms chunk sequence. */
            int ret = cu_opus_decode_frame(st, NULL, 0, pcm + pcm_count * st->channels, IMIN(frame_size - pcm_count, F20), 0);
            if (ret < 0)
                return ret;
            pcm_count += ret;
        } while (pcm_count < frame_size);
        st->last_packet_duration = pcm_count;
        return pcm_count;
    } else if (len < 0)
        return OPUS_BAD_ARG;

    packet_mode = opus_packet_get_mode(data);
    packet_bandwidth = opus_packet_get_bandwidth(data);
    packet_frame_size = opus_packet_get_samples_per_frame(data, st->Fs);
    packet_stream_channels = opus_packet_get_nb_channels(data);

    count = opus_packet_parse_impl(data, len, 0, &toc, NULL,
                                   size, &offset, NULL, NULL, NULL);
    if (count < 0)
        return count;

    data += offset;

    if (count * packet_frame_size > frame_size)
        return OPUS_BUFFER_TOO_SMALL;

    st->mode = packet_mode;
    st->bandwidth = packet_bandwidth;
    st->frame_size = packet_frame_size;
    st->stream_channels = packet_stream_channels;

    nb_samples = 0;
    for (i = 0; i < count; i++)
    {
        int ret;
        ret = cu_opus_decode_frame(st, data, size[i], pcm + nb_samples * st->channels, frame_size - nb_samples, 0);
        if (ret < 0)
            return ret;
        data += size[i];
        nb_samples += ret;
    }
    st->last_packet_duration = nb_samples;
    return nb_samples;
}

#endif /* CUOPUS_OPUS_CUH */
