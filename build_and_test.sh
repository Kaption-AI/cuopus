#!/usr/bin/env bash
# cuopus — build (and smoke-test) from the repo root.
#
# Usage: ./build_and_test.sh [lib|phase1|all]     (default: lib)
#
#   lib    = regenerate device/gen from vendor/opus-1.5.2 and build the two
#            production artifacts: out/libcuopus.so (in-process library) and
#            out/cuopus_decode (CLI + daemon). Compilation only — no GPU needed.
#   phase1 = the self-contained phase-1 bit-exactness gate: encode a test
#            corpus, validate the OGG index against ffprobe, then run the same
#            144-symbol range-decoder schedule through real libopus entdec on
#            CPU and the CUDA port on GPU, compared word-for-word on every
#            packet. Needs a GPU and ffmpeg on the host.
#
# Builds run inside a throwaway Docker CUDA container (any image with CUDA
# 12.x+ nvcc works; override with CUOPUS_IMG). If you have nvcc locally, set
# CUOPUS_NO_DOCKER=1. Only sm_120 has been validated; override CUOPUS_ARCH
# at your own risk (bit-exactness is expected but unverified elsewhere).
#
# The phase-2/3 full-decoder gates (1.62 M packets / 1.07 B samples vs real
# --enable-fixed-point libopus, 0 mismatches) used a reference-decoder driver
# and private audio corpora that are not shipped in this repo; their harness
# sources (harness2.cu, harness_host.cpp) and full methodology are — see
# PHASE2_RESULTS.md / PHASE3_RESULTS.md.
set -euo pipefail
cd "$(dirname "$0")"
TARGET="${1:-lib}"

IMG="${CUOPUS_IMG:-nvcr.io/nvidia/tensorrt-llm/release:1.2.1}"
ARCH="${CUOPUS_ARCH:-sm_120}"
OPUS_URL=https://downloads.xiph.org/releases/opus/opus-1.5.2.tar.gz
# Any audio file ffmpeg can read; ~8 min of speech-like audio is synthesized
# if unset (phase1 only).
TEST_AUDIO="${CUOPUS_TEST_AUDIO:-}"

run_build() {  # run a build command, in the container or bare
  if [ "${CUOPUS_NO_DOCKER:-0}" = "1" ]; then
    bash -c "$1"
  else
    docker run --rm -v "$PWD":/w -w /w "$IMG" bash -c "$1"
  fi
}

run_gpu() {    # run a command that needs the GPU
  if [ "${CUOPUS_NO_DOCKER:-0}" = "1" ]; then
    bash -c "$1"
  else
    docker run --rm --gpus all --shm-size=2g -v "$PWD":/w -w /w "$IMG" bash -c "$1"
  fi
}

vendor_check() {
  if [ ! -f vendor/opus-1.5.2/celt/entdec.c ]; then
    echo "vendor/opus-1.5.2 missing — fetching $OPUS_URL"
    mkdir -p vendor
    curl -fsSL -o vendor/opus-1.5.2.tar.gz "$OPUS_URL"
    tar -C vendor -xzf vendor/opus-1.5.2.tar.gz
  fi
}

lib() {
  echo "=== [lib 1/2] regenerate device/gen from vendor/opus-1.5.2 ==="
  vendor_check
  python3 device/port_silk.py
  echo "=== [lib 2/2] build libcuopus.so + cuopus_decode (nvcc, -arch=$ARCH) ==="
  mkdir -p out
  run_build "
    set -e
    nvcc -O2 -arch=$ARCH -rdc=true -DCUOPUS_CUDA -DFIXED_POINT=1 -DOPUS_BUILD \
      -Idevice -Idevice/gen -Idevice/gen/silk -Idevice/gen/celt -Idevice/gen/include \
      -Xcompiler -fPIC -shared libcuopus.cu -o out/libcuopus.so
    nvcc -O2 -arch=$ARCH -rdc=true -DCUOPUS_CUDA -DFIXED_POINT=1 -DOPUS_BUILD \
      -Idevice -Idevice/gen -Idevice/gen/silk -Idevice/gen/celt -Idevice/gen/include \
      decode_cli.cu -o out/cuopus_decode"
  ls -la out/libcuopus.so out/cuopus_decode
  echo "BUILD COMPLETE"
}

phase1() {
  echo "=== [P1 1/5] test corpus (OGG Opus, 20 ms + 120 ms frames) ==="
  mkdir -p testdata out
  if [ ! -f testdata/corpus_16k_f32.raw ]; then
    if [ -n "$TEST_AUDIO" ]; then
      ffmpeg -y -loglevel error -i "$TEST_AUDIO" -f f32le -ar 16000 -ac 1 \
        testdata/corpus_16k_f32.raw
    else
      # synthetic speech-ish corpus: AM/FM-modulated tones + pink noise.
      # (Opus mode selection is bitrate-driven, so this exercises SILK-WB
      # at 24 kbps just like speech does.)
      ffmpeg -y -loglevel error -filter_complex \
        "sine=frequency=210:duration=477[a];anoisesrc=d=477:color=pink:amplitude=0.28[b];[a][b]amix=inputs=2,tremolo=f=3.1:d=0.8,aresample=16000,pan=mono|c0=c0" \
        -f f32le -ar 16000 testdata/corpus_16k_f32.raw
    fi
  fi
  for fd in 20 120; do
    [ -f testdata/corpus_${fd}ms.ogg ] || ffmpeg -y -loglevel error -f f32le -ar 16000 -ac 1 \
      -i testdata/corpus_16k_f32.raw -c:a libopus -b:a 24k -frame_duration $fd testdata/corpus_${fd}ms.ogg
  done

  echo "=== [P1 2/5] OGG page index: validate vs ffprobe, extract packet blobs ==="
  python3 ogg_index.py validate testdata/corpus_20ms.ogg
  python3 ogg_index.py validate testdata/corpus_120ms.ogg
  python3 ogg_index.py extract testdata/corpus_20ms.ogg  testdata/packets_20ms.blob
  python3 ogg_index.py extract testdata/corpus_120ms.ogg testdata/packets_120ms.blob

  echo "=== [P1 3/5] vendor libopus 1.5.2 (reference range decoder) ==="
  vendor_check

  echo "=== [P1 4/5] build phase-1 harness (nvcc, -arch=$ARCH) ==="
  run_build "
    nvcc -O3 -arch=$ARCH -Xcompiler -fopenmp \
      -Ivendor/opus-1.5.2/celt -Ivendor/opus-1.5.2/include \
      harness.cu vendor/opus-1.5.2/celt/entdec.c vendor/opus-1.5.2/celt/entcode.c \
      -lgomp -o out/cuopus_harness"

  echo "=== [P1 5/5] bit-exactness + throughput (all packets, both frame sizes) ==="
  run_gpu "
    echo '--- 20 ms frames ---'
    ./out/cuopus_harness testdata/packets_20ms.blob 10
    echo '--- 120 ms frames ---'
    ./out/cuopus_harness testdata/packets_120ms.blob 10"
  echo "ALL PHASE-1 CHECKS COMPLETE"
}

case "$TARGET" in
  lib)    lib ;;
  phase1) phase1 ;;
  all)    lib; phase1 ;;
  *) echo "unknown target: $TARGET (want lib|phase1|all)"; exit 2 ;;
esac
