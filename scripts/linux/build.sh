#!/bin/bash
# Build llama-bench from llama.cpp source on Linux, in one of the configurations this
# investigation compared. Clones the source if it is not already there.
#
#   native    GGML_NATIVE=ON - tuned for the build machine's CPU. Not distributable.
#   v3        GGML_NATIVE=OFF with -march=x86-64-v3 - AVX2-era baseline, portable to any
#             2013-or-later x86 host, so one build can serve a fleet.
#   v3vnni   as v3 plus -mavxvnni - isolates what AVX-VNNI alone is worth, since -march=native
#             on this CPU also enables AVX-IFMA and AVX-VNNI-INT8.
#   replica   llama-install.sh's forced settings, which reproduce the llama.app binary.
#
#   scripts/linux/build.sh <native|v3|replica> [tag] [cuda-arch]

set -euo pipefail
CONFIG="${1:?usage: build.sh <native|v3|replica> [tag] [cuda-arch]}"
TAG="${2:-b10107}"
CUDA_ARCH="${3:-120}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="${LLAMA_SRC:-$REPO_ROOT/src}"
BUILD="${LLAMA_BUILD:-$REPO_ROOT/build-linux-$CONFIG}"

# nvcc is not on PATH in a default CUDA install
for d in /usr/local/cuda/bin /usr/local/cuda-13.3/bin /usr/local/cuda-13/bin; do
    [ -x "$d/nvcc" ] && export PATH="$d:$PATH" && break
done
command -v nvcc >/dev/null || { echo "nvcc not found - install the CUDA Toolkit" >&2; exit 1; }
CUDA_ROOT="$(dirname "$(dirname "$(command -v nvcc)")")"

if [ ! -f "$SRC/CMakeLists.txt" ]; then
    echo "cloning llama.cpp $TAG ..."
    git clone --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp.git "$SRC"
fi
echo "source:  $SRC @ $(git -C "$SRC" rev-parse HEAD)"
echo "nvcc:    $(nvcc --version | tail -2 | head -1)"
echo "config:  $CONFIG"

COMMON=(
    -DCMAKE_BUILD_TYPE=Release
    -DGGML_CUDA=ON
    "-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH"
    "-DCUDAToolkit_ROOT=$CUDA_ROOT"
    -DLLAMA_BUILD_TESTS=OFF
    -DLLAMA_BUILD_EXAMPLES=OFF
    -DLLAMA_BUILD_SERVER=OFF
    -DLLAMA_BUILD_APP=OFF
    -DLLAMA_BUILD_UI=OFF
    -DLLAMA_CURL=OFF
)

# llama-install.sh's CMakeLists.txt forces these. The ISA options must be listed explicitly:
# GGML_NATIVE=OFF alone leaves ggml's INS_ENB enabling them.
INSTALL_SH_FORCED=(
    -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DGGML_LTO=OFF -DGGML_CCACHE=OFF
    -DBUILD_SHARED_LIBS=OFF -DGGML_STATIC=ON
    -DGGML_SSE42=OFF -DGGML_AVX=OFF -DGGML_AVX2=OFF
    -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_BMI2=OFF
)

case "$CONFIG" in
    native)  SPECIFIC=() ;;                                   # GGML_NATIVE defaults to ON
    v3)      SPECIFIC=(-DGGML_NATIVE=OFF
                       -DCMAKE_C_FLAGS=-march=x86-64-v3
                       -DCMAKE_CXX_FLAGS=-march=x86-64-v3) ;;
    v3vnni)  SPECIFIC=(-DGGML_NATIVE=OFF
                       "-DCMAKE_C_FLAGS=-march=x86-64-v3 -mavxvnni"
                       "-DCMAKE_CXX_FLAGS=-march=x86-64-v3 -mavxvnni") ;;
    replica) SPECIFIC=("${INSTALL_SH_FORCED[@]}") ;;
    *)       echo "unknown config: $CONFIG" >&2; exit 1 ;;
esac

cmake -S "$SRC" -B "$BUILD" "${COMMON[@]}" "${SPECIFIC[@]}"

# CONFIGURE_ONLY=1 stops after configure - used to check a config's flags without paying for
# the 6-15 minute CUDA build.
if [ -n "${CONFIGURE_ONLY:-}" ]; then
    echo "configure only - stopping here"
    exit 0
fi

cmake --build "$BUILD" --target llama-bench llama-completion -j"$(nproc)"
ls -la "$BUILD/bin/llama-bench" "$BUILD/bin/llama-completion"
