#!/bin/sh

set -e

# --- Defaults ----------------------------------------------------
USER_CC=""
USER_CFLAGS=""
USER_LD_LIBS=""
USER_CUDA_HOME=""

# Feature toggles: default values (all off)
ENABLE_DEBUG=0
ENABLE_OPENMP=0
ENABLE_GPU=0       # 0 = off unless --enable-gpu

# --- Parse command‑line args ------------------------------------
for arg in "$@"; do
  case $arg in
    --cc=*)           USER_CC="${arg#*=}"         ;;
    --cflags=*)       USER_CFLAGS="${arg#*=}"     ;;
    --ld-libs=*)      USER_LD_LIBS="${arg#*=}"    ;;
    --cuda-home=*)    USER_CUDA_HOME="${arg#*=}"  ;;
    --enable-debug)   ENABLE_DEBUG=1               ;;
    --enable-openmp)  ENABLE_OPENMP=1              ;;
    --enable-gpu)     ENABLE_GPU=1                 ;;
    -h|--help)
      cat <<EOF
Usage: $0 [OPTIONS]

  --cc=CC               set C compiler (default: gcc)
  --cflags="FLAGS"      add to default CFLAGS
  --ld-libs="LIBS"      add to default LD_LIBS
  --cuda-home=DIR       manually set CUDA install DIR

  --enable-debug        turn on debug mode (-DDEBUG)
  --enable-openmp       turn on OpenMP support
  --enable-gpu          turn on GPU support
  -h, --help            show this help and exit
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      echo "Try '$0 --help'." >&2
      exit 1
      ;;
  esac
done

# --- 1) Determine CUDA_DIR if GPU enabled ---------------------
NVCC=""
if [ "$ENABLE_GPU" -eq 1 ]; then
  if [ -n "$USER_CUDA_HOME" ]; then
    CUDA_DIR="$USER_CUDA_HOME"
    NVCC="$CUDA_DIR/bin/nvcc"
  elif command -v nvcc >/dev/null 2>&1; then
    NVCC_PATH=$(command -v nvcc)
    CUDA_DIR=$(dirname "$(dirname "$(readlink -f "$NVCC_PATH")")")
    NVCC=nvcc
  else
    CUDA_DIR=""
  fi
  [ -n "$CUDA_DIR" ] || ENABLE_GPU=0
else
  CUDA_DIR=""
fi

# --- 2) Decide compiler and base flags --------------------------
CC="${USER_CC:-gcc}"

# Base CFLAGS
CFLAGS="-Wall -O2"
[ $ENABLE_OPENMP -eq 1 ] && CFLAGS="$CFLAGS -DOMP -fopenmp"
[ $ENABLE_DEBUG  -eq 1 ] && CFLAGS="$CFLAGS -DDEBUG"
[ $ENABLE_GPU    -eq 1 ] && CFLAGS="$CFLAGS -DCUDA"
[ -n "$USER_CFLAGS" ] && CFLAGS="$CFLAGS $USER_CFLAGS"

# Base LD_LIBS
LD_LIBS="-lm"
[ $ENABLE_GPU  -eq 1 ] && LD_LIBS="$LD_LIBS -L$CUDA_DIR/lib64 -lcudart"
[ -n "$USER_LD_LIBS" ] && LD_LIBS="$LD_LIBS $USER_LD_LIBS"

# --- 3) Summary -----------------------------------------------
echo "Configuration summary:"
echo "  CC           = $CC"
echo "  Debug        = $( [ $ENABLE_DEBUG -eq 1 ] && echo on || echo off )"
echo "  OpenMP       = $( [ $ENABLE_OPENMP -eq 1 ] && echo on || echo off )"
echo "  GPU support  = $( [ $ENABLE_GPU -eq 1 ] && echo on || echo off )"
echo "  CUDA_DIR     = ${CUDA_DIR:-<none>}"
echo "  NVCC         = ${NVCC:-<none>}"
echo "  CFLAGS       = $CFLAGS"
echo "  LD_LIBS      = $LD_LIBS"

# --- 4) Generate Makefile ---------------------------------------
sed \
  -e "s|@CC@|$CC|g" \
  -e "s|@CUDA_DIR@|$CUDA_DIR|g" \
  -e "s|@CFLAGS@|$CFLAGS|g" \
  -e "s|@LD_LIBS@|$LD_LIBS|g" \
  -e "s|@NVCC@|$NVCC|g" \
  Makefile.in > Makefile

echo "=> Makefile generated. Run 'make' to build."
