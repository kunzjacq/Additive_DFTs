#!/usr/bin/env bash
#
# Full benchmark of the binary polynomial products: gf2x, Mateer-Gao ("plain"), and the fused
# variant of Mateer-Gao ("fused"), for output sizes 2**8 to 2**MAX bits, in-place variants.
#
# The largest sizes need a lot of RAM: the in-place product of size 2**u bits needs
# 3.5 * 2**(u-3) bytes (14 GiB for u = 35, 28 GiB for u = 36, 56 GiB for u = 37), so for the
# largest sizes this script should be run from a text console (no desktop environment, no other
# application), on an otherwise idle machine, as timings are also disturbed by other processes.
#
# Typical duration for sizes 8 to 37 with gf2x up to size 35: about 45 minutes, of which about
# 15 minutes for gf2x at sizes 34 and 35, and 15 minutes for sizes 36 and 37.
#
# Needs a compiled gf2x (see README.md): by default in ./gf2x/install (CUSTOM_LIBS=<dir> to change).
#
# Results: in directory OUT (default bench-results/<date>): plain.log and fused.log (output of
# product_test), results.csv, info.txt (machine description); the table is also printed.
#
# usage: ./benchmark.sh [-s MIN] [-m MAX] [-g GF2X_MAX] [-o OUT] [-b BUILD_DIR] [-y]
#   -s MIN       smallest size (log2 of the number of bits of the result), default 8
#   -m MAX       largest size. Default: the largest size (at most 37) that fits in the available RAM
#   -g GF2X_MAX  largest size at which gf2x is timed, default 35 (gf2x could not be run above 35 with 64GB of RAM)
#   -o OUT       output directory
#   -b BUILD_DIR build directory, default build-bench
#   -y           do not ask for confirmation

set -euo pipefail
export LC_ALL=C  # decimal point, in particular (awk output is parsed as CSV)

cd "$(dirname "$0")"
SRC="$PWD"

MIN=8
MAX=""
GF2X_MAX=35
OUT=""
BUILD="build-bench"
YES=0

usage() { sed -n '/^# usage:/,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-1}"; }

while getopts "s:m:g:o:b:yh" opt; do
  case "$opt" in
    s) MIN="$OPTARG" ;;
    m) MAX="$OPTARG" ;;
    g) GF2X_MAX="$OPTARG" ;;
    o) OUT="$OPTARG" ;;
    b) BUILD="$OPTARG" ;;
    y) YES=1 ;;
    h) usage 0 ;;
    *) usage 1 ;;
  esac
done

[ -n "$OUT" ] || OUT="bench-results/$(date +%Y%m%d-%H%M%S)"

# ---- memory
avail_kb=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
avail=$((avail_kb * 1024))
margin=$((1 << 30))
# bytes needed by the in-place product of size 2**u bits: 3.5 * 2**(u-3) = 7 * 2**(u-4)
need() { echo $((7 << ($1 - 4))); }
gib() { awk -v b="$1" 'BEGIN {printf "%.1f", b / 1073741824}'; }

if [ -z "$MAX" ]; then
  MAX=$MIN
  while [ "$MAX" -lt 37 ] && [ $(( $(need $((MAX + 1))) + margin )) -le "$avail" ]; do MAX=$((MAX + 1)); done
fi
if [ "$MAX" -lt "$MIN" ]; then echo "empty range of sizes"; exit 1; fi
if [ $(( $(need "$MAX") + margin )) -gt "$avail" ]; then
  echo "WARNING: size $MAX needs $(gib "$(need "$MAX")") GiB, only $(gib "$avail") GiB are available."
  echo "         The process is likely to be killed by the system."
fi
[ "$GF2X_MAX" -le "$MAX" ] || GF2X_MAX=$MAX

if [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ] && [ "$MAX" -ge 36 ]; then
  echo "WARNING: a graphical session seems to be running (DISPLAY/WAYLAND_DISPLAY is set)."
  echo "         For sizes of 2**36 and more, run this script from a text console instead."
fi

echo "Sizes: $MIN to $MAX (gf2x up to $GF2X_MAX); available RAM: $(gib "$avail") GiB;" \
     "needed for size $MAX: $(gib "$(need "$MAX")") GiB."
echo "Output directory: $OUT"
if [ "$YES" -ne 1 ]; then
  read -r -p "Continue? [y/N] " answer
  case "$answer" in y|Y|yes) ;; *) exit 0 ;; esac
fi

# ---- build
if [ -z "${CUSTOM_LIBS:-}" ] && [ -d "$SRC/gf2x/install" ]; then export CUSTOM_LIBS="$SRC/gf2x/install"; fi
cmake -S "$SRC" -B "$BUILD" -D CMAKE_BUILD_TYPE=Release > /dev/null
cmake --build "$BUILD" --parallel --target product_test > "$BUILD/build.log" 2>&1 \
  || { echo "build failed, see $BUILD/build.log (is gf2x installed? see README.md)"; exit 1; }
BIN="$BUILD/product_test"

mkdir -p "$OUT"
{
  echo "date: $(date)"
  echo "git: $(git rev-parse --short HEAD 2> /dev/null || echo unknown)$(git diff --quiet 2> /dev/null || echo ' (modified)')"
  echo "cpu: $(lscpu | sed -n 's/^Model name: *//p')"
  echo "memory: $(gib "$(awk '/^MemTotal:/ {print $2 * 1024}' /proc/meminfo)") GiB total, $(gib "$avail") GiB available"
  echo "kernel: $(uname -sr)"
  echo "compiler: $(c++ --version | head -1)"
  echo "sizes: $MIN to $MAX, gf2x up to $GF2X_MAX"
} > "$OUT/info.txt"

# ---- runs. gf2x is timed in the plain run; the fused run only compares with gf2x up to size 25
# (to check its results).
status=0
SIZES="$MIN-$MAX"
echo "Plain variant..."
"$BIN" --sizes "$SIZES" --gf2x-max "$GF2X_MAX" > "$OUT/plain.log" 2>&1 || { echo "plain run FAILED (see $OUT/plain.log)"; status=1; }
echo "Fused variant..."
fused_gf2x=$((GF2X_MAX < 25 ? GF2X_MAX : 25))
"$BIN" --fused --no-degree-test --sizes "$SIZES" --gf2x-max "$fused_gf2x" > "$OUT/fused.log" 2>&1 || { echo "fused run FAILED (see $OUT/fused.log)"; status=1; }

# ---- results
awk '
  FNR == 1 { file++ }
  $1 == "RESULT" {
    if (file == 1) { g[$2] = $3; p[$2] = $4 } else { f[$2] = $4 }
    sizes[$2] = 1
    if (lo == "" || $2 + 0 < lo) lo = $2 + 0
    if (hi == "" || $2 + 0 > hi) hi = $2 + 0
  }
  END {
    print "log2_size,gf2x,plain,fused,fused_over_plain_speedup,gf2x_over_fused_ratio"
    for (s = lo; s <= hi; s++) {
      if (!(s in sizes)) continue
      printf "%s,%s,%s,%s,", s, (g[s] > 0 ? g[s] : ""), p[s], f[s]
      printf "%s,%s\n", (f[s] > 0 ? sprintf("%.2f", p[s] / f[s]) : ""), (g[s] > 0 && f[s] > 0 ? sprintf("%.2f", g[s] / f[s]) : "")
    }
  }' "$OUT/plain.log" "$OUT/fused.log" > "$OUT/results.csv"

echo
echo "Times in seconds (in-place products; ratio = gf2x time / fused time: > 1 if fused is faster):"
column -s, -t "$OUT/results.csv" 2> /dev/null || cat "$OUT/results.csv"
echo
grep -h "Wrong result\|failed" "$OUT/plain.log" "$OUT/fused.log" || true
echo "Results in $OUT"
exit $status
