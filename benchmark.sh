#!/usr/bin/env bash
#
# Full benchmark of the binary polynomial products: gf2x, Mateer-Gao ("plain"), and the optimized
# variant of Mateer-Gao ("optimized"), for output sizes 2**8 to 2**MAX bits, in-place variants.
#
# The largest sizes need a lot of RAM: the in-place product of size 2**u bits needs
# 3.5 * 2**(u-3) bytes (14 GiB for u = 35, 28 GiB for u = 36, 56 GiB for u = 37), so for the
# largest sizes this script should be run from a text console (no desktop environment, no other
# application), on an otherwise idle machine, as timings are also disturbed by other processes.
#
# Typical duration for sizes 8 to 37 with gf2x up to size 35: about 45 minutes, of which about
# 15 minutes for gf2x at sizes 34 and 35, and 15 minutes for sizes 36 and 37.
#
# gf2x (see README.md) is optional: by default it is searched for in ./gf2x/install (CUSTOM_LIBS=<dir> to
# change). If it is not found, product_bench is used instead of product_test: only the Mateer-Gao products
# are timed (the gf2x columns are empty) and the results are not checked.
#
# Results: in directory OUT (default bench-results/<date>): plain.log and optimized.log (output of
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

echo "Sizes: $MIN to $MAX (gf2x, if available, up to $GF2X_MAX); available RAM: $(gib "$avail") GiB;" \
     "needed for size $MAX: $(gib "$(need "$MAX")") GiB."
echo "Output directory: $OUT"
if [ "$YES" -ne 1 ]; then
  read -r -p "Continue? [y/N] " answer
  case "$answer" in y|Y|yes) ;; *) exit 0 ;; esac
fi

# ---- build
if [ -z "${CUSTOM_LIBS:-}" ] && [ -d "$SRC/gf2x/install" ]; then export CUSTOM_LIBS="$SRC/gf2x/install"; fi
cmake -S "$SRC" -B "$BUILD" -D CMAKE_BUILD_TYPE=Release > /dev/null
# product_test (which times gf2x and checks the results against it) is only built if gf2x was found;
# otherwise, product_bench is used and only the Mateer-Gao products are timed.
if grep -q '^GF2X_LIBRARY:FILEPATH=.*NOTFOUND' "$BUILD/CMakeCache.txt"; then
  HAS_GF2X=0; TARGET=product_bench
  GF2X_WARNING="WARNING: gf2x was not found (see README.md): gf2x is not timed, and the results of the products are NOT checked."
  echo "$GF2X_WARNING"
else
  HAS_GF2X=1; TARGET=product_test
fi
cmake --build "$BUILD" --parallel --target "$TARGET" > "$BUILD/build.log" 2>&1 \
  || { echo "build failed, see $BUILD/build.log"; exit 1; }
BIN="$BUILD/$TARGET"

mkdir -p "$OUT"
{
  echo "date: $(date)"
  echo "git: $(git rev-parse --short HEAD 2> /dev/null || echo unknown)$(git diff --quiet 2> /dev/null || echo ' (modified)')"
  echo "cpu: $(lscpu | sed -n 's/^Model name: *//p')"
  echo "memory: $(gib "$(awk '/^MemTotal:/ {print $2 * 1024}' /proc/meminfo)") GiB total, $(gib "$avail") GiB available"
  echo "kernel: $(uname -sr)"
  echo "compiler: $(c++ --version | head -1)"
  if [ "$HAS_GF2X" -eq 1 ]; then echo "sizes: $MIN to $MAX, gf2x up to $GF2X_MAX"; else echo "sizes: $MIN to $MAX, no gf2x (timings only, results not checked)"; fi
} > "$OUT/info.txt"

# ---- runs. gf2x is timed in the plain run; the optimized run only compares with gf2x up to size 25
# (to check its results).
status=0
SIZES="$MIN-$MAX"
optimized_gf2x=$((GF2X_MAX < 25 ? GF2X_MAX : 25))
if [ "$HAS_GF2X" -eq 1 ]; then
  PLAIN_OPTS=(--gf2x-max "$GF2X_MAX")
  OPT_OPTS=(--no-degree-test --gf2x-max "$optimized_gf2x")
else
  PLAIN_OPTS=()
  OPT_OPTS=()
fi
echo "Plain variant..."
"$BIN" --sizes "$SIZES" "${PLAIN_OPTS[@]}" > "$OUT/plain.log" 2>&1 || { echo "plain run FAILED (see $OUT/plain.log)"; status=1; }
echo "Optimized variant..."
"$BIN" --optimized --sizes "$SIZES" "${OPT_OPTS[@]}" > "$OUT/optimized.log" 2>&1 || { echo "optimized run FAILED (see $OUT/optimized.log)"; status=1; }

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
    print "log2_size,gf2x,plain,optimized,optimized_over_plain_speedup,gf2x_over_optimized_ratio"
    for (s = lo; s <= hi; s++) {
      if (!(s in sizes)) continue
      printf "%s,%s,%s,%s,", s, (g[s] > 0 ? g[s] : ""), p[s], f[s]
      printf "%s,%s\n", (f[s] > 0 ? sprintf("%.2f", p[s] / f[s]) : ""), (g[s] > 0 && f[s] > 0 ? sprintf("%.2f", g[s] / f[s]) : "")
    }
  }' "$OUT/plain.log" "$OUT/optimized.log" > "$OUT/results.csv"

echo
echo "Times in seconds (in-place products; ratio = gf2x time / optimized time: > 1 if optimized is faster):"
column -s, -t "$OUT/results.csv" 2> /dev/null || cat "$OUT/results.csv"
echo
grep -h "Wrong result\|failed" "$OUT/plain.log" "$OUT/optimized.log" || true
[ "$HAS_GF2X" -eq 1 ] || echo "$GF2X_WARNING"
echo "Results in $OUT"
exit $status
