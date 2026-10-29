# Code guide

A map of the source tree, written from a read-through of the code. For the mathematics, see
[algo_fft_en.md](algo_fft_en.md). For usage, benchmarks and build instructions, see
[README.md](README.md).

## Build targets

Defined in `CMakeLists.txt` (C++20, C99, little-endian only; `-march=native` and LTO are on by
default).

| Target | Sources | Built with |
|---|---|---|
| `product_test` | `product_test.cpp`, `utils.cpp`, `libfft/mg.cpp` | all compilers; links `gf2x` for GCC/Clang only |
| `fft_test` | `fft_test.cpp`, `utils.cpp`, `libfft/additive_fft.cpp`, `libfft/mg.cpp`, `libcantor/*.cpp` | GCC/Clang only |
| `cantor_basis_test` | `cantor_test.cpp`, `libcantor/*.cpp` | GCC/Clang only |

CMake options: `LTO` (default ON) and `USE_BOOST` (default OFF). With `USE_BOOST=ON`, CMake runs
`find_package(Boost 1.65)`, and if Boost is found, `defs.h` (generated from `defs.h.in`) defines
`Boost_FOUND`. That enables the 128 to 2048-bit word types in `include/cantor.h`. Without Boost,
`uint128_t` is `__uint128_t`.

`gf2x` is searched for in `<source>/lib` and in `$CUSTOM_LIBS/lib`. The vendored `include/gf2x.h`
is only the header.

Files not referenced by any target: `libfft/mateer_gao.cpp` (non-recursive tree-walk helpers
`DFTS` and `DFTS_incomplete_tree`), `libfft/mg_t.cpp` (includes a `mg_t.h` that does not exist,
so it cannot compile as is), `libfft/finite_field.c`, `libfft/utils.c`, and `include/finite_field.h`.
`include/mg_templates.h`, `include/mg.h`, `include/helpers.hpp` and the other headers are used.

## Concepts

- **Field and word type.** A field GF(2^n) is identified by its element type `word`
  (`uint8_t` to `uint64_t`, plus larger Boost types). `c_b_t<word>` (in `cantor.h`) holds the
  per-type constants: `n`, `word_logsize` (log2 n), and the helper types `half_type` and `log_type`.
- **Three bases.**
  - *beta*: the Cantor basis. Element i of the field is the sum of beta_k over the set bits k
    of i, so offsets and DFT points are plain integers.
  - *gamma*: a tower basis built from successive degree-2 extensions. This is where fast field
    multiplication lives.
  - *mult*: the polynomial basis 1, a, a^2, ... modulo a primitive polynomial. It is used for
    the PCLMULQDQ-based multiplication, `multiply_mult_repr`.
  - The `beta_to_gamma`, `beta_to_mult` and related conversions go between them.
- **Interleaving.** Most DFT routines take a `logstride` and process 2^logstride interleaved
  polynomials at once. This keeps the recursion shallow and each level's data contiguous.
- **Truncated DFTs.** Given a polynomial with at most 2^logsize coefficients, these compute only
  the first 2^logsize values (at an optional block offset). The Mateer-Gao recursion is templated
  on `s`, where 2^(2^s) bounds the block size.

## Directory layout

### `include/` and `libcantor/`: field construction

- `cantor.h` defines `cantor_basis_bare<word>` and `cantor_basis_full<word>`. `cantor_basis<word>`
  selects one of them: `full` for `uint8_t` and `uint16_t`, `bare` for everything else.
  - `full` builds the complete gamma multiplication table and offers reference products.
  - `bare` stores only gamma_i^2 and the beta_i, for large fields.
  - Both provide `multiply`, `square`, `inverse`, `trace` and the basis conversions.
  - The `*_mult_repr` methods (PCLMUL products in the mult basis) exist only in `bare`. The
    constructor builds everything, including the discrete-log tables.
- `libcantor/cantor_basis_bare.cpp` and `cantor_basis_full.cpp` implement these and contain the
  explicit template instantiations. `libcantor/helpers.cpp` and `include/helpers.hpp` hold bit
  utilities and linear-algebra helpers, with `_cppinteger` variants for Boost types.
- `cantor_test.cpp` checks the construction against hard-coded targets (a known 2048-bit value
  when Boost is available) and compares against reference products.

### `include/` and `libfft/`: the DFTs

- `additive_fft.h` and `libfft/additive_fft.cpp` define the class `additive_fft<word>`, the
  Von zur Gathen-Gerhard family:
  - `fft_direct` and `fft_direct_exp` evaluate naively and serve as references.
  - `additive_fft_ref_in_place` and `additive_fft_fast_in_place` are the full DFTs. The fast
    variant needs a buffer of 2^(m-2) words.
  - `additive_fft_rev_ref_in_place` and `additive_fft_rev_fast_in_place` are the inverses.
  - `vzgg_mateer_gao_combination` is for degree d much larger than the number of outputs 2^m. It
    first reduces by the m-th reductor sigma_m (which has few terms), then runs the truncated
    Mateer-Gao DFT.
  - `fold_polynomial` pre-reduces by X^(2^n - 1) - 1 when the degree reaches the multiplicative
    order.
  - `max_dft_size` is capped at min(40, n).
  - Euclidean division by the sparse sigma_i polynomials is in `gf2_extension_polynomials.h`
    (reference and optimized versions). `interleave.h` has in-place and buffered interleaving
    used by the fast variants.
- `decompose_taylor.h` holds the Taylor decomposition in base (x^tau - x), in recursive and
  iterative forms with `_reverse` counterparts. The `_alt` iterative variants are the ones the
  Mateer-Gao code calls.
- `mateer_gao.h` is the generic (any `word`) Mateer-Gao implementation. It is header-only
  templates.
  - `fft_aux_ref*` are the recursive kernels (full, truncated, offset, reverse, and `_mult`
    variants that work in the mult representation).
  - `fft_mateer`, `fft_mateer_truncated`, `fft_mateer_truncated_reverse` and the `_mult`
    versions are the entry points.
  - `fft_mateer_truncated_mult_smalldegree` is the small-degree-input optimization: it does one
    Taylor step, replicates the result and runs one smaller DFT per block.
  - `fft_mateer<word, s>` requires n <= 64 and s <= 5. The truncated entry points allow s <= 6.
- `polynomial_product.h` is a generic (any `word`) binary-polynomial product on top of
  `mateer_gao.h`. It is not used by any build target. The specialized one is below.

### The fast product (`mg.h`, `libfft/mg.cpp`, `mg_templates.h`)

This is what `product_test` benchmarks. It is specialized for GF(2^64) and uses PCLMULQDQ.

- The field is GF(2^64) with minimal polynomial x^64 + x^4 + x^3 + x + 1 (`product`,
  `product_batch`: four products per iteration).
- `init_beta_table()` derives the beta basis in the mult representation from a start value and
  the relation beta_i = beta_{i+1}^2 + beta_{i+1}, retrying until beta_0 = 1. It is called lazily
  by the product functions.
- Binary polynomials are cut into 32-bit chunks, each placed in a 64-bit word (`expand`). After
  the pointwise product and the inverse DFT, `contract` xors the overlapping halves back
  together. Both are little-endian only.
- `mg_core<s, logstride>` is the truncated forward DFT kernel. `mg_reverse_core` is its inverse.
  `mg_smalldegree<s>` is the forward DFT for inputs shorter than the output.
  `mg_smalldegree_with_buf<s>` does the same for the second operand and multiplies the result
  pointwise into the first operand's buffer.
- `mg_binary_polynomial_multiply` (out of place) and `mg_binary_polynomial_multiply_in_place`
  (result in `p1`, which must hold twice the result size rounded up to a power of 2) are the
  public entry points. `naive_product` is a PCLMUL schoolbook reference.
- Compile-time switch: `GRAY_CODE` (off) selects a Gray-code iteration order over the rows.

### The fused variant (`mg_fused.h`, `libfft/mg_fused.cpp`)

A copy of `mg.cpp` in namespace `mg_fused`, with the following changes (the original is meant to stay
a 'pure' implementation).

- `mg_core`, `mg_reverse_core`, `mg_smalldegree` and `mg_smalldegree_with_buf` take a `cut` argument:
  0 (plain), 16 or 4. With `cut = 16` (`s == 3`, `logstride == 0`) or `cut = 4` (`s == 2`), the loop
  on rows is skipped in the forward and in the reverse DFT. `mg_cut_level` selects 16 if all DFTs
  have at least 2**5 values, else 4 if at least 2**3, else 0 (`-DMG_CUT_LEVEL=n` forces a level).
- Rows are 16 (or 4) coefficients in the monomial basis, to be evaluated on a coset a + GF(16) of the
  subfield GF(16) (the Cantor basis spans subfields: x**R - x vanishes on GF(R) for R = 2**(2**j)).
  DFT, pointwise product and inverse DFT on a row are therefore a product modulo
  x**R + x + c, c = a**R + a, which is `offsets_mult[log2 R]` in `mg_core` (and, for row q, c xor the
  beta image of q). `ring_rows` / `ring_mul` / `ring_mul2` compute these products with Karatsuba on
  unreduced 128-bit clmul results, then reduce (r_k = p_k + p_{k+R-1} + c p_{k+R}), two rows at once in
  a ymm register when `MG_USE_YMM_CLMUL` is defined.
- `product_batch` is replaced by `mg_ring_rows` when `cut != 0`.
- Base case: when the operands have at most 2**`MG_FUSED_BASE_LOGN` (7) 64-bit words, the two public functions
  call `base_product` instead (no DFT, no allocation): Karatsuba on unreduced `pclmulqdq` products with a
  schoolbook base case of `MG_FUSED_BASE_KT` (4) words, `kmul` unrolled up to 16 words, `kmul_big` (recursive,
  `alloca`) above. The result words are written to the output directly (the result of the in-place
  function is written over `p1` after the operands were copied). Crossover with the DFT-based product, measured on a
  5950X: base case 1.3x faster at 2**14 bits, equal at 2**15, slower above. `-DMG_FUSED_BASE_LOGN=-1` disables it.
- Measured on a Ryzen 9 5950X, a row product is about 2x (R = 4, xmm) to 3.3x (R = 16, ymm, two rows)
  faster than the DFT chain, and slower for R = 256 (3.5x).

### Tests and utilities

- `product_test.cpp` multiplies random polynomials of degree 2^(lsz-1) - 1. The default list is
  `{8..19, 25, 30, 34}`. gf2x comparison is limited to lsz <= 25 (`gf2x_max_logsize`). Settings:
  `test_in_place_variant`, `test_naive_product`, `benchmark`, `min_time` (2 s, runs are timed
  by batches of at least `batch_min_time` = 1 ms) and `max_runs`. Exit code 0 means all comparisons passed. It requires the CPU flags AVX2, PCLMULQDQ,
  POPCNT and BMI1, and exits with code 2 otherwise.
  - `degree_test` first compares the in-place and out-of-place products of both variants with a naive product for all degrees d1 >= d2 up to 300, with minimal buffers and checking that the bits above the degree of the result are zero. It covers the operands of at most 64 bits and the degrees that are not of the form 2**k - 1.
- `fft_test.cpp` runs six tests, switchable through `run_tests[]`: VzGG DFT against direct
  evaluation, its reverse, truncated Mateer-Gao (and the VzGG/MG combination), the reverse
  truncated Mateer-Gao, Taylor decomposition, and DFT inversion. Top-of-file knobs: `word`
  (`uint32_t` or `uint64_t` only, because the mult-representation functions exist for those),
  `allocated_buf_size` (1 GB), `check_correctness`, `benchmark`, `full_mg_tests`,
  `full_dft_test_size_limit` and the `log_sz` list `{20, 24}`. It only checks for SSE2 and
  PCLMULQDQ.
- `utils.cpp` and `include/utils.h` provide CPUID feature detection (`cpu_features`) and
  printing and comparison helpers. `include/timer.h` is a small timer. `include/base.h` defines
  `restr` and `UNUSED`.

### Data files

`2020-10-results_not_in_place.csv`, `2021-05-results_in_place.csv` and `runtimes.ods` hold the
raw benchmark numbers. The README table matches the in-place CSV (a `0` in the gf2x column means
out of memory). `times.png` and `speed_ratio.png` are the plots the README embeds.

## Things to know

- **Memory use of the product.** A product whose result has 2^L bits uses a DFT buffer of 2^L / 4
  bytes, which is twice the result size. In-place, the second operand's expansion buffer is
  as large as the result. This matches the README's numbers.
- **Vector code paths.** When the build defines `__VPCLMULQDQ__` and `__AVX2__` (true with
  `-march=native` on Zen 3+ / Ice Lake+), `mg_templates.h` defines `MG_USE_YMM_CLMUL`, and
  `eval_degree1` and `product_batch` use 256-bit `_mm256_clmulepi64_epi128` (two products per
  instruction); `product_test` then also requires the CPU flag VPCLMULQDQ. Otherwise a 128-bit
  path is used. `eval_degree1_ref` is the original scalar-reduction version (select it with
  `MG_USE_REF_EVAL_DEGREE1`).
- **Reduction by table.** In the products, after `a*b` and `hi*minpoly`, the remaining 4-bit
  spill is reduced with a 16-entry `pshufb` table (`mg_lut128`) instead of a third clmul.
- **Row kernels.** `mg_row4_forward` / `mg_row4_reverse` (`libfft/mg.cpp`) handle the lowest
  recursion level (`s = 1`, `logstride = 0`, rows of 4 words) entirely in two registers.
  `-DMG_ROW4=0` disables them.
- **Large fields.** Truncated DFT loop indexes are `uint64_t`, so word types beyond 64 bits work
  for Cantor-basis construction but not reliably for DFTs.
