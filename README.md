# Additive Discrete Fourier Transforms (DFTs)
A C++20 implementation of several additive DFT algorithms over F2, and their application to fast binary polynomial multiplication.

The implementation targets the x86_64 architecture with the SSE2, SSE4.1, POPCNT and PCLMULQDQ instruction sets (the build uses `-march=native`; `product_test` additionally requires AVX2 and BMI1, and also VPCLMULQDQ if the build targets it, i.e. on CPUs such as Zen 3+ or Intel Ice Lake+). Test programs showcase the different algorithms and test their correctness. See [CODE_GUIDE.md](CODE_GUIDE.md) for a map of the source code.

Algorithms implemented are: Von zur Gathen - Gerhard algorithm with Cantor bases (also called the Wang-Zhu-Cantor algorithm), and the Mateer-Gao algorithm. Both of them are described in Todd Mateer's Master's Thesis (available at https://tigerprints.clemson.edu/all_dissertations/231/). Reverse algorithms are also implemented.

Truncated versions of these algorithms are available to perform only a part of the DFT computation. Provided with an input polynomial degree < v = 2<sup>u</sup>, the v first coefficients of the DFT can be computed efficiently (if the underlying finite field has size w, the time to compute the truncated DFT is less than v/w times the full DFT time). Truncated reverse transforms have the same complexity and build a polynomial of degree < v that takes the provided values.

A combination of Wang-Zhu-Cantor and Mateer-Gao is also implemented to tackle the case of the computation of u values of a polynomial of degree d where u is much smaller than d.

This project is about experimenting with these DFT algorithms. Reference implementations that are close to the high-level description of the algorithms are therefore kept alongside more optimized versions that are more difficult to read. As a result there is some redundancy in the source code.

## Application to fast product of binary polynomials

Mateer-Gao DFT (and crucially, its truncated implementation) is used to implement fast multiplication of binary polynomials (see https://cr.yp.to/f2mult.html for an outline of the method used). DFTs are used to evaluate the polynomials, then the values obtained for both polynomials are multiplied pointwise, and finally the product is built from the values with an inverse DFT). It is tested and benched against the fastest product implementation known to the author, from library `gf2x`, version 1.3 (https://gitlab.inria.fr/gf2x/gf2x).

The DFT approach is tested by multiplying polynomials of degree 2<sup>u-1</sup> - 1 for values of u s.t. the computations fit into memory. By default `product_test` runs u = 8...19, 25, 30 and 34, and compares results with gf2x only up to u = 25; the list of sizes and these limits are constants at the top of `product_test.cpp`. On an AMD 5950X, the fused variant is faster than gf2x from u = 11, with a growing advantage for large sizes (see the results below). The plain variant is a slower reference implementation. Both use less memory than gf2x for large sizes. 
Both an in-place and a not-in-place product variant are provided. The in-place variant uses less memory and is slightly faster.

### Fused variant

Two implementations of the product are provided. The fused variant (`libfft/mg_fused.cpp`, `include/mg_fused.h`, namespace `mg_fused`) is the fastest; the plain variant (`libfft/mg.cpp`, `include/mg.h`), described [below](#plain-variant), is a direct implementation of the Mateer-Gao algorithm. Both have the same interface and give the same results; the fused variant is 1.1 to 15 times faster than the plain one (the largest gains are at the smallest sizes). It is a copy of `mg.cpp` that adds optimizations and leaves the original untouched. It is faster than gf2x from 2<sup>11</sup> bits on.

**Principle.** The DFT is computed recursively on a square array: the Taylor decomposition and DFTs on columns first, then DFTs on rows. In the product, the forward DFTs of both operands, the pointwise product and the first stages of the inverse DFT are chained, so at the end of the forward DFTs the rows are polynomials of degree < R (R = 4, 16 or 256 words) evaluated on a coset a + GF(R) of the subfield GF(R) of GF(2<sup>64</sup>), which is possible because the beta basis is a Cantor basis. For a row, "DFT of both operands, pointwise product, inverse DFT" is therefore just a product of two polynomials of degree < R modulo the polynomial vanishing on the coset, x<sup>R</sup> + x + c, with c = a<sup>R</sup> + a (c is already known to the DFT code: it is the offset of the next recursion level). The fused variant does not perform these DFT stages on rows of 16 words (4 words for the smallest sizes): it computes the product of the rows directly.

**Implementation.** A row product is computed with Karatsuba on carry-less 64x64-bit products that are left unreduced (the reduction is linear over F2 and done once at the end), then reduced modulo x<sup>R</sup> + x + c, which costs one multiplication by c per coefficient since x<sup>R</sup> = x + c. When the CPU supports it (see above), two rows are processed at once with 256-bit `vpclmulqdq`; otherwise a 128-bit version is used. The direct product beats the chain of DFTs for R = 4 and 16 (by a factor of 2 to 3 on a row) but not for R = 256, hence the choice of these row sizes. See `CODE_GUIDE.md` for more details.

**Base case.** For small products (operands of at most 2<sup>15</sup> bits, i.e. 512 64-bit words, so results of up to 2<sup>16</sup> bits), the DFT is not used at all: the product is computed directly by Karatsuba on carry-less 64x64-bit products (one `pclmulqdq` for operands of at most 64 bits, three for 128 bits, and so on). The operands are processed as arrays of 128-bit digits (2 words) kept in vector registers, which makes the additions and the reassembly of the result cheap; this is the organization of the hand-crafted small products of gf2x. This is faster than the DFT-based product up to 2<sup>16</sup> bits (about 1.15 times faster at 2<sup>16</sup>, and more than 10 times faster at 2<sup>8</sup>, compared to the fused DFT-based product), and slower above (1.2 times at 2<sup>17</sup>). This crossover was set by experiment (macro `MG_FUSED_BASE_LOGN` in `mg_fused.cpp`; `MG_FUSED_BASE_LOGN=-1` disables the base case).

**Taylor decompositions.** The fused variant also uses faster Taylor decompositions (`include/mg_taylor_opt.h`): two levels of the recursion are merged into one step, and the decomposition of rows of 8 and 16 words is done in AVX2 registers. They are worth 3 to 6% of the time of a product at sizes of 2<sup>18</sup> bits and more (where the DFT is used), and give the same results as the original functions. The plain variant still uses the original ones. See [taylor_merge.md](taylor_merge.md).

**Use.** The functions of `mg_fused.h` are drop-in replacements for those of `mg.h`. `product_test` uses them if the constant `use_fused_variant` at the top of `product_test.cpp` is set to `true`. Both variants are tested by `degree_test` in `product_test`.

**Limitations and origin of the base case.** All the timings in this document are for polynomials of degree 2<sup>u-1</sup> - 1, i.e. for sizes that are powers of 2. Unlike gf2x, the implementation (both variants) is not optimized for other sizes: the size of the DFTs, and the size of the operands in the base case, are rounded up to a power of 2, so that the time of a product only increases in steps, at the powers of 2, and the product of two polynomials of a size just above a power of 2 takes about as long as a product of the next power of 2. gf2x on the other hand has code specialized for many sizes (1 to 9 words, then a choice of Toom-Cook variants for each size), and its time increases more smoothly. The organization of the base case (Karatsuba on 128-bit digits kept in vector registers) was inspired by the hand-crafted small products of gf2x, but the code was written independently and is not copied from gf2x.

#### Fused variant: product computation time results

Experiments on a PC with an AMD Ryzen 9 5950X CPU and 64GB of DDR4 RAM @1800MHz (3600 MT/s), with gf2x 1.3, are summarized below. Results are for the in-place product variant. The code was compiled with gcc 13.3 using `-march=native` (which enables the 256-bit `vpclmulqdq` code path on this CPU). All figures, for both variants and for gf2x, come from one run of the script `benchmark.sh` (the raw numbers are in `2026-10-results_in_place.csv`). Each time is an average over repeated runs, which are timed in batches (a batch lasts at least 1 ms, and runs are repeated until 2 s have elapsed; for the largest sizes, this means a single run), as reading the clock takes about 25 ns, which is not negligible for the smallest sizes. The results of the plain variant are checked against gf2x at every size up to 2<sup>35</sup>, and those of the fused variant up to 2<sup>25</sup> (`benchmark.sh` limits the comparison in the fused run to save time; the fused times were not compared with gf2x results above that). gf2x was not run for u = 36 and 37 (not enough memory); the in-place product of size 2<sup>36</sup> needs 28 GiB and that of size 2<sup>37</sup> needs 56 GiB (see the memory requirements below), which is close to the 64GB available.

|Output log2 size|gf2x (sec.)|fused Mateer-Gao (sec.)|speed ratio (gf2x time / fused time)|
|:----:|:----:|:----:|:----:|
|8| 6.451e-09| 6.497e-09| 0.99|
|9| 9.516e-09| 1.288e-08| 0.74|
|10| 2.016e-08| 2.417e-08| 0.83|
|11| 9.491e-08| 6.641e-08| 1.43|
|12| 2.795e-07| 1.999e-07| 1.40|
|13| 8.113e-07| 5.923e-07| 1.37|
|14| 2.503e-06| 1.778e-06| 1.41|
|15| 7.762e-06| 5.392e-06| 1.44|
|16| 2.282e-05| 1.632e-05| 1.40|
|17| 6.455e-05| 4.648e-05| 1.39|
|18| 0.0001793| 0.0001018| 1.76|
|19| 0.0004933| 0.0002203| 2.24|
|20| 0.001261| 0.0004674| 2.70|
|21| 0.003501| 0.001037| 3.38|
|22| 0.00715| 0.002186| 3.27|
|23| 0.01605| 0.004945| 3.25|
|24| 0.03914| 0.01044| 3.75|
|25| 0.08253| 0.02202| 3.75|
|26| 0.1965| 0.04765| 4.12|
|27| 0.3784| 0.1013| 3.74|
|28| 0.9347| 0.2442| 3.83|
|29| 2.497| 0.559| 4.47|
|30| 3.761| 1.234| 3.05|
|31| 9.937| 2.833| 3.51|
|32| 22.08| 6.355| 3.47|
|33| 87.75| 14.19| 6.18|
|34| 130.8| 31.47| 4.16|
|35| 572.3| 68.93| 8.30|
|36| not enough memory| 150.7| -|
|37| not enough memory| 329.1| -|

![Execution times](times.png)

![Speed ratio](speed_ratio.png)

The graphs also show the plain variant (grey dashed lines), described below. The fused variant is as fast as gf2x at 2<sup>8</sup> bits and 1.2 to 1.35 times slower at 2<sup>9</sup> and 2<sup>10</sup>, where gf2x uses hand-crafted code for each size. It is faster than gf2x from 2<sup>11</sup> on: 1.4 times faster up to 2<sup>17</sup>, then 1.8 times at 2<sup>18</sup>, 2.2 times at 2<sup>19</sup> and 2.7 times at 2<sup>20</sup>; from 2<sup>21</sup> to 2<sup>35</sup> it is 3 to 8 times faster. The gf2x times are irregular for large sizes (the ratio is 3.05 at 2<sup>30</sup> and 6.18 at 2<sup>33</sup>, for instance); the Mateer-Gao times, on the other hand, grow regularly, by a factor of about 2.2 each time the size doubles. The product of size 2<sup>37</sup> takes 329 s.

### Plain variant

The plain product (`libfft/mg.cpp`, `include/mg.h`) is a direct implementation of the Mateer-Gao DFT, without the optimizations of the fused variant. It is kept as a reference, as it is closer to the description of the algorithm (see [CODE_GUIDE.md](CODE_GUIDE.md)). Its memory requirements are the same as those of the fused variant. Its times, measured as described above in the same run, are compared to those of the fused variant below:

|Output log2 size|plain Mateer-Gao (sec.)|fused Mateer-Gao (sec.)|fused is faster than plain by a factor|
|:----:|:----:|:----:|:----:|
|8| 9.546e-08| 6.497e-09| 14.69|
|9| 1.716e-07| 1.288e-08| 13.33|
|10| 3.524e-07| 2.417e-08| 14.58|
|11| 7.439e-07| 6.641e-08| 11.20|
|12| 1.585e-06| 1.999e-07| 7.93|
|13| 3.357e-06| 5.923e-07| 5.67|
|14| 7.106e-06| 1.778e-06| 4.00|
|15| 1.55e-05| 5.392e-06| 2.87|
|16| 3.245e-05| 1.632e-05| 1.99|
|17| 6.849e-05| 4.648e-05| 1.47|
|18| 0.0001462| 0.0001018| 1.44|
|19| 0.0003095| 0.0002203| 1.40|
|20| 0.000659| 0.0004674| 1.41|
|21| 0.001402| 0.001037| 1.35|
|22| 0.002979| 0.002186| 1.36|
|23| 0.006423| 0.004945| 1.30|
|24| 0.01346| 0.01044| 1.29|
|25| 0.0281| 0.02202| 1.28|
|26| 0.059| 0.04765| 1.24|
|27| 0.1249| 0.1013| 1.23|
|28| 0.2987| 0.2442| 1.22|
|29| 0.6726| 0.559| 1.20|
|30| 1.466| 1.234| 1.19|
|31| 3.145| 2.833| 1.11|
|32| 7.037| 6.355| 1.11|
|33| 15.64| 14.19| 1.10|
|34| 34.7| 31.47| 1.10|
|35| 75.71| 68.93| 1.10|
|36| 169.4| 150.7| 1.12|
|37| 370.8| 329.1| 1.13|

The fused variant is 13 to 15 times faster than the plain one up to 2<sup>10</sup> bits, 2.9 to 11 times faster from 2<sup>11</sup> to 2<sup>15</sup>, 2 times faster at 2<sup>16</sup>, and the difference then decreases: 1.4 to 1.5 times at 2<sup>17</sup> to 2<sup>20</sup>, 1.2 times at 2<sup>27</sup> to 2<sup>30</sup>, and 1.1 times from 2<sup>31</sup> on.

**Where the differences come from.** They follow from the three optimizations of the fused variant described above:
* *Small sizes (up to 2<sup>16</sup> bits):* the plain variant always runs the full DFT machinery (expansion of the operands into 64-bit words, three DFTs, pointwise product, contraction), whose fixed cost dominates for small products, whereas the fused variant multiplies directly with Karatsuba on carry-less products (base case). This is the source of the factors of 10 and more.
* *Medium and large sizes:* the fused variant replaces the last stages of the DFTs, on rows of 16 words (4 for the smallest sizes), together with the pointwise product, by a direct product of the rows modulo x<sup>R</sup> + x + c, which is cheaper than the chain of DFT, product and inverse DFT on such a row (by a factor of 2 to 3 on a row). The faster Taylor decompositions add 3 to 6%. The gain is the same fraction of work at each size, but it is a smaller part of the total as the DFT grows, since the number of stages grows with the size while the number of stages replaced is fixed; this is why the advantage decreases from 1.4 to 1.1 as the size grows.

### Product memory requirements

The functions of `mg.h` and `mg_fused.h` have the same memory requirements (the base case of the fused variant, for small products, only uses a few tens of KB of stack).

**Notations.** A product q = p<sub>1</sub> p<sub>2</sub> is computed, where the storage of p<sub>1</sub> and p<sub>2</sub> together uses n bytes (one bit per coefficient). The storage of q is then at most n bytes. Let m be the smallest power of 2 such that n ≤ m, i.e. 2<sup>k-1</sup> < n ≤ 2<sup>k</sup> = m. The DFTs are computed on arrays of 64-bit words holding 32 bits of the polynomial each, so the array that holds the DFT of a polynomial of the size of q uses twice the size of q, rounded up to a power of 2, i.e. 2m bytes.

**Out-of-place variant** (`mg_binary_polynomial_multiply`). The function allocates two arrays of 2m bytes (one DFT array per operand), in addition to the buffers that store p<sub>1</sub>, p<sub>2</sub> and q (n bytes for p<sub>1</sub> and p<sub>2</sub>, at most n bytes for q). In total, M = 2n + 4m bytes of RAM are needed, with 5m < M ≤ 6m.

**In-place variant** (`mg_binary_polynomial_multiply_in_place`). The result q is stored into the buffer of p<sub>1</sub>, which must be large enough to hold the DFT: 2m bytes. The buffer of p<sub>2</sub> is modified, and the function allocates an additional array to compute the DFT of p<sub>2</sub> (by blocks), of twice the size of p<sub>2</sub> rounded up to a power of 2. Exchanging p<sub>1</sub> and p<sub>2</sub> if necessary, we can assume deg(p<sub>2</sub>) ≤ deg(p<sub>1</sub>); the size of p<sub>2</sub> is then at most m/2 bytes. The in-place product therefore needs at most m/2 (p<sub>2</sub>) + 2m (p<sub>1</sub> and q) + 2 × m/2 (additional array) = 3.5m bytes of RAM, which is much less than the out-of-place variant (up to 6m).

**Example.** The memory used by a product of two polynomials of 2<sup>u-1</sup> bits, whose result has 2<sup>u</sup> bits (the sizes of the tables above), is, in GiB (1 GiB = 2<sup>30</sup> bytes; here n = m = 2<sup>u-3</sup> bytes):

|Output log2 size u|input (p<sub>1</sub> or p<sub>2</sub>)|result q|in-place variant (3.5m)|out-of-place variant (6m)|
|:----:|:----:|:----:|:----:|:----:|
|32|0.25|0.5|1.75|3|
|33|0.5|1|3.5|6|
|35|2|4|14|24|
|36|4|8|28|48|
|37|8|16|56|96|

For u = 37, i.e. a product of two polynomials of degree 2<sup>36</sup>-1, each input polynomial is 8 GiB and the result is 16 GiB. With the out-of-place variant, buffers of 2 × 2 × 16 GiB = 64 GiB are additionally needed, for a total of 96 GiB. With the in-place variant, the buffer for p<sub>1</sub> and the result is 2 × 16 GiB = 32 GiB, p<sub>2</sub> takes 8 GiB, and an additional array of 2 × 8 GiB = 16 GiB is needed. Overall 8 + 16 + 32 = 56 GiB of RAM are needed.

### Product CPU requirements

The polynomial product code uses a fast implementation of multiplication in GF(2<sup>64</sup>), which is implemented multiple ways in the source, using PCLMULQDQ, POPCNT and SSE intrinsics. `product_test` checks at startup that the CPU reports AVX2, BMI1, POPCNT and PCLMULQDQ (and VPCLMULQDQ when the build targets it, see `MG_USE_YMM_CLMUL` in `include/mg_templates.h`), and exits otherwise; `fft_test` checks for SSE2 and PCLMULQDQ, plus AVX2 and VPCLMULQDQ when the build targets them (with `-march=native`, gcc vectorizes the PCLMULQDQ loops of the generic code with 256-bit `vpclmulqdq`).

## Cantor bases for large field sizes

Cantor bases are available for field sizes above 2<sup>64</sup> thanks to the optional use of `Boost::multiprecision` which implements large integers (enabled with the CMake option `-D USE_BOOST=ON`, off by default). With this library, the cantor basis test program tests the construction in finite fields of size up to 2<sup>2048</sup>. In theory, truncated Wang-Zhu-Cantor DFT could work in such large fields, however it is not tested and does not work due to overflows in the handling of loop indexes, which are all of type `uint64_t`. There is probably little interest to use these field sizes for truncated DFTs; full DFTs are of course totally unrealistic in these cases.

## Targets
There are three targets which can be built for 64-bit Linux or Windows (with MSVC, only `product_test` is built):
  * `product_test` which benches Mateer-Gao polynomial products (using the finite field GF(2<sup>64</sup>) for DFTs) and checks them against gf2x; its options (`--fused`, `--sizes`, `--gf2x-max`, `--help`) select the variant and the sizes; the script `benchmark.sh` builds it and runs the full benchmark of the tables below (plain and fused variants, gf2x), best from a text console for the largest sizes;
  * `fft_test` which enables to test and benchmark the DFT algorithms discussed above. The binary field and the size of a buffer used in tests determine which tests can be run; these parameters can be adjusted at compile-time at the top of `fft_test.cpp`. The word type must be `uint32_t` or `uint64_t`. If `USE_BOOST` is enabled and `Boost::multiprecision` (https://www.boost.org/doc/libs/1_73_0/libs/multiprecision/doc/html/index.html) is found, Cantor bases and DFT objects can be instantiated for all sizes for which large integers are available from this library. DFT algorithms are not tested (see above) and currently most likely do not work correctly in these large fields. 
  * `cantor_basis_test` which performs various consistency checks on cantor basis construction.

## Build instructions
Under Linux, gcc or clang are able to build all targets. Under windows, MinGW gcc (for instance from MSYS2) can be used, whereas MSVC will only build a limited version of `product_test` that does not use `gf2x`. 

The build process uses Cmake. For instance, an optimized build with gcc using a single-configuration generator such as make, in build directory \$BUILD_DIR and source dir \$SOURCE_DIR, is achieved with:

    cmake -S $SOURCE_DIR -B $BUILD_DIR -D CMAKE_BUILD_TYPE=Release
    cmake --build $BUILD_DIR --parallel

Multi-configuration generators need different options; see https://stackoverflow.com/questions/7724569/debug-vs-release-in-cmake/64719718#64719718. 

Target `product_test` requires a compiled version of gf2x for the target platform to be placed in a subdirectory of the source dir named `lib/`. Alternatively, if an environment variable `CUSTOM_LIBS` is defined, the directory `$CUSTOM_LIBS/lib` will also be searched for gf2x. Link-time optimization is on by default (disable with `-D LTO=OFF`).
