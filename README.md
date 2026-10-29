# Additive Discrete Fourier Transforms (DFTs)
A C++20 implementation of several additive DFT algorithms over F2, and their application to fast binary polynomial multiplication.

The implementation targets the x86_64 architecture with the SSE2, SSE4.1, POPCNT and PCLMULQDQ instruction sets (the build uses `-march=native`; `product_test` additionally requires AVX2 and BMI1, and also VPCLMULQDQ if the build targets it, i.e. on CPUs such as Zen 3+ or Intel Ice Lake+). Test programs showcase the different algorithms and test their correctness. See [CODE_GUIDE.md](CODE_GUIDE.md) for a map of the source code.

Algorithms implemented are: Von zur Gathen - Gerhard algorithm with Cantor bases (also called the Wang-Zhu-Cantor algorithm), and the Mateer-Gao algorithm. Both of them are described in Todd Mateer's Master's Thesis (available at https://tigerprints.clemson.edu/all_dissertations/231/). Reverse algorithms are also implemented.

Truncated versions of these algorithms are available to perform only a part of the DFT computation. Provided with an input polynomial degree < v = 2<sup>u</sup>, the v first coefficients of the DFT can be computed efficiently (if the underlying finite field has size w, the time to compute the truncated DFT is less than v/w times the full DFT time). Truncated reverse transforms have the same complexity and build a polynomial of degree < v that takes the provided values.

A combination of Wang-Zhu-Cantor and Mateer-Gao is also implemented to tackle the case of the computation of u values of a polynomial of degree d where u is much smaller than d.

This project is about experimenting with these DFT algorithms. Reference implementations that are close to the high-level description of the algorithms are therefore kept alongside more optimized versions that are more difficult to read. As a result there is some redundancy in the source code.

## Application to fast product of binary polynomials

Mateer-Gao DFT (and crucially, its truncated implementation) is used to implement fast multiplication of binary polynomials (see https://cr.yp.to/f2mult.html for an outline of the method used). DFTs are used to evaluate the polynomials, then the values obtained for both polynomials are multiplied pointwise, and finally the product is built from the values with an inverse DFT). It is tested and benched against the fastest product implementation known to the author, from library `gf2x`, version 1.3 (https://gitlab.inria.fr/gf2x/gf2x).

The DFT approach is tested by multiplying polynomials of degree 2<sup>u-1</sup> - 1 for values of u s.t. the computations fit into memory. By default `product_test` runs u = 8...19, 25, 30 and 34, and compares results with gf2x only up to u = 25; the list of sizes and these limits are constants at the top of `product_test.cpp`. With the plain variant, it is slower than gf2x for small sizes (u < 18 on an AMD 5950X), and faster beyond that (see the fused variant below). It also uses less memory for large sizes. 
Both an in-place and a not-in-place product variant are provided. The in-place variant uses less memory and is slightly faster.

### Product computation time results

Experiments on a PC with an AMD Ryzen 9 5950X CPU and 64GB of DDR4 RAM @1800MHz (3600 MT/s), with gf2x 1.3, are summarized below. Results are for the in-place product variant. The code was compiled with gcc using `-march=native` (which enables the 256-bit `vpclmulqdq` code path on this CPU). Each time is an average over repeated runs, which are timed in batches (a batch lasts at least 1 ms, and runs are repeated until 2 s have elapsed), as reading the clock takes about 25 ns, which is not negligible for the smallest sizes. The figures for u = 10...30 were obtained by running `product_test` with `gf2x_max_logsize = 30` and the size list set to u = 8...30; those for u = 31...35 were obtained with a previous version of `product_test` that timed each run (which makes no difference at these sizes). The raw numbers are in `2026-10-results_in_place.csv`. Sizes above 2<sup>35</sup> were not measured (see the memory requirements below: the in-place product of size 2<sup>36</sup> needs 28 GiB, and that of size 2<sup>37</sup> needs 56 GiB, which is close to the 64GB available).

|Output log2 size|gf2x product time (sec.)|Mateer-Gao product time (sec.)|
|:----:|:----:|:----:|
|10| 2.046e-08| 3.397e-07|
|11| 8.967e-08| 7.335e-07|
|12| 2.663e-07| 1.585e-06|
|13| 8.018e-07| 3.322e-06|
|14| 2.447e-06| 7.171e-06|
|15| 7.89e-06| 1.553e-05|
|16| 2.296e-05| 3.218e-05|
|17| 6.528e-05| 6.825e-05|
|18| 0.0001819| 0.000145|
|19| 0.0004993| 0.0003072|
|20| 0.001584| 0.000665|
|21| 0.003457| 0.001381|
|22| 0.007146| 0.003006|
|23| 0.01583| 0.006355|
|24| 0.03877| 0.01325|
|25| 0.08161| 0.02786|
|26| 0.1932| 0.058|
|27| 0.3703| 0.1239|
|28| 0.9148| 0.3017|
|29| 2.449| 0.6817|
|30| 3.67| 1.484|
|31| 9.827| 3.221|
|32| 22.11| 7.256|
|33| 87.79| 15.9|
|34| 133.1| 35.14|
|35| 574.1| 76.79|

![Execution times](https://github.com/kunzjacq/Additive_DFTs/blob/master/times.png?raw=true)

![Speed ratio](https://github.com/kunzjacq/Additive_DFTs/blob/master/speed_ratio.png?raw=true)

### Fused variant

The plain product above (`libfft/mg.cpp`, `include/mg.h`) is a direct implementation of the Mateer-Gao algorithm. A second implementation, `libfft/mg_fused.cpp` (`include/mg_fused.h`, namespace `mg_fused`), has the same interface and gives the same results, but is 1.05 to 15 times faster (the largest gains are at the smallest sizes); it is a copy of `mg.cpp` that adds optimizations and leaves the original untouched.

**Principle.** The DFT is computed recursively on a square array: the Taylor decomposition and DFTs on columns first, then DFTs on rows. In the product, the forward DFTs of both operands, the pointwise product and the first stages of the inverse DFT are chained, so at the end of the forward DFTs the rows are polynomials of degree < R (R = 4, 16 or 256 words) evaluated on a coset a + GF(R) of the subfield GF(R) of GF(2<sup>64</sup>), which is possible because the beta basis is a Cantor basis. For a row, "DFT of both operands, pointwise product, inverse DFT" is therefore just a product of two polynomials of degree < R modulo the polynomial vanishing on the coset, x<sup>R</sup> + x + c, with c = a<sup>R</sup> + a (c is already known to the DFT code: it is the offset of the next recursion level). The fused variant does not perform these DFT stages on rows of 16 words (4 words for the smallest sizes): it computes the product of the rows directly.

**Implementation.** A row product is computed with Karatsuba on carry-less 64x64-bit products that are left unreduced (the reduction is linear over F2 and done once at the end), then reduced modulo x<sup>R</sup> + x + c, which costs one multiplication by c per coefficient since x<sup>R</sup> = x + c. When the CPU supports it (see above), two rows are processed at once with 256-bit `vpclmulqdq`; otherwise a 128-bit version is used. The direct product beats the chain of DFTs for R = 4 and 16 (by a factor of 2 to 3 on a row) but not for R = 256, hence the choice of these row sizes. See `CODE_GUIDE.md` for more details.

**Base case.** For small products (operands of at most 2<sup>15</sup> bits, i.e. 512 64-bit words, so results of up to 2<sup>16</sup> bits), the DFT is not used at all: the product is computed directly by Karatsuba on carry-less 64x64-bit products (one `pclmulqdq` for operands of at most 64 bits, three for 128 bits, and so on). The operands are processed as arrays of 128-bit digits (2 words) kept in vector registers, which makes the additions and the reassembly of the result cheap; this is the organization of the hand-crafted small products of gf2x. This is faster than the DFT-based product up to 2<sup>16</sup> bits (about 1.15 times faster at 2<sup>16</sup>, and more than 10 times faster at 2<sup>8</sup>, compared to the fused DFT-based product), and slower above (1.2 times at 2<sup>17</sup>). This crossover was set by experiment (macro `MG_FUSED_BASE_LOGN` in `mg_fused.cpp`; `MG_FUSED_BASE_LOGN=-1` disables the base case).

**Limitations and origin of the base case.** All the timings in this document are for polynomials of degree 2<sup>u-1</sup> - 1, i.e. for sizes that are powers of 2. Unlike gf2x, the implementation (both variants) is not optimized for other sizes: the size of the DFTs, and the size of the operands in the base case, are rounded up to a power of 2, so that the time of a product only increases in steps, at the powers of 2, and the product of two polynomials of a size just above a power of 2 takes about as long as a product of the next power of 2. gf2x on the other hand has code specialized for many sizes (1 to 9 words, then a choice of Toom-Cook variants for each size), and its time increases more smoothly. The organization of the base case (Karatsuba on 128-bit digits kept in vector registers) was inspired by the hand-crafted small products of gf2x, but the code was written independently and is not copied from gf2x.

**Use.** The functions of `mg_fused.h` are drop-in replacements for those of `mg.h`. `product_test` uses them if the constant `use_fused_variant` at the top of `product_test.cpp` is set to `true`. Both variants are tested by `degree_test` in `product_test`.

**Results.** Same machine as above (AMD 5950X, in-place product, same measurement method; these are the same figures as the first two columns of the table above for the sizes in common). Plain and fused variants were measured in separate runs of `product_test` (for sizes up to 2<sup>20</sup>, the fused column was measured again after the base case was added), and the results of both variants are checked against gf2x at each size. For gf2x, the smallest of the measured times is given. For sizes 2<sup>31</sup> to 2<sup>35</sup>, only the fused variant was measured again (one run per size, in-place, with `use_fused_variant = true`); the gf2x and plain Mateer-Gao columns are those of the table above:

|Output log2 size|gf2x (sec.)|Mateer-Gao (sec.)|fused Mateer-Gao (sec.)|speedup of fused over plain|speed ratio fused / gf2x|
|:----:|:----:|:----:|:----:|:----:|:----:|
|8| 6.505e-09| 9.498e-08| 6.508e-09| 14.59| 1.00|
|9| 9.667e-09| 1.742e-07| 1.303e-08| 13.37| 0.74|
|10| 2.046e-08| 3.397e-07| 2.396e-08| 14.18| 0.85|
|11| 8.967e-08| 7.335e-07| 6.535e-08| 11.22| 1.37|
|12| 2.663e-07| 1.585e-06| 1.981e-07| 8.00| 1.34|
|13| 8.018e-07| 3.322e-06| 5.865e-07| 5.66| 1.37|
|14| 2.414e-06| 7.171e-06| 1.761e-06| 4.07| 1.37|
|15| 7.832e-06| 1.553e-05| 5.354e-06| 2.90| 1.46|
|16| 2.273e-05| 3.218e-05| 1.612e-05| 2.00| 1.41|
|17| 6.528e-05| 6.825e-05| 4.961e-05| 1.38| 1.32|
|18| 0.0001795| 0.000145| 0.000109| 1.33| 1.65|
|19| 0.0004964| 0.0003072| 0.0002358| 1.30| 2.11|
|20| 0.001584| 0.000665| 0.0004991| 1.33| 3.17|
|21| 0.003457| 0.001381| 0.001066| 1.29| 3.24|
|22| 0.007146| 0.003006| 0.002372| 1.27| 3.01|
|23| 0.01583| 0.006355| 0.005072| 1.25| 3.12|
|24| 0.03877| 0.01325| 0.01074| 1.23| 3.61|
|25| 0.08161| 0.02786| 0.02264| 1.23| 3.60|
|26| 0.1932| 0.058| 0.04771| 1.22| 4.05|
|27| 0.3703| 0.1239| 0.1026| 1.21| 3.61|
|28| 0.9148| 0.3017| 0.2563| 1.18| 3.57|
|29| 2.449| 0.6817| 0.5878| 1.16| 4.17|
|30| 3.67| 1.484| 1.297| 1.14| 2.83|
|31| 9.827| 3.221| 2.987| 1.08| 3.29|
|32| 22.11| 7.256| 6.718| 1.08| 3.29|
|33| 87.79| 15.9| 15.08| 1.05| 5.82|
|34| 133.1| 35.14| 33.2| 1.06| 4.01|
|35| 574.1| 76.79| 72.86| 1.05| 7.88|

The fused variant is 8 to 15 times faster than the plain one up to 2<sup>10</sup> bits, thanks to the base case, 3 to 11 times faster from 2<sup>11</sup> to 2<sup>15</sup>, 2 times faster at 2<sup>16</sup>, and the advantage then decreases from 1.3x at 2<sup>20</sup> to 1.1 - 1.2x at 2<sup>28</sup> to 2<sup>30</sup> and 1.05 - 1.08x from 2<sup>31</sup> to 2<sup>35</sup>, where the last DFT stages weigh little. The fused variant is as fast as gf2x at 2<sup>8</sup> and 1.2 to 1.35 times slower at 2<sup>9</sup> and 2<sup>10</sup>, where gf2x uses hand-crafted code for each size. It is faster than gf2x from 2<sup>11</sup> on (1.3 to 1.5 times faster up to 2<sup>17</sup>, then 1.7 times at 2<sup>18</sup> and more than 2 times from 2<sup>19</sup>). The plain variant is faster than gf2x from 2<sup>18</sup> on.

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

The polynomial product code uses a fast implementation of multiplication in GF(2<sup>64</sup>), which is implemented multiple ways in the source, using PCLMULQDQ, POPCNT and SSE intrinsics. `product_test` checks at startup that the CPU reports AVX2, BMI1, POPCNT and PCLMULQDQ (and VPCLMULQDQ when the build targets it, see `MG_USE_YMM_CLMUL` in `include/mg_templates.h`), and exits otherwise; `fft_test` only checks for SSE2 and PCLMULQDQ.

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
