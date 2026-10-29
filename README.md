# Additive Discrete Fourier Transforms (DFTs)
A C++20 implementation of several additive DFT algorithms over F2, and their application to fast binary polynomial multiplication.

The implementation targets the x86_64 architecture with the SSE2, SSE4.1, POPCNT and PCLMULQDQ instruction sets (the build uses `-march=native`; `product_test` additionally requires AVX2 and BMI1, and also VPCLMULQDQ if the build targets it, i.e. on CPUs such as Zen 3+ or Intel Ice Lake+). Test programs showcase the different algorithms and test their correctness. See [CODE_GUIDE.md](CODE_GUIDE.md) for a map of the source code.

Algorithms implemented are: Von zur Gathen - Gerhard algorithm with Cantor bases (also called the Wang-Zhu-Cantor algorithm), and the Mateer-Gao algorithm. Both of them are described in Todd Mateer's Master's Thesis (available at https://tigerprints.clemson.edu/all_dissertations/231/). Reverse algorithms are also implemented.

Truncated versions of these algorithms are available to perform only a part of the DFT computation. Provided with an input polynomial degree < v = 2<sup>u</sup>, the v first coefficients of the DFT can be computed efficiently (if the underlying finite field has size w, the time to compute the truncated DFT is less than v/w times the full DFT time). Truncated reverse transforms have the same complexity and build a polynomial of degree < v that takes the provided values.

A combination of Wang-Zhu-Cantor and Mateer-Gao is also implemented to tackle the case of the computation of u values of a polynomial of degree d where u is much smaller than d.

This project is about experimenting with these DFT algorithms. Reference implementations that are close to the high-level description of the algorithms are therefore kept alongside more optimized versions that are more difficult to read. As a result there is some redundancy in the source code.

## Application to fast product of binary polynomials

Mateer-Gao DFT (and crucially, its truncated implementation) is used to implement fast multiplication of binary polynomials (see https://cr.yp.to/f2mult.html for an outline of the method used). DFTs are used to evaluate the polynomials, then the values obtained for both polynomials are multiplied pointwise, and finally the product is built from the values with an inverse DFT). It is tested and benched against the fastest product implementation known to the author, from library `gf2x`, version 1.3 (https://gitlab.inria.fr/gf2x/gf2x).

The DFT approach is tested by multiplying polynomials of degree 2<sup>u-1</sup> - 1 for values of u s.t. the computations fit into memory. By default `product_test` runs u = 8...19, 25, 30 and 34, and compares results with gf2x only up to u = 25; the list of sizes and these limits are constants at the top of `product_test.cpp`. It is slower than gf2x for small sizes (u < 18 on an AMD 5950X), and faster beyond that. It also uses less memory for large sizes. 
Both an in-place and a not-in-place product variant are provided. The in-place variant uses less memory and is slightly faster.

### Product computation time results

Experiments on a PC with an AMD Ryzen 9 5950X CPU and 64GB of DDR4 RAM @1800MHz (3600 MT/s), with gf2x 1.3, are summarized below. Results are for the in-place product variant. The code was compiled with gcc using `-march=native` (which enables the 256-bit `vpclmulqdq` code path on this CPU). Each time is an average over repeated runs (repeated until 2 s have elapsed, at most 100 runs per size). These figures were obtained by running `product_test` with `gf2x_max_logsize = 35` and the size list set to u = 10...35. The raw numbers are in `2026-10-results_in_place.csv`. The 64GB of RAM do not allow the in-place product of size 2<sup>36</sup> to be computed.

|Output log2 size|gf2x product time (sec.)|Mateer-Gao product time (sec.)|
|:----:|:----:|:----:|
|10| 6.8e-08| 4.817e-07|
|11| 1.378e-07| 9.135e-07|
|12| 3.536e-07| 1.957e-06|
|13| 9.404e-07| 4.114e-06|
|14| 2.906e-06| 8.765e-06|
|15| 8.354e-06| 1.681e-05|
|16| 2.381e-05| 3.662e-05|
|17| 6.841e-05| 7.173e-05|
|18| 0.000185| 0.0001506|
|19| 0.0005032| 0.000318|
|20| 0.001591| 0.0006849|
|21| 0.00347| 0.001415|
|22| 0.007261| 0.003094|
|23| 0.01587| 0.006454|
|24| 0.0391| 0.01352|
|25| 0.08211| 0.02848|
|26| 0.1951| 0.05957|
|27| 0.3747| 0.1275|
|28| 0.9282| 0.3168|
|29| 2.464| 0.6889|
|30| 3.672| 1.518|
|31| 9.827| 3.221|
|32| 22.11| 7.256|
|33| 87.79| 15.9|
|34| 133.1| 35.14|
|35| 574.1| 76.79|

![Execution times](https://github.com/kunzjacq/Additive_DFTs/blob/master/times.png?raw=true)

![Speed ratio](https://github.com/kunzjacq/Additive_DFTs/blob/master/speed_ratio.png?raw=true)

### Product memory requirements

Assume that a product q = p<sub>1</sub> p<sub>2</sub> is computed, and that the storage of p<sub>1</sub> and p<sub>2</sub> uses n bytes, 2<sup>k-1</sup> < n ≤ 2<sup>k</sup> = m. Then the storage of q also uses at most n bytes. 

Besides the buffer to store the inputs and the result, the out-of-place Mateer-Gao product function uses buffers of size 4m. Overall, it therefore needs M = 2n + 4m RAM bytes, 5m < M ≤ 6m.

The in-place variant stores the result q into the input buffer for p<sub>1</sub>. This array must be of size 2m; an additional buffer twice the storage size of p<sub>2</sub>, rounded to the next power of two, is allocated. Exchanging p<sub>1</sub> and p<sub>2</sub> if necessary, we have that deg(p<sub>2</sub>) ≤ deg(p<sub>1</sub>). Then the storage size for p<sub>2</sub> is ≤ m/2, and the in-place product function uses at most m/2 + 2m + 2\* m/2 = 3.5m bytes of RAM, which is much better than the out-of-place function.

As an example, when processing the largest example above, i.e. a product of two polynomials of degree 2<sup>36</sup>-1, each input polynomial is 8GB, and the result is 16GB. With the out-of-place variant, a buffer of size 4 \* 16GB = 64GB is additionally needed, for a total of 96GB. With the in-place variant, the buffer size for the result must be 2 \* 16GB = 32GB; an additional buffer of size 2 \* 8GB = 16GB is needed. Overall 8+16+32GB = 56GB of RAM are needed.

### Product CPU requirements

The polynomial product code uses a fast implementation of multiplication in GF(2<sup>64</sup>), which is implemented multiple ways in the source, using PCLMULQDQ, POPCNT and SSE intrinsics. `product_test` checks at startup that the CPU reports AVX2, BMI1, POPCNT and PCLMULQDQ (and VPCLMULQDQ when the build targets it, see `MG_USE_YMM_CLMUL` in `include/mg_templates.h`), and exits otherwise; `fft_test` only checks for SSE2 and PCLMULQDQ.

## Cantor bases for large field sizes

Cantor bases are available for field sizes above 2<sup>64</sup> thanks to the optional use of `Boost::multiprecision` which implements large integers (enabled with the CMake option `-D USE_BOOST=ON`, off by default). With this library, the cantor basis test program tests the construction in finite fields of size up to 2<sup>2048</sup>. In theory, truncated Wang-Zhu-Cantor DFT could work in such large fields, however it is not tested and does not work due to overflows in the handling of loop indexes, which are all of type `uint64_t`. There is probably little interest to use these field sizes for truncated DFTs; full DFTs are of course totally unrealistic in these cases.

## Targets
There are three targets which can be built for 64-bit Linux or Windows (with MSVC, only `product_test` is built):
  * `product_test` which benches Mateer-Gao polynomial products (using the finite field GF(2<sup>64</sup>) for DFTs) and checks them against gf2x;
  * `fft_test` which enables to test and benchmark the DFT algorithms discussed above. The binary field and the size of a buffer used in tests determine which tests can be run; these parameters can be adjusted at compile-time at the top of `fft_test.cpp`. The word type must be `uint32_t` or `uint64_t`. If `USE_BOOST` is enabled and `Boost::multiprecision` (https://www.boost.org/doc/libs/1_73_0/libs/multiprecision/doc/html/index.html) is found, Cantor bases and DFT objects can be instantiated for all sizes for which large integers are available from this library. DFT algorithms are not tested (see above) and currently most likely do not work correctly in these large fields. 
  * `cantor_basis_test` which performs various consistency checks on cantor basis construction.

## Build instructions
Under Linux, gcc or clang are able to build all targets. Under windows, MinGW gcc (for instance from MSYS2) can be used, whereas MSVC will only build a limited version of `product_test` that does not use `gf2x`. 

The build process uses Cmake. For instance, an optimized build with gcc using a single-configuration generator such as make, in build directory \$BUILD_DIR and source dir \$SOURCE_DIR, is achieved with:

    cmake -S $SOURCE_DIR -B $BUILD_DIR -D CMAKE_BUILD_TYPE=Release
    cmake --build $BUILD_DIR --parallel

Multi-configuration generators need different options; see https://stackoverflow.com/questions/7724569/debug-vs-release-in-cmake/64719718#64719718. 

Target `product_test` requires a compiled version of gf2x for the target platform to be placed in a subdirectory of the source dir named `lib/`. Alternatively, if an environment variable `CUSTOM_LIBS` is defined, the directory `$CUSTOM_LIBS/lib` will also be searched for gf2x. Link-time optimization is on by default (disable with `-D LTO=OFF`).
