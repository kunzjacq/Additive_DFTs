# Additive Discrete Fourier Transforms (DFTs)
A C++17 implementation of several additive DFT algorithms over F2, and their application to fast binary polynomial multiplication.

The implementation targets the x86_64 architecture with SSE2 and PCLMULQDQ instruction sets. Test programs showcase the different algorithms and test their correctness.

Algorithms implemented are: Von zur Gathen - Gehrard algorithm with Cantor bases (also called the Wang-Zhu-Cantor algorithm),  and the Mateer-Gao algorithm. Both of them are described in Todd Mateer Master Thesis (available at https://tigerprints.clemson.edu/all_dissertations/231/). Reverse algorithms are also implemented.

Truncated versions of these algorithms are available to perform only a part of the DFT computation. Provided with an input polynomial degree < v = 2<sup>u</sup>, the v first coefficients of the DFT can be computed efficiently (if the underlying finite field has size w, the time to compute the truncated DFT is less than v/w times the full DFT time). Truncated reverse transforms have the same complexity and build a polynomial of degree < v that takes the provided values.

A combination of Wang-Zhu-Cantor and Mateer-Gao is also implemented to tackle the case of the computation of u values of a polynomial of degree d where u is much smaller than d.

This project is about experimenting with these DFT algorithms. Reference implementations that are close to the high-level description of the algorithms are therefore kept alongside more optimized versions that are more difficult to read. As a result there is some redundancy in the source code.

## Application to fast product of binary polynomials

Mateer-Gao DFT (and crucially, its truncated implementation) is used to implement fast multiplication of binary polynomials (see https://cr.yp.to/f2mult.html for an outline of the method used. DFTs are used to evaluate the polynomials, then the values obtained for both polynomials are multiplied pointwise, and finally the product is built from the values with an inverse DFT). It is tested and benched against the fastest product implementation known to the author, from library `gf2x` (https://gforge.inria.fr/projects/gf2x/).

The DFT approach is tested by multiplying polynomials of degree 2<sup>u-1</sup> - 1 for values of u s.t. the computations fit into memory. It is slower than gf2x for small sizes (u < 20), and faster beyond that. It also uses less memory for large sizes. 
Both an in-place and a not-in-place product variant are provided. The in-place variant uses less memory and is slightly faster.

### Product computation time results

Experiments on an AMD 3900X PC with 64GB of RAM are summarized below. Results are for the in-place product variant.

|Output log2 size|gf2x product time (sec.)|Mateer-Gao product time (sec.)|
|:----:|:----:|:----:|
|10| 1.533e-07| 1.54592e-06|
|11| 2.94e-07| 2.84344e-06|
|12| 7.6872e-07| 6.32339e-06|
|13| 2.15473e-06| 1.3588e-05|
|14| 6.30239e-06| 2.98442e-05|
|15| 1.98352e-05| 4.07306e-05|
|16| 2.80883e-05| 7.07753e-05|
|17| 8.11709e-05| 0.000152324|
|18| 0.000224136| 0.000328689|
|19| 0.000624297| 0.000702538|
|20| 0.00192079| 0.001496|
|21| 0.00385513| 0.00318693|
|22| 0.0080079| 0.00681072|
|23| 0.0177371| 0.0144946|
|24| 0.044928| 0.0303629|
|25| 0.0908105| 0.0637364|
|26| 0.216681| 0.135941|
|27| 0.434778| 0.292499|
|28| 1.03921| 0.630294|
|29| 2.66044| 1.34558|
|30| 4.48929| 2.85855|
|31| 12.127| 5.99039|
|32| 27.1416| 12.6645|
|33| 103.736| 26.927|
|34| 166.724| 57.2199|
|35| 682.815| 120.14|
|36| not enough memory| 251.969|
|37| not enough memory| 532.96|

![Execution times](https://github.com/kunzjacq/Additive_DFTs/blob/master/times.png?raw=true)

![Speed ratio](https://github.com/kunzjacq/Additive_DFTs/blob/master/speed_ratio.png?raw=true)

### Product memory requirements

Assume that a product q = p<sub>1</sub> p<sub>2</sub> is computed, and that the storage of p<sub>1</sub> and p<sub>2</sub> uses n bytes, 2<sup>k-1</sup> < n ≤ 2<sup>k</sup> = m. Then the storage of q also uses at most n bytes. 

Besides the buffer to store the inputs and the result, the out-of-place Mateer-Gao product function uses buffers of size 4m. Overall, it therefore needs M = 2n + 4m RAM bytes, 5m < M ≤ 6m.

The in-place variant stores the result q into the input buffer for p<sub>1</sub>. This array must be of size 2m; an additional buffer twice the storage size of p<sub>2</sub>, rounded to the next power of two, is allocated. Exchanging p<sub>1</sub> and p<sub>2</sub> if necessary, we have that deg(p<sub>2</sub>) ≤ deg(p<sub>1</sub>). Then the storage size for p<sub>2</sub> is ≤ m/2, and the in-place product function uses at most m/2 + 2m + 2\* m/2 = 3.5m bytes of RAM, which is much better than the out-of-place function.

As an example, when processing the largest example above, i.e. a product of two polynomials of degree 2<sup>36</sup>-1, each input polynomial is 8GB, and the result is 16GB. With the out-of-place variant, a buffer of size 4 \* 16GB = 64GB is additionally needed, for a total of 96GB. With the in-place variant, the buffer size for the result must be 2 \* 16GB = 32GB; an additional buffer of size 2 \* 8GB = 16GB is needed. Overall 8+16+32GB = 56GB of RAM are needed.

### Product CPU requirements

The polynomial product code uses a fast implementation of multiplication in GF(2<sup>64</sup>), which is implemented multiple ways in the source, using AVX2, BMI1, POPCNT and PCLMULQDQ instructions. The availability of these instructions is checked by the test programs.

## Cantor bases for large field sizes

Cantor bases are available for field sizes above 2<sup>64</sup> thanks to the optional use of `Boost::multiprecision` which implements large integers. With this library, the cantor basis test program tests the construction in finite fields of size up to 2<sup>2048</sup>. In theory, truncated Wang-Zhu-Cantor DFT could work in such large fields, however it is not tested and does not work due to overflows in the handling of loop indexes, which are all of type `uint64_t`. There is probably little interest to use these field sizes for truncated DFTs; full DFTs are of course totally unrealistic in these cases.

## Targets
There are three targets which can be built for 64-bit Linux or Windows:
  * `product_test` which benches Mateer-Gao polynomial products (using the finite field GF(2<sup>64</sup>) for DFTs) and checks them against gf2x;
  * `fft_test` which enables to test and benchmark the DFT algorithms discussed above. The binary field and the size of a buffer used in tests determine which tests can be run; these parameters can be adjusted at compile-time at the top of `fft_test.cpp`. If `Boost::multiprecision` (https://www.boost.org/doc/libs/1_73_0/libs/multiprecision/doc/html/index.html) is detected, Cantor bases and DFT objects can be instantiated for all sizes for which large integers are available from this library. DFT algorithms are not tested (see above) and currently most likely do not work correctly in these large fields. 
  * `cantor_basis_test` which performs various consistency checks on cantor basis construction.

For target `fft_test`,

## Build instructions
Under Linux, gcc or clang are able to build all targets. Under windows, MinGW gcc (for instance from MSYS2) can be used, whereas MSVC will only build a limited version of `product_test` that does not use `gf2x`. 

The build process uses Cmake. For instance, an optimized build with gcc using a single-configuration generator such as make, in build directory $BUILD_DIR and source dir $SOURCE_DIR, is achieved with:

    cmake -S $SOURCE_DIR -B $BUILD_DIR -D CMAKE_BUILD_TYPE=Release
    cmake --build $BUILD_DIR --parallel

Multi-configuration generators need different options; see https://stackoverflow.com/questions/7724569/debug-vs-release-in-cmake/64719718#64719718. 

Target `product_test` requires a compiled version of gf2x for the target platform to be placed in a subdirectory of the source dir named `lib/`. Alternatively, if an environment variable `CUSTOM_LIBS` is defined and contains a valid directory path, it will be used to look for gf2x.





