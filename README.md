# Additive_DFTs
A C++17 implementation of several additive DFT algorithms over F2, and their application to fast binary polynomial multiplication. 
 
The implementation targets the x86_64 architecture with SSE2 and PCLMULQDQ instruction sets. Test programs showcase the different algorithms and test their correctness.
 
Algorithms implemented are: Von zur Gathen - Gehrard algorithm with Cantor bases (also called the Wang-Zhu-Cantor algorithm),  and the Mateer-Gao algorithm. Both of them are described in Todd Mateer Master Thesis (https://tigerprints.clemson.edu/all_dissertations/231/). Reverse algorithms are also implemented.

Truncated versions of these algorithms are available to perform only a part of the DFT computation. Provided an input polynomial degree < v = 2<sup>u</sup>, the v first coefficients of the DFT can be computed efficiently (if the field has size w, the time to compute the truncated DFT is less than v/w the time to compute the full DFT). Truncated reverse transforms have the same complexity and build a polynomial of degree < v that takes the provided values.

A combination of Wang-Zhu-Cantor and Mateer-Gao is also implemented to tackle the case of the computation of u values of a polynomial of degree d where u is much smaller than d.

This project is about experimenting with these DFT algorithms. Reference implementations that are close to the high-level description of the algorithms are therefore kept alongside more optimized versions that are more difficult to read. As a result there is some redundancy in the source code.

## Binary polynomial product

Mateer-Gao DFT (and crucially, its truncated implementation) is used to implement a fast multiplication of binary polynomials (see https://cr.yp.to/f2mult.html). It is tested and benched against the fastest implementation known to the author performing this operation, in library `gf2x` (https://gforge.inria.fr/projects/gf2x/). 

The DFT approach is slower than gf2x for small sizes (< 2^20), and faster beyond that, at least in the case which was tested, where the polynomials multiplied have equal degree just below a power of 2.  It also appears to use less memory for large sizes. 

Experiments on an AMD 3900X machine are summarized below.

|Output size|gf2x product time (sec.)|Mateer-Gao product time (sec.)|
|:----:|:----:|:----:|
|10| 1.04702e-07| 1.55653e-06|
|11| 2.68164e-07| 2.89191e-06|
|12| 7.27469e-07| 6.32747e-06|
|13| 2.1147e-06| 6.86151e-06|
|14| 3.01368e-06| 1.5177e-05|
|15| 9.6984e-06| 3.36861e-05|
|16| 2.74011e-05| 7.2759e-05|
|17| 7.94745e-05| 0.00015739|
|18| 0.000217923| 0.000341152|
|19| 0.000608785| 0.000730794|
|20| 0.00187449| 0.00161085|
|21| 0.00383994| 0.00344927|
|22| 0.00792871| 0.00736224|
|23| 0.0173911| 0.0157705|
|24| 0.0446058| 0.0321648|
|25| 0.0897099| 0.0685952|
|26| 0.212486| 0.145032|
|27| 0.424994| 0.316625|
|28| 1.03537| 0.686158|
|29| 2.60567| 1.45992|
|30| 4.4542| 3.08745|
|31| 11.9174| 6.4638|
|32| 26.4949| 13.6163|
|33| 101.108| 28.7695|
|34| 160.492| 60.9985|
|35| 678.952| 129.156|
|36| not enough memory| 271.887|

In log scale, the speed comparison is pictured below.

![Execution times](https://github.com/kunzjacq/Additive_DFTs/blob/master/times.png?raw=true)

The speed ratio between gf2x and Mateer-Gao is pictured below.

![Speed ratio](https://github.com/kunzjacq/Additive_DFTs/blob/master/speed_ratio.png?raw=true)

Polynomial product uses a fast implementation of GF(2^64) that is implemented with SSE2 and PCLMULQDQ. The availability of these instruction sets is not checked at runtime.

## Cantor bases for large field sizes

Cantor bases are available for field sizes above 2^64 thanks to the optional use of Boost::multiprecision which implements large integers. With this library, the cantor basis test program tests the construction in finite fields of size up to 2^{2048}. In theory, truncated Wang-Zhu-Cantor DFT could work in such large fields, however it is not tested and does not work due to overflows in the handling of loop indexes, which are all of type uint64_t. There is probably little interest to use these field sizes for truncated DFTs; full DFTs are of course totally unrealistic in these cases.

## Build instructions
A compiled version of gf2x for the target platform must be placed in `lib/` (for linux) or `lib_mingw/` (for windows). For windows, MinGW gcc usage is assumed, for instance with the version that comes with MSYS2. cmake is needed, the usual cmake build process applies (in a build dir separate from the source dir $SOURCE_DIR):

    cmake $SOURCE_DIR
    make

To make an optimized build, add `-DCMAKE_BUILD_TYPE=Release` to the cmake invocation.

There are three targets: 
  * `product_test` which benches Mateer-Gao polynomial products and checks them against gf2x;
  * `fft_test` which enables to test and benchmark the DFT algorithms discussed above. If Boost::multiprecision   (https://www.boost.org/doc/libs/1_73_0/libs/multiprecision/doc/html/index.html) is detected, Cantor bases and DFT objects can be instantiated for all sizes for which large integers are available from this library. DFT algorithms are not tested (see above);
  * `cantor_basis_test` which performs various consistency checks on cantor basis construction.

The binary field used in tests and the size of the buffer used, which determine what tests can be run, can be adjusted at compile-time at the top of `fft_test.cpp`. Polynomial product always uses GF(2^64).

## TODO
  * check the availability of the instructions used (SSE2, PCLMULQDQ) at runtime, before attempting to use them.

