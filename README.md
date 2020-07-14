# Additive_DFTs
A C++17 implementation of several additive DFT algorithms over F2, and their application to fast binary polynomial multiplication. 
 
The implementation targets the x86_64 architecture with SSE2 and PCLMULQDQ instruction sets. A test program showcases the different algorithms and tests their correctness.
 
Algorithms implemented are: Von zur Gathen - Gehrard algorithm with Cantor bases (also called the Wang-Zhu-Cantor algorithm), and the Mateer-Gao algorithm. All of them are described in Todd Mateer Master Thesis (https://tigerprints.clemson.edu/all_dissertations/231/). Reverse algorithms are also implemented.

Truncated versions of these algorithms are available to perform only a part of the DFT computation.

A combination of Wang-Zhu-Cantor and Mateer-Gao is also implemented to tackle the case of the computation of u values of a polynomial of degree d where u is much smaller than d.

Mateer-Gao DFT is used to implement a fast multiplication of binary polynomials (see https://cr.yp.to/f2mult.html). It is tested and benched against a very fast algorithm to perform such multiplications, implemented in library `gf2x` (https://gforge.inria.fr/projects/gf2x/). The DFT approach is slower for small sizes (< 2^20 - 2^25), and faster beyond that, at least in the tested case where the polynomials multiplied have equal degree just below a power of 2.  It uses more memory however, because it needs to store the DFT of the result which is twice as large as the result itself, and a second DFT of the same size. This could be improved but at least one DFT must be stored at some point. Efficient handling of the general case of polynomials of different degrees would probably require a clever splitting of the input polynomials.

Polynomial product uses a fast implementation of GF(2^64) that is implemented with SSE2 and PCLMULQDQ. The availability of these instruction sets is not checked at runtime.

This project is more about experimenting with these DFT algorithms than trying to compete with gf2x or any other optimized polynomial computation library. As such, reference implementations that are close to the high-level description of the algorithms are kept alongside more optimized versions that are more difficult to read. As a result there is some redundancy in the source code.

Cantor bases are available for field sizes above 2^64 thanks to the optional use of Boost::multiprecision which implements large integers. With this library, the cantor basis test program tests the construction in finite fields of size up to 2^{2048}. In theory, truncated Wang-Zhu-Cantor DFT could work in such large fields, however it is not tested and does not work due to overflows in the handling of loop indexes, which are all of type uint64_t. There is probably little interest to use these field sizes for truncated DFTs; full DFTs are of course totally unrealistic in these cases.

## Build instructions
A compiled version of gf2x for the target platform must be placed in `lib/` (for linux) or `lib_mingw/` (for windows). For windows, MinGW gcc usage is assumed, for instance with the version that comes with MSYS2. cmake is needed, the usual cmake build process applies (in a build dir separate from the source dir $SOURCE_DIR):

    cmake $SOURCE_DIR
    make

To make an optimized build, add `-DCMAKE_BUILD_TYPE=Release` to the cmake invocation.

There are two targets: `cantor_basis_test` which performs various consistency checks on cantor basis construction; and `fft_test` which enables to test and benchmark the DFT algorithms and polynomial product discussed above. If Boost::multiprecision   (https://www.boost.org/doc/libs/1_73_0/libs/multiprecision/doc/html/index.html) is detected, Cantor bases and DFT objects can be instantiated for all sizes for which large integers are available from this library. DFT algorithms are not tested (see above).

The binary field used in tests and the size of the buffer used, which determine what tests can be run, can be adjusted at compile-time at the top of `fft_test.cpp`. Polynomial product always uses GF(2^64).

## TODO
  * check the availability of the instructions used (SSE2, PCLMULQDQ) at runtime, before attempting to use them.
  * more documentation.
