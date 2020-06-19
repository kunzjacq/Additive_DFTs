# Additive_DFTs
A C++17 implementation of several additive DFT algorithms over F2.
 
A test program showcases the different algorithms and tests their correctness.
 
Algorithms implemented are: Von zur Gathen - Gehrard algorithm with Cantor bases (also called the Wang-Zhu-Cantor algorithm), and the Mateer-Gao algorithm. All of them are described in Todd Mateer Master Thesis (https://tigerprints.clemson.edu/all_dissertations/231/). Reverse algorithms are also implemented.

Truncated versions of these algorithms are available to perform only a part of the DFT computation.

A combination of Wang-Zhu-Cantor and Mateer-Gao is also implemented to tackle the case of the computation of u values of a polynomial of degree d where u is much smaller than d.

Mateer-Gao DFT is used to implement a fast multiplication of binary polynomials (see https://cr.yp.to/f2mult.html). It is tested and benched against a very fast algorithm to perform such multiplications, implemented in library `gf2x` (https://gforge.inria.fr/projects/gf2x/). Overall the DFT approach is slower, but for large size (> 2^{30}) the speed ratio is about 2, which is not so bad since the DFT implementation does not use any assembly code. For small sizes the ratio is much higher however; memory requirements of the DFT implementation are also higher, mainly due to the need to convert the polynomials between several representations before doing the actual DFT work. 

This project is more about experimenting with these nice algorithms than trying to compete with gf2x or any other optimized polynomial computation library. As such, reference implementations that are close to the high-level description of the algorithms are kept alongside more optimized versions that are more difficult to read.

Cantor bases are available for field sizes above 2^{64} thanks to Boost large integers (which is optional). The cantor basis test program checks the construction up to 2^{2048}. In theory, truncated Wang-Zhu-Cantor DFT should work in such large fields, however it was not tested and some bugs may be present due to overflows in the handling of loop indexes, which are all of type uint64_t. There is probably little interest to use these field sizes for truncated DFTs; full DFTs are of course totally unrealistic for these sizes.

## Build instructions
A compiled version of gf2x for the target platform must be placed in `lib/` (for linux) or `lib_mingw` (for windows/). For windows, MinGW gcc usage is assumed, for instance with the version that comes with MSYS2. cmake is needed, the usual cmake build process applies (in a build dir separate from the source dir $SOURCE_DIR):

    cmake $SOURCE_DIR
    make

To make an optimized build, add `-DCMAKE_BUILD_TYPE=Release` to the cmake invocation.

There are two targets: `cantor_basis_test` which performs various consistency checks on cantor basis construction; and `fft_test` which enables to test and benchmark the DFT algorithms discussed above. If Boost::multiprecision   (https://www.boost.org/doc/libs/1_73_0/libs/multiprecision/doc/html/index.html) is detected, Cantor bases and DFT objects can be instantiated for all sizes for which large integers from this library are available. They are not tested however.

The binary field used in tests and the size of the buffer used, which determine what tests can be run, can be adjusted at the top of `fft_test.cpp`.

## TODO
  * implement more combinations of several algorithms to improve performance on truncated cases.
  * optimize the underlying finite field multiplications, with assembly and/or batch computing functions.
