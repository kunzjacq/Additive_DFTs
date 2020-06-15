# Additive_DFTs
A C++17 implementation of several additive DFT algorithms over F2.
 
A test program showcases the different algorithms and tests their correctness.
 
Algorithms implemented are: Von zur Gathen - Gehrard algorithm with Cantor bases (also called the Wang-Zhu-Cantor algorithm), and the Mateer-Gao algorithm. All of them are described in Todd Mateer Master Thesis. Reverse algorithms are also implemented.

Truncated versions of these algorithms are available to perform only a part of the DFT computation.

A combination of Wang-Zhu-Cantor and Mateer-Gao is also implemented to tackle the case of the computation of u values of a polynomial of degree d where u is much smaller than d.

Mateer-Gao DFT is used to implement a fast multiplication of binary polynomials. It is tested and benched against a very fast algorithm to perform such multiplications, implemented in library `gf2x` (https://gforge.inria.fr/projects/gf2x/). Overall the DFT approach is slower, but for large size the speed ratio is about 10, which is not so bad since the DFT implementation does not use any assembly. For small sizes the ratio is much higher however; memory requirements of the DFT implementation are also higher, mainly due to the need to convert the polynomials between several representations before doing the actual DFT work. 

This project is more about experimenting with these nice algorithms than trying to compete with gf2x or any other optimized polynomial computation library. As such, reference implementations that are close to the high-level description of the algorithms are kept alongsize more optimized versions that are more difficult to read.

## Build instructions
A compiled version of gf2x for the target platform must be placed in `lib/` (for linux) or `lib_mingw` (for windows/). For windows, MinGW gcc usage is assumed, for instance with the version that comes with MSYS2. cmake is needed, the usual cmake build process applies (in a build dir separate from the source dir $SOURCE_DIR):
    cmake $SOURCE_DIR/CMake
    make

To make an optimized build, add `-DCMAKE_BUILD_TYPE=Release` to the cmake invocation.

The binary field used in tests in the size of the buffer used, which determine what tests can be run, can be adjusted at the top of `fft_test.cpp`.

## TODO
  * simplify the test program to make it easier to change the tests and benchmarks that are run.
  * implement more combinations of several algorithms to improve performance on truncated cases.
  * optimize the underlying finite field multiplications, with assembly and/or batch computing functions.
