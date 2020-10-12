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
|10| 1.09431e-07| 1.38589e-06|
|11| 2.62482e-07| 2.52093e-06|
|12| 7.15466e-07| 5.61545e-06|
|13| 2.03284e-06| 7.47628e-06|
|14| 2.99103e-06| 1.4035e-05|
|15| 9.9116e-06| 3.16259e-05|
|16| 2.81112e-05| 6.83873e-05|
|17| 8.04921e-05| 0.00014863|
|18| 0.000221933| 0.000323124|
|19| 0.000619341| 0.0006945|
|20| 0.00191297| 0.00149074|
|21| 0.00386728| 0.00319287|
|22| 0.00798317| 0.00689116|
|23| 0.0175718| 0.0147847|
|24| 0.044663| 0.0310474|
|25| 0.0901284| 0.0653282|
|26| 0.214669| 0.138846|
|27| 0.431616| 0.299029|
|28| 1.03791| 0.651955|
|29| 2.62693| 1.39348|
|30| 4.54754| 2.96513|
|31| 12.119| 6.23943|
|32| 27.0756| 13.1931|
|33| 101.915| 27.8231|
|34| 161.692| 59.1067|
|35| 671.035| 124.084|
|36| not enough memory| 263.476|
|37| not enough memory| 558.934|

![Execution times](https://github.com/kunzjacq/Additive_DFTs/blob/master/times.png?raw=true)

![Speed ratio](https://github.com/kunzjacq/Additive_DFTs/blob/master/speed_ratio.png?raw=true)

### Product memory requirements

Assume that a product q = p<sub>1</sub> p<sub>2</sub> is computed, and that the storage of p<sub>1</sub> and p<sub>2</sub> uses n bytes, 2<sup>k-1</sup> < n ≤ 2<sup>k</sup> = m. Then the storage of q also uses at most n bytes. 

Besides the buffer to store the inputs and the result, the out-of-place Mateer-Gao product function uses buffers of size 4m. Overall, it therefore needs M = 2n + 4m RAM bytes, 5m < M ≤ 6m.

The in-place variant stores the result q into the input buffer for p<sub>1</sub>. This array must be of size 2m; an additional buffer twice the storage size of p<sub>2</sub>, rounded to the next power of two, is allocated. Exchanging p<sub>1</sub> and p<sub>2</sub> if necessary, we have that deg(p<sub>2</sub>) ≤ deg(p<sub>1</sub>). Then the storage size for p<sub>2</sub> is ≤ m/2, and the in-place product function uses at most m/2 + 2m + 2\* m/2 = 3.5m bytes of RAM, which is much better than the out-of-place function.

As an example, when processing the largest example above, i.e. a product of two polynomials of degree 2<sup>36</sup>-1, each input polynomial is 8GB, and the result is 16GB. With the out-of-place variant, a buffer of size 4 \* 16GB = 64GB is additionally needed, for a total of 96GB. With the in-place variant, the buffer size for the result must be 2 \* 16GB = 32GB; an additional buffer of size 2 \* 8GB = 16GB is needed. Overall 8+16+32GB = 56GB of RAM are needed.

### Product CPU requirements

The polynomial product code uses a fast implementation of multiplication in GF(2<sup>64</sup>) that uses SSE2 and PCLMULQDQ instruction sets. Their availability is checked at the beginning of the test programs.

## Cantor bases for large field sizes

Cantor bases are available for field sizes above 2<sup>64</sup> thanks to the optional use of `Boost::multiprecision` which implements large integers. With this library, the cantor basis test program tests the construction in finite fields of size up to 2<sup>2048</sup>. In theory, truncated Wang-Zhu-Cantor DFT could work in such large fields, however it is not tested and does not work due to overflows in the handling of loop indexes, which are all of type `uint64_t`. There is probably little interest to use these field sizes for truncated DFTs; full DFTs are of course totally unrealistic in these cases.

## Build instructions
Test programs can be built for 64-bit Linux or Windows. For windows, MinGW gcc usage is assumed, for instance with the version that comes with MSYS2. Cmake is needed. The usual cmake build process applies (in a build directory separate from the source dir $SOURCE_DIR):

    cmake $SOURCE_DIR
    make

To make an optimized build, add `-DCMAKE_BUILD_TYPE=Release` to the cmake invocation.

There are three targets:
  * `product_test` which benches Mateer-Gao polynomial products and checks them against gf2x;
  * `fft_test` which enables to test and benchmark the DFT algorithms discussed above. If `Boost::multiprecision` (https://www.boost.org/doc/libs/1_73_0/libs/multiprecision/doc/html/index.html) is detected, Cantor bases and DFT objects can be instantiated for all sizes for which large integers are available from this library. DFT algorithms are not tested (see above) and currently cannot work correctly in these fields;
  * `cantor_basis_test` which performs various consistency checks on cantor basis construction.

For target `fft_test`, the binary field and the size of a buffer used in tests determine which tests can be run. These parameters can be adjusted at compile-time at the top of `fft_test.cpp`.

Target `product_test` requires a compiled version of gf2x for the target platform to be placed in `lib/` (for Linux) or `lib_mingw/` (for Windows). The finite field used is GF(2<sup>64</sup>).


