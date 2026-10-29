#pragma once

#include "base.h"

/**
 * Variant of the Mateer-Gao binary polynomial product of mg.h (see libfft/mg_fused.cpp).
 * Same interface and results as the functions of mg.h, in namespace mg_fused.
 * The last DFT stages on rows of 4 or 16 words, the pointwise product and the first stages of the
 * inverse DFT are replaced by direct products modulo x**R + x + c. Products of polynomials of at most 2**13 bits are computed
 * directly (Karatsuba on carry-less products). Faster than mg.h's for all result sizes of 2**6 bits or more.
 */
namespace mg_fused {

void mg_binary_polynomial_multiply(uint64_t *p1, uint64_t *p2, uint64_t *result, uint64_t d1, uint64_t d2);

void mg_binary_polynomial_multiply_in_place(uint64_t *p1, uint64_t *p2, uint64_t d1, uint64_t d2);

} // namespace mg_fused
