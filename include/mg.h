#pragma once

#include "base.h"

void naive_product(uint64_t* p1u, uint64_t n1, uint64_t* p2u, uint64_t n2, uint64_t* qu);

/**
 * @brief binary_polynomial_multiply
 * out-of-place multiplication of binary polynomials using Mateer-Gao DFT: result <- p1 × p2.
 * @param p1
 * 64-bit buffer with 1st polynomial p1, of degree d1. Buffer should be readable up to index i = d1 / 64.
 * @param p2
 * same for second polynomial p2, of degree d2. Buffer should be readable up to index i = d2 / 64.
 * @param result
 * 64-bit buffer for result, of size at least (d1 + d2) / 64 + 1.
 * @param d1
 * degree of 1st polynomial.
 * @param d2
 * degree of 2nd polynomial.
 */
void mg_binary_polynomial_multiply(uint64_t *p1, uint64_t *p2, uint64_t *result, uint64_t d1, uint64_t d2);

/**
 * @brief mg_binary_polynomial_multiply_in_place
 * in-place multiplication of binary polynomials using Mateer-Gao DFT: p1 <- p1 × p2.
 * Assumes p1 can hold twice the size of the result polynomial, rounded to the next power of 2.
 * result is returned in p1, p2 is modified during computations.
 * Since the size of p1 only depends on the result size, p2 should be set to the smallest degree
 * polynomial to be multiplied.
 * @param p1
 * 64-bit array for input polynomial p1 and output
 * @param p2
 * 64-bit array for second input polynomial
 * @param d1
 * degree of p1
 * @param d2
 * degree of p2
 */

void mg_binary_polynomial_multiply_in_place (uint64_t *p1, uint64_t *p2, uint64_t d1, uint64_t d2);

void init_alt_table();
