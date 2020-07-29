#pragma once

#include <cstdint>
#include <cstddef>

class mateer_gao_polynomial_product{
    uint64_t* m_beta_to_mult_table;
    uint64_t* m_mult_beta_pow_table;
public:
  mateer_gao_polynomial_product();
  ~mateer_gao_polynomial_product();


  /**
   * @brief binary_polynomial_multiply
   * multiplication of binary polynomials using Mateer-Gao DFT.
   * In this version, the required buffers are allocated internally.
   * @param p1
   * buffer with 1st polynomial, of degree d1. Buffer should be readable up to index i = d1 / 8.
   * @param p2
   * same for second polynomial, of degree d2. Buffer should be readable up to index i = d2 / 8.
   * @param result
   * Buffer for result, of byte size at least (d1+d2) / 8 + 1.
   * @param d1
   * degree of 1st polynomial.
   * @param d2
   * degree of 2nd polynomial.
   */

  void binary_polynomial_multiply(uint8_t *p1, uint8_t *p2, uint8_t *result, uint64_t d1, uint64_t d2);
};
