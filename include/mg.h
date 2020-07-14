#pragma once

#include <cstdint>
#include <cstddef>

class mateer_gao_polynomial_product{
    uint64_t* m_beta_to_mult_table;
public:
  mateer_gao_polynomial_product();
  ~mateer_gao_polynomial_product();
  void binary_polynomial_multiply(uint8_t *p1, uint8_t *p2, uint8_t *result, uint64_t *b1, uint64_t *b2, size_t d1, size_t d2, unsigned int logsize);
};
