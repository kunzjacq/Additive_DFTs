#pragma once

#include <cstdint>

#include "cantor.h"
#include "gf2_extension_polynomials.h"

template<class word>
void interleave_quarter_buffer(word* buf, word* p_poly, int logsz);

template <class word>
class additive_fft
{
public:
  static constexpr int max_dft_size = min<unsigned int>(40uL, c_b_t<word>::n);
  static constexpr int cst_coeff_divide = max_dft_size>>1;
  word* cst_coefficients_l;
  word* cst_coefficients_h;
  unsigned int m;
  uint64_t* packed_degrees;
  unsigned int* num_coefficients;
  cantor_basis<word>* m_c_b;
  additive_fft(cantor_basis<word> *c_b, unsigned int p_log_bound);
  ~additive_fft();

  /**
   * @brief fft_direct
   * Evaluates input polynomial for all values of the interval (in beta representation)
   * 2**m_log_bound * blk_offset + i, i=0 ... 2**m_log_bound - 1
   * The polynomial, given in dense forme by p_poly and p_poly_degree, is assumed
   * to be given in gamma representation.
   * more explicitly, in po_result[i] the value
   * P(beta_to_gamma(2**m_log_bound * blk_offset + i)) is stored.
   * @param p_poly
   * @param p_poly_degree
   * @param po_result
   * @param blk_offset
   */

  void fft_direct(const word *p_poly, uint64_t p_poly_degree, word *po_result, uint64_t blk_offset = 0) const;
  void fft_direct_exp(const word* p_poly, uint64_t p_poly_degree, const word& x, word* po_result) const;

  /**
   * @brief additive_fft_ref_in_place, additive_fft_fast_in_place
   * Computes in-place the same result as fft_direct, using the Von zur Gathen-Gerhard additive FFT
   * of input polynomial, explained for instance in Todd Mateer PhD thesis
   * (https://tigerprints.clemson.edu/all_dissertations/231/)
   * or at https://cr.yp.to/f2mult.html.
   * The algorithm also uses Cantor bases to simplify Euclidean reductions. These bases ensure that
   * the reductors have all their coefficients equal to 1, except the constant coefficient.
   *
   * If polynomial degree >= field multiplicative order, the input polynomial is pre-reduced by
   * X**multiplicative_order - 1.
   * The output buffer is assumed to be of size at least 2**m to be able to hold the result.
   * @param p_poly: input polynomial, and result
   * @param p_poly_degree: degree of the input polynomial
   * @param blk_offset: as in fft_direct
   */
  void additive_fft_ref_in_place(word* p_poly, uint64_t p_poly_degree, uint64_t blk_offset = 0) const;
  void additive_fft_fast_in_place(word* p_poly, uint64_t p_poly_degree, word* p_buf, uint64_t blk_offset = 0) const;

  /**
   * @brief additive_fft_rev_ref_in_place, additive_fft_rev_fast_in_place
   * builds a polynomial of degree < 2**m that has the prescribed output values on
   *
   * Inverse of additive_fft_{ref,fast}_in_place when the input to these functions is a polynomial
   * of degree < 2**m.
   * @param p_values: input values, and output polynomial.
   * @param blk_index: defines the range corresponding to the input values.
   *        p_values[i] = P(beta_to_gamma(2**m_log_bound * blk_offset + i)), where 0 <= i < 2**m.
   */
  void additive_fft_rev_ref_in_place(word *p_values, uint64_t blk_index) const;
  void additive_fft_rev_fast_in_place(word *p_values, word* p_buf, uint64_t blk_index) const;
  void prepare_polynomials();
  void print_fft_polynomials();
  void evaluate_polynomial_additive_FFT(
      word *p_poly,
      word p_num_terms,
      word *po_result);
};

#ifdef HAS_UINT2048
extern template class additive_fft<uint2048_t>;
#endif
#ifdef HAS_UINT1024
extern template class additive_fft<uint1024_t>;
#endif
#ifdef HAS_UINT512
extern template class additive_fft<uint512_t>;
#endif
#ifdef HAS_UINT256
extern template class additive_fft<uint256_t>;
#endif

extern template class additive_fft<uint128_t>;

extern template class additive_fft<uint64_t>;

extern template class additive_fft<uint32_t>;

extern template class additive_fft<uint16_t>;

extern template class additive_fft<uint8_t>;
