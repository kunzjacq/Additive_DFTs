#pragma once

#include "cantor.h"
#include "gf2_extension_polynomials.h"

template <class word>
class additive_fft
{
public:
  word* buf;
  word* cst_coefficients;
  word* coeffs_to_cancel;
  unsigned int m;
  unsigned int* packed_logdegrees;
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
   * @brief additive_fft_ref
   * Computes the same result as fft_direct, using the Von zur Gathen-Gerhard additive FFT
   * of input polynomial, explained for instance in Todd Mateer PhD thesis
   * (https://tigerprints.clemson.edu/all_dissertations/231/)
   * or at https://cr.yp.to/f2mult.html.
   * The algorithm also uses Cantor bases to simplify Euclidean reductions. These bases ensure that
   * the reductors have all their coefficients equal to 1, except the constant coefficient.
   *
   * polynomial degree >= field multiplicative order is not supported if m = m_log_bound.
   * This can be overcome if necessary by pre-reducing the input polynomial by
   * X**multiplicative_order - 1.
   * (with current implementation, such a plynomial would cause a buffer overflow
   * if m = m_log_bound : the dividend in the first euclidean division would be larger than the
   * internal buffer in which it is to be stored.)
   * (if m > m_log_bound, the first euclidean division computed does not store the dividend,
   * hence the overflow does not occur.)
   * @param p_poly
   * @param p_poly_degree
   * must satisfy p_poly_degree < 2**n - 1 if the field has 2**n elements.
   * @param po_result
   * @param blk_offset
   */
  void additive_fft_ref(word* p_poly, uint64_t p_poly_degree, word *po_result, uint64_t blk_offset = 0) const;

  /**
   * @brief additive_fft_rev_ref
   * Inverse of additive_fft_ref: rebuilds the polynomial from the output values.
   * @param p_values
   * @param po_result
   * @param blk_index
   */
  void additive_fft_rev_ref( word* p_values, word* po_result, uint64_t blk_index)  const;
  void additive_fft_fast(word* p_poly, uint64_t p_poly_degree, word *po_result, uint64_t blk_offset = 0) const;
  void prepare_polynomials();
  void print_fft_polynomials();
  void evaluate_polynomial_additive_FFT(word *p_poly,
      word p_num_terms,
      word *po_result
      );
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
