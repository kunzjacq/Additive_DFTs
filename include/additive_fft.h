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
  uint64_t* packed_degrees;
  unsigned int* num_coefficients;
  cantor_basis<word>* m_c_b;
  additive_fft(cantor_basis<word> *c_b);
  ~additive_fft();

  /**
   * @brief fft_direct
   * Evaluates input polynomial for all values of the interval (in beta representation)
   * 2**m * blk_index ^ i, i=0 ... 2**m - 1
   * The polynomial is given in dense forme by p_poly and p_poly_degree.
   * All coefficients and results are in gamma basis.
   * On output, po_result[i] = P(beta_to_gamma(2**m * blk_index ^ i))
   * @param p_poly: input polynomial coefficients
   * @param p_poly_degree: input polynomial degree
   * @param m: log2 of interval size for output values
   * @param po_result: result
   * @param blk_index: index of interval for output values (see formula above)
   */

  void fft_direct(
      const word *p_poly,
      uint64_t p_poly_degree,
      uint32_t m,
      word *po_result,
      uint64_t blk_index = 0) const;

  /**
   * @brief fft_direct_exp
   * Evaluates input polynomial for x**i, i=0 ... 2**m - 1
   * The polynomial is given in dense forme by p_poly and p_poly_degree.
   * x, all coefficients, and results are in gamma basis.
   * On output, po_result[i] = P(x**i), i=0 ... 2**m - 1
   * @param p_poly: input polynomial coefficients
   * @param p_poly_degree: input polynomial degree
   * @param m: log2 of interval size for output values
   * @param x: element in gamma representation whose images of powers are computed
   * @param po_result: result
   */

  void fft_direct_exp(
      const word* p_poly,
      uint64_t p_poly_degree,
      uint32_t m,
      const word& x,
      word* po_result) const;

  /**
   * @brief additive_fft_ref_in_place, additive_fft_fast_in_place
   * Evaluates input polynomial for all values of the interval (in beta representation)
   * 2**m * blk_index ^ i, i=0 ... 2**m - 1, as fft_direct.
   * Uses the Von zur Gathen-Gerhard additive DFT algorithm, explained for instance in
   * Todd Mateer PhD thesis (https://tigerprints.clemson.edu/all_dissertations/231/)
   * or at https://cr.yp.to/f2mult.html.
   * The algorithm also uses Cantor bases to simplify Euclidean reductions. These bases ensure that
   * the reductors have all their coefficients equal to 1, except the constant coefficient.
   * If polynomial degree >= field multiplicative order, the input polynomial is pre-reduced by
   * X**multiplicative_order - 1.
   * The input/output buffer is assumed to be of size at least 2**m (which may ne larger than the
   * input polynomial) to be able to hold the result.
   * @param p_poly: input polynomial, and result
   * @param p_poly_degree: as in fft_direct
   * @param m: as in fft_direct
   * @param: p_buf (for fast version): buffer of size 2**(m-2)
   * @param blk_index: as in fft_direct
   */
  void additive_fft_ref_in_place(
      word* p_poly,
      uint64_t p_poly_degree,
      uint32_t m,
      uint64_t blk_index = 0) const;
  void additive_fft_fast_in_place(
      word* p_poly,
      uint64_t p_poly_degree,
      uint32_t m,
      word* p_buf,
      uint64_t blk_index = 0) const;

  /**
   * @brief additive_fft_rev_ref_in_place, additive_fft_rev_fast_in_place
   * builds a polynomial of degree < 2**m that has the prescribed output values on the
   * input values 2**m * blk_index + i, i - 0 ... 2**m - 1.
   * In other words, outputs a polynomial P of degree < 2**m s.t.
   * P(beta_to_gamma(2**m * blk_index + i)) = p_values[i], i = 0 ... 2**m - 1.
   * THese functions are therefore the inverse of additive_fft_{ref,fast}_in_place when the input to
   * these functions is a polynomial of degree < 2**m.
   * @param p_values: input values, and output polynomial.
   * @param m: the input log2 interval size and log2 degree of output polynomial.
   * @param: p_buf (for fast version): buffer of size 2**(m-2)
   * @param blk_index: defines the range corresponding to the input values (see above).
   *
   */
  void additive_fft_rev_ref_in_place(
      word *p_values,
      uint32_t m,
      uint64_t blk_index = 0) const;
  void additive_fft_rev_fast_in_place(
      word *p_values,
      uint32_t m,
      word* p_buf,
      uint64_t blk_index = 0) const;

  void prepare_polynomials();
  void print_fft_polynomials();
  void evaluate_polynomial_additive_FFT(
      word *p_poly,
      word p_num_terms,
      word *po_result);
};

/**
 * reduces in-place a polynomial by X**multiplicative order - 1,
 * where multiplicative_order refers to the multiplicative order of the binary field
 * with elements of type word, i.e., multiplicative order = 2**n - 1 if word has n bits.
 */

template<class word>
uint64_t fold_polynomial(word* p_poly, uint64_t p_poly_degree)
{
  // deal with cases where the polynomial degree exceeds the multiplicative order of elements in
  // the finite field : fold the polynomial by reducing it by X**multiplicative order - 1.
  // this can only occur if c_b_t<word>::n <= 64, since the degree is a 64-bit value.
  // Degrees around or beyond 2**64 are unrealistic anyway.

  if constexpr(c_b_t<word>::n == 64)
  {
    constexpr uint64_t mult_order = ~(0uLL);  // 2**n-1
    if(p_poly_degree == mult_order)
    {
      p_poly[0]^=p_poly[mult_order];
      p_poly_degree = mult_order - 1;
    }
  }
  else if constexpr(c_b_t<word>::n < 64)
  {
    constexpr uint64_t mult_order = ~(static_cast<word>(0));  // 2**n-1
    if(p_poly_degree >= mult_order)
    {
      uint64_t k = p_poly_degree / mult_order;

      for(uint64_t j = 0; j + k*mult_order < p_poly_degree + 1; j++)
      {
        p_poly[j] ^= p_poly[j+k*mult_order];
      }

      for(uint64_t i = 1; i < k; i++)
      {
        for(uint64_t j = 0; j < mult_order; j++)
        {
          p_poly[j] ^= p_poly[j+i*mult_order];
        }
      }
      p_poly_degree = mult_order - 1;
    }
  }
  return p_poly_degree;
}

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
