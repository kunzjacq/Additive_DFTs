#pragma once

#include "cantor.h"

#include "helpers.hpp"

#include <cassert>

template <class word>
word power(cantor_basis<word>* c_b, const word& p_val, const word& p_exp)
{
  if(p_exp == 0) return 1;
  int pos = c_b_t<word>::n-1;
  word res = p_val;
  while((pos > 0) && !bit(p_exp,pos)) pos--;
  for(; pos != 0; pos--)
  {
    res = c_b->square(res);
    if(bit(p_exp,pos-1)) res = c_b->multiply(res, p_val);
  }
  return res;
}


/**
 * @brief evaluate_sparse_polynomial_ref
 * evaluates a sparse polynomial at one point.
 * same as evaluate_sparse_polynomial, but does not use any log or exp table.
 *
 * @param p_f
 * finite field definition
 * @param p_coeffs
 * monomial coefficients.
 * @param p_degrees
 * monomial degrees.
 * @param p_num_coeffs
 * number of monomials.
 * @param p_y
 * point where the polynomial is evaluated.
 * @return
 * the value of the polynomial.
 */


template <class word>
word evaluate_sparse_poly(
    cantor_basis<word>* c_b,
    const word* p_coeffs,
    const word* p_degrees,
    uint64_t p_num_monomials,
    const word& p_y)
{
  uint64_t j;
  word res = 0;
  word curr_exp = 0;
  word curr_power = 1; // = p_y**curr_exp at all times
  if(p_y == 0 && p_num_monomials > 0)
  {
    if(p_degrees[0] == 0) res = p_coeffs[0];
  }
  else
  {
    for(j = 0; j < p_num_monomials; j++)
    {
      if(p_coeffs[j])
      {
        curr_power = c_b->multiply(curr_power, power<word>(c_b, p_y, p_degrees[j] - curr_exp));
        curr_exp = p_degrees[j];
        res ^= c_b->multiply(curr_power, p_coeffs[j]);
      }
    }
  }
  return res;
}

#if 1
/**
 * @brief euclidean_division_ref
 * euclidean division of a dense polynomial by a sparse polynomial in a binary field described
 * by a Canotr base.
 * @param c_b
 * cantor basis object
 * @param p_reductee
 * coefficients of polynomial to reduce.
 * @param p_reductee_degree
 * degree of polynomial to reduce.
 * @param p_reductor_coeffs
 * reductor monomial coefficients
 * @param p_reductor_degrees
 * reductor monomial degrees
 * @param p_reductor_num_terms
 * number of terms in reductor.
 * @param po_remainder
 * output array for remainder, of size (reductor degree).
 * @param po_dividend
 * output array for dividend, or 0. The dividend is written if po_dividend != 0.
 * if this array is nonzero, it must be of size s = p_reductee_degree - (reductor degree) + 1.
 * coefficients above this index will not be touched; therefore if the input buffer is larger
 * than that bound, it is the responsibility of the calling code to clear the coefficients
 * of index >= s.
 */

template <class word>
void euclidean_division_ref(
    cantor_basis<word>* c_b,
    const word* p_reductee,
    uint64_t p_reductee_degree,
    const word* p_reductor_coeffs,
    const word* p_reductor_degrees,
    uint64_t p_reductor_num_terms,
    word* po_remainder,
    word* po_dividend)
{
  uint64_t i, k;
  uint64_t reductor_degree = p_reductor_degrees[p_reductor_num_terms - 1];
  word reductor_head = p_reductor_coeffs[p_reductor_num_terms - 1];
  word reductor_head_inv = c_b->inverse(reductor_head);

  if (reductor_degree > p_reductee_degree)
  {
    for(i = 0; i < p_reductee_degree + 1; i++) po_remainder[i] = p_reductee[i];
    for(; i < reductor_degree; i++)             po_remainder[i] = 0;
    return;
  }
  for(i = 0; i < reductor_degree; i++) po_remainder[i] = 0;
  if (po_dividend)
  {
    for(i = 0; i < p_reductee_degree - reductor_degree + 1; i++) po_dividend[i] = 0;
  }

  uint64_t i_mod_reductor_degree = p_reductee_degree % reductor_degree;
  for (i = p_reductee_degree; i > reductor_degree - 1; i--)
  {
    // here, and in the whole loop, i_mod_reductor_degree = i mod reductor_degree;
    // introduce a new monomial of the reductee
    const word coeff_to_cancel = po_remainder[i_mod_reductor_degree] ^ p_reductee[i];
    po_remainder[i_mod_reductor_degree] = 0;
    if (coeff_to_cancel)
    {
      word mult_coeff = c_b->multiply(coeff_to_cancel, reductor_head_inv);
      // add mult_coeff * X**(i-reductor_degree) * reductor to reductee
      if (po_dividend) po_dividend[i - reductor_degree] = mult_coeff;
      for (k = 0; k < p_reductor_num_terms - 1; k++)
      {
        const word reductor_coeff = p_reductor_coeffs[k];
        if(reductor_coeff)
        {
          uint64_t idx = p_reductor_degrees[k] + i_mod_reductor_degree;
          if(idx >= reductor_degree) idx -= reductor_degree;
          // below at all times,
          // idx = (p_reductor_degrees[k] + i) % reductor_degree
          po_remainder[idx] ^= c_b->multiply(reductor_coeff, mult_coeff);
        }
      }
    }
    if(i_mod_reductor_degree == 0) i_mod_reductor_degree = reductor_degree;
    i_mod_reductor_degree--;
  }
  for (i = 0; i < reductor_degree; i++) po_remainder[i] ^= p_reductee[i];
}
#endif

/**
 * @brief euclidean_division
 * euclidean division of a dense polynomial by a sparse polynomial in a binary field described
 * by a Cantor base. The reductor is assumed to have all its coefficients equal to 1, except maybe
 * the constant coefficient provided separately; the degrees of the non-constant monomials are
 * powers of 2 whose logs are given in p_reductor_logdegrees.
 * @param c_b
 * cantor basis object
 * @param p_reductee
 * coefficients of polynomial to reduce.
 * @param p_reductee_degree
 * degree of polynomial to reduce.
 * @param p_reductor_logdegrees
 * reductor non-constant monomials log2 of degrees.
 * @param p_reductor_const_term
 * constant term in reductor.
 * @param p_reductor_num_terms
 * number of non-constant terms in reductor.
 * @param p_reductor_degree
 * degree of reductor. Should be equal to (1uLL << p_reductor_logdegrees[p_reductor_num_terms - 1]).
 * this relation is checked in debug mode (if NDEBUG is not defined).
 * @param po_remainder
 * output array for remainder, of size equal to reductor degree.
 * @param po_dividend
 * output array for dividend, or 0. The dividend is written if po_dividend != 0.
 * if this array is nonzero, it must be of size s = p_reductee_degree - (reductor degree) + 1.
 * coefficients above this index will not be touched; therefore if the input buffer is larger
 * than that bound, it is the responsibility of the calling code to clear the coefficients
 * of index >= s.
 */


template <class word>
void euclidean_division(
    cantor_basis<word>* c_b,
    const word* p_reductee,
    uint64_t p_reductee_degree,
    const unsigned int* p_reductor_logdegrees,
    const word& p_reductor_const_term,
    uint64_t p_reductor_num_terms,
    uint64_t p_reductor_degree,
    word* po_remainder,
    word* po_dividend)
{
  uint64_t i, k;
  assert((1uLL << p_reductor_logdegrees[p_reductor_num_terms - 1]) == p_reductor_degree);

  if (p_reductor_degree > p_reductee_degree)
  {
    for(i = 0; i < p_reductee_degree + 1; i++) po_remainder[i] = p_reductee[i];
    for(     ; i < p_reductor_degree    ; i++) po_remainder[i] = 0;
    return;
  }
  for(i = 0; i < p_reductor_degree; i++) po_remainder[i] = 0;
  if (po_dividend)
  {
    for(i = 0; i < p_reductee_degree - p_reductor_degree + 1; i++) po_dividend[i] = 0;
  }

  uint64_t i_mod_reductor_degree = p_reductee_degree % p_reductor_degree;
  for (i = p_reductee_degree; i > p_reductor_degree - 1; i--)
  {
    // here, and in the whole loop, i_mod_reductor_degree = i mod p_reductor_degree;
    // introduce a new monomial of the reductee
    const word coeff_to_cancel = po_remainder[i_mod_reductor_degree] ^ p_reductee[i];
    if (coeff_to_cancel)
    {
      // add coeff_to_cancel* X**(i-p_reductor_degree) * reductor to reductee
      if (po_dividend) po_dividend[i - p_reductor_degree] = coeff_to_cancel;
      // constant term
      po_remainder[i_mod_reductor_degree] = c_b->multiply(p_reductor_const_term, coeff_to_cancel);
      // non-constant terms
      for (k = 0; k < p_reductor_num_terms - 1; k++)
      {
        uint64_t idx = (1uLL << p_reductor_logdegrees[k]) + i_mod_reductor_degree;
        if(idx >= p_reductor_degree) idx -= p_reductor_degree;
        // below at all times,
        // idx = (p_reductor_degrees[k] + i) % reductor_degree
        po_remainder[idx] ^= coeff_to_cancel;
      }
    }
    else
    {
      po_remainder[i_mod_reductor_degree] = 0;
    }
    if(i_mod_reductor_degree == 0) i_mod_reductor_degree = p_reductor_degree;
    i_mod_reductor_degree--;
  }
  for (i = 0; i < p_reductor_degree; i++) po_remainder[i] ^= p_reductee[i];
}
