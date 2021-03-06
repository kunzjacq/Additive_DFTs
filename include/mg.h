#pragma once

#include <cstdint> // for uint64_t and other types
#include <cstddef>
#include <cassert> // for assert

#include <immintrin.h> // for x86_64 intrinsics

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

/**
 * @brief eval_degree1
 * evaluates in-place 2**logstride degree 1 polynomials in val and val^1,
 * in multiplicative representation.
 *
 *  with
 const uint64_t stride = 1uLL << logstride

 * if (!reverse), equivalent to

 for(uint64_t i = 0; i < stride; i++)
 {
   // computes (u,v) where
   // u = f_0 + f1 * w,
   // v = f_0 + f1 * (w + 1) = u + f_1
   // f_0 = poly[i], f_1 = p[i + stride]
   p[i]          ^= product(p[i + stride], val);
   p[i + stride] ^= p[i];
  }

  * if(reverse), performs the reverse operation, i.e.
  for(uint64_t i = 0; i < stride; i++)
  {
   p[i + stride] ^= p[i];
   p[i]          ^= product(p[i + stride], val);
  }

 * @param reverse
 * if true, reverses the polynomial evaluation to rebuild coefficients.
 * @param logstride
 * the log2 number of polynomials.
 * @param val
 * the value which the polynomials are evaluated at.
 * @param p
 * a pointer to the array containing the interleaved polynomials.
 */
template<bool reverse, int logstride>
static inline void eval_degree1(const uint64_t val, uint64_t* p)
{
  constexpr uint64_t minpoly = 0x1b; // x**64 = x**4 + x**3 + x + 1
  // x**64 + x**4 + x**3 + x + 1 is primitive over GF(2)
  // it is the minimal polynomial of the multiplicative generator

  __m128i xa = _mm_set_epi64x(minpoly, val);
  if constexpr(logstride == 0)
  {
    __m128i xb, xc;
    if constexpr(reverse) p[1] ^= p[0];
    xb = _mm_set1_epi64x(p[1]);
    xb = _mm_clmulepi64_si128(xa, xb, 0x00);
    xc = _mm_clmulepi64_si128(xa, xb, 0x11);
    xb = _mm_xor_si128(xb, xc);
    xc = _mm_clmulepi64_si128(xa, xc, 0x11);
    xb = _mm_xor_si128(xb, xc);
    p[0] ^= xb[0];
    if constexpr(!reverse) p[1] ^= p[0];

  }
  else if constexpr(logstride == 1)
  {
    __m128i xb1, xb2, xc1, xc2;
    if constexpr(reverse)
    {
      p[2] ^= p[0];
      p[3] ^= p[1];
    }
    xb1 = _mm_set1_epi64x(p[2]);
    xb2 = _mm_set1_epi64x(p[3]);
    xb1 = _mm_clmulepi64_si128(xa, xb1, 0x00);
    xb2 = _mm_clmulepi64_si128(xa, xb2, 0x00);
    xc1 = _mm_clmulepi64_si128(xa, xb1, 0x11);
    xc2 = _mm_clmulepi64_si128(xa, xb2, 0x11);
    xb1 = _mm_xor_si128(xb1, xc1);
    xb2 = _mm_xor_si128(xb2, xc2);
    xc1 = _mm_clmulepi64_si128(xa, xc1, 0x11);
    xc2 = _mm_clmulepi64_si128(xa, xc2, 0x11);
    xb1 = _mm_xor_si128(xb1, xc1);
    xb2 = _mm_xor_si128(xb2, xc2);
    p[0] ^= xb1[0];
    p[1] ^= xb2[0];
    if constexpr(!reverse)
    {
      p[2] ^= p[0];
      p[3] ^= p[1];
    }
  }
  else
  {
    const uint64_t stride = 1uLL << logstride;
    for(uint64_t i = 0; i < stride; i += 4)
    {

      __m128i xb1, xb2, xb3, xb4, xc1, xc2, xc3, xc4;
      if constexpr(reverse)
      {
        p[i + stride]     ^= p[i];
        p[i + stride + 1] ^= p[i + 1];
        p[i + stride + 2] ^= p[i + 2];
        p[i + stride + 3] ^= p[i + 3];
      }
      xb1 = _mm_set1_epi64x(p[i + stride]);
      xb2 = _mm_set1_epi64x(p[i + stride + 1]);
      xb3 = _mm_set1_epi64x(p[i + stride + 2]);
      xb4 = _mm_set1_epi64x(p[i + stride + 3]);
      xb1 = _mm_clmulepi64_si128(xa, xb1, 0x00);
      xb2 = _mm_clmulepi64_si128(xa, xb2, 0x00);
      xb3 = _mm_clmulepi64_si128(xa, xb3, 0x00);
      xb4 = _mm_clmulepi64_si128(xa, xb4, 0x00);
      xc1 = _mm_clmulepi64_si128(xa, xb1, 0x11);
      xc2 = _mm_clmulepi64_si128(xa, xb2, 0x11);
      xc3 = _mm_clmulepi64_si128(xa, xb3, 0x11);
      xc4 = _mm_clmulepi64_si128(xa, xb4, 0x11);
      xb1 = _mm_xor_si128(xb1, xc1);
      xb2 = _mm_xor_si128(xb2, xc2);
      xb3 = _mm_xor_si128(xb3, xc3);
      xb4 = _mm_xor_si128(xb4, xc4);
      xc1 = _mm_clmulepi64_si128(xa, xc1, 0x11);
      xc2 = _mm_clmulepi64_si128(xa, xc2, 0x11);
      xc3 = _mm_clmulepi64_si128(xa, xc3, 0x11);
      xc4 = _mm_clmulepi64_si128(xa, xc4, 0x11);
      xb1 = _mm_xor_si128(xb1, xc1);
      xb2 = _mm_xor_si128(xb2, xc2);
      xb3 = _mm_xor_si128(xb3, xc3);
      xb4 = _mm_xor_si128(xb4, xc4);
      p[i]              ^= xb1[0];
      p[i + 1]          ^= xb2[0];
      p[i + 2]          ^= xb3[0];
      p[i + 3]          ^= xb4[0];
      if constexpr(!reverse)
      {
        p[i + stride]     ^= p[i];
        p[i + stride + 1] ^= p[i + 1];
        p[i + stride + 2] ^= p[i + 2];
        p[i + stride + 3] ^= p[i + 3];
      }
    }
  }
}

/**
 * @brief mg_decompose_taylor_recursive
 * in-place taylor decomposition of input interleaved polynomials pointed by 'p'.
 * there are 2**'logstride' polynomials, of degree 2**'logsize'-1.
 * It is assumed that
 *  t >= 1
 *  logsize > t && logsize <= 2*t
 * For tau = 2**t, (tau >= 2),
 * each polynomial is rewritten as sum_i f_i (x**tau - x)**i
 * where each f_i is of degree < tau. f_i = 0 if i >= 2**(logsize - t).
 * f_i replaces coefficents of degree i*tau, ..., (i+1)*tau - 1 in each initial polynomial.
 * @param logstride
 * @param t
 * @param logsize
 * @param p
 */
template<unsigned int logstride, unsigned int t, class T>
inline void mg_decompose_taylor_recursive(
    unsigned int logsize,
    T* p)
{
  static_assert(t >= 1);
  assert(logsize > t && logsize <= 2*t);
  //decompose_taylor_one_step
  uint64_t delta_s   = 1uLL << (logstride + logsize - 1 - t); // logn - 1 - t >= 0
  const uint64_t n_s = 1uLL  << (logstride + logsize);
  const uint64_t m_s = n_s >> 1;
  T* q = p + delta_s - m_s;
  // m_s > 0, hence the loop below is not infinite
  for(uint64_t i = n_s - 1; i > m_s - 1; i--) q[i] ^= p[i];

  if(logsize > t + 1)
  {
    mg_decompose_taylor_recursive<logstride, t, T>(logsize - 1, p);
    mg_decompose_taylor_recursive<logstride, t, T>(logsize - 1, p + m_s);
  }
}

/**
 * @brief mg_decompose_taylor_reverse_recursive
 * reverse function of mg_decompose_taylor_recursive.
 * @param logstride
 * @param t
 * @param logsize
 * @param p
 */
template<unsigned int logstride, unsigned int t>
inline void mg_decompose_taylor_reverse_recursive(
    unsigned int logsize,
    uint64_t* p)
{
  static_assert(t >= 1);
  assert(logsize > t && logsize <= 2*t);
  const uint64_t n_s = 1uLL  << (logstride + logsize);
  const uint64_t m_s = n_s >> 1;
  if(logsize > t + 1)
  {
    mg_decompose_taylor_reverse_recursive<logstride, t>(logsize - 1, p);
    mg_decompose_taylor_reverse_recursive<logstride, t>(logsize - 1, p + m_s);
  }

  //decompose_taylor_one_step_reverse
  uint64_t delta_s   = 1uLL << (logstride + logsize - 1 - t); // logn - 1 - t >= 0
  uint64_t* q = p + delta_s - m_s;
  for(uint64_t i = m_s; i < n_s; i++) q[i] ^= p[i];
}
