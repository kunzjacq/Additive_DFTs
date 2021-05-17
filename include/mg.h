#pragma once

#include <cstdint> // for uint64_t and other types
#include <cstddef>
#include <cassert> // for assert
#include <memory> // for assume_aligned

#include <immintrin.h> // for x86_64 intrinsics

#ifdef __GNUC__
// unused, _mm_set_epi64x is preferred
#define m128_extract(v,idx) (v)[idx]
#define restr __restrict__
#else
#ifdef _MSC_VER
// unused, _mm_set_epi64x is preferred
#define m128_extract(v,idx) (v).m128i_i64[idx]
#define restr __restrict
#else
#error "unsupported compiler"
#endif
#endif

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
static inline void eval_degree1(const uint64_t val,  uint64_t* restr pu)
{
  uint64_t* restr p = (uint64_t*) std::assume_aligned<16>(pu);

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
    p[0] ^= _mm_extract_epi64(xb, 0);
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
    xb2 = _mm_set_epi64x(p[3], p[2]);
    xb1 = _mm_clmulepi64_si128(xa, xb2, 0x00);
    xb2 = _mm_clmulepi64_si128(xa, xb2, 0x10);
    xc1 = _mm_clmulepi64_si128(xa, xb1, 0x11);
    xc2 = _mm_clmulepi64_si128(xa, xb2, 0x11);
    xb1 = _mm_xor_si128(xb1, xc1);
    xb2 = _mm_xor_si128(xb2, xc2);
    xc1 = _mm_clmulepi64_si128(xa, xc1, 0x11);
    xc2 = _mm_clmulepi64_si128(xa, xc2, 0x11);
    xb1 = _mm_xor_si128(xb1, xc1);
    xb2 = _mm_xor_si128(xb2, xc2);
    p[0] ^= _mm_extract_epi64(xb1, 0);
    p[1] ^= _mm_extract_epi64(xb2, 0);
    if constexpr(!reverse)
    {
      p[2] ^= p[0];
      p[3] ^= p[1];
    }
  }
  else //if constexpr(logstride == 2) // best larger stride on ryzen 2 processors
  {
    constexpr uint64_t stride = 1uLL << logstride;
    uint64_t* restr q = std::assume_aligned<16>(p + stride);
    for(uint64_t i = 0; i < stride; i += 4)
    {
      __m128i xb[4], xc[4];
#if 1
      xb[0] = _mm_set_epi64x(q[1], q[0]); // here _mm_set_epi64x is faster than _mm_load_si128 (ryzen 2)
      xb[2] = _mm_set_epi64x(q[3], q[2]);
#else
      xb[0] = _mm_load_si128((__m128i*)(q));
      xb[2] = _mm_load_si128((__m128i*)(q)+1);
#endif
      if constexpr(reverse)
      {
        // q[i] ^= p[i], i = 0 ... 3
        xb[1] = _mm_load_si128((__m128i*)(p));
        xb[3] = _mm_load_si128((__m128i*)(p)+1);
        xb[0] = _mm_xor_si128(xb[0], xb[1]);
        xb[2] = _mm_xor_si128(xb[2], xb[3]);
        _mm_store_si128((__m128i*)(q),   xb[0]);
        _mm_store_si128((__m128i*)(q)+1, xb[2]);
      }
      xb[1] = _mm_clmulepi64_si128(xa, xb[0], 0x10);
      xb[3] = _mm_clmulepi64_si128(xa, xb[2], 0x10);
      xb[0] = _mm_clmulepi64_si128(xa, xb[0], 0x00);
      xb[2] = _mm_clmulepi64_si128(xa, xb[2], 0x00);
      xc[0] = _mm_clmulepi64_si128(xa, xb[0], 0x11);
      xc[1] = _mm_clmulepi64_si128(xa, xb[1], 0x11);
      xc[2] = _mm_clmulepi64_si128(xa, xb[2], 0x11);
      xc[3] = _mm_clmulepi64_si128(xa, xb[3], 0x11);
      xb[0] = _mm_xor_si128(xb[0], xc[0]);
      xb[1] = _mm_xor_si128(xb[1], xc[1]);
      xb[2] = _mm_xor_si128(xb[2], xc[2]);
      xb[3] = _mm_xor_si128(xb[3], xc[3]);
      xc[0] = _mm_clmulepi64_si128(xa, xc[0], 0x11);
      xc[1] = _mm_clmulepi64_si128(xa, xc[1], 0x11);
      xc[2] = _mm_clmulepi64_si128(xa, xc[2], 0x11);
      xc[3] = _mm_clmulepi64_si128(xa, xc[3], 0x11);
      xb[0] = _mm_xor_si128(xb[0], xc[0]);
      xb[1] = _mm_xor_si128(xb[1], xc[1]);
      xb[2] = _mm_xor_si128(xb[2], xc[2]);
      xb[3] = _mm_xor_si128(xb[3], xc[3]);
      // code below equivalent to p[i]  ^= low half of xb[i], i = 0 ... 3
      //<
      xb[0] =_mm_unpacklo_epi64(xb[0], xb[1]);
      xb[2] =_mm_unpacklo_epi64(xb[2], xb[3]);
      xb[1] = _mm_load_si128((__m128i*)(p)); // here _mm_load_si128 is faster than _mm_set_epi64x
      xb[3] = _mm_load_si128((__m128i*)(p)+1);
      xb[0] = _mm_xor_si128(xb[0], xb[1]);
      xb[2] = _mm_xor_si128(xb[2], xb[3]);
      _mm_store_si128((__m128i*)(p),   xb[0]); //16-byte aligned store
      _mm_store_si128((__m128i*)(p)+1, xb[2]);
      //>
      if constexpr(!reverse)
      {
        // q[i] ^= p[i], i = 0 ... 3
        xb[1] = _mm_load_si128((__m128i*)(q)  );
        xb[3] = _mm_load_si128((__m128i*)(q)+1);
        xb[0] = _mm_xor_si128(xb[0], xb[1]);
        xb[2] = _mm_xor_si128(xb[2], xb[3]);
        _mm_store_si128((__m128i*)(q),   xb[0]);
        _mm_store_si128((__m128i*)(q)+1, xb[2]);
      }
      p += 4;
      q += 4;
    }
  }
#if 0
  else //if constexpr(logstride == 3) // best larger stride on intel core 8 processors
  {
    constexpr uint64_t stride = 1uLL << logstride;
    uint64_t* restr q = std::assume_aligned<16>(p + stride);
    for(uint64_t i = 0; i < stride; i += 8)
    {
      __m128i xb[8], xc[8];
#if 0
      xb[0] = _mm_set_epi64x(q[1], q[0]); // here _mm_load_si128 is faster than _mm_set_epi64x (intel core 8)
      xb[2] = _mm_set_epi64x(q[3], q[2]);
      xb[4] = _mm_set_epi64x(q[5], q[4]);
      xb[6] = _mm_set_epi64x(q[7], q[6]);
#else
      xb[0] = _mm_load_si128((__m128i*)(q));
      xb[2] = _mm_load_si128((__m128i*)(q)+1);
      xb[4] = _mm_load_si128((__m128i*)(q)+2);
      xb[6] = _mm_load_si128((__m128i*)(q)+3);
#endif
      if constexpr(reverse)
      {
        // q[i] ^= p[i], i = 0 ... 7
        xb[1] = _mm_load_si128((__m128i*)(p));
        xb[3] = _mm_load_si128((__m128i*)(p)+1);
        xb[5] = _mm_load_si128((__m128i*)(p)+2);
        xb[7] = _mm_load_si128((__m128i*)(p)+3);
        xb[0] = _mm_xor_si128(xb[0], xb[1]);
        xb[2] = _mm_xor_si128(xb[2], xb[3]);
        xb[4] = _mm_xor_si128(xb[4], xb[5]);
        xb[6] = _mm_xor_si128(xb[6], xb[7]);
        _mm_store_si128((__m128i*)(q),   xb[0]);
        _mm_store_si128((__m128i*)(q)+1, xb[2]);
        _mm_store_si128((__m128i*)(q)+2, xb[4]);
        _mm_store_si128((__m128i*)(q)+3, xb[6]);
      }
      xb[1] = _mm_clmulepi64_si128(xa, xb[0], 0x10);
      xb[3] = _mm_clmulepi64_si128(xa, xb[2], 0x10);
      xb[5] = _mm_clmulepi64_si128(xa, xb[4], 0x10);
      xb[7] = _mm_clmulepi64_si128(xa, xb[6], 0x10);
      xb[0] = _mm_clmulepi64_si128(xa, xb[0], 0x00);
      xb[2] = _mm_clmulepi64_si128(xa, xb[2], 0x00);
      xb[4] = _mm_clmulepi64_si128(xa, xb[4], 0x00);
      xb[6] = _mm_clmulepi64_si128(xa, xb[6], 0x00);
      xc[0] = _mm_clmulepi64_si128(xa, xb[0], 0x11);
      xc[1] = _mm_clmulepi64_si128(xa, xb[1], 0x11);
      xc[2] = _mm_clmulepi64_si128(xa, xb[2], 0x11);
      xc[3] = _mm_clmulepi64_si128(xa, xb[3], 0x11);
      xc[4] = _mm_clmulepi64_si128(xa, xb[4], 0x11);
      xc[5] = _mm_clmulepi64_si128(xa, xb[5], 0x11);
      xc[6] = _mm_clmulepi64_si128(xa, xb[6], 0x11);
      xc[7] = _mm_clmulepi64_si128(xa, xb[7], 0x11);
      xb[0] = _mm_xor_si128(xb[0], xc[0]);
      xb[1] = _mm_xor_si128(xb[1], xc[1]);
      xb[2] = _mm_xor_si128(xb[2], xc[2]);
      xb[3] = _mm_xor_si128(xb[3], xc[3]);
      xb[4] = _mm_xor_si128(xb[4], xc[4]);
      xb[5] = _mm_xor_si128(xb[5], xc[5]);
      xb[6] = _mm_xor_si128(xb[6], xc[6]);
      xb[7] = _mm_xor_si128(xb[7], xc[7]);
      xc[0] = _mm_clmulepi64_si128(xa, xc[0], 0x11);
      xc[1] = _mm_clmulepi64_si128(xa, xc[1], 0x11);
      xc[2] = _mm_clmulepi64_si128(xa, xc[2], 0x11);
      xc[3] = _mm_clmulepi64_si128(xa, xc[3], 0x11);
      xc[4] = _mm_clmulepi64_si128(xa, xc[4], 0x11);
      xc[5] = _mm_clmulepi64_si128(xa, xc[5], 0x11);
      xc[6] = _mm_clmulepi64_si128(xa, xc[6], 0x11);
      xc[7] = _mm_clmulepi64_si128(xa, xc[7], 0x11);
      xb[0] = _mm_xor_si128(xb[0], xc[0]);
      xb[1] = _mm_xor_si128(xb[1], xc[1]);
      xb[2] = _mm_xor_si128(xb[2], xc[2]);
      xb[3] = _mm_xor_si128(xb[3], xc[3]);
      xb[4] = _mm_xor_si128(xb[4], xc[4]);
      xb[5] = _mm_xor_si128(xb[5], xc[5]);
      xb[6] = _mm_xor_si128(xb[6], xc[6]);
      xb[7] = _mm_xor_si128(xb[7], xc[7]);

      // code below equivalent to p[i]  ^= low half of xb[i], i = 0 ... 7
      //<
      xb[0] =_mm_unpacklo_epi64(xb[0], xb[1]);
      xb[2] =_mm_unpacklo_epi64(xb[2], xb[3]);
      xb[4] =_mm_unpacklo_epi64(xb[4], xb[5]);
      xb[6] =_mm_unpacklo_epi64(xb[6], xb[7]);
      xb[1] = _mm_load_si128((__m128i*)(p)); // here _mm_load_si128 is faster than _mm_set_epi64x
      xb[3] = _mm_load_si128((__m128i*)(p)+1);
      xb[5] = _mm_load_si128((__m128i*)(p)+2);
      xb[7] = _mm_load_si128((__m128i*)(p)+3);
      xb[0] = _mm_xor_si128(xb[0], xb[1]);
      xb[2] = _mm_xor_si128(xb[2], xb[3]);
      xb[4] = _mm_xor_si128(xb[4], xb[5]);
      xb[6] = _mm_xor_si128(xb[6], xb[7]);
      _mm_store_si128((__m128i*)(p),   xb[0]); //16-byte aligned store
      _mm_store_si128((__m128i*)(p)+1, xb[2]);
      _mm_store_si128((__m128i*)(p)+2, xb[4]);
      _mm_store_si128((__m128i*)(p)+3, xb[6]);
      //>
      if constexpr(!reverse)
      {
        // q[i] ^= p[i], i = 0 ... 7
        xb[1] = _mm_load_si128((__m128i*)(q));
        xb[3] = _mm_load_si128((__m128i*)(q)+1);
        xb[5] = _mm_load_si128((__m128i*)(q)+2);
        xb[7] = _mm_load_si128((__m128i*)(q)+3);
        xb[0] = _mm_xor_si128(xb[0], xb[1]);
        xb[2] = _mm_xor_si128(xb[2], xb[3]);
        xb[4] = _mm_xor_si128(xb[4], xb[5]);
        xb[6] = _mm_xor_si128(xb[6], xb[7]);
        _mm_store_si128((__m128i*)(q),   xb[0]);
        _mm_store_si128((__m128i*)(q)+1, xb[2]);
        _mm_store_si128((__m128i*)(q)+2, xb[4]);
        _mm_store_si128((__m128i*)(q)+3, xb[6]);
      }
      p+=8;
      q+=8;
    }
  }
#endif
#if 0
  else // logstride >=4
  {
    constexpr uint64_t stride = 1uLL << logstride;
    uint64_t* restr q = std::assume_aligned<16>(p + stride);
    for(uint64_t i = 0; i < stride; i += 16)
    {
      __m128i xb[16], xc[16];
      xb[0] = _mm_load_si128((__m128i*)(q));
      xb[2] = _mm_load_si128((__m128i*)(q)+1);
      xb[4] = _mm_load_si128((__m128i*)(q)+2);
      xb[6] = _mm_load_si128((__m128i*)(q)+3);
      xb[8] = _mm_load_si128((__m128i*)(q)+4);
      xb[10] = _mm_load_si128((__m128i*)(q)+5);
      xb[12] = _mm_load_si128((__m128i*)(q)+6);
      xb[14] = _mm_load_si128((__m128i*)(q)+7);
      if constexpr(reverse)
      {
        // q[i] ^= p[i], i = 0 ... 7
        xb[1] = _mm_load_si128((__m128i*)(p));
        xb[3] = _mm_load_si128((__m128i*)(p)+1);
        xb[5] = _mm_load_si128((__m128i*)(p)+2);
        xb[7] = _mm_load_si128((__m128i*)(p)+3);
        xb[9] = _mm_load_si128((__m128i*)(p)+4);
        xb[11] = _mm_load_si128((__m128i*)(p)+5);
        xb[13] = _mm_load_si128((__m128i*)(p)+6);
        xb[15] = _mm_load_si128((__m128i*)(p)+7);

        xb[0] = _mm_xor_si128(xb[0], xb[1]);
        xb[2] = _mm_xor_si128(xb[2], xb[3]);
        xb[4] = _mm_xor_si128(xb[4], xb[5]);
        xb[6] = _mm_xor_si128(xb[6], xb[7]);
        xb[8] = _mm_xor_si128(xb[8], xb[9]);
        xb[10] = _mm_xor_si128(xb[10], xb[11]);
        xb[12] = _mm_xor_si128(xb[12], xb[13]);
        xb[14] = _mm_xor_si128(xb[14], xb[15]);

        _mm_store_si128((__m128i*)(q),   xb[0]);
        _mm_store_si128((__m128i*)(q)+1, xb[2]);
        _mm_store_si128((__m128i*)(q)+2, xb[4]);
        _mm_store_si128((__m128i*)(q)+3, xb[6]);
        _mm_store_si128((__m128i*)(q)+4, xb[8]);
        _mm_store_si128((__m128i*)(q)+5, xb[10]);
        _mm_store_si128((__m128i*)(q)+6, xb[12]);
        _mm_store_si128((__m128i*)(q)+7, xb[14]);
      }
      xb[0x1] = _mm_clmulepi64_si128(xa, xb[0x0], 0x10);
      xb[0x3] = _mm_clmulepi64_si128(xa, xb[0x2], 0x10);
      xb[0x5] = _mm_clmulepi64_si128(xa, xb[0x4], 0x10);
      xb[0x7] = _mm_clmulepi64_si128(xa, xb[0x6], 0x10);
      xb[0x9] = _mm_clmulepi64_si128(xa, xb[0x8], 0x10);
      xb[0xb] = _mm_clmulepi64_si128(xa, xb[0xa], 0x10);
      xb[0xd] = _mm_clmulepi64_si128(xa, xb[0xc], 0x10);
      xb[0xf] = _mm_clmulepi64_si128(xa, xb[0xe], 0x10);

      xb[0x0] = _mm_clmulepi64_si128(xa, xb[0x0], 0x00);
      xb[0x2] = _mm_clmulepi64_si128(xa, xb[0x2], 0x00);
      xb[0x4] = _mm_clmulepi64_si128(xa, xb[0x4], 0x00);
      xb[0x6] = _mm_clmulepi64_si128(xa, xb[0x6], 0x00);
      xb[0x8] = _mm_clmulepi64_si128(xa, xb[0x8], 0x00);
      xb[0xa] = _mm_clmulepi64_si128(xa, xb[0xa], 0x00);
      xb[0xc] = _mm_clmulepi64_si128(xa, xb[0xc], 0x00);
      xb[0xe] = _mm_clmulepi64_si128(xa, xb[0xe], 0x00);

      xc[0x0] = _mm_clmulepi64_si128(xa, xb[0x0], 0x11);
      xc[0x1] = _mm_clmulepi64_si128(xa, xb[0x1], 0x11);
      xc[0x2] = _mm_clmulepi64_si128(xa, xb[0x2], 0x11);
      xc[0x3] = _mm_clmulepi64_si128(xa, xb[0x3], 0x11);
      xc[0x4] = _mm_clmulepi64_si128(xa, xb[0x4], 0x11);
      xc[0x5] = _mm_clmulepi64_si128(xa, xb[0x5], 0x11);
      xc[0x6] = _mm_clmulepi64_si128(xa, xb[0x6], 0x11);
      xc[0x7] = _mm_clmulepi64_si128(xa, xb[0x7], 0x11);
      xc[0x8] = _mm_clmulepi64_si128(xa, xb[0x8], 0x11);
      xc[0x9] = _mm_clmulepi64_si128(xa, xb[0x9], 0x11);
      xc[0xa] = _mm_clmulepi64_si128(xa, xb[0xa], 0x11);
      xc[0xb] = _mm_clmulepi64_si128(xa, xb[0xb], 0x11);
      xc[0xc] = _mm_clmulepi64_si128(xa, xb[0xc], 0x11);
      xc[0xd] = _mm_clmulepi64_si128(xa, xb[0xd], 0x11);
      xc[0xe] = _mm_clmulepi64_si128(xa, xb[0xe], 0x11);
      xc[0xf] = _mm_clmulepi64_si128(xa, xb[0xf], 0x11);

      xb[0x0] = _mm_xor_si128(xb[0x0], xc[0x0]);
      xb[0x1] = _mm_xor_si128(xb[0x1], xc[0x1]);
      xb[0x2] = _mm_xor_si128(xb[0x2], xc[0x2]);
      xb[0x3] = _mm_xor_si128(xb[0x3], xc[0x3]);
      xb[0x4] = _mm_xor_si128(xb[0x4], xc[0x4]);
      xb[0x5] = _mm_xor_si128(xb[0x5], xc[0x5]);
      xb[0x6] = _mm_xor_si128(xb[0x6], xc[0x6]);
      xb[0x7] = _mm_xor_si128(xb[0x7], xc[0x7]);
      xb[0x8] = _mm_xor_si128(xb[0x8], xc[0x8]);
      xb[0x9] = _mm_xor_si128(xb[0x9], xc[0x9]);
      xb[0xa] = _mm_xor_si128(xb[0xa], xc[0xa]);
      xb[0xb] = _mm_xor_si128(xb[0xb], xc[0xb]);
      xb[0xc] = _mm_xor_si128(xb[0xc], xc[0xc]);
      xb[0xd] = _mm_xor_si128(xb[0xd], xc[0xd]);
      xb[0xe] = _mm_xor_si128(xb[0xe], xc[0xe]);
      xb[0xf] = _mm_xor_si128(xb[0xf], xc[0xf]);

      xc[0x0] = _mm_clmulepi64_si128(xa, xc[0x0], 0x11);
      xc[0x1] = _mm_clmulepi64_si128(xa, xc[0x1], 0x11);
      xc[0x2] = _mm_clmulepi64_si128(xa, xc[0x2], 0x11);
      xc[0x3] = _mm_clmulepi64_si128(xa, xc[0x3], 0x11);
      xc[0x4] = _mm_clmulepi64_si128(xa, xc[0x4], 0x11);
      xc[0x5] = _mm_clmulepi64_si128(xa, xc[0x5], 0x11);
      xc[0x6] = _mm_clmulepi64_si128(xa, xc[0x6], 0x11);
      xc[0x7] = _mm_clmulepi64_si128(xa, xc[0x7], 0x11);
      xc[0x8] = _mm_clmulepi64_si128(xa, xc[0x8], 0x11);
      xc[0x9] = _mm_clmulepi64_si128(xa, xc[0x9], 0x11);
      xc[0xa] = _mm_clmulepi64_si128(xa, xc[0xa], 0x11);
      xc[0xb] = _mm_clmulepi64_si128(xa, xc[0xb], 0x11);
      xc[0xc] = _mm_clmulepi64_si128(xa, xc[0xc], 0x11);
      xc[0xd] = _mm_clmulepi64_si128(xa, xc[0xd], 0x11);
      xc[0xe] = _mm_clmulepi64_si128(xa, xc[0xe], 0x11);
      xc[0xf] = _mm_clmulepi64_si128(xa, xc[0xf], 0x11);

      xb[0x0] = _mm_xor_si128(xb[0x0], xc[0x0]);
      xb[0x1] = _mm_xor_si128(xb[0x1], xc[0x1]);
      xb[0x2] = _mm_xor_si128(xb[0x2], xc[0x2]);
      xb[0x3] = _mm_xor_si128(xb[0x3], xc[0x3]);
      xb[0x4] = _mm_xor_si128(xb[0x4], xc[0x4]);
      xb[0x5] = _mm_xor_si128(xb[0x5], xc[0x5]);
      xb[0x6] = _mm_xor_si128(xb[0x6], xc[0x6]);
      xb[0x7] = _mm_xor_si128(xb[0x7], xc[0x7]);
      xb[0x8] = _mm_xor_si128(xb[0x8], xc[0x8]);
      xb[0x9] = _mm_xor_si128(xb[0x9], xc[0x9]);
      xb[0xa] = _mm_xor_si128(xb[0xa], xc[0xa]);
      xb[0xb] = _mm_xor_si128(xb[0xb], xc[0xb]);
      xb[0xc] = _mm_xor_si128(xb[0xc], xc[0xc]);
      xb[0xd] = _mm_xor_si128(xb[0xd], xc[0xd]);
      xb[0xe] = _mm_xor_si128(xb[0xe], xc[0xe]);
      xb[0xf] = _mm_xor_si128(xb[0xf], xc[0xf]);

      // code below equivalent to p[i]  ^= low half of xb[i], i = 0 ... f
      //<
      xb[0x0] =_mm_unpacklo_epi64(xb[0x0], xb[0x1]);
      xb[0x2] =_mm_unpacklo_epi64(xb[0x2], xb[0x3]);
      xb[0x4] =_mm_unpacklo_epi64(xb[0x4], xb[0x5]);
      xb[0x6] =_mm_unpacklo_epi64(xb[0x6], xb[0x7]);
      xb[0x8] =_mm_unpacklo_epi64(xb[0x8], xb[0x9]);
      xb[0xa] =_mm_unpacklo_epi64(xb[0xa], xb[0xb]);
      xb[0xc] =_mm_unpacklo_epi64(xb[0xc], xb[0xd]);
      xb[0xe] =_mm_unpacklo_epi64(xb[0xe], xb[0xf]);


      xb[0x1] = _mm_load_si128((__m128i*)(p)+0x0);
      xb[0x3] = _mm_load_si128((__m128i*)(p)+0x1);
      xb[0x5] = _mm_load_si128((__m128i*)(p)+0x2);
      xb[0x7] = _mm_load_si128((__m128i*)(p)+0x3);
      xb[0x9] = _mm_load_si128((__m128i*)(p)+0x4);
      xb[0xb] = _mm_load_si128((__m128i*)(p)+0x5);
      xb[0xd] = _mm_load_si128((__m128i*)(p)+0x6);
      xb[0xf] = _mm_load_si128((__m128i*)(p)+0x7);

      xb[0x0] = _mm_xor_si128(xb[0x0], xb[0x1]);
      xb[0x2] = _mm_xor_si128(xb[0x2], xb[0x3]);
      xb[0x4] = _mm_xor_si128(xb[0x4], xb[0x5]);
      xb[0x6] = _mm_xor_si128(xb[0x6], xb[0x7]);
      xb[0x8] = _mm_xor_si128(xb[0x8], xb[0x9]);
      xb[0xa] = _mm_xor_si128(xb[0xa], xb[0xb]);
      xb[0xc] = _mm_xor_si128(xb[0xc], xb[0xd]);
      xb[0xe] = _mm_xor_si128(xb[0xe], xb[0xf]);

      _mm_store_si128((__m128i*)(p),   xb[0x0]); //16-byte aligned store
      _mm_store_si128((__m128i*)(p)+1, xb[0x2]);
      _mm_store_si128((__m128i*)(p)+2, xb[0x4]);
      _mm_store_si128((__m128i*)(p)+3, xb[0x6]);
      _mm_store_si128((__m128i*)(p)+4, xb[0x8]);
      _mm_store_si128((__m128i*)(p)+5, xb[0xa]);
      _mm_store_si128((__m128i*)(p)+6, xb[0xc]);
      _mm_store_si128((__m128i*)(p)+7, xb[0xe]);
      //>

      if constexpr(!reverse)
      {
        // q[i] ^= p[i], i = 0 ... 7
        xb[1] = _mm_load_si128((__m128i*)(q));
        xb[3] = _mm_load_si128((__m128i*)(q)+1);
        xb[5] = _mm_load_si128((__m128i*)(q)+2);
        xb[7] = _mm_load_si128((__m128i*)(q)+3);
        xb[9] = _mm_load_si128((__m128i*)(q)+4);
        xb[11] = _mm_load_si128((__m128i*)(q)+5);
        xb[13] = _mm_load_si128((__m128i*)(q)+6);
        xb[15] = _mm_load_si128((__m128i*)(q)+7);

        xb[0] = _mm_xor_si128(xb[0], xb[1]);
        xb[2] = _mm_xor_si128(xb[2], xb[3]);
        xb[4] = _mm_xor_si128(xb[4], xb[5]);
        xb[6] = _mm_xor_si128(xb[6], xb[7]);
        xb[8] = _mm_xor_si128(xb[8], xb[9]);
        xb[10] = _mm_xor_si128(xb[10], xb[11]);
        xb[12] = _mm_xor_si128(xb[12], xb[13]);
        xb[14] = _mm_xor_si128(xb[14], xb[15]);

        _mm_store_si128((__m128i*)(q),   xb[0]);
        _mm_store_si128((__m128i*)(q)+1, xb[2]);
        _mm_store_si128((__m128i*)(q)+2, xb[4]);
        _mm_store_si128((__m128i*)(q)+3, xb[6]);
        _mm_store_si128((__m128i*)(q)+4, xb[8]);
        _mm_store_si128((__m128i*)(q)+5, xb[10]);
        _mm_store_si128((__m128i*)(q)+6, xb[12]);
        _mm_store_si128((__m128i*)(q)+7, xb[14]);
      }
      p+=0x10;
      q+=0x10;
    }
  }

#endif
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
    T* restr pu)
{
  T* restr p = (T*) std::assume_aligned<16>(pu);
  static_assert(t >= 1);
  assert(logsize > t && logsize <= 2*t);
  //decompose_taylor_one_step
  const uint64_t delta_s   = 1uLL << (logstride + logsize - 1 - t); // logn - 1 - t >= 0
  const uint64_t n_s = 1uLL  << (logstride + logsize);
  const uint64_t m_s = n_s >> 1;
  T* q = p + delta_s - m_s;
  // m_s > 0, hence the loop below is not infinite
  // see however the discussion in mg_decompose_taylor_reverse_recursive about undefined behavior
  // in the loop
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
  // 'warning: iteration 2305843009213693951 invokes undefined behavior [-Waggressive-loop-optimizations]'
  // 2305843009213693951 = 0x1fffffffffffffff = 2**61 - 1
  // since sizeof(uint64_t) = 8, this is an index value where p[i] necessarily wraps
  // (depending on the value of p, it may of course wrap sooner).
  for(uint64_t i = m_s; i < n_s; i++) q[i] ^= p[i];
}
