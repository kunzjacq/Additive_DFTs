#pragma once

#include <cstddef>
#include <cassert> // for assert
#include <memory> // for assume_aligned

#include <immintrin.h> // for x86_64 intrinsics

#include "base.h"


#ifdef __GNUC__
// unused, _mm_set_epi64x is preferred
#define m128_extract(v,idx) (v)[idx]
#else
#ifdef _MSC_VER
// unused, _mm_set_epi64x is preferred
#define m128_extract(v,idx) (v).m128i_i64[idx]
#else
#error "unsupported compiler"
#endif
#endif

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
static inline void eval_degree1_ref(const uint64_t val,  uint64_t* restr pu)
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
        xb[1] = _mm_load_si128((__m128i*)(q));
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


// Optimized eval_degree1 kernels.
// - The reduction of the 128-bit product modulo the field polynomial uses one carry-less
//   multiplication and a table lookup (see below) instead of two carry-less multiplications.
// - If the compiler targets VPCLMULQDQ and AVX2 (e.g. -march=native on a CPU that has them),
//   the wide-stride kernels (logstride >= 2) use 256-bit carry-less multiplications, i.e. two
//   field products per instruction.
// Define MG_USE_REF_EVAL_DEGREE1 to use the simple reference kernel eval_degree1_ref instead.
#if defined(__VPCLMULQDQ__) && defined(__AVX2__)
#define MG_USE_YMM_CLMUL
#endif
// Reduction of the 128-bit product (hi:lo) mod x^64 + x^4 + x^3 + x + 1 (m = 0x1b):
//   hi*m is 68 bits: low 64 bits h0 and a 4-bit spill h1 (< 16);  result = lo ^ h0 ^ (h1*m)
//   h1*m depends on a nibble only: 16-entry table, applied with pshufb instead of a 3rd clmul.
static inline __m128i mg_lut128()
{
  return _mm_setr_epi8(0, 27, 54, 45, 108, 119, 90, 65, 216, 195, 238, 245, 180, 175, 130, 153);
}

template<bool reverse, int logstride>
static inline void eval_degree1_new(const uint64_t val, uint64_t* restr pu)
{
  uint64_t* restr p = (uint64_t*) std::assume_aligned<16>(pu);
  constexpr uint64_t minpoly = 0x1b;
  const __m128i xa  = _mm_set_epi64x(minpoly, val);
  const __m128i lut = mg_lut128();
  if constexpr(logstride == 0)
  {
    if constexpr(reverse) p[1] ^= p[0];
    __m128i xb = _mm_set1_epi64x(p[1]);
    xb = _mm_clmulepi64_si128(xa, xb, 0x00);
    __m128i xc = _mm_clmulepi64_si128(xa, xb, 0x11);
    xb = _mm_xor_si128(xb, xc);
    xc = _mm_srli_si128(_mm_shuffle_epi8(lut, xc), 8);
    xb = _mm_xor_si128(xb, xc);
    p[0] ^= _mm_cvtsi128_si64(xb);
    if constexpr(!reverse) p[1] ^= p[0];
  }
  else if constexpr(logstride == 1)
  {
    __m128i P = _mm_load_si128((__m128i*)(p));
    __m128i Q = _mm_load_si128((__m128i*)(p) + 1);
    if constexpr(reverse)
    {
      Q = _mm_xor_si128(Q, P);
      _mm_store_si128((__m128i*)(p) + 1, Q);
    }
    __m128i e  = _mm_clmulepi64_si128(xa, Q, 0x00);
    __m128i o  = _mm_clmulepi64_si128(xa, Q, 0x10);
    __m128i xe = _mm_clmulepi64_si128(xa, e, 0x11);
    __m128i xo = _mm_clmulepi64_si128(xa, o, 0x11);
    e = _mm_xor_si128(e, xe);
    o = _mm_xor_si128(o, xo);
    __m128i r = _mm_xor_si128(_mm_unpacklo_epi64(e, o),
                              _mm_unpackhi_epi64(_mm_shuffle_epi8(lut, xe), _mm_shuffle_epi8(lut, xo)));
    P = _mm_xor_si128(P, r);
    _mm_store_si128((__m128i*)(p), P);
    if constexpr(!reverse)
    {
      Q = _mm_xor_si128(Q, P);
      _mm_store_si128((__m128i*)(p) + 1, Q);
    }
  }
  else
  {
    constexpr uint64_t stride = 1uLL << logstride;
    uint64_t* restr q = p + stride;
#ifdef MG_USE_YMM_CLMUL
    const __m256i xa2  = _mm256_broadcastsi128_si256(xa);
    const __m256i lut2 = _mm256_broadcastsi128_si256(lut);
    constexpr int U = (logstride >= 3) ? 2 : 1; // 256-bit vectors (4 elements each) per iteration
    for(uint64_t i = 0; i < stride; i += 4 * U)
    {
      __m256i P[U], Q[U];
      for(int u = 0; u < U; u++)
      {
        P[u] = _mm256_loadu_si256((__m256i*)(p + 4*u));
        Q[u] = _mm256_loadu_si256((__m256i*)(q + 4*u));
        if constexpr(reverse)
        {
          Q[u] = _mm256_xor_si256(Q[u], P[u]);
          _mm256_storeu_si256((__m256i*)(q + 4*u), Q[u]);
        }
      }
      for(int u = 0; u < U; u++)
      {
        __m256i e  = _mm256_clmulepi64_epi128(xa2, Q[u], 0x00);
        __m256i o  = _mm256_clmulepi64_epi128(xa2, Q[u], 0x10);
        __m256i xe = _mm256_clmulepi64_epi128(xa2, e, 0x11);
        __m256i xo = _mm256_clmulepi64_epi128(xa2, o, 0x11);
        e = _mm256_xor_si256(e, xe);
        o = _mm256_xor_si256(o, xo);
        __m256i r = _mm256_xor_si256(_mm256_unpacklo_epi64(e, o),
                      _mm256_unpackhi_epi64(_mm256_shuffle_epi8(lut2, xe), _mm256_shuffle_epi8(lut2, xo)));
        P[u] = _mm256_xor_si256(P[u], r);
        _mm256_storeu_si256((__m256i*)(p + 4*u), P[u]);
        if constexpr(!reverse)
        {
          Q[u] = _mm256_xor_si256(Q[u], P[u]);
          _mm256_storeu_si256((__m256i*)(q + 4*u), Q[u]);
        }
      }
      p += 4 * U;
      q += 4 * U;
    }
#else
    for(uint64_t i = 0; i < stride; i += 4)
    {
      __m128i P0 = _mm_load_si128((__m128i*)(p)),     P1 = _mm_load_si128((__m128i*)(p) + 1);
      __m128i Q0 = _mm_load_si128((__m128i*)(q)),     Q1 = _mm_load_si128((__m128i*)(q) + 1);
      if constexpr(reverse)
      {
        Q0 = _mm_xor_si128(Q0, P0);
        Q1 = _mm_xor_si128(Q1, P1);
        _mm_store_si128((__m128i*)(q),     Q0);
        _mm_store_si128((__m128i*)(q) + 1, Q1);
      }
      __m128i e0 = _mm_clmulepi64_si128(xa, Q0, 0x00), o0 = _mm_clmulepi64_si128(xa, Q0, 0x10);
      __m128i e1 = _mm_clmulepi64_si128(xa, Q1, 0x00), o1 = _mm_clmulepi64_si128(xa, Q1, 0x10);
      __m128i xe0 = _mm_clmulepi64_si128(xa, e0, 0x11), xo0 = _mm_clmulepi64_si128(xa, o0, 0x11);
      __m128i xe1 = _mm_clmulepi64_si128(xa, e1, 0x11), xo1 = _mm_clmulepi64_si128(xa, o1, 0x11);
      e0 = _mm_xor_si128(e0, xe0); o0 = _mm_xor_si128(o0, xo0);
      e1 = _mm_xor_si128(e1, xe1); o1 = _mm_xor_si128(o1, xo1);
      __m128i r0 = _mm_xor_si128(_mm_unpacklo_epi64(e0, o0),
                     _mm_unpackhi_epi64(_mm_shuffle_epi8(lut, xe0), _mm_shuffle_epi8(lut, xo0)));
      __m128i r1 = _mm_xor_si128(_mm_unpacklo_epi64(e1, o1),
                     _mm_unpackhi_epi64(_mm_shuffle_epi8(lut, xe1), _mm_shuffle_epi8(lut, xo1)));
      P0 = _mm_xor_si128(P0, r0);
      P1 = _mm_xor_si128(P1, r1);
      _mm_store_si128((__m128i*)(p),     P0);
      _mm_store_si128((__m128i*)(p) + 1, P1);
      if constexpr(!reverse)
      {
        Q0 = _mm_xor_si128(Q0, P0);
        Q1 = _mm_xor_si128(Q1, P1);
        _mm_store_si128((__m128i*)(q),     Q0);
        _mm_store_si128((__m128i*)(q) + 1, Q1);
      }
      p += 4;
      q += 4;
    }
#endif
  }
}

template<bool reverse, int logstride>
static inline void eval_degree1(const uint64_t val, uint64_t* restr pu)
{
#ifdef MG_USE_REF_EVAL_DEGREE1
  eval_degree1_ref<reverse, logstride>(val, pu);
#else
  eval_degree1_new<reverse, logstride>(val, pu);
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
