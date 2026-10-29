#pragma once

#include <cstdint>
#include <cassert>
#include <memory>
#include <immintrin.h>

#include "base.h"

/**
 * Optimized versions of mg_decompose_taylor_recursive and mg_decompose_taylor_reverse_recursive
 * (see mg_templates.h for their specification), used by the optimized variant of the product
 * (libfft/mg_fused.cpp). The results are identical to those of the original functions.
 *
 * Two optimizations, described in taylor_merge.md:
 * - two levels of the recursion are merged into a single step (for t >= 2), written as 8
 *   additions of ranges between disjoint parts of the array, two of which are done in a single loop;
 * - for t = 2 and logstride = 0 (rows of 8 and 16 words), the whole decomposition is done in AVX2
 *   registers (4 words per register), without loops.
 */

// dst[i] ^= src[i], i < len, for disjoint ranges
template<class T>
static inline void taylor_xor_range(T* restr dst, const T* restr src, uint64_t len)
{
  for(uint64_t i = 0; i < len; i++) dst[i] ^= src[i];
}

// dst[i] ^= s1[i] ^ s2[i], i < len, for disjoint ranges
template<class T>
static inline void taylor_xor2(T* restr dst, const T* restr s1, const T* restr s2, uint64_t len)
{
  for(uint64_t i = 0; i < len; i++) dst[i] ^= s1[i] ^ s2[i];
}

#ifdef __AVX2__
// rows of 8 and 16 words (t = 2, logstride = 0), in AVX2 registers. For 16 words, the quarters
// A, B, C, D of the array (see taylor_merge.md, g = 1) are one register each.
static inline __m256i taylor_shift_up(__m256i x)   // (x0, x1, x2, x3) -> (0, x0, x1, x2)
{
  return _mm256_blend_epi32(_mm256_permute4x64_epi64(x, 0x93), _mm256_setzero_si256(), 0x03);
}
static inline __m256i taylor_lane3_to_lane0(__m256i x)   // -> (x3, 0, 0, 0)
{
  return _mm256_blend_epi32(_mm256_setzero_si256(), _mm256_permute4x64_epi64(x, 0xff), 0x03);
}
static inline __m256i taylor_hi_to_lo(__m256i x)   // -> (x2, x3, 0, 0)
{
  return _mm256_permute2x128_si256(x, x, 0x81);
}
static inline __m256i taylor_lo_to_hi(__m256i x)   // -> (0, 0, x0, x1)
{
  return _mm256_permute2x128_si256(x, x, 0x08);
}

static inline void taylor_row16_forward(uint64_t* p)
{
  __m256i A = _mm256_loadu_si256((const __m256i*) p);
  __m256i B = _mm256_loadu_si256((const __m256i*) (p + 4));
  __m256i C = _mm256_loadu_si256((const __m256i*) (p + 8));
  __m256i D = _mm256_loadu_si256((const __m256i*) (p + 12));
  const __m256i Cm = _mm256_xor_si256(C, taylor_hi_to_lo(D));                                    // step 1
  B = _mm256_xor_si256(B, _mm256_xor_si256(taylor_hi_to_lo(Cm), taylor_lo_to_hi(D)));            // steps 2, 3
  B = _mm256_xor_si256(B, taylor_lane3_to_lane0(B));                                              // step 4
  A = _mm256_xor_si256(A, _mm256_xor_si256(taylor_lo_to_hi(Cm), taylor_shift_up(B)));            // steps 5, 6
  D = _mm256_xor_si256(D, taylor_lane3_to_lane0(D));                                              // step 7
  C = _mm256_xor_si256(Cm, taylor_shift_up(D));                                                   // step 8
  _mm256_storeu_si256((__m256i*) p, A);
  _mm256_storeu_si256((__m256i*) (p + 4), B);
  _mm256_storeu_si256((__m256i*) (p + 8), C);
  _mm256_storeu_si256((__m256i*) (p + 12), D);
}

static inline void taylor_row16_reverse(uint64_t* p)
{
  __m256i A = _mm256_loadu_si256((const __m256i*) p);
  __m256i B = _mm256_loadu_si256((const __m256i*) (p + 4));
  __m256i C = _mm256_loadu_si256((const __m256i*) (p + 8));
  __m256i D = _mm256_loadu_si256((const __m256i*) (p + 12));
  const __m256i Cm = _mm256_xor_si256(C, taylor_shift_up(D));                                     // undo step 8
  D = _mm256_xor_si256(D, taylor_lane3_to_lane0(D));                                              // undo step 7
  A = _mm256_xor_si256(A, _mm256_xor_si256(taylor_shift_up(B), taylor_lo_to_hi(Cm)));            // undo steps 6, 5
  B = _mm256_xor_si256(B, taylor_lane3_to_lane0(B));                                              // undo step 4
  B = _mm256_xor_si256(B, _mm256_xor_si256(taylor_lo_to_hi(D), taylor_hi_to_lo(Cm)));            // undo steps 3, 2
  C = _mm256_xor_si256(Cm, taylor_hi_to_lo(D));                                                   // undo step 1
  _mm256_storeu_si256((__m256i*) p, A);
  _mm256_storeu_si256((__m256i*) (p + 4), B);
  _mm256_storeu_si256((__m256i*) (p + 8), C);
  _mm256_storeu_si256((__m256i*) (p + 12), D);
}

static inline void taylor_row8_forward(uint64_t* p)
{
  const __m256i lo = _mm256_loadu_si256((const __m256i*) p);
  __m256i hi = _mm256_loadu_si256((const __m256i*) (p + 4));
  hi = _mm256_xor_si256(hi, taylor_lane3_to_lane0(hi));
  _mm256_storeu_si256((__m256i*) p, _mm256_xor_si256(lo, taylor_shift_up(hi)));
  _mm256_storeu_si256((__m256i*) (p + 4), hi);
}

static inline void taylor_row8_reverse(uint64_t* p)
{
  __m256i lo = _mm256_loadu_si256((const __m256i*) p);
  __m256i hi = _mm256_loadu_si256((const __m256i*) (p + 4));
  lo = _mm256_xor_si256(lo, taylor_shift_up(hi));
  hi = _mm256_xor_si256(hi, taylor_lane3_to_lane0(hi));
  _mm256_storeu_si256((__m256i*) p, lo);
  _mm256_storeu_si256((__m256i*) (p + 4), hi);
}
#endif

/**
 * @brief mg_decompose_taylor_opt
 * same specification as mg_decompose_taylor_recursive.
 */
template<unsigned int logstride, unsigned int t, class T>
inline void mg_decompose_taylor_opt(
    unsigned int logsize,
    T* restr pu)
{
  T* restr p = (T*) std::assume_aligned<16>(pu);
  static_assert(t >= 1);
  assert(logsize > t && logsize <= 2*t);
#ifdef __AVX2__
  if constexpr(logstride == 0 && t == 2 && sizeof(T) == 8)
  {
    if(logsize == 4) taylor_row16_forward((uint64_t*) p); else taylor_row8_forward((uint64_t*) p);
    return;
  }
#endif
  const uint64_t n_s = 1uLL << (logstride + logsize);
  if constexpr(t >= 2)
  {
    if(logsize >= t + 2)
    {
      // two levels at once, see taylor_merge.md. A, B, C, D: quarters of the array.
      // g = delta / 2, where delta is the shift of the first level
      const uint64_t q_s = n_s >> 2;
      const uint64_t g_s = 1uLL << (logstride + logsize - 2 - t);
      T* restr A = p;
      T* restr B = p + q_s;
      T* restr C = p + 2 * q_s;
      T* restr D = p + 3 * q_s;
      taylor_xor_range(C, D + q_s - 2 * g_s, 2 * g_s);
      taylor_xor_range(B, C + q_s - 2 * g_s, 2 * g_s);
      taylor_xor_range(B + 2 * g_s, D, q_s - 2 * g_s);
      taylor_xor_range(B, B + q_s - g_s, g_s);
      taylor_xor_range(A + g_s, B, g_s);                               // steps 5 and 6
      taylor_xor2(A + 2 * g_s, C, B + g_s, q_s - 2 * g_s);
      taylor_xor_range(D, D + q_s - g_s, g_s);
      taylor_xor_range(C + g_s, D, q_s - g_s);
      if(logsize > t + 2)
      {
        mg_decompose_taylor_opt<logstride, t, T>(logsize - 2, A);
        mg_decompose_taylor_opt<logstride, t, T>(logsize - 2, B);
        mg_decompose_taylor_opt<logstride, t, T>(logsize - 2, C);
        mg_decompose_taylor_opt<logstride, t, T>(logsize - 2, D);
      }
      return;
    }
  }
  // one level
  const uint64_t delta_s = 1uLL << (logstride + logsize - 1 - t);
  const uint64_t m_s = n_s >> 1;
  T* q = p + delta_s - m_s;
  for(uint64_t i = n_s - 1; i > m_s - 1; i--) q[i] ^= p[i];
  if(logsize > t + 1)
  {
    mg_decompose_taylor_opt<logstride, t, T>(logsize - 1, p);
    mg_decompose_taylor_opt<logstride, t, T>(logsize - 1, p + m_s);
  }
}

/**
 * @brief mg_decompose_taylor_reverse_opt
 * same specification as mg_decompose_taylor_reverse_recursive.
 */
template<unsigned int logstride, unsigned int t, class T>
inline void mg_decompose_taylor_reverse_opt(
    unsigned int logsize,
    T* restr p)
{
  static_assert(t >= 1);
  assert(logsize > t && logsize <= 2*t);
#ifdef __AVX2__
  if constexpr(logstride == 0 && t == 2 && sizeof(T) == 8)
  {
    if(logsize == 4) taylor_row16_reverse((uint64_t*) p); else taylor_row8_reverse((uint64_t*) p);
    return;
  }
#endif
  const uint64_t n_s = 1uLL << (logstride + logsize);
  if constexpr(t >= 2)
  {
    if(logsize >= t + 2)
    {
      const uint64_t q_s = n_s >> 2;
      const uint64_t g_s = 1uLL << (logstride + logsize - 2 - t);
      T* restr A = p;
      T* restr B = p + q_s;
      T* restr C = p + 2 * q_s;
      T* restr D = p + 3 * q_s;
      if(logsize > t + 2)
      {
        mg_decompose_taylor_reverse_opt<logstride, t, T>(logsize - 2, A);
        mg_decompose_taylor_reverse_opt<logstride, t, T>(logsize - 2, B);
        mg_decompose_taylor_reverse_opt<logstride, t, T>(logsize - 2, C);
        mg_decompose_taylor_reverse_opt<logstride, t, T>(logsize - 2, D);
      }
      // the additions of the forward function, in reverse order
      taylor_xor_range(C + g_s, D, q_s - g_s);
      taylor_xor_range(D, D + q_s - g_s, g_s);
      taylor_xor2(A + 2 * g_s, C, B + g_s, q_s - 2 * g_s);
      taylor_xor_range(A + g_s, B, g_s);
      taylor_xor_range(B, B + q_s - g_s, g_s);
      taylor_xor_range(B + 2 * g_s, D, q_s - 2 * g_s);
      taylor_xor_range(B, C + q_s - 2 * g_s, 2 * g_s);
      taylor_xor_range(C, D + q_s - 2 * g_s, 2 * g_s);
      return;
    }
  }
  // one level
  const uint64_t m_s = n_s >> 1;
  if(logsize > t + 1)
  {
    mg_decompose_taylor_reverse_opt<logstride, t, T>(logsize - 1, p);
    mg_decompose_taylor_reverse_opt<logstride, t, T>(logsize - 1, p + m_s);
  }
  const uint64_t delta_s = 1uLL << (logstride + logsize - 1 - t);
  T* q = p + delta_s - m_s;
  for(uint64_t i = m_s; i < n_s; i++) q[i] ^= p[i];
}
