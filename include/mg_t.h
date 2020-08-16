#pragma once

#include <cstdint>
#include <cstddef>

#include <immintrin.h>

#include "mg.h"

/**
* @brief sq_iter
* Iterated squaring in GF(2**64). See 'product'.
* returns 'a' squared 'iter' times.
* @param a
* @return (...(a**2)**2...)**2, with 'iter' squarings.
*/
template<unsigned int iter>
static inline uint64_t sq_iter(const uint64_t&a)
{
  constexpr uint64_t minpoly = 0x1b;
  __m128i xp, xa, xc;
  xp = _mm_set1_epi64x(minpoly);
  xa = _mm_set1_epi64x(a);
  for(unsigned int i = 0; i < iter; i++)
  {
    xa = _mm_clmulepi64_si128(xa, xa, 0x00);
    xc = _mm_clmulepi64_si128(xp, xa, 0x10);
    xa = _mm_xor_si128(xa, xc);
    xc = _mm_clmulepi64_si128(xp, xc, 0x10);
    xa = _mm_xor_si128(xa, xc);
  }
  return xa[0];
}

template<unsigned int logstride, unsigned int t, unsigned int logsize>
inline void mgt_decompose_taylor_recursive(uint64_t* p)
{
  static_assert(t >= 1);
  static_assert(logsize > t && logsize <= 2*t);
  //decompose_taylor_one_step
  constexpr uint64_t delta_s   = 1uLL << (logstride + logsize - 1 - t); // logn - 1 - t >= 0
  constexpr uint64_t n_s = 1uLL  << (logstride + logsize);
  constexpr uint64_t m_s = n_s >> 1;
  uint64_t* q = p + delta_s - m_s;
  // m_s > 0, hence the loop below is not infinite
  for(uint64_t i = n_s - 1; i > m_s - 1; i--) q[i] ^= p[i];

  if constexpr(logsize > t + 1)
  {
    mgt_decompose_taylor_recursive<logstride, t, logsize - 1>(p);
    mgt_decompose_taylor_recursive<logstride, t, logsize - 1>(p + m_s);
  }
}

template<unsigned int logstride, unsigned int t, unsigned int logsize>
inline void mgt_decompose_taylor_reverse_recursive(uint64_t* p)
{
  static_assert(t >= 1);
  static_assert(logsize > t && logsize <= 2*t);
  constexpr uint64_t n_s = 1uLL  << (logstride + logsize);
  constexpr uint64_t m_s = n_s >> 1;
  if constexpr(logsize > t + 1)
  {
    mgt_decompose_taylor_reverse_recursive<logstride, t, logsize - 1>(p);
    mgt_decompose_taylor_reverse_recursive<logstride, t, logsize - 1>(p + m_s);
  }

  //decompose_taylor_one_step_reverse
  constexpr uint64_t delta_s   = 1uLL << (logstride + logsize - 1 - t); // logn - 1 - t >= 0
  uint64_t* q = p + delta_s - m_s;
  for(uint64_t i = m_s; i < n_s; i++) q[i] ^= p[i];
}

template <int s, int logstride, int logsize, bool first_taylor_done>
inline void mgtt_core(uint64_t* poly, const uint64_t* offsets_mult)
// log2 size of each stride to be processed;
// should be <= 2**s, the input is assumed to be of size 2**logsize and only the 2**logsize first output values are computed
{
  if constexpr(s == 0)
  {
    eval_degree1<false, logstride>(offsets_mult[0], poly);
  }
  else
  {
    constexpr unsigned int t = 1 << (s - 1);
    if constexpr(logsize <= t)
    {
      mgtt_core<s - 1, logstride, logsize, first_taylor_done>(poly, offsets_mult);
    }
    else
    {
      // here 2*t >= logsize > t
      // on input: there are 2**'logstride' interleaved series of size 'eta' = 2**(2*logsize-t);
      if constexpr(!first_taylor_done)
      {
        //decompose_taylor_recursive(logstride, 2 * t, t, 1uLL << logsize, poly);
        mgt_decompose_taylor_recursive<logstride, t, logsize>(poly);
      }
      // fft on columns
      // if logsize >= t, each fft should process 2**(logsize - t) values
      // i.e. logsize' = logsize - t
      // offset' = offset >> t;
      mgtt_core<s - 1, logstride + t, logsize - t, false>(poly, offsets_mult + t);
      uint64_t offsets_local[t];
      for(unsigned int j = 0; j < t; j++) offsets_local[j] = offsets_mult[j];
      // 2*t >= logsize > t, therefore tau < 2**t
      const uint64_t tau   = 1uLL <<  (logsize - t);
      for(uint64_t i = 0; i < tau; i++)
      {
        // at each iteration, offsets_local[j] = beta_to_mult(offset + (i << (t-j))), j < t
        mgtt_core<s - 1, logstride, t, false>(poly, offsets_local);
        const long int h = _mm_popcnt_u64(i^(i+1)); // i^(i+1) is a power of 2 - 1
        for(unsigned int j = 0; j < t; j++)
        {
          int tp = t - j;
          offsets_local[j] ^= beta_over_mult_cumulative[h + tp] ^ beta_over_mult_cumulative[tp];
        }
        poly += 1uLL << (t + logstride);
      }
    }
  }
}

template <int s, unsigned int logstride, unsigned int logsize>
void mgtt_reverse_core(uint64_t* poly, const uint64_t* offsets_mult)
{
  if constexpr(s == 0)
  {
    eval_degree1<true, logstride>(offsets_mult[0], poly);
  }
  else
  {
    constexpr unsigned int t = 1 << (s - 1);
    if constexpr(logsize <= t)
    {
      mgtt_reverse_core<s - 1, logstride, logsize>(poly, offsets_mult);
    }
    else
    {
      const uint64_t tau   = 1uLL <<  (logsize - t);
      const uint64_t row_size = 1uLL << (t + logstride);
      uint64_t* poly_loc = poly;
      uint64_t offsets_local[t];
      for(unsigned int j = 0; j < t; j++) offsets_local[j] = offsets_mult[j];
      for(uint64_t i = 0; i < tau; i++)
      {
        // at each iteration, offsets_local[j] = beta_to_mult(offset + (i << (t-j))), j < t
        mgtt_reverse_core<s - 1, logstride, t>(poly_loc, offsets_local);
        long int h = _mm_popcnt_u64(i^(i+1));
        for(unsigned int j = 0; j < t; j++)
        {
          int tp = t - j;
          offsets_local[j] ^= beta_over_mult_cumulative[h + tp] ^ beta_over_mult_cumulative[tp];
        }
        poly_loc += row_size;
      }
      // reverse fft on columns
      //offset' = offset >> t;
      mgtt_reverse_core<s - 1, logstride + t, logsize - t>(poly, offsets_mult + t);
      mgt_decompose_taylor_reverse_recursive<logstride, t, logsize>(poly);
    }
  }
}


template <int s, int logstride, bool first_taylor_done>
inline void mgt_core(
    uint64_t* poly,
    uint64_t* offsets_mult,
    unsigned int logsize)
{
  assert(logsize < 40);
  if(logsize == 1) mgtt_core<s, logstride, 1, first_taylor_done>(poly, offsets_mult);
  if constexpr(s >= 1)
  {
    if(logsize == 2) mgtt_core<s, logstride, 2, first_taylor_done>(poly, offsets_mult);
  }
  if constexpr(s >= 2)
  {
    if(logsize == 3) mgtt_core<s, logstride, 3, first_taylor_done>(poly, offsets_mult);
    else if(logsize == 4) mgtt_core<s, logstride, 4, first_taylor_done>(poly, offsets_mult);
  }
  if constexpr(s >= 3)
  {
    if(logsize == 5) mgtt_core<s, logstride, 5, first_taylor_done>(poly, offsets_mult);
    else if(logsize == 6) mgtt_core<s, logstride, 6, first_taylor_done>(poly, offsets_mult);
    else if(logsize == 7) mgtt_core<s, logstride, 7, first_taylor_done>(poly, offsets_mult);
    else if(logsize == 8) mgtt_core<s, logstride, 8, first_taylor_done>(poly, offsets_mult);
  }
  if constexpr(s >= 4)
  {
    if(logsize == 9) mgtt_core<s, logstride, 9, first_taylor_done>(poly, offsets_mult);
    else if(logsize == 10) mgtt_core<s, logstride, 10, first_taylor_done>(poly, offsets_mult);
    else if(logsize == 11) mgtt_core<s, logstride, 11, first_taylor_done>(poly, offsets_mult);
    else if(logsize == 12) mgtt_core<s, logstride, 12, first_taylor_done>(poly, offsets_mult);
    else if(logsize == 13) mgtt_core<s, logstride, 13, first_taylor_done>(poly, offsets_mult);
    else if(logsize == 14) mgtt_core<s, logstride, 14, first_taylor_done>(poly, offsets_mult);
    else if(logsize == 15) mgtt_core<s, logstride, 15, first_taylor_done>(poly, offsets_mult);
    else if(logsize == 16) mgtt_core<s, logstride, 16, first_taylor_done>(poly, offsets_mult);
}
  if constexpr(s >= 5)
  {
    if(logsize == 17) mgtt_core<s, logstride, 17, first_taylor_done>(poly, offsets_mult);
    else if(logsize == 18) mgtt_core<s, logstride, 18, first_taylor_done>(poly, offsets_mult);
    else if(logsize == 19) mgtt_core<s, logstride, 19, first_taylor_done>(poly, offsets_mult);
    else if(logsize == 20) mgtt_core<s, logstride, 20, first_taylor_done>(poly, offsets_mult);
    else if(logsize == 21) mgtt_core<s, logstride, 21, first_taylor_done>(poly, offsets_mult);
    else if(logsize == 22) mgtt_core<s, logstride, 22, first_taylor_done>(poly, offsets_mult);
    else if(logsize == 23) mgtt_core<s, logstride, 23, first_taylor_done>(poly, offsets_mult);
    else if(logsize == 24) mgtt_core<s, logstride, 24, first_taylor_done>(poly, offsets_mult);
    else if(logsize == 25) mgtt_core<s, logstride, 25, first_taylor_done>(poly, offsets_mult);
    else if(logsize == 26) mgtt_core<s, logstride, 26, first_taylor_done>(poly, offsets_mult);
    else if(logsize == 27) mgtt_core<s, logstride, 27, first_taylor_done>(poly, offsets_mult);
    else if(logsize == 28) mgtt_core<s, logstride, 28, first_taylor_done>(poly, offsets_mult);
    else if(logsize == 29) mgtt_core<s, logstride, 29, first_taylor_done>(poly, offsets_mult);
    else if(logsize == 30) mgtt_core<s, logstride, 30, first_taylor_done>(poly, offsets_mult);
    else if(logsize == 31) mgtt_core<s, logstride, 31, first_taylor_done>(poly, offsets_mult);
    else if(logsize == 32) mgtt_core<s, logstride, 32, first_taylor_done>(poly, offsets_mult);
  }
  if constexpr(s >= 6)
  {
    if(logsize == 33) mgtt_core<s, logstride, 33, first_taylor_done>(poly, offsets_mult);
    else if(logsize == 34) mgtt_core<s, logstride, 34, first_taylor_done>(poly, offsets_mult);
    else if(logsize == 35) mgtt_core<s, logstride, 35, first_taylor_done>(poly, offsets_mult);
    else if(logsize == 36) mgtt_core<s, logstride, 36, first_taylor_done>(poly, offsets_mult);
    else if(logsize == 37) mgtt_core<s, logstride, 37, first_taylor_done>(poly, offsets_mult);
    else if(logsize == 38) mgtt_core<s, logstride, 38, first_taylor_done>(poly, offsets_mult);
    else if(logsize == 39) mgtt_core<s, logstride, 39, first_taylor_done>(poly, offsets_mult);
  }
}

template <int s, int logstride>
inline void mgt_reverse_core(
    uint64_t* poly,
    uint64_t* offsets_mult,
    unsigned int logsize)
{
  assert(logsize < 40);
  if(logsize == 1) mgtt_reverse_core<s, logstride, 1>(poly, offsets_mult);
  if constexpr(s >= 1)
  {
  if(logsize == 2) mgtt_reverse_core<s, logstride, 2>(poly, offsets_mult);
  }
  if constexpr(s >= 2)
  {
    if(logsize == 3) mgtt_reverse_core<s, logstride, 3>(poly, offsets_mult);
    else if(logsize == 4) mgtt_reverse_core<s, logstride, 4>(poly, offsets_mult);
  }
  if constexpr(s >= 3)
  {
    if(logsize == 5) mgtt_reverse_core<s, logstride, 5>(poly, offsets_mult);
    else if(logsize == 6) mgtt_reverse_core<s, logstride, 6>(poly, offsets_mult);
    else if(logsize == 7) mgtt_reverse_core<s, logstride, 7>(poly, offsets_mult);
    else if(logsize == 8) mgtt_reverse_core<s, logstride, 8>(poly, offsets_mult);
  }
  if constexpr(s >= 4)
  {
    if(logsize == 9) mgtt_reverse_core<s, logstride, 9>(poly, offsets_mult);
    else if(logsize == 10) mgtt_reverse_core<s, logstride, 10>(poly, offsets_mult);
    else if(logsize == 11) mgtt_reverse_core<s, logstride, 11>(poly, offsets_mult);
    else if(logsize == 12) mgtt_reverse_core<s, logstride, 12>(poly, offsets_mult);
    else if(logsize == 13) mgtt_reverse_core<s, logstride, 13>(poly, offsets_mult);
    else if(logsize == 14) mgtt_reverse_core<s, logstride, 14>(poly, offsets_mult);
    else if(logsize == 15) mgtt_reverse_core<s, logstride, 15>(poly, offsets_mult);
    else if(logsize == 16) mgtt_reverse_core<s, logstride, 16>(poly, offsets_mult);
  }
  if constexpr(s >= 5)
  {
    if(logsize == 17) mgtt_reverse_core<s, logstride, 17>(poly, offsets_mult);
    else if(logsize == 18) mgtt_reverse_core<s, logstride, 18>(poly, offsets_mult);
    else if(logsize == 19) mgtt_reverse_core<s, logstride, 19>(poly, offsets_mult);
    else if(logsize == 20) mgtt_reverse_core<s, logstride, 20>(poly, offsets_mult);
    else if(logsize == 21) mgtt_reverse_core<s, logstride, 21>(poly, offsets_mult);
    else if(logsize == 22) mgtt_reverse_core<s, logstride, 22>(poly, offsets_mult);
    else if(logsize == 23) mgtt_reverse_core<s, logstride, 23>(poly, offsets_mult);
    else if(logsize == 24) mgtt_reverse_core<s, logstride, 24>(poly, offsets_mult);
    else if(logsize == 25) mgtt_reverse_core<s, logstride, 25>(poly, offsets_mult);
    else if(logsize == 26) mgtt_reverse_core<s, logstride, 26>(poly, offsets_mult);
    else if(logsize == 27) mgtt_reverse_core<s, logstride, 27>(poly, offsets_mult);
    else if(logsize == 28) mgtt_reverse_core<s, logstride, 28>(poly, offsets_mult);
    else if(logsize == 29) mgtt_reverse_core<s, logstride, 29>(poly, offsets_mult);
    else if(logsize == 30) mgtt_reverse_core<s, logstride, 30>(poly, offsets_mult);
    else if(logsize == 31) mgtt_reverse_core<s, logstride, 31>(poly, offsets_mult);
    else if(logsize == 32) mgtt_reverse_core<s, logstride, 32>(poly, offsets_mult);
  }
  if constexpr(s >= 6)
  {
    if(logsize == 33) mgtt_reverse_core<s, logstride, 33>(poly, offsets_mult);
    else if(logsize == 34) mgtt_reverse_core<s, logstride, 34>(poly, offsets_mult);
    else if(logsize == 35) mgtt_reverse_core<s, logstride, 35>(poly, offsets_mult);
    else if(logsize == 36) mgtt_reverse_core<s, logstride, 36>(poly, offsets_mult);
    else if(logsize == 37) mgtt_reverse_core<s, logstride, 37>(poly, offsets_mult);
    else if(logsize == 38) mgtt_reverse_core<s, logstride, 38>(poly, offsets_mult);
    else if(logsize == 39) mgtt_reverse_core<s, logstride, 39>(poly, offsets_mult);
  }
}
