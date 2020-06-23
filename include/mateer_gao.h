#pragma once

#include <cstdint>
#include <cassert>

#include "decompose_taylor.h"
#include "cantor.h"

// s = log log of block size (for recursion: initialized with field log-log size)
template <class word, int s>
void fft_aux_ref(
    cantor_basis<word>* c_b,
    word* poly,
    word j, // block index to process
    unsigned int logstride)
{
  if constexpr(s == 0)
  {
    const uint64_t stride = 1uL << logstride;
    const word val = c_b->beta_to_gamma(j << 1);
    for(uint64_t i = 0; i < stride; i++)
    {
      // computes (u,v) where
      // u = f_0 + f1 * w_{2.j},
      // v = f_0 + f1 * w_{2.j+1} = f_0 + f1 *(w_{2.j} + 1) = u + f_1
      // f_0 = poly[i], f_1 = poly[i + stride]
      poly[i]          ^= c_b->multiply(poly[i + stride], val);
      poly[i + stride] ^= poly[i];
    }
  }
  else
  {
    constexpr unsigned int t = 1   << (s - 1);
    const uint64_t tau   = 1uL <<  t;
    const uint64_t eta   = 1uL << (2 * t);
    decompose_taylor(logstride, 2 * t, t, eta, poly);
    // on input: there are 'stride' series of size 'eta' = 'tau' * 'tau';
    // elt k of series i is in poly[i + stride * k].
    // do fft on columns.
    // series i is split into 'tau' series of size 'tau', s.t. element a of series b is element of index
    // b + a tau in original series.
    // There are tau * stride subseries.
    // Consider element of index A=a < tau in subseries S of index B < tau * stride.
    // Series S is mapped as the b = (B - i) / stride subseries of the initial series of
    // index i = B mod stride. With this choice,
    // i = B mod stride
    // k = b + a tau =(B - i) / stride + A * tau; k * stride = B - i + A * tau * stride
    // => poly[i + k * stride] = poly[i + B - i + A * tau * stride] = poly[B + A * tau * stride]
    // => performing FFTs on all subseries can be performed as a recursive call with
    // with stride' = tau*stride, i.e. logstride' = logstride + t
    // hence the recursive call
    fft_aux_ref<word, s-1>(c_b, poly, j, logstride + t);
    const uint64_t row_size = 1uL << (t + logstride);
    for(uint64_t i = 0; i < tau; i++)
    {
      word j_loop = (j << t) | i;
      fft_aux_ref<word, s-1>(c_b, poly, j_loop, logstride);
      poly += row_size;
    }
  }
}

template <class word, int s>
void fft_aux_ref_truncated(
    cantor_basis<word>* c_b,
    word* poly,
    word j, // block index to process
    unsigned int logstride,
    unsigned int logsize)
// log2 size of each stride to be processed;
// should be <= 2**s, the input is assumed to be of size 2**logsize and only the 2**logsize first output values are computed
{
  if constexpr(s == 0)
  {
    const uint64_t stride = 1uL << logstride;
    const word val = c_b->beta_to_gamma(j << 1);
    for(uint64_t i = 0; i < stride; i++)
    {
      // computes (u,v) where
      // u = f_0 + f1 * w_{2.j},
      // v = f_0 + f1 * w_{2.j+1} = f_0 + f1 *(w_{2.j} + 1) = u + f_1
      // f_0 = poly[i], f_1 = poly[i + stride]
      poly[i]          ^= c_b->multiply(poly[i + stride], val);
      poly[i + stride] ^= poly[i];
    }
  }
  else
  {
    const unsigned int t = 1 << (s - 1);
    if(logsize <= t)
    {
      fft_aux_ref_truncated<word, s - 1>(c_b, poly, j << t, logstride, logsize);
    }
    else
    {
      const uint64_t tau   = 1uL <<  (logsize - t);
      const uint64_t eta   = 1uL << logsize;
      // on input: there are 2**'logstride' interleaved series of size 'eta' = 2**(2*logsize-t);
      decompose_taylor_iterative_alt(logstride, 2 * t, t, eta, poly);
      // fft on columns
      // if logsize >= t, each fft should process 2**(logsize - t) values
      // i.e. logsize' = logsize - t
      fft_aux_ref_truncated<word, s - 1>(c_b, poly, j, logstride + t, logsize - t);
      const uint64_t row_size = 1uL << (t + logstride);
      for(uint64_t i = 0; i < tau; i++)
      {
        word j_loop = (j << t) | i;
        fft_aux_ref_truncated<word, s - 1>(c_b, poly, j_loop, logstride, t);
        poly += row_size;
      }
    }
  }
}

template <class word, int s>
void fft_aux_ref_truncated_mult(
    cantor_basis<word>* c_b,
    word* poly,
    word j, // block index to process
    unsigned int logstride,
    unsigned int logsize)
// log2 size of each stride to be processed;
// should be <= 2**s, the input is assumed to be of size 2**logsize and only the 2**logsize first output values are computed
{
  if constexpr(s == 0)
  {
    const uint64_t stride = 1uL << logstride;
    const word val = c_b->beta_to_mult(j << 1);
    for(uint64_t i = 0; i < stride; i++)
    {
      // computes (u,v) where
      // u = f_0 + f1 * w_{2.j},
      // v = f_0 + f1 * w_{2.j+1} = f_0 + f1 *(w_{2.j} + 1) = u + f_1
      // f_0 = poly[i], f_1 = poly[i + stride]
      poly[i]          ^= c_b->multiply_mult_repr(poly[i + stride], val);
      poly[i + stride] ^= poly[i];
    }
  }
  else
  {
    const unsigned int t = 1 << (s - 1);
    if(logsize <= t)
    {
      fft_aux_ref_truncated_mult<word, s - 1>(c_b, poly, j << t, logstride, logsize);
    }
    else
    {
      const uint64_t tau   = 1uL <<  (logsize - t);
      const uint64_t eta   = 1uL << logsize;
      // on input: there are 2**'logstride' interleaved series of size 'eta' = 2**(2*logsize-t);
      decompose_taylor_iterative_alt(logstride, 2 * t, t, eta, poly);
      // fft on columns
      // if logsize >= t, each fft should process 2**(logsize - t) values
      // i.e. logsize' = logsize - t
      fft_aux_ref_truncated_mult<word, s - 1>(c_b, poly, j, logstride + t, logsize - t);
      const uint64_t row_size = 1uL << (t + logstride);
      for(uint64_t i = 0; i < tau; i++)
      {
        word j_loop = (j << t) | i;
        fft_aux_ref_truncated_mult<word, s - 1>(c_b, poly, j_loop, logstride, t);
        poly += row_size;
      }
    }
  }
}

template <class word, int s>
void fft_aux_ref_truncated_reverse(
    cantor_basis<word>* c_b,
    word* poly,
    word j, // block index to process
    unsigned int logstride,
    unsigned int logsize)
// log2 size of each stride to be processed;
// should be <= 2**s, the input is assumed to be of size 2**logsize and only the 2**logsize first output values are computed
{
  if constexpr(s == 0)
  {
    const uint64_t stride = 1uL << logstride;
    const word val = c_b->beta_to_gamma(j << 1);
    for(uint64_t i = 0; i < stride; i++)
    {
      poly[i + stride] ^= poly[i];
      poly[i]          ^= c_b->multiply(poly[i + stride], val);
    }
  }
  else
  {
    constexpr unsigned int t = 1 << (s - 1);
    if(logsize <= t)
    {
      fft_aux_ref_truncated_reverse<word, s - 1>(c_b, poly, j << t, logstride, logsize);
    }
    else
    {
      const uint64_t tau   = 1uL <<  (logsize - t);
      const uint64_t eta   = 1uL << logsize;
      const uint64_t row_size = 1uL << (t + logstride);
      word* poly_loc = poly;
      for(uint64_t i = 0; i < tau; i++)
      {
        word j_loop = (j << t) | i;
        fft_aux_ref_truncated_reverse<word, s - 1>(c_b, poly_loc, j_loop, logstride, t);
        poly_loc += row_size;
      }
      // reverse fft on columns
      fft_aux_ref_truncated_reverse<word, s - 1>(c_b, poly, j, logstride + t, logsize - t);
      decompose_taylor_reverse_iterative_alt(logstride, 2 * t, t, eta, poly);
    }
  }
}

template <class word, int s>
void fft_aux_ref_truncated_reverse_mult(
    cantor_basis<word>* c_b,
    word* poly,
    word j, // block index to process
    unsigned int logstride,
    unsigned int logsize)
// log2 size of each stride to be processed;
// should be <= 2**s, the input is assumed to be of size 2**logsize and only the 2**logsize first output values are computed
{
  if constexpr(s == 0)
  {
    const uint64_t stride = 1uL << logstride;
    const word val = c_b->beta_to_mult(j << 1);
    for(uint64_t i = 0; i < stride; i++)
    {
      poly[i + stride] ^= poly[i];
      poly[i]          ^= c_b->multiply_mult_repr(poly[i + stride], val);
    }
  }
  else
  {
    constexpr unsigned int t = 1 << (s - 1);
    if(logsize <= t)
    {
      fft_aux_ref_truncated_reverse_mult<word, s - 1>(c_b, poly, j << t, logstride, logsize);
    }
    else
    {
      const uint64_t tau   = 1uL <<  (logsize - t);
      const uint64_t eta   = 1uL << logsize;
      const uint64_t row_size = 1uL << (t + logstride);
      word* poly_loc = poly;
      for(uint64_t i = 0; i < tau; i++)
      {
        word j_loop = (j << t) | i;
        fft_aux_ref_truncated_reverse_mult<word, s - 1>(c_b, poly_loc, j_loop, logstride, t);
        poly_loc += row_size;
      }
      // reverse fft on columns
      fft_aux_ref_truncated_reverse_mult<word, s - 1>(c_b, poly, j, logstride + t, logsize - t);
      decompose_taylor_reverse_iterative_alt(logstride, 2 * t, t, eta, poly);
    }
  }
}


template <class word, int s>
void fft_aux_fast(
    cantor_basis<word>* c_b,
    word* poly,
    word j,
    unsigned int logstride,
    unsigned int log_outer_parallelism)
{
  if constexpr(s == 0)
  {
    uint64_t stride = 1uL << logstride;
    word val = c_b->beta_to_gamma(j << 1);
    const uint64_t outer_parallelism = 1uL << (log_outer_parallelism + 1);
    const unsigned int num_bytes = (log_outer_parallelism >> 3) + 1;
    if (num_bytes == 1)
    {
      for(uint64_t k = 0; k < outer_parallelism; k += 2)
      {
        for(uint64_t i = 0; i < stride; i++)
        {
          poly[i]          ^= c_b->multiply(poly[i + stride], val);
          poly[i + stride] ^= poly[i];
        }
        val ^= c_b->template beta_to_gamma_byte<0> (k^(k + 2));
        poly += 2*stride;
      }
    }
    else
    {
      // k = (kr << 8)^kl
      for(uint64_t ku = 0; ku < outer_parallelism >> 8; ku++)
      {
        for(uint32_t kl = 0; kl < 0x100uL; kl += 2)
        {
          for(uint32_t i = 0; i < stride; i++)
          {
            poly[i]          ^= c_b->multiply(poly[i + stride], val);
            poly[i + stride] ^= poly[i];
          }
          val ^= c_b->template beta_to_gamma_byte<0> (kl^(kl + 2));
          poly += 2*stride;
        }
        val ^= c_b->beta_to_gamma((ku^(ku+1))<<8, num_bytes);
      }
    }
  }
  else
  {
    const unsigned int t = 1 << (s - 1);
    // const uint64_t eta = 1uL << (2 * t);
    // replaced the eta value in-line to avoid overflow when s = 6 (word = uint64_t).
    const uint64_t outer_block_size = 1uL << (2 * t + logstride);
    word* poly_loc = poly;
    for(uint64_t loop = 0; loop < (1uL << log_outer_parallelism); loop++)
    {
      // perform taylor decomposition in-line
      for(unsigned int interval_logsize = 2*t; interval_logsize > t; interval_logsize--)
      {
        const size_t n_s = 1uL << (interval_logsize + logstride);
        const uint64_t delta_s = 1uL << (interval_logsize - 1 - t + logstride);
        const uint64_t m_s = n_s >> 1;
        const size_t num_iter_a = delta_s;
        const size_t num_iter_b = m_s - delta_s;
        word *p1 = poly_loc + delta_s, *p2 = poly_loc + (n_s - delta_s), *p3 = poly_loc + m_s;
        //for(size_t i = 0; i < (eta >> interval_logsize); i ++)
        for(size_t i = 0; i < (1uLL << (2*t - interval_logsize)); i ++)
        {
          for(uint64_t p = 0; p < num_iter_a; p++) p1[p] ^= p2[p];
          for(uint64_t p = 0; p < num_iter_b; p++) p1[p] ^= p3[p];
          for(uint64_t p = 0; p < num_iter_a; p++) p3[p] ^= p2[p];
          p1 += n_s;
          p2 += n_s;
          p3 += n_s;
        }
      }
      uint64_t j_loop = j|loop;
      fft_aux_fast<word, s-1>(c_b, poly_loc, j_loop,        logstride + t, 0);
      fft_aux_fast<word, s-1>(c_b, poly_loc, (j_loop << t), logstride,     t);
      poly_loc += outer_block_size;
    }
  }
}

template <class word, int s>
void fft_aux_fast_truncated(
    cantor_basis<word>* c_b,
    word* poly,
    word j,
    unsigned int logstride,
    unsigned int log_outer_parallelism,
    unsigned int logsize)
{
  if constexpr(s == 0)
  {
    if(logsize == 0) return;
    uint64_t stride = 1uL << logstride;
    word val = c_b->beta_to_gamma(j << 1);
    const uint64_t outer_parallelism = 1uL << (log_outer_parallelism + 1);
    const unsigned int num_bytes = (log_outer_parallelism >> 3) + 1;
    if (num_bytes == 1)
    {
      for(uint64_t k = 0; k < outer_parallelism; k += 2)
      {
        for(uint64_t i = 0; i < stride; i++)
        {
          poly[i]          ^= c_b->multiply(poly[i + stride], val);
          poly[i + stride] ^= poly[i];
        }
        val ^= c_b->template beta_to_gamma_byte<0> (k^(k + 2));
        poly += 2*stride;
      }
    }
    else
    {
      // k = (kr << 8)^kl
      for(uint64_t ku = 0; ku < outer_parallelism >> 8; ku++)
      {
        for(uint32_t kl = 0; kl < 0x100uL; kl += 2)
        {
          for(uint32_t i = 0; i < stride; i++)
          {
            poly[i]          ^= c_b->multiply(poly[i + stride], val);
            poly[i + stride] ^= poly[i];
          }
          val ^= c_b->template beta_to_gamma_byte<0> (kl^(kl + 2));
          poly += 2*stride;
        }
        val ^= c_b->beta_to_gamma((ku^(ku+1))<<8, num_bytes);
      }
    }
  }
  else
  {
    const unsigned int t = 1 << (s - 1);
    if(logsize <= t)
    {
      fft_aux_fast_truncated<word, s-1>(c_b, poly, j << t, logstride, 0, logsize);
    }
    else
    {
      //const uint64_t eta   = 1uLL << logsize;
      // replaced the eta value in-line as in fft_aux_fast although no overflow is possible here
      // (assuming logsize < 64, which seems safe)
      const uint64_t outer_block_size = 1uL << (2 * t + logstride);
      for(uint64_t loop = 0; loop < (1uL << log_outer_parallelism); loop++)
      {
        // perform taylor decomposition in-line
        for(unsigned int interval_logsize = logsize; interval_logsize > t; interval_logsize--)
        {
          const size_t n_s = 1uL << (interval_logsize + logstride);
          const uint64_t delta_s = 1uL << (interval_logsize - 1 - t + logstride);
          const uint64_t m_s = n_s >> 1;
          const size_t num_iter_a = delta_s;
          const size_t num_iter_b = m_s - delta_s;
          word *p1 = poly + delta_s, *p2 = poly + (n_s - delta_s), *p3 = poly + m_s;
          for(size_t i = 0; i < (1uLL << (logsize - interval_logsize)); i ++)
          {
            for(uint64_t p = 0; p < num_iter_a; p++) p1[p] ^= p2[p];
            for(uint64_t p = 0; p < num_iter_b; p++) p1[p] ^= p3[p];
            for(uint64_t p = 0; p < num_iter_a; p++) p3[p] ^= p2[p];
            p1 += n_s;
            p2 += n_s;
            p3 += n_s;
          }
        }
        uint64_t j_loop = j|loop;
        fft_aux_fast_truncated<word, s-1>(c_b, poly, j_loop,        logstride + t, 0, logsize - t);
        fft_aux_fast_truncated<word, s-1>(c_b, poly, (j_loop << t), logstride,     logsize - t, t);
        poly += outer_block_size;
      }
    }
  }
}

template <class word, int s>
void fft_aux_fast_truncated_reverse(
    cantor_basis<word>* c_b,
    word* poly,
    word j,
    unsigned int logstride,
    unsigned int log_outer_parallelism,
    unsigned int logsize)
{
  if constexpr(s == 0)
  {
    if(logsize == 0) return;
    uint64_t stride = 1uL << logstride;
    word val = c_b->beta_to_gamma(j << 1);
    const uint64_t outer_parallelism = 1uL << (log_outer_parallelism + 1);
    const unsigned int num_bytes = (log_outer_parallelism >> 3) + 1;
    if (num_bytes == 1)
    {
      for(uint64_t k = 0; k < outer_parallelism; k += 2)
      {
        for(uint64_t i = 0; i < stride; i++)
        {
          poly[i + stride] ^= poly[i];
          poly[i]          ^= c_b->multiply(poly[i + stride], val);
        }
        val ^= c_b->template beta_to_gamma_byte<0> (k^(k + 2));
        poly += 2*stride;
      }
    }
    else
    {
      // k = (kr << 8)^kl
      for(uint64_t ku = 0; ku < outer_parallelism >> 8; ku++)
      {
        for(uint32_t kl = 0; kl < 0x100uL; kl += 2)
        {
          for(uint32_t i = 0; i < stride; i++)
          {
            poly[i + stride] ^= poly[i];
            poly[i]          ^= c_b->multiply(poly[i + stride], val);
          }
          val ^= c_b->template beta_to_gamma_byte<0> (kl^(kl + 2));
          poly += 2*stride;
        }
        val ^= c_b->beta_to_gamma((ku^(ku+1))<<8, num_bytes);
      }
    }
  }
  else
  {
    const unsigned int t = 1 << (s - 1);
    if(logsize <= t)
    {
      fft_aux_fast_truncated_reverse<word, s-1>(c_b, poly, j, logstride, log_outer_parallelism, logsize);
    }
    else
    {
      const uint64_t eta   = 1uL << logsize;
      const uint64_t outer_block_size = 1uL << (2 * t + logstride);
      for(uint64_t loop = 0; loop < (1uL << log_outer_parallelism); loop++)
      {
        uint64_t j_loop = j|loop;
        fft_aux_fast_truncated_reverse<word, s-1>(c_b, poly, (j_loop << t), logstride,     logsize - t, t);
        fft_aux_fast_truncated_reverse<word, s-1>(c_b, poly, j_loop,        logstride + t, 0, logsize - t);
        // perform reverse taylor decomposition in-line
        for(unsigned int interval_logsize = t + 1; interval_logsize <= logsize; interval_logsize++)
        {
          const size_t n_s = 1uL << (interval_logsize + logstride);
          const uint64_t delta_s = 1uL << (interval_logsize - 1 - t + logstride);
          const uint64_t m_s = n_s >> 1;

          word *p1 = poly ;
          for(size_t i = 0; i < (eta >> interval_logsize); i ++)
          {
            for(size_t p = m_s; p < n_s; p++) p1[p - m_s + delta_s] ^= p1[p];
            p1 += n_s;
          }
        }
        poly += outer_block_size;
      }
    }
  }
}

/* performs a full DFT of size 2**(2**s) = u of a polynomial with coefficients of type 'word'
 * expressed in the finite field implemented by cantor basis c_b.
 * s = 3: u=2**8
 * s = 4: u=2**16
 * s = 5: u=2**32
 * s = 6: u=2**64
 * input polynomial should have at most u coefficients, i.e. have degree at most u-1.
 * no optimization is performed for small degree polynomials (the degree is not given as
 * a parameter to the algorithm)
 * with s = 6, variant fft_aux_ref does not work (uint64_t eta = 1uL<<(1<<s) overflows)
 * s = 6 cannot be used realistically anyway (the input/output is 2**67 bytes!!!)
 * therefore s=6 is forbidden.
 * u = 2**(2**s) and the finite field size f can be controlled independently: the cases
 * u > s, u = s (the most natural) and u < s work provided the condition on the input
 * polynomial degree is met.
*/
template <class word, int s>
void fft_mateer(cantor_basis<word>* c_b, word* poly)
{
  static_assert(c_b_t<word>::n<=64);
  static_assert(s<=5);
#if 0
  fft_aux_ref<word>(c_b, poly, s, 0, 0);
#else
  fft_aux_fast<word, s>(c_b, poly, 0, 0, 0);
#endif
}

/* as fft_mateer, but computes only the first v=2**logsize values, with v <= u = 2**(2**s).
 * input polynomial should have at most v coefficients, i.e. have degree at most v-1.
 * s=6 can be used. beyond s=6, since all variables controlling loops are uint64_t, some
 * overflows may occur during the computation of loop indexes, even if these indexes are
 * small, and the computation may fail.
 * s can be set at the minimum value s.t v <= 2**(2**s). The computation time does not depend
 * much on s and it can therefore be set at min(6, log of word size) safely even if this
 * results in a value that is too large.
 */
template <class word, int s>
void fft_mateer_truncated(cantor_basis<word>* c_b, word* poly, unsigned int logsize)
{
  static_assert(s<=6);
  if(logsize > (1u << s)) logsize = 1 << s;
#if 1
  fft_aux_ref_truncated<word, s>(c_b, poly, 0, 0, logsize);
#else
  fft_aux_fast_truncated<word, s>(c_b, poly, 0, 0, 0, logsize);
#endif
}

template <class word, int s>
void fft_mateer_truncated_mult(cantor_basis<word>* c_b, word* poly, unsigned int logsize)
{
  static_assert(s<=6);
  if(logsize > (1u << s)) logsize = 1 << s;
  fft_aux_ref_truncated_mult<word, s>(c_b, poly, 0, 0, logsize);
}

/* Reverse function of fft_mateer_truncated.
 *
 */
template <class word, int s>
void fft_mateer_truncated_reverse(cantor_basis<word>* c_b, word* poly, unsigned int logsize)
{
  static_assert(s<=6);
  if(logsize > (1u << s)) logsize = 1 << s;
#if 1
  fft_aux_ref_truncated_reverse<word,s>(c_b, poly, 0, 0, logsize);
#else
  fft_aux_fast_truncated_reverse<word, s>(c_b, poly, 0, 0, 0, logsize);
#endif
}

template <class word, int s>
void fft_mateer_truncated_reverse_mult(cantor_basis<word>* c_b, word* poly, unsigned int logsize)
{
  static_assert(s<=6);
  if(logsize > (1u << s)) logsize = 1 << s;
  fft_aux_ref_truncated_reverse_mult<word,s>(c_b, poly, 0, 0, logsize);
}
