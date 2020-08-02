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
    const uint64_t stride = 1uLL << logstride;
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
    const uint64_t tau   = 1uLL <<  t;
    const uint64_t eta   = 1uLL << (2 * t);
    decompose_taylor_recursive(logstride, 2 * t, t, eta, poly);
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
    const uint64_t row_size = 1uLL << (t + logstride);
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
    const uint64_t stride = 1uLL << logstride;
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
      const uint64_t tau   = 1uLL <<  (logsize - t);
      const uint64_t eta   = 1uLL << logsize;
      // on input: there are 2**'logstride' interleaved series of size 'eta' = 2**(2*logsize-t);
      decompose_taylor_iterative_alt(logstride, 2 * t, t, eta, poly);
      // fft on columns
      // if logsize >= t, each fft should process 2**(logsize - t) values
      // i.e. logsize' = logsize - t
      fft_aux_ref_truncated<word, s - 1>(c_b, poly, j, logstride + t, logsize - t);
      const uint64_t row_size = 1uLL << (t + logstride);
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
void fft_aux_ref_truncated_offset_alt(
    cantor_basis<word>* c_b,
    word* poly,
    word j, // block index to process ('outer' offset); 1 = offset of 2^(2^s)
    unsigned int logstride,
    unsigned int logsize,
    uint64_t offset) // 'inner' offset, < 2^(2^s)
// log2 size of each stride to be processed;
// should be <= 2**s, the input is assumed to be of size 2**logsize and only the 2**logsize first output values are computed
{
  if constexpr(s == 0)
  {
    const uint64_t stride = 1uLL << logstride;
    const word val = c_b->beta_to_gamma(offset^(j << 1));
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
    uint64_t mask = (1uLL << t) - 1;
    if(logsize <= t)
    {
      fft_aux_ref_truncated_offset_alt<word, s - 1>(c_b, poly, (j << t) + (offset >> t), logstride, logsize, offset & mask);
    }
    else
    {
      const uint64_t tau   = 1uLL <<  (logsize - t);
      const uint64_t eta   = 1uLL << logsize;
      // on input: there are 2**'logstride' interleaved series of size 'eta' = 2**(2*logsize-t);
      decompose_taylor_iterative_alt(logstride, 2 * t, t, eta, poly);
      // fft on columns
      // if logsize >= t, each fft should process 2**(logsize - t) values
      // i.e. logsize' = logsize - t
      fft_aux_ref_truncated_offset_alt<word, s - 1>(c_b, poly, j, logstride + t, logsize - t, offset>>t);
      const uint64_t row_size = 1uLL << (t + logstride);
      for(uint64_t i = 0; i < tau; i++)
      {
        word j_loop = (j << t) + i + (offset >> t);
        fft_aux_ref_truncated_offset_alt<word, s - 1>(c_b, poly, j_loop, logstride, t, offset & mask);
        poly += row_size;
      }
    }
  }
}

template <class word, int s>
void fft_aux_ref_truncated_offset(
    cantor_basis<word>* c_b,
    word* poly,
    word offset, // offset; should be less than 2**n - 2**logsize, n=2**s, so as to avoid a wrap of the indexes.
 // should be a multiple of 2**logsize.
    unsigned int logstride,
    unsigned int logsize)
// log2 size of each stride to be processed;
// should be <= 2**s, the input is assumed to be of size 2**logsize and only the 2**logsize first output values are computed
{
  if constexpr(s == 0)
  {
    const uint64_t stride = 1uLL << logstride;
    const word val = c_b->beta_to_gamma(offset);
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
      fft_aux_ref_truncated_offset<word, s - 1>(c_b, poly, offset, logstride, logsize);
    }
    else
    {
      const uint64_t tau   = 1uLL <<  (logsize - t);
      const uint64_t eta   = 1uLL << logsize;
      // on input: there are 2**'logstride' interleaved series of size 'eta' = 2**(2*logsize-t);
      decompose_taylor_iterative_alt(logstride, 2 * t, t, eta, poly);
      // fft on columns
      // if logsize >= t, each fft should process 2**(logsize - t) values
      // i.e. logsize' = logsize - t
      fft_aux_ref_truncated_offset<word, s - 1>(c_b, poly, offset >> t, logstride + t, logsize - t);
      const uint64_t row_size = 1uLL << (t + logstride);
      for(uint64_t i = 0; i < tau; i++)
      {
        fft_aux_ref_truncated_offset<word, s - 1>(c_b, poly, offset + (i << t), logstride, t);
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
    const uint64_t stride = 1uLL << logstride;
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
      const uint64_t tau   = 1uLL <<  (logsize - t);
      const uint64_t eta   = 1uLL << logsize;
      const uint64_t row_size = 1uLL << (t + logstride);
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
void fft_aux_ref_truncated_reverse_offset(
    cantor_basis<word>* c_b,
    word* poly,
    word offset, // block index to process
    unsigned int logstride,
    unsigned int logsize)
// log2 size of each stride to be processed;
// should be <= 2**s, the input is assumed to be of size 2**logsize and only the 2**logsize first output values are computed
{
  if constexpr(s == 0)
  {
    const uint64_t stride = 1uLL << logstride;
    const word val = c_b->beta_to_gamma(offset);
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
      fft_aux_ref_truncated_reverse_offset<word, s - 1>(c_b, poly, offset, logstride, logsize);
    }
    else
    {
      const uint64_t tau   = 1uLL <<  (logsize - t);
      const uint64_t eta   = 1uLL << logsize;
      const uint64_t row_size = 1uLL << (t + logstride);
      word* poly_loc = poly;
      for(uint64_t i = 0; i < tau; i++)
      {
        fft_aux_ref_truncated_reverse_offset<word, s - 1>(c_b, poly_loc, offset + (i << t), logstride, t);
        poly_loc += row_size;
      }
      // reverse fft on columns
      fft_aux_ref_truncated_reverse_offset<word, s - 1>(c_b, poly, offset >> t, logstride + t, logsize - t);
      decompose_taylor_reverse_iterative_alt(logstride, 2 * t, t, eta, poly);
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
    const uint64_t stride = 1uLL << logstride;
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
      const uint64_t tau   = 1uLL <<  (logsize - t);
      const uint64_t eta   = 1uLL << logsize;
      // on input: there are 2**'logstride' interleaved series of size 'eta' = 2**(2*logsize-t);
      decompose_taylor_iterative_alt(logstride, 2 * t, t, eta, poly);
      // fft on columns
      // if logsize >= t, each fft should process 2**(logsize - t) values
      // i.e. logsize' = logsize - t
      fft_aux_ref_truncated_mult<word, s - 1>(c_b, poly, j, logstride + t, logsize - t);
      const uint64_t row_size = 1uLL << (t + logstride);
      for(uint64_t i = 0; i < tau; i++)
      {
        word j_loop = (j << t) | i;
        fft_aux_ref_truncated_mult<word, s - 1>(c_b, poly, j_loop, logstride, t);
        poly += row_size;
      }
    }
  }
}

/**
 * computes the 2**logsize first values of 2**logstride interleaved polynomials on input
 * with <= 2**logsize coefficients
 * i.e. performes 2**logstride interleaved partial DFTS
 * => acts on an array of size 2**(logstride+logsize)
 */
template <class word, int s>
inline void fft_aux_ref_truncated_mult_offset(
    cantor_basis<word>* c_b,
    word* poly,
    word offset,
    unsigned int logstride,
    unsigned int logsize,
    bool first_taylor_done = false)
// log2 size of each stride to be processed;
// should be <= 2**s, the input is assumed to be of size 2**logsize and only the 2**logsize first output values are computed
{
  if constexpr(s == 0)
  {
    const uint64_t stride = 1uLL << logstride;
    const word val = c_b->beta_to_mult(offset);
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
    constexpr unsigned int t = 1 << (s - 1);
    if(logsize <= t)
    {
      fft_aux_ref_truncated_mult_offset<word, s - 1>(c_b, poly, offset, logstride, logsize, first_taylor_done);
    }
    else
    {
      const uint64_t tau   = 1uLL <<  (logsize - t);
      const uint64_t eta   = 1uLL << logsize;
      // on input: there are 2**'logstride' interleaved series of size 'eta' = 2**(2*logsize-t);
      if(!first_taylor_done) decompose_taylor_iterative_alt(logstride, 2 * t, t, eta, poly);
      // fft on columns
      // if logsize >= t, each fft should process 2**(logsize - t) values
      // i.e. logsize' = logsize - t
      fft_aux_ref_truncated_mult_offset<word, s - 1>(c_b, poly, offset >> t, logstride + t, logsize - t, false);
      const uint64_t row_size = 1uLL << (t + logstride);
      for(uint64_t i = 0; i < tau; i++)
      {
        fft_aux_ref_truncated_mult_offset<word, s - 1>(c_b, poly, offset + (i << t), logstride, t, false);
        poly += row_size;
      }
    }
  }
}

template <class word>
void fft_mateer_truncated_mult_smalldegree(
    cantor_basis<word>* c_b,
    word* poly, // of degree < 2**logsize_prime
    unsigned int logsizeprime, // <= logsize
    unsigned int logsize
    )
{
  constexpr unsigned int s = c_b_t<word>::word_logsize;
  unsigned int sprime = s;
  while(sprime > 0 && logsizeprime <= (1u << (sprime-1))) sprime--;
  if(sprime == 0) return; // FIXME: not handled
  const unsigned int t = 1 << (sprime - 1);
  const uint64_t eta   = 1uLL << logsizeprime;
  decompose_taylor_iterative_alt(0, 2 * t, t, eta, poly);
  for(uint64_t i = 1; i < 1uLL << (logsize - logsizeprime); i++)
  {
    for(uint64_t j = 0; j < 1uLL << logsizeprime; j++)
    {
      poly[j + (i << logsizeprime)] = poly[j];
    }
  }

  for(uint64_t i = 0; i < 1uLL << (logsize - logsizeprime); i++)
  {
    fft_aux_ref_truncated_mult_offset<word, s>(c_b, poly + (i << logsizeprime), i << logsizeprime, 0, logsizeprime, true);
  }
}


template <class word, int s>
inline void fft_aux_ref_truncated_reverse_mult(
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
    const uint64_t stride = 1uLL << logstride;
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
      const uint64_t tau   = 1uLL <<  (logsize - t);
      const uint64_t eta   = 1uLL << logsize;
      const uint64_t row_size = 1uLL << (t + logstride);
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
void fft_aux_ref_truncated_reverse_mult_offset(
    cantor_basis<word>* c_b,
    word* poly,
    word offset,
    unsigned int logstride,
    unsigned int logsize)
// log2 size of each stride to be processed;
// should be <= 2**s, the input is assumed to be of size 2**logsize and only the 2**logsize first output values are computed
{
  if constexpr(s == 0)
  {
    const uint64_t stride = 1uLL << logstride;
    const word val = c_b->beta_to_mult(offset);
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
      fft_aux_ref_truncated_reverse_mult_offset<word, s - 1>(c_b, poly, offset, logstride, logsize);
    }
    else
    {
      const uint64_t tau   = 1uLL <<  (logsize - t);
      const uint64_t eta   = 1uLL << logsize;
      const uint64_t row_size = 1uLL << (t + logstride);
      word* poly_loc = poly;
      for(uint64_t i = 0; i < tau; i++)
      {
        fft_aux_ref_truncated_reverse_mult_offset<word, s - 1>(c_b, poly_loc, offset + (i << t), logstride, t);
        poly_loc += row_size;
      }
      // reverse fft on columns
      fft_aux_ref_truncated_reverse_mult_offset<word, s - 1>(c_b, poly, offset >> t, logstride + t, logsize - t);
      decompose_taylor_reverse_iterative_alt(logstride, 2 * t, t, eta, poly);
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
  fft_aux_ref<word>(c_b, poly, s, 0, 0);
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
void fft_mateer_truncated(cantor_basis<word>* c_b, word* poly, unsigned int logsize, word offset = 0)
{
  static_assert(s<=6);
  if(logsize > (1u << s)) logsize = 1 << s;
  //fft_aux_ref_truncated<word, s>(c_b, poly, 0, 0, logsize);
  fft_aux_ref_truncated_offset<word, s>(c_b, poly, offset << logsize, 0, logsize);
  //fft_aux_ref_truncated_offset_alt<word, s>(c_b, poly, 0, 0, logsize, offset << logsize);
}

template <class word, int s>
void fft_mateer_truncated_mult(cantor_basis<word>* c_b, word* poly, unsigned int logsize, word offset = 0)
{
  static_assert(s<=6);
  if(logsize > (1u << s)) logsize = 1 << s;
  //fft_aux_ref_truncated_mult<word, s>(c_b, poly, 0, 0, logsize);
  fft_aux_ref_truncated_mult_offset<word, s>(c_b, poly, offset << logsize, 0, logsize);
}

/* Reverse functions.
 *
 */
template <class word, int s>
void fft_mateer_truncated_reverse(cantor_basis<word>* c_b, word* poly, unsigned int logsize, word offset = 0)
{
  static_assert(s<=6);
  if(logsize > (1u << s)) logsize = 1 << s;
  //fft_aux_ref_truncated_reverse<word,s>(c_b, poly, 0, 0, logsize);
  fft_aux_ref_truncated_reverse_offset<word,s>(c_b, poly, offset << logsize, 0, logsize);
}

template <class word, int s>
void fft_mateer_truncated_reverse_mult(cantor_basis<word>* c_b, word* poly, unsigned int logsize, word offset = 0)
{
  static_assert(s<=6);
  if(logsize > (1u << s)) logsize = 1 << s;
  //fft_aux_ref_truncated_reverse_mult<word, s>(c_b, poly, 0, 0, logsize);
  fft_aux_ref_truncated_reverse_mult_offset<word, s>(c_b, poly, offset << logsize, 0, logsize);
}
