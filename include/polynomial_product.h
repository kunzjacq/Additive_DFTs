#pragma once

#include "additive_fft.h"
#include "mateer_gao.h"

/**
 * @brief binary_polynomial_multiply
 * multiplication of binary polynomials using a fft of logsize 'logsize' in binary field whose
 * elements are words. Does not work for word = uint8_t (in that case half-bytes would have to
 * be copied to / from fft buffers, and the current code only knows how to copy full bytes).
 * @param c_b
 * @param p1
 * buffer with 1st polynomial, of degree d1. Should be readable up to index i = d1 / 8
 * @param p2
 * @param result
 * @param b1
 * @param b2
 * @param d1
 * @param d2
 * @param logsize
 */

template <class word>
void binary_polynomial_to_words(cantor_basis<word>* c_b, uint8_t* p, word* res, size_t d, size_t fft_size)
{
  constexpr unsigned int bytes_per_word = c_b_t<word>::n >> 4;
  size_t j = 0, k = 0;
  for(size_t i = 0; i < d / 8 + 1; i++)
  {
    if(k == 0) res[j]  = static_cast<word>(p[i]);
    else       res[j] |= static_cast<word>(p[i]) << (8 * k);
    k++;
    if(k == bytes_per_word)
    {
      res[j] = c_b->mult_to_gamma(res[j]);
      j++;
      k = 0;
    }
  }
  for(; j < fft_size;j++) res[j] = 0;
}

template<>
void binary_polynomial_to_words(cantor_basis<uint8_t>* c_b, uint8_t* p, uint8_t* res, size_t d, size_t fft_size)
{
  size_t i;
  for(i = 0; i < d / 8 + 1; i++)
  {
    res[2*i]   = c_b->mult_to_gamma(p[i] & 0xF);
    res[2*i+1] = c_b->mult_to_gamma(p[i] >> 4);
  }
  i = 2*i;
  for(; i < fft_size; i++) res[i] = 0;
}

template <class word>
void words_to_binary_polynomial(cantor_basis<word>* c_b, word* buf, uint8_t* p, size_t d, size_t fft_size)
{
  constexpr unsigned int bytes_per_word = c_b_t<word>::n >> 4;
  constexpr unsigned int bits_per_word  = c_b_t<word>::n >> 1;
  constexpr word u = 1;
  constexpr word low_mask = (u << bits_per_word) - 1;
  size_t bound = d / bits_per_word + 1;
  for(size_t i = 0; i  < bound; i++)
  {
    buf[i] = c_b->gamma_to_mult(buf[i]);
  }
  word up = 0;
  for(size_t i = 0; i < bound; i++)
  {
    word w = buf[i];
    word low = (w & low_mask) ^ up;
    up  = w >> bits_per_word;
    for(unsigned int j = 0; j < bytes_per_word; j++)
    {
      p[i    * bytes_per_word + j] = (low >> (8*j)) & 0xFF;
    }
  }

  for(unsigned int j = 0; j < d / 8 + 1 - bound * bytes_per_word; j++)
  {
    p[ bound * bytes_per_word + j] = (up  >> (8*j)) & 0xFF;
  }
  for(size_t i = d / 8 + 1; i < fft_size; i++) p[i] = 0;
}

template <>
void words_to_binary_polynomial(cantor_basis<uint8_t>* c_b, uint8_t* buf, uint8_t* p, size_t d, size_t fft_size)
{
  constexpr unsigned int bits_per_word  = 4;
  constexpr uint8_t low_mask = (1 << bits_per_word) - 1;
  size_t bound = d / bits_per_word + 1;
  for(size_t i = 0; i  < bound; i++)
  {
    buf[i] = c_b->gamma_to_mult(buf[i]);
  }
  uint8_t up = 0;
  for(size_t i = 0; i < bound; i++)
  {
    uint8_t w = buf[i];
    uint8_t low = (w & low_mask) ^ up;
    up = w >> bits_per_word;
    if((i&1) == 0)
    {
      p[i >> 1] = low;
    }
    else
    {
      p[i >> 1] |= (low << 4);
    }
  }
  if((bound & 1) == 0)
  {
    p[bound >> 1] = up;
  }
  else
  {
    p[bound >> 1] |= (up << 4);
  }
  for(size_t i = (bound >> 1) + 1; i < fft_size; i++) p[i] = 0;
}

template <class word>
void binary_polynomial_multiply(
    cantor_basis<word>* c_b,
    uint8_t* p1, uint8_t* p2, uint8_t* result, word* b1, word* b2,
    size_t d1, size_t d2, unsigned int logsize)
{
  constexpr unsigned int s = c_b_t<word>::word_logsize;
  // max number of words to store result:
  // = (d1 + d2 + 1 + bits_per_word - 1) / bits_per_word = (d1 + d2) / bits_per_word + 1;
#ifndef NDEBUG
  constexpr unsigned int bits_per_word  = c_b_t<word>::n >> 1;
  const size_t bound = (d1+d2) / bits_per_word + 1;
#endif
  const size_t sz = (1uL << logsize);
  assert(sz >= bound);
  assert(logsize <= c_b_t<word>::n);

  binary_polynomial_to_words(c_b, p1, b1, d1, sz);
  binary_polynomial_to_words(c_b, p2, b2, d2, sz);

  // instead of using mateer-gao directly, we could have used
  //additive_fft<word> fft(c_b);
  //fft.vzgg_mateer_gao_combination(b1, d1/bits_per_word + 1 , logsize);
  //fft.vzgg_mateer_gao_combination(b2, d2/bits_per_word + 1 , logsize);
  // but this does not help since in this case, the degree of the polynomials transformed
  // is less that 2**logsize.
  fft_mateer_truncated<word,s>(c_b, b1, logsize);
  fft_mateer_truncated<word,s>(c_b, b2, logsize);
  for(size_t i = 0; i < sz; i++) b1[i] = c_b->multiply(b1[i], b2[i]);
  fft_mateer_truncated_reverse<word,s>(c_b, b1, logsize);

  words_to_binary_polynomial(c_b, b1, result, d1 + d2, sz);
}
