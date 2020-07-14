#pragma once

#include "additive_fft.h"
#include "mateer_gao.h"

template <class word>
uint64_t binary_polynomial_to_words_mult(uint8_t* p, word* res, size_t d, size_t fft_size)
{
  constexpr unsigned int bytes_per_word = c_b_t<word>::n >> 4;
  constexpr int num_bits_half = sizeof(typename c_b_t<word>::half_type)*8;
  uint64_t dw = d / num_bits_half + 1;
  size_t j = 0, k = 0;
  for(size_t i = 0; i < d / 8 + 1; i++)
  {
    if(k == 0) res[j]  = static_cast<word>(p[i]);
    else       res[j] |= static_cast<word>(p[i]) << (8 * k);
    k++;
    if(k == bytes_per_word)
    {
      j++;
      k = 0;
    }
  }
  for(; j < fft_size;j++) res[j] = 0;
  return dw;
}

template <class word>
uint64_t binary_polynomial_to_words_mult_little_endian(uint8_t* p, word* res, uint64_t d, size_t fft_size)
{
  auto source = reinterpret_cast<typename c_b_t<word>::half_type*>(p);
  constexpr int num_bits_half = sizeof(typename c_b_t<word>::half_type)*8;
  uint64_t dw = d / num_bits_half + 1;
  uint64_t j = 0;
  // little endian
  for(j = 0; j < dw; j++) res[j] = source[j];
  for(; j < fft_size;j++) res[j] = 0;
  return dw;
}

template<>
uint64_t binary_polynomial_to_words_mult(uint8_t* p, uint8_t* res, uint64_t d, size_t fft_size)
{
  constexpr int num_bits_half = 4;
  uint64_t dw = d / num_bits_half + 1;
  size_t i;
  for(i = 0; i < d / 8 + 1; i++)
  {
    res[2*i]   = p[i] & 0xF;
    res[2*i+1] = p[i] >> 4;
  }
  i = 2*i;
  for(; i < fft_size; i++) res[i] = 0;
  return dw;
}

template<>
uint64_t binary_polynomial_to_words_mult_little_endian(uint8_t* p, uint8_t* res, uint64_t d, size_t fft_size)
{
  return binary_polynomial_to_words_mult<uint8_t>(p, res, d, fft_size);
}

template <class word>
void words_to_binary_polynomial_mult_little_endian(word* buf, uint8_t* p, uint64_t d)
{
  constexpr unsigned int bits_per_half_word  = c_b_t<word>::n >> 1;
  const size_t bound = d / bits_per_half_word + 1;
  word* dest_even = reinterpret_cast<word*>(p);
  word* dest_odd  = reinterpret_cast<word*>(p + (bits_per_half_word >> 3));
  for(uint64_t i = 0; i < d/8+1; i++) p[i] = 0;
  for(uint64_t i = 0; i < bound; i++)
  {
    if((i&1)==0)
    {
      dest_even[i >> 1] ^= buf[i];
    }
    else
    {
      dest_odd[i >> 1]  ^= buf[i];
    }
  }
}

template <class word>
void words_to_binary_polynomial_mult(word* buf, uint8_t* p, uint64_t d)
{
  constexpr unsigned int bytes_per_word = c_b_t<word>::n >> 4;
  constexpr unsigned int bits_per_word  = c_b_t<word>::n >> 1;
  constexpr word u = 1;
  constexpr word low_mask = (u << bits_per_word) - 1;
  size_t bound = d / bits_per_word + 1;
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
}


template <>
void words_to_binary_polynomial_mult(uint8_t* buf, uint8_t* p, uint64_t d)
{
  constexpr unsigned int bits_per_word  = 4;
  constexpr uint8_t low_mask = (1 << bits_per_word) - 1;
  size_t bound = d / bits_per_word + 1;
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
}

template<>
void words_to_binary_polynomial_mult_little_endian(uint8_t* buf, uint8_t* p, uint64_t d)
{
  words_to_binary_polynomial_mult<uint8_t>(buf, p, d);
}

/**
 * @brief binary_polynomial_multiply
 * multiplication of binary polynomials using a fft of logsize 'logsize' in binary field whose
 * elements are words.
 * @param c_b cantor basis<word> needed by FFTs
 * @param
 * buffer with 1st polynomial, of degree d1. Buffer should be readable up to index i = d1 / 8.
 * @param p2
 * same for second polynomial, of degree d2. Buffer should be readable up to index i = d2 / 8.
 * @param result
 * Buffer for result, of size at least (d1+d2) / 8 + 1.
 * @param b1
 * buffer for DFT of 1st polynomial, of size 2**logsize.
 * @param b2
 * buffer for DFT of 2nd polynomial, of size 2**logsize.
 * @param d1
 * degree of 1st polynomial.
 * @param d2
 * degree of 2nd polynomial.
 * @param logsize
 * The log2 size of the DFTs performed. 2**logsize should be >= (d1 + d2 + 1) / num_coeffs_per_word
 * for the result to be correct, where num_coeffs_per_word is half the number of bits in a word.
 */

template <class word>
void binary_polynomial_multiply(
    cantor_basis<word>* c_b,
    uint8_t* p1, uint8_t* p2, uint8_t* result, word* b1, word* b2,
    size_t d1, size_t d2, unsigned int logsize)
{
  constexpr unsigned int s = c_b_t<word>::word_logsize;
  const size_t sz = (1uLL << logsize);
  uint64_t w1 = binary_polynomial_to_words_mult_little_endian(p1, b1, d1, sz);
  uint64_t w2 = binary_polynomial_to_words_mult_little_endian(p2, b2, d2, sz);
  int logsizeprime = logsize;
  while(1uLL << logsizeprime >= 2 * w1) logsizeprime--;
  fft_mateer_truncated_mult_smalldegree<word>(c_b, b1, logsizeprime, logsize);
  logsizeprime = logsize;
  while(1uLL << logsizeprime >= 2 * w2) logsizeprime--;
  fft_mateer_truncated_mult_smalldegree<word>(c_b, b2, logsizeprime, logsize);
  for(size_t i = 0; i < sz; i++) b1[i] = c_b->multiply_mult_repr(b1[i], b2[i]);
  fft_mateer_truncated_reverse_mult<word,s>(c_b, b1, logsize);
  words_to_binary_polynomial_mult_little_endian(b1, result, d1 + d2);
}
