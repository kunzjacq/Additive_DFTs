#include "mg.h"

#include <memory>    // for unique_ptr
#include <algorithm> // for min, max

#include <immintrin.h>

// uncomment to use the version of Mateer-Gao DFT where the log block size is a template parameter,
// (see mg_t.h). Surprisingly, the templatized version is slower than the regular implementation.
// #define USE_TEMPLATIZED

#ifdef USE_TEMPLATIZED
#include "mg_t.h"
#endif

using namespace std;

/**
 * both 'contract' and 'expand' can be used with dest = source,
 * or with disjoint dest and source arrays. Their implementation is little-endian only.
 **/

/**
 * @brief expand
 * Space each 32-bit word of 'source' with a null 32-bit word into 'dest'.
 * Pad the result with zeros up to 'fft_size' 64-bit words.
 * 'source' is a binary polyomial of degree 'd', that uses with 'd' / 32 + 1 32-bit words.
 * @param dest
 * destination 64-bit array
 * @param source
 * source 64-bit array
 * @param d
 * degree of the input polynomial
 * @param fft_size
 * padding size
 */
static void expand(uint64_t* dest, uint64_t* source, uint64_t d, uint64_t fft_size)
{
  auto source32 = reinterpret_cast<uint32_t*>(source);
  uint64_t num_dwords = d / 32 + 1; // d + 1 coefficients, rounded above to a multiple of 32
  uint64_t j = 0;
  // little endian
  for(j = 0; j < num_dwords; j++) dest[num_dwords - 1 - j] = source32[num_dwords - 1 - j];
  for(; j < fft_size;j++) dest[j] = 0;
}

/**
 * @brief contract
 * Contract a polynomial over GF(2**64), xoring each upper half of each 64-bit with the lower
 * half of the next 64-bit word.
 * @param dest
 * 64-bit destination array
 * @param source
 * 64-bit source array
 * @param d
 * input polynomial degree
 */
static void contract(uint64_t* dest, uint64_t* source, uint64_t d)
{
  const size_t num_dwords = d / 32 + 1;
  uint32_t* dest32   = reinterpret_cast<uint32_t*>(dest);
  uint32_t* source32 = reinterpret_cast<uint32_t*>(source);
  dest32[0] = source32[0];
  for(uint64_t i = 1; i < num_dwords; i++)
  {
    dest32[i] = source32[2*i - 1] ^ source32[2*i];
  }
}

/**
 * @brief product
 * × in GF(2**64), implemented with element u of minimal polynomial x**64 - x**4 + x**3 + x + 1.
 * 1 is represented by 0x1, u by 0x2.
 * @param a
 * @param b
 * @return a × b
 */
static inline uint64_t product(const uint64_t&a, const uint64_t&b)
{
  constexpr uint64_t minpoly = 0x1b; // x**64 = x**4 + x**3 + x + 1
  // x**64 + x**4 + x**3 + x + 1 is primitive over GF(2)
  // it is the minimal polynomial of the multiplicative generator
  __m128i xa, xb, xc;
  xa = _mm_set_epi64x(minpoly, a);
  xb = _mm_set1_epi64x(b);
  xb = _mm_clmulepi64_si128(xa, xb, 0x00);
  xc = _mm_clmulepi64_si128(xa, xb, 0x11);
  xb = _mm_xor_si128(xb, xc);
  xc = _mm_clmulepi64_si128(xa, xc, 0x11);
  xb = _mm_xor_si128(xb, xc);
  return xb[0];
}

/**
 * @brief product_batch
 * performs 2**logsize products in GF(2**64). a[i] <- a[i] × b[i].
 * assumes logsize >=2 (since products are performed in groups of 4)
 * @param a_ptr
 * source and destination array
 * @param b_ptr
 * source array
 * @param logsize
 * log2 of number of products to perform
 */
static void product_batch(uint64_t* a_ptr, uint64_t* b_ptr, unsigned int logsize)
{
  const uint64_t sz = 1uLL << logsize;
  assert(logsize >= 2);

  constexpr uint64_t minpoly = 0x1b;
  __m128i xp, xa1, xa2, xa3, xa4, xb1, xb2, xb3, xb4, xc1, xc2, xc3, xc4;
  xp[0] = minpoly;
  for(uint64_t i = 0; i < sz; i += 4)
  {
    xa1 = _mm_set1_epi64x(a_ptr[i]);
    xa2 = _mm_set1_epi64x(a_ptr[i + 1]);
    xa3 = _mm_set1_epi64x(a_ptr[i + 2]);
    xa4 = _mm_set1_epi64x(a_ptr[i + 3]);

    xb1 = _mm_set1_epi64x(b_ptr[i]);
    xb2 = _mm_set1_epi64x(b_ptr[i + 1]);
    xb3 = _mm_set1_epi64x(b_ptr[i + 2]);
    xb4 = _mm_set1_epi64x(b_ptr[i + 3]);
    xb1 = _mm_clmulepi64_si128(xa1, xb1, 0x00);
    xb2 = _mm_clmulepi64_si128(xa2, xb2, 0x00);
    xb3 = _mm_clmulepi64_si128(xa3, xb3, 0x00);
    xb4 = _mm_clmulepi64_si128(xa4, xb4, 0x00);
    xc1 = _mm_clmulepi64_si128(xp, xb1, 0x10);
    xc2 = _mm_clmulepi64_si128(xp, xb2, 0x10);
    xc3 = _mm_clmulepi64_si128(xp, xb3, 0x10);
    xc4 = _mm_clmulepi64_si128(xp, xb4, 0x10);
    xb1 = _mm_xor_si128(xb1, xc1);
    xb2 = _mm_xor_si128(xb2, xc2);
    xb3 = _mm_xor_si128(xb3, xc3);
    xb4 = _mm_xor_si128(xb4, xc4);
    xc1 = _mm_clmulepi64_si128(xp, xc1, 0x10);
    xc2 = _mm_clmulepi64_si128(xp, xc2, 0x10);
    xc3 = _mm_clmulepi64_si128(xp, xc3, 0x10);
    xc4 = _mm_clmulepi64_si128(xp, xc4, 0x10);
    xb1 = _mm_xor_si128(xb1, xc1);
    xb2 = _mm_xor_si128(xb2, xc2);
    xb3 = _mm_xor_si128(xb3, xc3);
    xb4 = _mm_xor_si128(xb4, xc4);
    a_ptr[i]     = xb1[0];
    a_ptr[i + 1] = xb2[0];
    a_ptr[i + 2] = xb3[0];
    a_ptr[i + 3] = xb4[0];
  }
}

/**
  @brief mg_core
 * computes the 2**logsize first values of 2**logstride interleaved polynomials given on input,
 * each with <= 2**logsize coefficients; i.e. performs 2**logstride interleaved partial DFTS.
 * processing is done in-place on an array of size 2**(logstride + logsize).
 * Output values computed correspond to input values whose beta representation is
 * offset ^ i, i=0 ... 2**logsize - 1, and offset is a multiple of 2**logsize.
 * the multiplicative representation of 'offset' is in offsets_mult[0] (see below).
 * @param s
 * a recursion parameter s.t. logsize <= 2**s.
 * @param logstride
 * the log2 number of interleaved polynomials on input.
 * @param mult_pow_table
 * mult_pow_table[i], i < 2**s, contains the value 2**(i+1) - 1 in beta representation.
 * (i.e. i consecutive bits set to 1 starting from 0) converted to multiplicative representation.
 * @param poly
 * the interleaved polynomials to process, in multiplivative representation,
 * with 2**logsize coefficients each.
 * @param offsets_mult
 * offsets_mult[0] is the offset 'offset' of the values computed in beta representation,
 * converted to multiplicative representation.
 * offsets_mult[j], j < 2**s, is (offset >> j) converted to multplicative representation.
 * @param logsize
 * the log2 size of the input polynomials, equal to the size of interval of values
 * computed for each polynomial.
 * @param first_taylor done
 * if true, the first taylor expansion to perform is skipped. enables an optimization when
 * the polynomial to process has small degree; see mg_smalldegree.
 */
template <int s, int logstride>
inline void mg_core(
    uint64_t* poly,
    const uint64_t* offsets_mult,
    unsigned int logsize,
    bool first_taylor_done)
{
  if constexpr(s == 0)
  {
    eval_degree1<false, logstride>(offsets_mult[0], poly);
  }
  else
  {
    constexpr unsigned int t = 1 << (s - 1);
    if(logsize <= t)
    {
      mg_core<s - 1, logstride>(poly, offsets_mult, logsize, first_taylor_done);
    }
    else
    {
      // on input: there are 2**'logstride' interleaved series of size 'eta' = 2**(2*logsize-t);
      if(!first_taylor_done)
      {
        mg_decompose_taylor_recursive<logstride,t, uint64_t>(logsize, poly);
      }
      // fft on columns
      // if logsize >= t, each fft should process 2**(logsize - t) values
      // i.e. logsize' = logsize - t
      // offset' = offset >> t;
      mg_core<s - 1, logstride + t>(poly, offsets_mult + t, logsize - t, false);
      uint64_t offsets_local[t];
      for(unsigned int j = 0; j < t; j++) offsets_local[j] = offsets_mult[j];
      // 2*t >= logsize > t, therefore tau < 2**t
      const uint64_t tau   = 1uLL <<  (logsize - t);
      for(uint64_t i = 0; i < tau; i++)
      {
        // at each iteration, offsets_local[j] = beta_to_mult(offset + (i << (t-j))), j < t
        mg_core<s - 1, logstride>(poly, offsets_local, t, false);
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

/**
 * @brief mg_smalldegree
 * computes one truncated DFTs of an input polynomial of degree < 2**logsizeprime.
 * 2**logsize output values are computed, with logsizeprime <= logsize.
 * If logsizeprime < logsize, this function is more efficient than a direct call to mg_core.
 * the result is nevertheless the same as
 * mg_core<s, 0>(p, offsets_mult, logsize, false);
 * with offsets_mults[] all set to 0.
 * @param s
 * a recursion parameter s.t. logsize <= 2**s.
 * @param mult_pow_table
 * see the same argument for mg_core.
 * @param p
 * the input and output array, of size 2**logsize.
 * @param logsizeprime
 * a log2 bound on the number of coefficients of the input polynomial.
 * @param logsize
 * the log2 number of output values computed (with offset 0).
 */


template <unsigned int s>
void mg_smalldegree(
    uint64_t* p, // of degree < 2**logsize_prime
    unsigned int logsizeprime, // log2 bound on the polynomial number of coefficents, <= logsize
    unsigned int logsize // size of the DFT to be computed, <= 2**(2**s)
    )
{
  if constexpr(s > 0)
  {
    if(logsizeprime <= (1u << (s-1)))
    {
      mg_smalldegree<s-1>(p, logsizeprime, logsize);
      return;
    }
    constexpr unsigned int t = 1u << (s - 1);
    mg_decompose_taylor_recursive<0u, t, uint64_t>(logsizeprime, p);
    for(uint64_t i = 1; i < 1uLL << (logsize - logsizeprime); i++)
    {
      uint64_t* q = p + (i << logsizeprime);
      for(uint64_t j = 0; j < 1uLL << logsizeprime; j++) q[j] = p[j];
    }

    uint64_t offsets_mult[1 << s];
    for(int i = 0; i < (1 << s); i++) offsets_mult[i] = 0;

    for(uint64_t i = 0; i < 1uLL << (logsize - logsizeprime); i++)
    {
      uint64_t* q = p + (i << logsizeprime);
#ifdef USE_TEMPLATIZED
      mgt_core<s, 0, true>(p + (i << logsizeprime), offsets_mult, logsizeprime);
#else
      mg_core<s, 0>(q, offsets_mult, logsizeprime, true);
#endif
      const long unsigned int h = _mm_popcnt_u64(i^(i+1));
      // goal: xor to offsets_mult[j] to the multiplicative representation w of the value
      // v = (i^(i+1) << logsizeprime) >> j in beta representation,
      // 0 <= j < 2**s.
      // i^(i+1) = 2**h - 1.
      // beta_over_mult_cumulative[u] contains the multiplicative representation of
      // the value 2**u - 1 in beta representation, u <= 64.
      // (of hamming weight u)
      // case 1: if j >= h + logsizeprime, (i^(i+1) << logsizeprime) >> j = 0,
      // therefore w = 0.
      // case 2: if j < h + logsizeprime but j >= logsizeprime, v = 2**(h-(j-logsizeprime)) - 1,
      // therefore w = beta_over_mult_cumulative[h + logsizeprime - j].
      // case 3: if j < logsizeprime,
      // v = (2**h - 1) << (logsizeprime - j)
      //   = (2**(h+ logsizeprime - j) - 1) ^ (2**(logsizeprime - j) - 1),
      // therefore
      // w = beta_over_mult_cumulative[h + logsizeprime - j] ^ beta_over_mult_cumulative[logsizeprime - j].
      for(unsigned int j = 0; j < min<unsigned long>(1 << s, h + logsizeprime); j++)
      {
        offsets_mult[j] ^= beta_over_mult_cumulative[h + logsizeprime - j];
      }
      for(unsigned int j = 0; j < min<unsigned long>(1 << s, logsizeprime); j++)
      {
        offsets_mult[j] ^= beta_over_mult_cumulative[logsizeprime - j];
      }
    }
  }
  else
  {
    // s == 0, logsize = 0 or 1. do not optimize.
    uint64_t offsets_mult[1];
    offsets_mult[0] = 0;
#ifdef USE_TEMPLATIZED
    mgt_core<0, 0, false>(p, offsets_mult, logsize);
#else
    mg_core<0, 0>(p, offsets_mult, logsize, false);
#endif
  }
}

template <unsigned int s>
void mg_smalldegree_with_buf(
    uint64_t* p, // binary polynomial, unexpanded, of degree < 2**logsize_prime **once expanded**
    unsigned int logsizeprime, // log2 bound on the polynomial number of coefficents, <= logsize
    unsigned int logsize, // size of the DFT to be computed, <= 2**(2**s)
    uint64_t* main_buf // buffer of size 2**logsize, whose elements wil be termwise multiplied by DFT of p
    )
{
  if constexpr(s > 0)
  {
    if(logsizeprime <= (1u << (s-1)))
    {
      mg_smalldegree_with_buf<s-1>(p, logsizeprime, logsize, main_buf);
      return;
    }
    // logsizeprime > 2**(s-1) => logsizeprime > 1

    uint64_t* buf = new uint64_t[1uLL << logsizeprime];
    unique_ptr<uint64_t[]> _(buf);

    constexpr unsigned int t = 1u << (s - 1);

    const uint64_t szp = 1uLL << logsizeprime;
    // the natural order of operations is to expand p (expand every 32-bit word into q 64-bit word),
    // then apply mg_decompose_taylor_recursive on 64-bit words
    // this is equivalent to doing mg_decompose_taylor_recursive on 32-bit words, then expanding
    mg_decompose_taylor_recursive<0u, t, uint32_t>(logsizeprime, (uint32_t*)p);

    uint64_t offsets_mult[1 << s];
    for(int i = 0; i < (1 << s); i++) offsets_mult[i] = 0;

    for(uint64_t i = 0; i < 1uLL << (logsize - logsizeprime); i++)
    {
      expand(buf, p, szp * 32 - 1, szp);
      mg_core<s, 0>(buf, offsets_mult, logsizeprime, true);
      // logsizeprime >= 2 (see above)
      product_batch(main_buf + (i << logsizeprime), buf, logsizeprime);
      const long unsigned int h = _mm_popcnt_u64(i^(i+1));
      for(unsigned int j = 0; j < min<unsigned long>(1 << s, h + logsizeprime); j++)
      {
        offsets_mult[j] ^= beta_over_mult_cumulative[h + logsizeprime - j];
      }
      for(unsigned int j = 0; j < min<unsigned long>(1 << s, logsizeprime); j++)
      {
        offsets_mult[j] ^= beta_over_mult_cumulative[logsizeprime - j];
      }
    }
  }
  else
  {
    // s == 0, logsize = 0 or 1. do not optimize.
    uint64_t offsets_mult[1];
    offsets_mult[0] = 0;
    mg_core<0, 0>(p, offsets_mult, logsize, false);
    for(int i = 0; i < 1 << logsize; i++) main_buf[i] = product(main_buf[i], p[i]);
  }
}

/**
 * @brief mg_reverse_core
 * Reverse of mg_core. See mg_core for argument description.
 */
template <int s, int logstride>
void mg_reverse_core(
    uint64_t* poly,
    const uint64_t* offsets_mult,
    unsigned int logsize)
{
  if constexpr(s == 0)
  {
    eval_degree1<true, logstride>(offsets_mult[0], poly);
  }
  else
  {
    constexpr unsigned int t = 1 << (s - 1);
    if(logsize <= t)
    {
      mg_reverse_core<s - 1, logstride>(poly, offsets_mult, logsize);
    }
    else
    {
      const uint64_t row_size = 1uLL << (t + logstride);
      uint64_t* poly_loc = poly;
      uint64_t offsets_local[t];
      for(unsigned int j = 0; j < t; j++) offsets_local[j] = offsets_mult[j];
      // 2*t >= logsize > t, therefore tau < 2**t
      const uint64_t tau   = 1uLL <<  (logsize - t);
      for(uint64_t i = 0; i < tau; i++)
      {
        // at each iteration, offsets_local[j] = beta_to_mult(offset + (i << (t-j))), j < t
        mg_reverse_core<s - 1, logstride>(poly_loc, offsets_local, t);
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
      mg_reverse_core<s - 1, logstride + t>(poly, offsets_mult + t, logsize - t);
      mg_decompose_taylor_reverse_recursive<logstride, t>(logsize, poly);
    }
  }
}

void mg_binary_polynomial_multiply(uint64_t* p1, uint64_t* p2, uint64_t* result, uint64_t d1, uint64_t d2)
{
  constexpr unsigned int s = 6;
  uint64_t offsets_mult[1 << s];
  for(int i = 0; i < (1 << s); i++) offsets_mult[i] = 0;
  unsigned int logsize_result = 1;
  // size of result, in 32-bit words.
  while((1uLL << logsize_result) * 32 < (d1 + d2 + 1)) logsize_result++;
  const uint64_t sz = 1uLL << logsize_result;
  // a buffer twice the size of the result
  uint64_t* b1 = new uint64_t[sz];
  unique_ptr<uint64_t[]> _1(b1);
  // a second one
  uint64_t* b2 = new uint64_t[sz];
  unique_ptr<uint64_t[]> _2(b2);
  expand(b1, p1, d1, sz);
  expand(b2, p2, d2, sz);
  // size of p1, in 32-bit words.
  unsigned int logsize_1 = 0;
  while((1uLL << logsize_1) * 32 < (d1 + 1)) logsize_1++;
  // size of p2, in 32-bit words.
  unsigned int logsize_2 = 0;
  while((1uLL << logsize_2) * 32 < (d2 + 1)) logsize_2++;
  mg_smalldegree<s>(b1, logsize_1, logsize_result);
  mg_smalldegree<s>(b2, logsize_2, logsize_result);
  if(logsize_result < 2)
  {
    for(size_t i = 0; i < sz; i++) b1[i] = product(b1[i], b2[i]);
  }
  else
  {
    product_batch(b1, b2, logsize_result);
  }
#ifdef USE_TEMPLATIZED
  mgt_reverse_core<s,0>(b1, offsets_mult, logsize);
#else
  mg_reverse_core<s,0>(b1, offsets_mult, logsize_result);
#endif

  contract(result, b1, d1 + d2);
}

void mg_binary_polynomial_multiply_in_place (uint64_t* p1, uint64_t* p2, uint64_t d1, uint64_t d2)
{
  constexpr unsigned int s = 6;
  uint64_t offsets_mult[1 << s];
  for(int i = 0; i < (1 << s); i++) offsets_mult[i] = 0;
  // size of the result, in 32-bit words.
  unsigned int logsize_result = 0;
  while((1uLL << logsize_result) * 32 < (d1 + d2 + 1)) logsize_result++;

  // size of p1, in 32-bit words.
  unsigned int logsize_1 = 0;
  while((1uLL << logsize_1) * 32 < (d1 + 1)) logsize_1++;

  // size of p2, in 32-bit words.
  unsigned int logsize_2 = 0;
  while((1uLL << logsize_2) * 32 < (d2 + 1)) logsize_2++;

  const uint64_t sz_result = 1uLL << logsize_result;
  expand(p1, p1, d1, sz_result);
  mg_smalldegree<s>((uint64_t*) p1, logsize_1, logsize_result);

  mg_smalldegree_with_buf<s>(p2, logsize_2, logsize_result, p1);

  mg_reverse_core<s,0>(p1, offsets_mult, logsize_result);
  contract(p1, p1, d1 + d2);
}
