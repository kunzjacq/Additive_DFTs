#include "mg.h"

#include <cstdint> // for uint64_t and other types
#include <cassert> // for assert
#include <memory> // for unique_ptr
#include <algorithm> // for min, max

#include <immintrin.h>

using namespace std;

/**
  m_beta_over_mult[i] = beta_i in multplicative representation, i.e. in GF(2**64) as described in
  function 'product'. beta_i refers to the beta basis in mateer-gao algorithm.
  */
static constexpr uint64_t m_beta_over_mult[] = {
  0x1               , 0x19c9369f278adc02, 0xa181e7d66f5ff795, 0x447175c8e9f2810b,
  0x013b052fbd1cfb5d, 0xb7ea5a9705b771c0, 0x467698598926dc01, 0x1a9c05699898468f,
  0x109a9dd350b468b8, 0x184b3b707446faf8, 0xbd21077a71c52b4a, 0xfd4fb47b8220beec,
  0xe4b954b528d60802, 0xf3dc28c547403da8, 0xedb0cbd331e507da, 0x528a186f6c748cc3,
  0x2a4b189ca2c5b59,  0x029a7d210e532656, 0xada3c0442e57c3ed, 0xfc81451edeb89ab1,
  0xe0c6709608af557d, 0xf3a6d818653f0cad, 0x462ae6a22ce5e08f, 0x1e677e25acb98834,
  0x0d1c664464247894, 0x0ba8cdc203194db8, 0xa72be7361a8a2e7b, 0xe8f9a3bef7bf9a23,
  0xfdc592067c87cefb, 0x5619de8a10dc091c, 0xb00c8dfcd7e00c33, 0xef0bce0ca0b80a16,
  0xfbae750576464fad, 0xe39ee95e2988b6f0, 0x5fcbcfaf4fc958bc, 0xbb39bb36fb3a3a6e,
  0x49f4e995a5e6a19a, 0xb96e26d8986dcc00, 0xfd102dc9a745eace, 0xfc215193194874b9,
  0x4f3722442534c506, 0x14c452a09a1f6902, 0x056a9d1a427420eb, 0x1c4a2ee33d3f6ccc,
  0x0beb2000556e2d9d, 0xa623096706b4f000, 0xe9eb43c1f6c4800e, 0x4fea486d51515e3b,
  0xa2848cdf92322663, 0xf53bfc6ad613accb, 0x5d14a7c82a96db10, 0x0bb55122c488b32d,
  0x150e463f2440d67e, 0xb23320484d5035f4, 0x5bc3417ba85bf519, 0x10c52081c14d3417,
  0xb60c77a147cd8953, 0x5bf866708a444e09, 0x1449d1fdf01d5b76, 0xb230b2ee57674530,
  0x46fbb34fb404e77d, 0x1ea01f4ba902109e, 0x0d8989c26ace34e6, 0x8b7f48848818e45c
};

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
template<unsigned int logstride, unsigned int t>
inline void mg_decompose_taylor_recursive(
    unsigned int logsize,
    uint64_t* p)
{
  static_assert(t >= 1);
  assert(logsize > t && logsize <= 2*t);
  //decompose_taylor_one_step
  uint64_t delta_s   = 1uLL << (logstride + logsize - 1 - t); // logn - 1 - t >= 0
  const uint64_t n_s = 1uLL  << (logstride + logsize);
  const uint64_t m_s = n_s >> 1;
  uint64_t* q = p + delta_s - m_s;
  // m_s > 0, hence the loop below is not infinite
  for(uint64_t i = n_s - 1; i > m_s - 1; i--) q[i] ^= p[i];

  if(logsize > t + 1)
  {
    mg_decompose_taylor_recursive<logstride, t>(logsize - 1, p);
    mg_decompose_taylor_recursive<logstride, t>(logsize - 1, p + m_s);
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

/**
  @brief mg_aux
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
inline void mg_aux(
    uint64_t* mult_pow_table,
    uint64_t* poly,
    uint64_t* offsets_mult,
    unsigned int logsize,
    bool first_taylor_done)
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
    if(logsize <= t)
    {
      mg_aux<s - 1, logstride>(mult_pow_table, poly, offsets_mult, logsize, first_taylor_done);
    }
    else
    {
      // here 2*t >= logsize > t
      const uint64_t tau   = 1uLL <<  (logsize - t);
      // on input: there are 2**'logstride' interleaved series of size 'eta' = 2**(2*logsize-t);
      if(!first_taylor_done)
      {
        //decompose_taylor_recursive(logstride, 2 * t, t, 1uLL << logsize, poly);
        mg_decompose_taylor_recursive<logstride,t>(logsize,poly);
      }
      // fft on columns
      // if logsize >= t, each fft should process 2**(logsize - t) values
      // i.e. logsize' = logsize - t
      // it is faster to recompute offset_prime in mult representation from its beta
      // representation than performing the equivalent of >> t directly on mult representation
      // which could be done by
      //offset' = offset >> t;
      mg_aux<s - 1, logstride + t>(mult_pow_table, poly, offsets_mult + t, logsize - t, false);
      uint64_t offsets_local[1u << s];
      for(unsigned int i = 0; i < (1u << s); i++) offsets_local[i] = offsets_mult[i];
      for(uint64_t i = 0; i < tau; i++)
      {
        // at all times, offset_mult_prime = beta_to_mult(offset + (i << t), beta_table)
        // in multiplicative representation, at current depth and for iteration i, offset is offset[depth] ^ (i << t);
        mg_aux<s - 1, logstride>(mult_pow_table, poly, offsets_local, t, false);
        const long int h = _mm_popcnt_u64(i^(i+1)); // i^(i+1) is a power of 2 - 1
        for(unsigned int i = 0; i < (1u<<s); i++)
        {
          int tp = t - i;
          offsets_local[i] ^= mult_pow_table[h + tp] ^ mult_pow_table[tp];
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
 * If logsizeprime < logsize, this function is more efficient than a direct call to mg_aux.
 * @param s
 * a recursion parameter s.t. logsize <= 2**s.
 * @param mult_pow_table
 * see the same argument for mg_aux.
 * @param p
 * the input and output array, of size 2**logsize.
 * @param logsizeprime
 * a log2 bound on the number of coefficients of the input polynomial.
 * @param logsize
 * the log2 number of output values computed (with offset 0).
 */
template <unsigned int s>
void mg_smalldegree(
    uint64_t* mult_pow_table,
    uint64_t* p, // of degree < 2**logsize_prime
    unsigned int logsizeprime, // log2 bound on the polynomial number of coefficents, <= logsize
    unsigned int logsize // size of the DFT to be computed, <= 2**(2**s)
    )
{
  if constexpr(s > 0)
  {
    if(logsizeprime <= (1u << (s-1)))
    {
      mg_smalldegree<s-1>(mult_pow_table, p, logsizeprime, logsize);
      return;
    }
    constexpr unsigned int t = 1u << (s - 1);
    mg_decompose_taylor_recursive<0u, t>(logsizeprime, p);
    for(uint64_t i = 1; i < 1uLL << (logsize - logsizeprime); i++)
    {
      uint64_t* q = p + (i << logsizeprime);
      for(uint64_t j = 0; j < 1uLL << logsizeprime; j++) q[j] = p[j];
    }

    uint64_t offsets_mult[1 << s];
    for(int i = 0; i < (1 << s); i++) offsets_mult[i] = 0;

    for(uint64_t i = 0; i < 1uLL << (logsize - logsizeprime); i++)
    {
      //offset in beta repr: i << logsizeprime
      mg_aux<s, 0>(mult_pow_table, p + (i << logsizeprime), offsets_mult, logsizeprime, true);
      const long unsigned int h = _mm_popcnt_u64(i^(i+1)); // i^(i+1) is a power of 2 - 1
      // for j > h + logsizeprime, (i^(i+1) << i) >> logsizeprime = 0
      for(unsigned int j = 0; j <= min<unsigned long>(63, h + logsizeprime); j++)
      {
        offsets_mult[j] ^= mult_pow_table[h + logsizeprime - j];
      }
      for(unsigned int j = 0; j <= min<unsigned long>(63, logsizeprime); j++)
      {
        offsets_mult[j] ^= mult_pow_table[logsizeprime - j];
      }
    }
  }
  else
  {
    // s == 0, logsizeprime = 0 or 1
    uint64_t offsets_mult[1];
    offsets_mult[0] = 0;
    mg_aux<0, 0>(mult_pow_table, p, offsets_mult, logsizeprime, false);
  }
}

/**
 * @brief mg_reverse_aux
 * Reverse of mg_aux. See mg_aux for argument description.
 */
template <int s, int logstride>
void mg_reverse_aux(
    uint64_t* mult_pow_table,
    uint64_t* poly,
    uint64_t* offsets_mult,
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
      mg_reverse_aux<s - 1, logstride>(mult_pow_table, poly, offsets_mult, logsize);
    }
    else
    {
      const uint64_t tau   = 1uLL <<  (logsize - t);
      const uint64_t row_size = 1uLL << (t + logstride);
      uint64_t* poly_loc = poly;
      uint64_t offsets_local[1u << s];
      for(unsigned int i = 0; i < (1u << s); i++) offsets_local[i] = offsets_mult[i];
      for(uint64_t i = 0; i < tau; i++)
      {
        // offset_mult_prime = beta_to_mult(offset ^ (i << t), beta_table)
        // offset_prime = offset ^ (i << t);
        mg_reverse_aux<s - 1, logstride>(mult_pow_table, poly_loc, offsets_local, t);
        long int h = _mm_popcnt_u64(i^(i+1));
        for(unsigned int i = 0; i < (1u << s); i++)
        {
          int tp = t - i;
          offsets_local[i] ^= mult_pow_table[h + tp] ^ mult_pow_table[tp];
        }
        poly_loc += row_size;
      }
      // reverse fft on columns
      //offset' = offset >> t;
      mg_reverse_aux<s - 1, logstride + t>(mult_pow_table, poly, offsets_mult + t, logsize - t);
      mg_decompose_taylor_reverse_recursive<logstride, t>(logsize, poly);
    }
  }
}

static uint64_t expand(uint8_t* p, uint64_t* res, uint64_t d, size_t fft_size)
{
  auto source = reinterpret_cast<uint32_t*>(p);
  constexpr int num_bits_half = 32;
  uint64_t dw = d / num_bits_half + 1;
  uint64_t j = 0;
  // little endian
  for(j = 0; j < dw; j++) res[j] = source[j];
  for(; j < fft_size;j++) res[j] = 0;
  return dw;
}

static void contract(uint64_t* buf, uint8_t* p, uint64_t d)
{
  constexpr unsigned int bits_per_half_uint64_t  = 32;
  const size_t bound = d / bits_per_half_uint64_t + 1;
  uint64_t* dest_even = reinterpret_cast<uint64_t*>(p);
  uint64_t* dest_odd  = reinterpret_cast<uint64_t*>(p + (bits_per_half_uint64_t >> 3));
  for(uint64_t i = 0; i < d/8+1; i++) p[i] = 0;
  for(uint64_t i = 0; i < bound; i++)
  {
    if((i & 1) == 0)
    {
      dest_even[i >> 1] ^= buf[i];
    }
    else
    {
      dest_odd[i >> 1]  ^= buf[i];
    }
  }
}

/**
 * @brief mateer_gao_polynomial_product::binary_polynomial_multiply
 * computes the product of 2 binary polynomials pointed to by 'p1' and 'p2' and of degree 'd1'
 * and 'd2'.
 * result is stored in 'result'.
 * @param p1
 * @param p2
 * @param result
 * @param d1
 * @param d2
 */
void mateer_gao_polynomial_product::binary_polynomial_multiply(
    uint8_t* p1, uint8_t* p2, uint8_t* result, uint64_t d1, uint64_t d2)
{
  constexpr unsigned int s = 6;
  uint64_t offsets_mult[1 << s];
  for(int i = 0; i < (1 << s); i++) offsets_mult[i] = 0;
  unsigned int logsize = 1;
  while((1uLL<<logsize) * 32 < (d1 + d2 + 1)) logsize++;
  uint64_t* b1 = new uint64_t[1uLL<<logsize];
  unique_ptr<uint64_t[]> _1(b1);
  uint64_t* b2 = new uint64_t[1uLL<<logsize];
  unique_ptr<uint64_t[]> _2(b2);
  const uint64_t sz = 1uLL << logsize;
  uint64_t w1 = expand(p1, b1, d1, sz);
  uint64_t w2 = expand(p2, b2, d2, sz);
  int logsizeprime = logsize;
  while(1uLL << logsizeprime >= 2 * w1) logsizeprime--;
  mg_smalldegree<s>(m_mult_beta_pow_table, b1, logsizeprime, logsize);
  logsizeprime = logsize;
  while(1uLL << logsizeprime >= 2 * w2) logsizeprime--;
  mg_smalldegree<s>(m_mult_beta_pow_table, b2, logsizeprime, logsize);
  if(logsize < 2)
  {
    for(size_t i = 0; i < sz; i++) b1[i] = product(b1[i], b2[i]);
  }
  else
  {
    product_batch(b1, b2, logsize);
  }
  mg_reverse_aux<s,0>(m_mult_beta_pow_table, b1, offsets_mult, logsize);
  contract(b1, result, d1 + d2);
}

mateer_gao_polynomial_product::mateer_gao_polynomial_product():
  m_mult_beta_pow_table(new uint64_t[64])

{
  m_mult_beta_pow_table[0] = 0;
  for(int i = 1; i < 64; i++)
  {
    m_mult_beta_pow_table[i] = m_mult_beta_pow_table[i-1]^m_beta_over_mult[i-1];
  }
}

mateer_gao_polynomial_product::~mateer_gao_polynomial_product()
{
  delete[] m_mult_beta_pow_table;
  m_mult_beta_pow_table = nullptr;
}
