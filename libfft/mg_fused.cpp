#include <memory>    // for unique_ptr
#include <algorithm> // for min, max

#include <immintrin.h>
#include "mg_fused.h"

#include "mg_templates.h"

using namespace std;

// Mateer-Gao binary polynomial product with the last DFT stages, pointwise product and first
// inverse DFT stages fused into direct products modulo x**R + x + c (see 'ring_rows' below).
// This is a variant of mg.cpp, which is left unchanged; everything is in namespace mg_fused.
namespace mg_fused {

//#define GRAY_CODE

/**
 * @brief beta_over_mult
 * for i < 63,
 * beta_over_mult[i] = beta_over_mult[i+1] ^ beta_over_mult[i+1]**2 (*)
 * where the product is in GF(2**64), in multiplicative representation, i.e. in GF(2**64) as
 * described in function 'product'
 * and
 * beta_over_mult[0] = 1 (**)
 * .
 * This table used to be statically generated through another program.
 * Now it is generated from a known good value beta_over_mult[63] using equation (*) above.
 * A 'good value' is a value s.t. (**) is satisfied.
 * 1/2 of all possible starting values are good values, therefore it is not hard to find one.
 * initialization of beta_over_mult[] and beta_over_mult_cumulative[] is performed by function
 * init_beta_table().
 */
static uint64_t beta_over_mult[64];

static bool beta_is_initialized = false;

/** beta_over_mult_cumulative[i] = sum_{j = 0 ... i - 1} beta_i
 * in multplicative representation, i.e. in GF(2**64) as described in
 * function 'product'. beta_i refers to the beta basis in Mateer-Gao algorithm.
 */

static uint64_t beta_over_mult_cumulative[64];

/**
 * @brief expand
 * Space each 32-bit word of 'source' with a null 32-bit word into 'dest'.
 * Pad the result with zeros up to 'fft_size' 64-bit words.
 * 'source' is a binary polyomial of degree 'd', that uses with 'd' / 32 + 1 32-bit words.
 * Can be used with dest = source, or with disjoint dest and source arrays.
 * Implementation is little-endian only.
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
  // backward loop to handle the case where dest = source
  for(j = 0; j < num_dwords; j++) dest[num_dwords - 1 - j] = source32[num_dwords - 1 - j];
  for(; j < fft_size;j++) dest[j] = 0;
}

/**
 * @brief contract
 * Contract a polynomial over GF(2**64), xoring each upper half of each 64-bit with the lower
 * half of the next 64-bit word.
 * Can be used with dest = source, or with disjoint dest and source arrays.
 * Implementation is little-endian only.
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
  if(num_dwords & 1) dest32[num_dwords] = 0; // clear the upper half of the last 64-bit word of the result
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
  // p = x**64 + x**4 + x**3 + x + 1 is primitive over GF(2)
  // it is the minimal polynomial of the multiplicative generator g
  // which therefore satisfies g**64 + g**4 + g**3 + g + 1
  __m128i xa, xb, xc;
  xa = _mm_set_epi64x(minpoly, a); // xa_0 = a, xa_1 = minpoly
  xb = _mm_set1_epi64x(b); // xb_0 = b
  // below, * is the carry-less multiplication of 64-bit values which produces a 128-bit value
  // and which is implemented by the instruction PCLMUL in the amd64 instruction set
  // if 128-bit words are interpreted as polynomials evaluated at g, * and × coincide, but *
  // produces values that are larger than 64 bits and that must therefore be reduced mod p
  xb = _mm_clmulepi64_si128(xa, xb, 0x00); // xb = xa_0 * xb_0 = a * b
  xc = _mm_clmulepi64_si128(xa, xb, 0x11); // xc = xb_1 * minpoly
  // g**64 × xb_1 = minpoly × xb_1 as values in GF(2**64) (since p(g) = g**64 ^ minpoly = 0)
  // minpoly * xb_1 is a 68-bit value
  xb = _mm_xor_si128(xb, xc); // xb = xb ^ xc (the result is accumulated in xb_0)
  // xc_1 < 16: minpoly * xc_1 is read from a 16-entry table (pshufb) instead of a third clmul
  xc = _mm_srli_si128(_mm_shuffle_epi8(mg_lut128(), xc), 8);
  xb = _mm_xor_si128(xb, xc); // xb = xb ^ xc
  return _mm_extract_epi64(xb, 0); // return xb_0
}

static int init_beta_table()
{
  //constexpr uint64_t start_value = 0x8b7f48848818e45c; // start value previously used in table generated externally
  constexpr uint64_t start_value = 0x8000000000000000;
  beta_over_mult[63] = start_value;
  int iter = 0;
  do
  {
    beta_over_mult[63]++;
    for(int i = 62; i >= 0; i--)
    {
      uint64_t val = beta_over_mult[i+1];
      beta_over_mult[i] = val^product(val, val);
    }
    iter++;
  }
  while(beta_over_mult[0] != 1);

  beta_over_mult_cumulative[0] = 0;
  for(int i = 1; i < 64; i++)
  {
    beta_over_mult_cumulative[i] = beta_over_mult_cumulative[i-1]^beta_over_mult[i-1];
  }
  beta_is_initialized = true;
  return iter;
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
static void product_batch(uint64_t* restr a_ptru, uint64_t* restr b_ptru, unsigned int logsize)
{
  __m128i* restr a_ptr128 = (__m128i*) std::assume_aligned<16>(a_ptru);
  __m128i* restr b_ptr128 = (__m128i*) std::assume_aligned<16>(b_ptru);

  assert(logsize >= 2);
  const uint64_t sz = 1uLL << (logsize - 2);
#ifdef MG_USE_YMM_CLMUL
  // 4 products per iteration, 2 per 128-bit lane; same reduction as eval_degree1_new
  {
    constexpr uint64_t minpoly = 0x1b;
    const __m256i xp  = _mm256_set1_epi64x(minpoly);
    const __m128i lut128 = mg_lut128();
    const __m256i lut = _mm256_broadcastsi128_si256(lut128);
    __m256i* restr a256 = (__m256i*) a_ptru;
    const __m256i* restr b256 = (const __m256i*) b_ptru;
    for(uint64_t i = 0; i < sz; i++)
    {
      const __m256i xa = _mm256_loadu_si256(a256 + i);
      const __m256i xb = _mm256_loadu_si256(b256 + i);
      __m256i e  = _mm256_clmulepi64_epi128(xa, xb, 0x00);
      __m256i o  = _mm256_clmulepi64_epi128(xa, xb, 0x11);
      __m256i xe = _mm256_clmulepi64_epi128(e, xp, 0x01);
      __m256i xo = _mm256_clmulepi64_epi128(o, xp, 0x01);
      e = _mm256_xor_si256(e, xe);
      o = _mm256_xor_si256(o, xo);
      __m256i r = _mm256_xor_si256(_mm256_unpacklo_epi64(e, o),
          _mm256_unpackhi_epi64(_mm256_shuffle_epi8(lut, xe), _mm256_shuffle_epi8(lut, xo)));
      _mm256_storeu_si256(a256 + i, r);
    }
    return;
  }
#endif

  constexpr uint64_t minpoly = 0x1b;
  __m128i xp, xa1, xa2, xa3, xa4, xb1, xb2, xb3, xb4, xc1, xc2, xc3, xc4;

  xp = _mm_set1_epi64x(minpoly);
  for(uint64_t i = 0; i < sz; i ++)
  {
    xa1 = _mm_load_si128(a_ptr128);
    xa3 = _mm_load_si128(a_ptr128 + 1);
    xb1 = _mm_load_si128(b_ptr128);
    xb3 = _mm_load_si128(b_ptr128 + 1);
#if 1
    xa2 = _mm_shuffle_epi32(xa1, 0x4e); // put higher part of xa1 in lower part of xa2:
    // dwords 3, 2 in lower part, 3*4+2 = e,
    // dwords 1, 0 in higher part, 1 * 4 + 0 = 4
    xa4 = _mm_shuffle_epi32(xa3, 0x4e); // same with xa3 / xa4
    xb2 = _mm_shuffle_epi32(xb1, 0x4e); // same with xb1 / xb2
    xb4 = _mm_shuffle_epi32(xb3, 0x4e); // same with xb3 / xb4
#else
    xa2 = _mm_unpackhi_epi64(xa1, xb2); //xa1 = upper half of xa2; xb2 unused here
    xa4 = _mm_unpackhi_epi64(xa3, xb2); //xb2 unused here
    xb2 = _mm_unpackhi_epi64(xb1, xb2); //xb2 unused here
    xb4 = _mm_unpackhi_epi64(xb3, xb2); //xb2 unused here
#endif

    xb1 = _mm_clmulepi64_si128(xa1, xb1, 0x00);
    xb2 = _mm_clmulepi64_si128(xa2, xb2, 0x00);
    xb3 = _mm_clmulepi64_si128(xa3, xb3, 0x00);
    xb4 = _mm_clmulepi64_si128(xa4, xb4, 0x00);
    xc1 = _mm_clmulepi64_si128(xp,  xb1, 0x10);
    xc2 = _mm_clmulepi64_si128(xp,  xb2, 0x10);
    xc3 = _mm_clmulepi64_si128(xp,  xb3, 0x10);
    xc4 = _mm_clmulepi64_si128(xp,  xb4, 0x10);
    xb1 = _mm_xor_si128(xb1, xc1);
    xb2 = _mm_xor_si128(xb2, xc2);
    xb3 = _mm_xor_si128(xb3, xc3);
    xb4 = _mm_xor_si128(xb4, xc4);
    xc1 = _mm_clmulepi64_si128(xp, xc1, 0x10);
    xc2 = _mm_clmulepi64_si128(xp, xc2, 0x10);
    xc3 = _mm_clmulepi64_si128(xp, xc3, 0x10);
    xc4 = _mm_clmulepi64_si128(xp, xc4, 0x10);
    xb1 =_mm_unpacklo_epi64(xb1, xb2); // group lower halves of xb1 and xb2 into xb1
    xc1 =_mm_unpacklo_epi64(xc1, xc2);
    xb3 =_mm_unpacklo_epi64(xb3, xb4);
    xc3 =_mm_unpacklo_epi64(xc3, xc4);
    xb1 = _mm_xor_si128(xb1, xc1);
    xb3 = _mm_xor_si128(xb3, xc3);
    _mm_store_si128(a_ptr128,     xb1); //16-byte aligned store
    _mm_store_si128(a_ptr128 + 1, xb3);
    a_ptr128 += 2;
    b_ptr128 += 2;
  }
}


#ifndef MG_ROW4
#define MG_ROW4 1
#endif
// ---- register-resident kernels for rows of 4 words (s = 1, logsize = 2, logstride = 0)
// (valA * Q.lo, valB * Q.hi style helpers; xa* = (minpoly : val); reduction as in eval_degree1_new)
static inline __m128i mg_mul_vec2(__m128i xa, __m128i Q, __m128i lut)
{
  __m128i e  = _mm_clmulepi64_si128(xa, Q, 0x00);
  __m128i o  = _mm_clmulepi64_si128(xa, Q, 0x10);
  __m128i xe = _mm_clmulepi64_si128(xa, e, 0x11);
  __m128i xo = _mm_clmulepi64_si128(xa, o, 0x11);
  e = _mm_xor_si128(e, xe);
  o = _mm_xor_si128(o, xo);
  return _mm_xor_si128(_mm_unpacklo_epi64(e, o),
                       _mm_unpackhi_epi64(_mm_shuffle_epi8(lut, xe), _mm_shuffle_epi8(lut, xo)));
}

// (valA * SA[immA], valB * SB[immB]) with immA, immB in {0x00 (low qword), 0x10 (high qword)}
template<int immA, int immB>
static inline __m128i mg_mul_pair(__m128i xaA, __m128i xaB, __m128i SA, __m128i SB, __m128i lut)
{
  __m128i m0 = _mm_clmulepi64_si128(xaA, SA, immA);
  __m128i m1 = _mm_clmulepi64_si128(xaB, SB, immB);
  __m128i x0 = _mm_clmulepi64_si128(xaA, m0, 0x11);
  __m128i x1 = _mm_clmulepi64_si128(xaB, m1, 0x11);
  m0 = _mm_xor_si128(m0, x0);
  m1 = _mm_xor_si128(m1, x1);
  return _mm_xor_si128(_mm_unpacklo_epi64(m0, m1),
                       _mm_unpackhi_epi64(_mm_shuffle_epi8(lut, x0), _mm_shuffle_epi8(lut, x1)));
}

template<bool taylor>
static inline void mg_row4_forward(uint64_t* restr pu, const uint64_t* offsets_mult)
{
  uint64_t* restr p = (uint64_t*) std::assume_aligned<16>(pu);
  constexpr uint64_t minpoly = 0x1b;
  const uint64_t valA = offsets_mult[0];
  const uint64_t c1   = offsets_mult[1];
  const uint64_t valB = valA ^ beta_over_mult_cumulative[2] ^ beta_over_mult_cumulative[1];
  const __m128i lut = mg_lut128();
  __m128i A = _mm_load_si128((__m128i*)(p));
  __m128i B = _mm_load_si128((__m128i*)(p) + 1);
  if constexpr(taylor)
  {
    B = _mm_xor_si128(B, _mm_srli_si128(B, 8));  // (w2^w3, w3)
    A = _mm_xor_si128(A, _mm_slli_si128(B, 8));  // (w0, w1^w2^w3)
  }
  __m128i U = _mm_xor_si128(A, mg_mul_vec2(_mm_set_epi64x(minpoly, c1), B, lut));
  __m128i V = _mm_xor_si128(B, U);
  __m128i R = mg_mul_pair<0x10, 0x10>(_mm_set_epi64x(minpoly, valA), _mm_set_epi64x(minpoly, valB), U, V, lut);
  __m128i W = _mm_xor_si128(_mm_unpacklo_epi64(U, V), R);
  __m128i X = _mm_xor_si128(_mm_unpackhi_epi64(U, V), W);
  _mm_store_si128((__m128i*)(p),     _mm_unpacklo_epi64(W, X));
  _mm_store_si128((__m128i*)(p) + 1, _mm_unpackhi_epi64(W, X));
}

static inline void mg_row4_reverse(uint64_t* restr pu, const uint64_t* offsets_mult)
{
  uint64_t* restr p = (uint64_t*) std::assume_aligned<16>(pu);
  constexpr uint64_t minpoly = 0x1b;
  const uint64_t valA = offsets_mult[0];
  const uint64_t c1   = offsets_mult[1];
  const uint64_t valB = valA ^ beta_over_mult_cumulative[2] ^ beta_over_mult_cumulative[1];
  const __m128i lut = mg_lut128();
  __m128i A = _mm_load_si128((__m128i*)(p));
  __m128i B = _mm_load_si128((__m128i*)(p) + 1);
  __m128i W = _mm_unpacklo_epi64(A, B);
  __m128i X = _mm_unpackhi_epi64(A, B);
  X = _mm_xor_si128(X, W);
  __m128i R = mg_mul_pair<0x00, 0x10>(_mm_set_epi64x(minpoly, valA), _mm_set_epi64x(minpoly, valB), X, X, lut);
  W = _mm_xor_si128(W, R);
  __m128i U = _mm_unpacklo_epi64(W, X);
  __m128i V = _mm_unpackhi_epi64(W, X);
  V = _mm_xor_si128(V, U);
  U = _mm_xor_si128(U, mg_mul_vec2(_mm_set_epi64x(minpoly, c1), V, lut));
  U = _mm_xor_si128(U, _mm_slli_si128(V, 8));
  V = _mm_xor_si128(V, _mm_srli_si128(V, 8));
  _mm_store_si128((__m128i*)(p),     U);
  _mm_store_si128((__m128i*)(p) + 1, V);
}


// ---------------------------------------------------------------------------------------------
// Direct products of rows.
//
// In mg_core, once the Taylor decompositions and the DFTs on columns are done, the data is made of
// rows of R = 4 or 16 words (R = 2**(2**(s-1)), s = 2 or 3), which are polynomials of degree < R
// to be evaluated on a coset a + GF(R) of the subfield GF(R) of GF(2**64) (R = 2**(2**j), and the
// beta basis is a Cantor basis). The DFT of such a row, the pointwise product with another row's
// DFT, and the inverse DFT, together compute the product of both rows modulo
// prod_{u in GF(R)} (x - a - u) = x**R + x + c, with c = a**R + a. In beta representation c is
// (offset >> log2(R)) i.e. it is available as offsets_mult[log2(R)] in mg_core.
// That product can be computed directly with Karatsuba on carry-less 64x64 -> 128 bit products
// (kept unreduced, as the operations are linear over GF(2)), followed by a cheap reduction:
// x**R = x + c, i.e. r_k = p_k + p_{k+R-1} + c * p_{k+R} (no cascading).
// This is faster than the chain of 3 DFTs + pointwise product for R = 4 and 16, not for R = 256.
// ---------------------------------------------------------------------------------------------

static inline __m128i cl(uint64_t a, uint64_t b)
{
  return _mm_clmulepi64_si128(_mm_cvtsi64_si128((long long) a), _mm_cvtsi64_si128((long long) b), 0);
}

// out[0..2N-2] <- unreduced a * b, N-word polynomials. Karatsuba, schoolbook for N <= K.
template<int N, int K> static inline void kmul(const uint64_t* a, const uint64_t* b, __m128i* out)
{
  if constexpr(N == 1) out[0] = cl(a[0], b[0]);
  else if constexpr(N <= K)
  {
    for(int i = 0; i < 2 * N - 1; i++) out[i] = _mm_setzero_si128();
    for(int i = 0; i < N; i++)
      for(int j = 0; j < N; j++) out[i + j] = _mm_xor_si128(out[i + j], cl(a[i], b[j]));
  }
  else
  {
    constexpr int h = N / 2;
    uint64_t as[h], bs[h];
    __m128i p0[2 * h - 1], p2[2 * h - 1], pm[2 * h - 1];
    kmul<h, K>(a, b, p0);
    kmul<h, K>(a + h, b + h, p2);
    for(int i = 0; i < h; i++) {as[i] = a[i] ^ a[i + h]; bs[i] = b[i] ^ b[i + h];}
    kmul<h, K>(as, bs, pm);
    for(int i = 0; i < 2 * N - 1; i++) out[i] = _mm_setzero_si128();
    for(int i = 0; i < 2 * h - 1; i++)
    {
      out[i]         = _mm_xor_si128(out[i], p0[i]);
      out[i + 2 * h] = _mm_xor_si128(out[i + 2 * h], p2[i]);
      out[i + h]     = _mm_xor_si128(out[i + h], _mm_xor_si128(pm[i], _mm_xor_si128(p0[i], p2[i])));
    }
  }
}

// reduction of two unreduced 128-bit values e, o to two 64-bit values (lower, upper half of result)
static inline __m128i reduce_pair(__m128i e, __m128i o, __m128i xp, __m128i lut)
{
  __m128i xe = _mm_clmulepi64_si128(e, xp, 0x01);
  __m128i xo = _mm_clmulepi64_si128(o, xp, 0x01);
  e = _mm_xor_si128(e, xe);
  o = _mm_xor_si128(o, xo);
  return _mm_xor_si128(_mm_unpacklo_epi64(e, o),
      _mm_unpackhi_epi64(_mm_shuffle_epi8(lut, xe), _mm_shuffle_epi8(lut, xo)));
}

// out <- a * b mod (x**R + x + c) in GF(2**64)[x]. out can be equal to a or b.
template<int R, int K> static inline void ring_mul(const uint64_t* a, const uint64_t* b, uint64_t c, uint64_t* out)
{
  __m128i p[2 * R - 1];
  kmul<R, K>(a, b, p);
  const __m128i xp = _mm_set1_epi64x(0x1b), lut = mg_lut128(), cv = _mm_set1_epi64x((long long) c);
  __m128i t[2 * R];
  for(int m = R; m < 2 * R - 1; m += 2)
  {
    __m128i rv = reduce_pair(p[m], (m + 1 < 2 * R - 1) ? p[m + 1] : _mm_setzero_si128(), xp, lut);
    t[m]     = _mm_clmulepi64_si128(rv, cv, 0x00);
    t[m + 1] = _mm_clmulepi64_si128(rv, cv, 0x01);
  }
  for(int k = 0; k < R; k += 2)
  {
    __m128i s0 = p[k], s1 = p[k + 1];
    if(k >= 1)     s0 = _mm_xor_si128(s0, p[k + R - 1]);
    s1 = _mm_xor_si128(s1, p[k + R]);
    if(k <= R - 2) s0 = _mm_xor_si128(s0, t[k + R]);
    if(k + 1 <= R - 2) s1 = _mm_xor_si128(s1, t[k + 1 + R]);
    _mm_storeu_si128((__m128i*)(out + k), reduce_pair(s0, s1, xp, lut));
  }
}

#ifdef MG_USE_YMM_CLMUL
// same for two rows at once: lane 0 of each 256-bit vector is for the first row, lane 1 for the second.
// 'SoA' layout: coefficient i of a row is in the low 64 bits of the lane of vector number i.
static inline __m256i cl2(__m256i a, __m256i b) {return _mm256_clmulepi64_epi128(a, b, 0x00);}

template<int N, int K> static inline void kmul2(const __m256i* a, const __m256i* b, __m256i* out)
{
  if constexpr(N == 1) out[0] = cl2(a[0], b[0]);
  else if constexpr(N <= K)
  {
    for(int i = 0; i < 2 * N - 1; i++) out[i] = _mm256_setzero_si256();
    for(int i = 0; i < N; i++)
      for(int j = 0; j < N; j++) out[i + j] = _mm256_xor_si256(out[i + j], cl2(a[i], b[j]));
  }
  else
  {
    constexpr int h = N / 2;
    __m256i as[h], bs[h], p0[2 * h - 1], p2[2 * h - 1], pm[2 * h - 1];
    kmul2<h, K>(a, b, p0);
    kmul2<h, K>(a + h, b + h, p2);
    for(int i = 0; i < h; i++)
    {
      as[i] = _mm256_xor_si256(a[i], a[i + h]);
      bs[i] = _mm256_xor_si256(b[i], b[i + h]);
    }
    kmul2<h, K>(as, bs, pm);
    for(int i = 0; i < 2 * N - 1; i++) out[i] = _mm256_setzero_si256();
    for(int i = 0; i < 2 * h - 1; i++)
    {
      out[i]         = _mm256_xor_si256(out[i], p0[i]);
      out[i + 2 * h] = _mm256_xor_si256(out[i + 2 * h], p2[i]);
      out[i + h]     = _mm256_xor_si256(out[i + h],
          _mm256_xor_si256(pm[i], _mm256_xor_si256(p0[i], p2[i])));
    }
  }
}

static inline __m256i reduce_pair2(__m256i e, __m256i o, __m256i xp, __m256i lut)
{
  __m256i xe = _mm256_clmulepi64_epi128(e, xp, 0x01);
  __m256i xo = _mm256_clmulepi64_epi128(o, xp, 0x01);
  e = _mm256_xor_si256(e, xe);
  o = _mm256_xor_si256(o, xo);
  return _mm256_xor_si256(_mm256_unpacklo_epi64(e, o),
      _mm256_unpackhi_epi64(_mm256_shuffle_epi8(lut, xe), _mm256_shuffle_epi8(lut, xo)));
}

template<int R> static inline void load_rows(const uint64_t* r0, const uint64_t* r1, __m256i* x)
{
  for(int i = 0; i < R; i += 2)
  {
    __m256i v = _mm256_loadu2_m128i((const __m128i*)(r1 + i), (const __m128i*)(r0 + i));
    x[i]     = v;
    x[i + 1] = _mm256_srli_si256(v, 8);
  }
}

// out0 <- a0 * b0 mod (x**R + x + c0), out1 <- a1 * b1 mod (x**R + x + c1). outputs can be equal to inputs.
template<int R, int K> static inline void ring_mul2(
    const uint64_t* a0, const uint64_t* b0, uint64_t c0, uint64_t* out0,
    const uint64_t* a1, const uint64_t* b1, uint64_t c1, uint64_t* out1)
{
  __m256i a[R], b[R], p[2 * R - 1];
  load_rows<R>(a0, a1, a);
  load_rows<R>(b0, b1, b);
  kmul2<R, K>(a, b, p);
  const __m256i xp = _mm256_set1_epi64x(0x1b);
  const __m256i lut = _mm256_broadcastsi128_si256(mg_lut128());
  const __m256i cv = _mm256_set_epi64x((long long) c1, (long long) c1, (long long) c0, (long long) c0);
  __m256i t[2 * R];
  for(int m = R; m < 2 * R - 1; m += 2)
  {
    __m256i rv = reduce_pair2(p[m], (m + 1 < 2 * R - 1) ? p[m + 1] : _mm256_setzero_si256(), xp, lut);
    t[m]     = _mm256_clmulepi64_epi128(rv, cv, 0x00);
    t[m + 1] = _mm256_clmulepi64_epi128(rv, cv, 0x01);
  }
  for(int k = 0; k < R; k += 2)
  {
    __m256i s0 = p[k], s1 = p[k + 1];
    if(k >= 1)     s0 = _mm256_xor_si256(s0, p[k + R - 1]);
    s1 = _mm256_xor_si256(s1, p[k + R]);
    if(k <= R - 2) s0 = _mm256_xor_si256(s0, t[k + R]);
    if(k + 1 <= R - 2) s1 = _mm256_xor_si256(s1, t[k + 1 + R]);
    _mm256_storeu2_m128i((__m128i*)(out1 + k), (__m128i*)(out0 + k), reduce_pair2(s0, s1, xp, lut));
  }
}
#endif

// Karatsuba/schoolbook switch-over (in number of coefficients), found by experiment
static constexpr int kt_ring4  = 1;
static constexpr int kt_ring16 = 4;

/**
 * @brief ring_rows
 * a[i-th row of R words] <- a[i] * b[i] mod (x**R + x + c_i), i < 2**(logsize - log2(R)), with
 * c_i = c ^ (beta image of i); see above.
 * @param c
 * c_0, i.e. the multiplicative representation of (offset >> log2(R)) where 'offset' is the offset
 * of the first value in beta representation.
 */
template<int R, int K> static void ring_rows(uint64_t* a, const uint64_t* b, unsigned int logsize, uint64_t c)
{
  constexpr unsigned int lr = (R == 4) ? 2 : 4;
  const uint64_t nrows = 1uLL << (logsize - lr);
  for(uint64_t q = 0; q < nrows; q += 2)
  {
    // q is even: c_{q+1} = c_q ^ beta_0 (beta image of 1), then c_{q+2} = c_{q+1} ^ cumulative[h]
    const uint64_t c1 = c ^ beta_over_mult_cumulative[1];
#ifdef MG_USE_YMM_CLMUL
    ring_mul2<R, K>(a + R * q, b + R * q, c, a + R * q, a + R * (q + 1), b + R * (q + 1), c1, a + R * (q + 1));
#else
    ring_mul<R, K>(a + R * q, b + R * q, c, a + R * q);
    ring_mul<R, K>(a + R * (q + 1), b + R * (q + 1), c1, a + R * (q + 1));
#endif
    c = c1 ^ beta_over_mult_cumulative[_mm_popcnt_u64((q + 1) ^ (q + 2))];
  }
}

static void mg_ring_rows(uint64_t* a, const uint64_t* b, unsigned int logsize, uint64_t c, int cut)
{
  if(cut == 16) ring_rows<16, kt_ring16>(a, b, logsize, c);
  else          ring_rows<4,  kt_ring4>(a, b, logsize, c);
}

/**
 * @brief mg_cut_level
 * 'cut' parameter of the functions below: 0 for a plain product (as in mg.cpp), 16 or 4 to stop
 * DFTs before the last stages on rows of 16 or 4 words, and to replace these stages (and the
 * pointwise product and the first stages of the inverse DFT) by ring_rows.
 * Requires that all DFTs have at least 2**5 (resp. 2**3) values.
 */
static int mg_cut_level(unsigned int logsize_1, unsigned int logsize_2, unsigned int logsize_result)
{
  const unsigned int m = min(min(logsize_1, logsize_2), logsize_result);
#ifdef MG_CUT_LEVEL
  const int forced = MG_CUT_LEVEL;
  if(forced == 16 && m < 5) return (m >= 3) ? 4 : 0;
  if(forced == 4 && m < 3) return 0;
  return forced;
#else
  if(m >= 5) return 16;
  if(m >= 3) return 4;
  return 0;
#endif
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
    bool first_taylor_done,
    int cut = 0)
{
  if constexpr(s == 0)
  {
    eval_degree1<false, logstride>(offsets_mult[0], poly);
  }
#if MG_ROW4
  else if constexpr(s == 1 && logstride == 0)
  {
    if(logsize == 2)
    {
      if(first_taylor_done) mg_row4_forward<false>(poly, offsets_mult);
      else                  mg_row4_forward<true>(poly, offsets_mult);
    }
    else
    {
      mg_core<0, 0>(poly, offsets_mult, logsize, first_taylor_done);
    }
  }
#endif
  else
  {
    constexpr unsigned int t = 1 << (s - 1);
    if(logsize <= t)
    {
      mg_core<s - 1, logstride>(poly, offsets_mult, logsize, first_taylor_done, cut);
    }
    else
    {
      // on input: there are 2**'logstride' interleaved series of size 'eta' = 2**(2*logsize-t);
      if(!first_taylor_done)
      {
        mg_decompose_taylor_recursive<logstride, t, uint64_t>(logsize, poly);
      }
      // fft on columns
      // if logsize >= t, each fft should process 2**(logsize - t) values
      // i.e. logsize' = logsize - t
      // offset' = offset >> t;
      mg_core<s - 1, logstride + t>(poly, offsets_mult + t, logsize - t, false, cut);
      uint64_t offsets_local[t];
      for(unsigned int j = 0; j < t; j++) offsets_local[j] = offsets_mult[j];
      // 2*t >= logsize > t, therefore tau = 2**(logsize - t) <= 2**t
      const uint64_t tau   = 1uLL <<  (logsize - t);
#ifndef GRAY_CODE
      const uint64_t row_size = 1uLL << (t + logstride);
      for(uint64_t i = 0; i < (((logstride == 0) && ((s == 2 && cut == 4) || (s == 3 && cut == 16))) ? 0 : tau); i++)
      {
        // at each iteration, offsets_local[j] = beta_to_mult(offset + (i << (t-j))), j < t
        mg_core<s - 1, logstride>(poly, offsets_local, t, false, cut);
        const int h = (int) _mm_popcnt_u64(i^(i+1)); // i^(i+1) is a power of 2 - 1
        for(unsigned int j = 0; j < t; j++)
        {
          int tp = t - j;
          offsets_local[j] ^= beta_over_mult_cumulative[h + tp] ^ beta_over_mult_cumulative[tp];
        }
        poly += row_size;
      }
#else
      for(uint64_t k = 0; k < tau; k++)
      {
        mg_core<s - 1, logstride>(poly + ((k ^ (k >> 1)) << (t + logstride)), offsets_local, t, false);
        const int h = (int) _tzcnt_u64(~k) + t; // 1 << (h-t) is equal to k ^ (k >> 1) ^ (k+1) ^ ((k+1) >> 1)
        for(unsigned int j = 0; j < t; j++) offsets_local[j] ^= beta_over_mult[h - j];
      }
#endif
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
    unsigned int logsize, // size of the DFT to be computed, <= 2**(2**s)
    int cut = 0
    )
{
  if constexpr(s > 0)
  {
    if(logsizeprime <= (1u << (s-1)))
    {
      mg_smalldegree<s-1>(p, logsizeprime, logsize, cut);
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
      mg_core<s, 0>(q, offsets_mult, logsizeprime, true, cut);
      const int h = (int) _mm_popcnt_u64(i^(i+1));
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
    mg_core<0, 0>(p, offsets_mult, logsize, false);
  }
}

template <unsigned int s>
void mg_smalldegree_with_buf(
    uint64_t* p, // binary polynomial, unexpanded, of degree < 2**logsize_prime **once expanded**
    unsigned int logsizeprime, // log2 bound on the polynomial number of coefficents, <= logsize
    unsigned int logsize, // size of the DFT to be computed, <= 2**(2**s)
    uint64_t* main_buf, // buffer of size 2**logsize, whose elements wil be termwise multiplied by DFT of p
    int cut = 0
    )
{
  if constexpr(s > 0)
  {
    if(logsizeprime <= (1u << (s-1)))
    {
      mg_smalldegree_with_buf<s-1>(p, logsizeprime, logsize, main_buf, cut);
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
      mg_core<s, 0>(buf, offsets_mult, logsizeprime, true, cut);
      // logsizeprime >= 2 (see above)
      if(cut != 0) mg_ring_rows(main_buf + (i << logsizeprime), buf, logsizeprime, offsets_mult[cut == 16 ? 4 : 2], cut);
      else product_batch(main_buf + (i << logsizeprime), buf, logsizeprime);
      const int h = (int) _mm_popcnt_u64(i^(i+1));
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
    unsigned int logsize,
    int cut = 0)
{
  if constexpr(s == 0)
  {
    eval_degree1<true, logstride>(offsets_mult[0], poly);
  }
#if MG_ROW4
  else if constexpr(s == 1 && logstride == 0)
  {
    if(logsize == 2) mg_row4_reverse(poly, offsets_mult);
    else             mg_reverse_core<0, 0>(poly, offsets_mult, logsize);
  }
#endif
  else
  {
    constexpr unsigned int t = 1 << (s - 1);
    if(logsize <= t)
    {
      mg_reverse_core<s - 1, logstride>(poly, offsets_mult, logsize, cut);
    }
    else
    {
      uint64_t offsets_local[t];
      for(unsigned int j = 0; j < t; j++) offsets_local[j] = offsets_mult[j];
      // 2*t >= logsize > t, therefore tau < 2**t
      const uint64_t tau   = 1uLL <<  (logsize - t);
#ifndef GRAY_CODE
      const uint64_t row_size = 1uLL << (t + logstride);
      uint64_t* poly_loc = poly;
      for(uint64_t i = 0; i < (((logstride == 0) && ((s == 2 && cut == 4) || (s == 3 && cut == 16))) ? 0 : tau); i++)
      {
        // at each iteration, offsets_local[j] = beta_to_mult(offset + (i << (t-j))), j < t
        mg_reverse_core<s - 1, logstride>(poly_loc, offsets_local, t, cut);
        const int h = (int) _mm_popcnt_u64(i^(i+1));
        for(unsigned int j = 0; j < t; j++)
        {
          int tp = t - j;
          offsets_local[j] ^= beta_over_mult_cumulative[h + tp] ^ beta_over_mult_cumulative[tp];
        }
        poly_loc += row_size;
      }
#else
      for(uint64_t k = 0; k < tau; k++)
      {
        mg_reverse_core<s - 1, logstride>(poly + ((k ^ (k >> 1)) << (t + logstride)), offsets_local, t);
        const int h = (int) _tzcnt_u64(~k) + t; // 1 << (h-t) is equal to k ^ (k >> 1) ^ (k+1) ^ ((k+1) >> 1)
        for(unsigned int j = 0; j < t; j++) offsets_local[j] ^= beta_over_mult[h - j];
      }
#endif
      // reverse fft on columns
      //offset' = offset >> t;
      mg_reverse_core<s - 1, logstride + t>(poly, offsets_mult + t, logsize - t, cut);
      mg_decompose_taylor_reverse_recursive<logstride, t>(logsize, poly);
    }
  }
}

void mg_binary_polynomial_multiply(uint64_t* p1, uint64_t* p2, uint64_t* result, uint64_t d1, uint64_t d2)
{
  if(!beta_is_initialized) init_beta_table();
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
  // operands of at most 2 32-bit words are handled as 4-word polynomials unless the result is small
  if(logsize_result >= 2)
  {
    logsize_1 = max(logsize_1, 2u);
    logsize_2 = max(logsize_2, 2u);
  }
  const int cut = mg_cut_level(logsize_1, logsize_2, logsize_result);
  mg_smalldegree<s>(b1, logsize_1, logsize_result, cut);
  mg_smalldegree<s>(b2, logsize_2, logsize_result, cut);
  if(cut != 0) mg_ring_rows(b1, b2, logsize_result, 0, cut);
  else if(logsize_result < 2)
  {
    for(size_t i = 0; i < sz; i++) b1[i] = product(b1[i], b2[i]);
  }
  else
  {
    product_batch(b1, b2, logsize_result);
  }
  mg_reverse_core<s,0>(b1, offsets_mult, logsize_result, cut);
  contract(result, b1, d1 + d2);
}

void mg_binary_polynomial_multiply_in_place (uint64_t* p1, uint64_t* p2, uint64_t d1, uint64_t d2)
{
  if(!beta_is_initialized) init_beta_table();
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
  if(logsize_result < 2)
  {
    // the result has at most 64 bits: a single carry-less product
    __m128i x = _mm_clmulepi64_si128(_mm_cvtsi64_si128((long long) p1[0]), _mm_cvtsi64_si128((long long) p2[0]), 0);
    p1[0] = (uint64_t) _mm_cvtsi128_si64(x);
    return;
  }
  expand(p1, p1, d1, sz_result);
  // operands of at most 2 32-bit words are handled as 4-word polynomials
  logsize_1 = max(logsize_1, 2u);
  uint64_t small_p2[2] = {0, 0};
  if(logsize_2 < 2)
  {
    // p2 may not have room for 4 32-bit words: work on a copy
    small_p2[0] = p2[0];
    p2 = small_p2;
    logsize_2 = 2;
  }
  const int cut = mg_cut_level(logsize_1, logsize_2, logsize_result);
  mg_smalldegree<s>((uint64_t*) p1, logsize_1, logsize_result, cut);
  mg_smalldegree_with_buf<s>(p2, logsize_2, logsize_result, p1, cut);
  mg_reverse_core<s,0>(p1, offsets_mult, logsize_result, cut);
  contract(p1, p1, d1 + d2);
}

} // namespace mg_fused
