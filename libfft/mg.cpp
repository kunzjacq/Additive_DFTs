#include <memory>    // for unique_ptr
#include <algorithm> // for min, max

#include <immintrin.h>
#include "mg.h"


constexpr int inline_bound = 8;

#include "mg_templates.h"

using namespace std;

//#define GRAY_CODE

//#define NEW_DECOMPOSE_TAYLOR


/** beta_over_mult_cumulative[i] = sum_j = 0 ... i - 1 beta_i
 * in multplicative representation, i.e. in GF(2**64) as described in
 * function 'product'. beta_i refers to the beta basis in Mateer-Gao algorithm.
 */
static constexpr uint64_t beta_over_mult_cumulative[] = {
  0x0,               0x1,               0x19c9369f278adc03,0xb848d14948d52b96,
  0xfc39a481a127aa9d,0xfd02a1ae1c3b51c0,0x4ae8fb39198c2000, 0xc9e636090aafc01,
  0x160266090832ba8e,0x0698fbda5886d236,0x1ed3c0aa2cc028ce,0xa3f2c7d05d050384,
  0x5ebd73abdf25bd68,0xba04271ef7f3b56a,0x49d80fdbb0b388c2,0xa468c40881568f18,
  0xf6e2dc67ed2203db,0xf4466dee270e5882,0xf6dc10cf295d7ed4,0x5b7fd08b070abd39,
  0xa7fe9595d9b22788,0x4738e503d11d72f5,0xb49e3d1bb4227e58,0xf2b4dbb998c79ed7,
  0xecd3a59c347e16e3,0xe1cfc3d8505a6e77,0xea670e1a534323cf,0x4d4ce92c49c90db4,
  0xa5b54a92be769797,0x5870d894c2f1596c,0xe69061ed22d5070 ,0xbe658be205cd5c43,
  0x516e45eea5755655,0xaac030ebd33319f8,0x495ed9b5fabbaf08,0x1695161ab572f7b4,
  0xadacad2c4e48cdda,0xe45844b9ebae6c40,0x5d36626173c3a040,0xa0264fa8d4864a8e,
  0x5c071e3bcdce3e37,0x13303c7fe8fafb31,0x07f46edf72e59233,0x029ef3c53091b2d8,
  0x1ed4dd260daede14,0x153ffd2658c0f389,0xb31cf4415e740389,0x5af7b780a8b08387,
  0x151dffedf9e1ddbc,0xb79973326bd3fbdf,0x42a28f58bdc05714,0x1fb6289097568c04,
  0x140379b253de3f29,0x010d3f8d779ee957,0xb33e1fc53acedca3,0xe8fd5ebe929529ba,
  0xf8387e3f53d81dad,0x4e34099e141594fe,0x15cc6fee9e51daf7,0x0185be136e4c8181,
  0xb3b50cfd392bc4b1,0xf54ebfb28d2f23cc,0xebeea0f9242d3352,0xe667293b4ee307b4,
  0x6d1861bfc6fbe3e8
};

#ifdef GRAY_CODE
static constexpr uint64_t beta_over_mult[] = {
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
#endif

void naive_product(uint64_t* p1u, uint64_t n1, uint64_t* p2u, uint64_t n2, uint64_t* qu)
{
  __m128i* restr p1 = (__m128i*) std::assume_aligned<16>(p1u);
  __m128i* restr p2 = (__m128i*) std::assume_aligned<16>(p2u);
  __m128i* restr q  = (__m128i*) std::assume_aligned<16>(qu);
  for(uint64_t i = 0; i < n1 + n2; i++) qu[i] = 0;
  n1 = (n1 + 1) >> 1;
  n2 = (n2 + 1) >> 1;
  for(uint64_t i = 0; i < n1; i++)
  {
    __m128i x = _mm_load_si128(p1 + i);
    __m128i next = _mm_set1_epi8(0x0);
    for(uint64_t j = 0; j < n2; j++)
    {
      __m128i y = _mm_load_si128(p2 + j);
      __m128i z1 = _mm_clmulepi64_si128(x, y, 0x00);
      __m128i z2 = _mm_clmulepi64_si128(x, y, 0x01);
      //z1 ^= next;
      z1 = _mm_xor_si128(z1, next);
      __m128i z4 = _mm_clmulepi64_si128(x, y, 0x11);
      z2 = _mm_xor_si128(z2, _mm_clmulepi64_si128(x, y, 0x10));
      __m128i z3 = _mm_set_epi64x(_mm_extract_epi64(z2, 0), 0x0);
      __m128i z5 = _mm_set_epi64x(0x0, _mm_extract_epi64(z2, 1));
      z1 = _mm_xor_si128(z1, z3);
      next = _mm_xor_si128(z4, z5);
      q[i + j] = _mm_xor_si128(q[i + j], z1);
    }
    q[i + n2] = _mm_xor_si128(q[i + n2], next);
  }
}

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
  return _mm_extract_epi64(xb, 0);
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
    xa2 = _mm_shuffle_epi32(xa1, 0x4e); // put higher part of xa1 in xa2
    xa4 = _mm_shuffle_epi32(xa3, 0x4e); // (dwords 3, 2, in lower part, 3*4+2 = e,
    xb2 = _mm_shuffle_epi32(xb1, 0x4e); // 1, 0 in higher part, 1 * 4 + 0 = 4)
    xb4 = _mm_shuffle_epi32(xb3, 0x4e);
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

template <unsigned int logsize>
extern inline force_inline void mg_core_inlined(
    int logstride,
    uint64_t* poly,
    const uint64_t* offsets_mult,
    bool first_taylor_done)
{
  if constexpr(logsize <= 1)
  {
    eval_degree1<false>(logstride, offsets_mult[0], poly);
  }
  else
  {
    constexpr unsigned int t = comp_t(logsize); // power 2**u maximal s.t. < logsize
    // on input: there are 2**'logstride' interleaved series of size 2**(2*logsize-t)
    if(!first_taylor_done)
    {
#ifdef NEW_DECOMPOSE_TAYLOR
      mg_decompose_taylor_recursive_alt<logsize, uint64_t>(logstride, poly);
#else
      mg_decompose_taylor_recursive<t, uint64_t>(logsize, logstride, poly);
#endif
    }
    // fft on columns
    // if logsize >= t, each fft should process 2**(logsize - t) values
    // i.e. logsize' = logsize - t
    // offset' = offset >> t;
    mg_core_inlined<logsize - t>(logstride + t, poly, offsets_mult + t, false);
    uint64_t offsets_local[t];
    for(unsigned int j = 0; j < t; j++) offsets_local[j] = offsets_mult[j];
    // 2*t >= logsize > t, therefore tau = 2**(logsize - t) <= 2**t
    constexpr uint64_t tau   = 1uLL <<  (logsize - t);
#ifndef GRAY_CODE
    const uint64_t row_size = 1uLL << (t + logstride);
    for(uint64_t i = 0; i < tau; i++)
    {
      // at each iteration, offsets_local[j] = beta_to_mult(offset + (i << (t-j))), j < t
      mg_core_inlined<t>(logstride, poly, offsets_local, false);
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
      mg_core<t>(logstride, poly + ((k ^ (k >> 1)) << (t + logstride)), offsets_local, false);
      const int h = (int) _tzcnt_u64(~k) + t; // 1 << (h-t) is equal to k ^ (k >> 1) ^ (k+1) ^ ((k+1) >> 1)
      for(unsigned int j = 0; j < t; j++) offsets_local[j] ^= beta_over_mult[h - j];
    }
#endif
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
template <unsigned int logsize>
void mg_core(
    int logstride,
    uint64_t* poly,
    const uint64_t* offsets_mult,
    bool first_taylor_done)
{
  if constexpr(logsize <= 1)
  {
    eval_degree1<false>(logstride, offsets_mult[0], poly);
  }
  else
  {
    constexpr unsigned int t = comp_t(logsize); // power 2**u maximal s.t. < logsize
    // on input: there are 2**'logstride' interleaved series of size 2**(2*logsize-t)
    if(!first_taylor_done)
    {
#ifdef NEW_DECOMPOSE_TAYLOR
      mg_decompose_taylor_recursive_alt<logsize, uint64_t>(logstride, poly);
#else
      mg_decompose_taylor_recursive<t, uint64_t>(logsize, logstride, poly);
#endif
    }
    // fft on columns
    // if logsize >= t, each fft should process 2**(logsize - t) values
    // i.e. logsize' = logsize - t
    // offset' = offset >> t;
    constexpr int tp = logsize - t;
    if constexpr(tp <= inline_bound)
    {
      mg_core_inlined<logsize - t>(logstride + t, poly, offsets_mult + t, false);
    }
    else
    {
      mg_core<logsize - t>(logstride + t, poly, offsets_mult + t, false);
    }
    uint64_t offsets_local[t];
    for(unsigned int j = 0; j < t; j++) offsets_local[j] = offsets_mult[j];
    // 2*t >= logsize > t, therefore tau = 2**(logsize - t) <= 2**t
    constexpr uint64_t tau   = 1uLL <<  (logsize - t);
#ifndef GRAY_CODE
    const uint64_t row_size = 1uLL << (t + logstride);
    for(uint64_t i = 0; i < tau; i++)
    {
      // at each iteration, offsets_local[j] = beta_to_mult(offset + (i << (t-j))), j < t
      if constexpr(t <= inline_bound)
      {
        mg_core_inlined<t>(logstride, poly, offsets_local, false);
      }
      else
      {
        mg_core<t>(logstride, poly, offsets_local, false);
      }
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
      mg_core<t>(logstride, poly + ((k ^ (k >> 1)) << (t + logstride)), offsets_local, false);
      const int h = (int) _tzcnt_u64(~k) + t; // 1 << (h-t) is equal to k ^ (k >> 1) ^ (k+1) ^ ((k+1) >> 1)
      for(unsigned int j = 0; j < t; j++) offsets_local[j] ^= beta_over_mult[h - j];
    }
#endif
  }
}

#if 0
// experiments with generic constification


template <unsigned int logsize>
class mg_core_class
{
public:
  static void call(int logstride, uint64_t* poly, const uint64_t* offsets_mult,
                  bool first_taylor_done)
  {
    mg_core<logsize>(
        logstride,
        poly,
        offsets_mult,
        first_taylor_done);
  }
};

template <template<unsigned int, class ... Ts2> class T, class... Ts2, class...Ts>
void constify_alt2(unsigned int n, Ts ... args)
{
  if(n==0) T<0, Ts2...>::call(args ...);
  else if(n==1) T<1, Ts2...>::call(args...);
}

template <template<int> class T, class...Ts>
void constify_alt3(int n, Ts ... args)
{
  if(n==0) T<0>::call(args ...);
  else if(n==1) T<1>::call(args...);
}

void f(int logstride, int n, uint64_t* poly, const uint64_t* offsets_mult, bool first_taylor_done)
{
  //does not work because the compiler is unable to delimit Ts and Ts2
  //constify_alt2<mg_core_class,  int, uint64_t* , const uint64_t* , bool>(n, logstride, poly, offsets_mult, first_taylor_done);
  //works
  constify_alt3<mg_core_class,  int, uint64_t* , const uint64_t* , bool>(n, logstride, poly, offsets_mult, first_taylor_done);
}
#endif

void constify_mg(int logstride, int n, uint64_t* poly, const uint64_t* offsets_mult, bool first_taylor_done)
{
  if(n==0) mg_core<0>(logstride, poly, offsets_mult, first_taylor_done);
  else if(n==1) mg_core<1>(logstride, poly, offsets_mult, first_taylor_done);
  else if(n==2) mg_core<2>(logstride, poly, offsets_mult, first_taylor_done);
  else if(n==3) mg_core<3>(logstride, poly, offsets_mult, first_taylor_done);
  else if(n==4) mg_core<4>(logstride, poly, offsets_mult, first_taylor_done);
  else if(n==5) mg_core<5>(logstride, poly, offsets_mult, first_taylor_done);
  else if(n==6) mg_core<6>(logstride, poly, offsets_mult, first_taylor_done);
  else if(n==7) mg_core<7>(logstride, poly, offsets_mult, first_taylor_done);
  else if(n==8) mg_core<8>(logstride, poly, offsets_mult, first_taylor_done);
  else if(n==9) mg_core<9>(logstride, poly, offsets_mult, first_taylor_done);
  else if(n==10) mg_core<10>(logstride, poly, offsets_mult, first_taylor_done);
  else if(n==11) mg_core<11>(logstride, poly, offsets_mult, first_taylor_done);
  else if(n==12) mg_core<12>(logstride, poly, offsets_mult, first_taylor_done);
  else if(n==13) mg_core<13>(logstride, poly, offsets_mult, first_taylor_done);
  else if(n==14) mg_core<14>(logstride, poly, offsets_mult, first_taylor_done);
  else if(n==15) mg_core<15>(logstride, poly, offsets_mult, first_taylor_done);
  else if(n==16) mg_core<16>(logstride, poly, offsets_mult, first_taylor_done);
  else if(n==17) mg_core<17>(logstride, poly, offsets_mult, first_taylor_done);
  else if(n==18) mg_core<18>(logstride, poly, offsets_mult, first_taylor_done);
  else if(n==19) mg_core<19>(logstride, poly, offsets_mult, first_taylor_done);
  else if(n==20) mg_core<20>(logstride, poly, offsets_mult, first_taylor_done);
  else if(n==21) mg_core<21>(logstride, poly, offsets_mult, first_taylor_done);
  else if(n==22) mg_core<22>(logstride, poly, offsets_mult, first_taylor_done);
  else if(n==23) mg_core<23>(logstride, poly, offsets_mult, first_taylor_done);
  else if(n==24) mg_core<24>(logstride, poly, offsets_mult, first_taylor_done);
  else if(n==25) mg_core<25>(logstride, poly, offsets_mult, first_taylor_done);
  else if(n==26) mg_core<26>(logstride, poly, offsets_mult, first_taylor_done);
  else if(n==27) mg_core<27>(logstride, poly, offsets_mult, first_taylor_done);
  else if(n==28) mg_core<28>(logstride, poly, offsets_mult, first_taylor_done);
  else if(n==29) mg_core<29>(logstride, poly, offsets_mult, first_taylor_done);
  else if(n==30) mg_core<30>(logstride, poly, offsets_mult, first_taylor_done);
  else if(n==31) mg_core<31>(logstride, poly, offsets_mult, first_taylor_done);
  else if(n==32) mg_core<32>(logstride, poly, offsets_mult, first_taylor_done);
  else if(n==33) mg_core<33>(logstride, poly, offsets_mult, first_taylor_done);
  else if(n==34) mg_core<34>(logstride, poly, offsets_mult, first_taylor_done);
  else if(n==35) mg_core<35>(logstride, poly, offsets_mult, first_taylor_done);
  else if(n==36) mg_core<36>(logstride, poly, offsets_mult, first_taylor_done);
  else if(n==37) mg_core<37>(logstride, poly, offsets_mult, first_taylor_done);
  else if(n==38) mg_core<38>(logstride, poly, offsets_mult, first_taylor_done);
  else if(n==39) mg_core<39>(logstride, poly, offsets_mult, first_taylor_done);
  else if(n==40) mg_core<40>(logstride, poly, offsets_mult, first_taylor_done);
}

template<class T>
void constify_mg_decompose_taylor_recursive_alt(unsigned int logsize, unsigned int logstride, T* p)
{
  if(logsize==0) mg_decompose_taylor_recursive_alt<0, T>(logstride, p);
  else if(logsize==1) mg_decompose_taylor_recursive_alt<1>(logstride, p);
  else if(logsize==2) mg_decompose_taylor_recursive_alt<2>(logstride, p);
  else if(logsize==3) mg_decompose_taylor_recursive_alt<3>(logstride, p);
  else if(logsize==4) mg_decompose_taylor_recursive_alt<4>(logstride, p);
  else if(logsize==5) mg_decompose_taylor_recursive_alt<5>(logstride, p);
  else if(logsize==6) mg_decompose_taylor_recursive_alt<6>(logstride, p);
  else if(logsize==7) mg_decompose_taylor_recursive_alt<7>(logstride, p);
  else if(logsize==8) mg_decompose_taylor_recursive_alt<8>(logstride, p);
  else if(logsize==9) mg_decompose_taylor_recursive_alt<9>(logstride, p);
  else if(logsize==10) mg_decompose_taylor_recursive_alt<10>(logstride, p);
  else if(logsize==11) mg_decompose_taylor_recursive_alt<11>(logstride, p);
  else if(logsize==12) mg_decompose_taylor_recursive_alt<12>(logstride, p);
  else if(logsize==13) mg_decompose_taylor_recursive_alt<13>(logstride, p);
  else if(logsize==14) mg_decompose_taylor_recursive_alt<14>(logstride, p);
  else if(logsize==15) mg_decompose_taylor_recursive_alt<15>(logstride, p);
  else if(logsize==16) mg_decompose_taylor_recursive_alt<16>(logstride, p);
  else if(logsize==17) mg_decompose_taylor_recursive_alt<17>(logstride, p);
  else if(logsize==18) mg_decompose_taylor_recursive_alt<18>(logstride, p);
  else if(logsize==19) mg_decompose_taylor_recursive_alt<19>(logstride, p);
  else if(logsize==20) mg_decompose_taylor_recursive_alt<20>(logstride, p);
  else if(logsize==21) mg_decompose_taylor_recursive_alt<21>(logstride, p);
  else if(logsize==22) mg_decompose_taylor_recursive_alt<22>(logstride, p);
  else if(logsize==23) mg_decompose_taylor_recursive_alt<23>(logstride, p);
  else if(logsize==24) mg_decompose_taylor_recursive_alt<24>(logstride, p);
  else if(logsize==25) mg_decompose_taylor_recursive_alt<25>(logstride, p);
  else if(logsize==26) mg_decompose_taylor_recursive_alt<26>(logstride, p);
  else if(logsize==27) mg_decompose_taylor_recursive_alt<27>(logstride, p);
  else if(logsize==28) mg_decompose_taylor_recursive_alt<28>(logstride, p);
  else if(logsize==29) mg_decompose_taylor_recursive_alt<29>(logstride, p);
  else if(logsize==30) mg_decompose_taylor_recursive_alt<30>(logstride, p);
  else if(logsize==31) mg_decompose_taylor_recursive_alt<31>(logstride, p);
  else if(logsize==32) mg_decompose_taylor_recursive_alt<32>(logstride, p);
  else if(logsize==33) mg_decompose_taylor_recursive_alt<33>(logstride, p);
  else if(logsize==34) mg_decompose_taylor_recursive_alt<34>(logstride, p);
  else if(logsize==35) mg_decompose_taylor_recursive_alt<35>(logstride, p);
  else if(logsize==36) mg_decompose_taylor_recursive_alt<36>(logstride, p);
  else if(logsize==37) mg_decompose_taylor_recursive_alt<37>(logstride, p);
  else if(logsize==38) mg_decompose_taylor_recursive_alt<38>(logstride, p);
  else if(logsize==39) mg_decompose_taylor_recursive_alt<39>(logstride, p);
  else if(logsize==40) mg_decompose_taylor_recursive_alt<40>(logstride, p);
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
#ifdef NEW_DECOMPOSE_TAYLOR
    constify_mg_decompose_taylor_recursive_alt<uint64_t>(logsizeprime, 0u, p);
#else
    constexpr unsigned int t = 1u << (s - 1);
    mg_decompose_taylor_recursive<t, uint64_t>(logsizeprime, 0, p);
#endif

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
      //mg_core<s, 0>(q, offsets_mult, logsizeprime, true);
      constify_mg (0, logsizeprime, q, offsets_mult, true);
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
    //mg_core<0, 0>(p, offsets_mult, logsize, false);
    constify_mg(0, logsize, p, offsets_mult, true);
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

    const uint64_t szp = 1uLL << logsizeprime;
    // the natural order of operations is to expand p (expand every 32-bit word into q 64-bit word),
    // then apply mg_decompose_taylor_recursive on 64-bit words
    // this is equivalent to doing mg_decompose_taylor_recursive on 32-bit words, then expanding
#ifdef NEW_DECOMPOSE_TAYLOR
    constify_mg_decompose_taylor_recursive_alt<uint32_t>(logsizeprime, 0u, (uint32_t*)p);
#else
    constexpr unsigned int t = 1u << (s - 1);
    mg_decompose_taylor_recursive<t, uint32_t>(logsizeprime, 0u, (uint32_t*)p);
#endif
    uint64_t offsets_mult[1 << s];
    for(int i = 0; i < (1 << s); i++) offsets_mult[i] = 0;

    for(uint64_t i = 0; i < 1uLL << (logsize - logsizeprime); i++)
    {
      expand(buf, p, szp * 32 - 1, szp);
      //mg_core<s, 0>(buf, offsets_mult, logsizeprime, true);
      constify_mg (0, logsizeprime, buf, offsets_mult, true);
      // logsizeprime >= 2 (see above)
      product_batch(main_buf + (i << logsizeprime), buf, logsizeprime);
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
    //mg_core<0, 0>(p, offsets_mult, logsize, false);
    constify_mg(0, logsize, p, offsets_mult, false);
    for(int i = 0; i < 1 << logsize; i++) main_buf[i] = product(main_buf[i], p[i]);
  }
}

template <unsigned int logsize>
extern inline force_inline void mg_reverse_core_inlined(
    int logstride,
    uint64_t* poly,
    const uint64_t* offsets_mult)
{
  if constexpr(logsize <= 1)
  {
    eval_degree1<true>(logstride, offsets_mult[0], poly);
  }
  else
  {
    constexpr unsigned int t = comp_t(logsize);
    if constexpr(logsize <= t)
    {
      mg_reverse_core_inlined<logsize>(logstride, poly, offsets_mult);
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
      for(uint64_t i = 0; i < tau; i++)
      {
        // at each iteration, offsets_local[j] = beta_to_mult(offset + (i << (t-j))), j < t
        mg_reverse_core_inlined<t>(logstride, poly_loc, offsets_local);
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
        mg_reverse_core_inlined<t>(logstride, poly + ((k ^ (k >> 1)) << (t + logstride)), offsets_local);
        const int h = (int) _tzcnt_u64(~k) + t; // 1 << (h-t) is equal to k ^ (k >> 1) ^ (k+1) ^ ((k+1) >> 1)
        for(unsigned int j = 0; j < t; j++) offsets_local[j] ^= beta_over_mult[h - j];
      }
#endif
      // reverse fft on columns
      //offset' = offset >> t;
      mg_reverse_core_inlined<logsize - t>(logstride + t, poly, offsets_mult + t);
      mg_decompose_taylor_reverse_recursive<t>(logstride, logsize, poly);
    }
  }
}

/**
 * @brief mg_reverse_core
 * Reverse of mg_core. See mg_core for argument description.
 */
template <unsigned int logsize>
void mg_reverse_core(
    int logstride,
    uint64_t* poly,
    const uint64_t* offsets_mult)
{
  if constexpr(logsize <= 1)
  {
    eval_degree1<true>(logstride, offsets_mult[0], poly);
  }
  else
  {
    constexpr unsigned int t = comp_t(logsize);
    if constexpr(logsize <= t)
    {
      if constexpr(logsize <= inline_bound)
      {
        mg_reverse_core_inlined<logsize>(logstride, poly, offsets_mult);
      }
      else
      {
        mg_reverse_core<logsize>(logstride, poly, offsets_mult);
      }
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
      for(uint64_t i = 0; i < tau; i++)
      {
        // at each iteration, offsets_local[j] = beta_to_mult(offset + (i << (t-j))), j < t
        if constexpr(t <= inline_bound)
        {
          mg_reverse_core_inlined<t>(logstride, poly_loc, offsets_local);
        }
        else
        {
          mg_reverse_core<t>(logstride, poly_loc, offsets_local);
        }
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
        mg_reverse_core<t>(logstride, poly + ((k ^ (k >> 1)) << (t + logstride)), offsets_local);
        const int h = (int) _tzcnt_u64(~k) + t; // 1 << (h-t) is equal to k ^ (k >> 1) ^ (k+1) ^ ((k+1) >> 1)
        for(unsigned int j = 0; j < t; j++) offsets_local[j] ^= beta_over_mult[h - j];
      }
#endif
      // reverse fft on columns
      //offset' = offset >> t;
      mg_reverse_core<logsize - t>(logstride + t, poly, offsets_mult + t);
      mg_decompose_taylor_reverse_recursive<t>(logstride, logsize, poly);
    }
  }
}

void constify_mgr(int n, int logstride, uint64_t* poly, const uint64_t* offsets_mult)
{
  if(n==0) mg_reverse_core<0>(logstride, poly, offsets_mult);
  else if(n==1) mg_reverse_core<1>(logstride, poly, offsets_mult);
  else if(n==2) mg_reverse_core<2>(logstride, poly, offsets_mult);
  else if(n==3) mg_reverse_core<3>(logstride, poly, offsets_mult);
  else if(n==4) mg_reverse_core<4>(logstride, poly, offsets_mult);
  else if(n==5) mg_reverse_core<5>(logstride, poly, offsets_mult);
  else if(n==6) mg_reverse_core<6>(logstride, poly, offsets_mult);
  else if(n==7) mg_reverse_core<7>(logstride, poly, offsets_mult);
  else if(n==8) mg_reverse_core<8>(logstride, poly, offsets_mult);
  else if(n==9) mg_reverse_core<9>(logstride, poly, offsets_mult);
  else if(n==10) mg_reverse_core<10>(logstride, poly, offsets_mult);
  else if(n==11) mg_reverse_core<11>(logstride, poly, offsets_mult);
  else if(n==12) mg_reverse_core<12>(logstride, poly, offsets_mult);
  else if(n==13) mg_reverse_core<13>(logstride, poly, offsets_mult);
  else if(n==14) mg_reverse_core<14>(logstride, poly, offsets_mult);
  else if(n==15) mg_reverse_core<15>(logstride, poly, offsets_mult);
  else if(n==16) mg_reverse_core<16>(logstride, poly, offsets_mult);
  else if(n==17) mg_reverse_core<17>(logstride, poly, offsets_mult);
  else if(n==18) mg_reverse_core<18>(logstride, poly, offsets_mult);
  else if(n==19) mg_reverse_core<19>(logstride, poly, offsets_mult);
  else if(n==20) mg_reverse_core<20>(logstride, poly, offsets_mult);
  else if(n==21) mg_reverse_core<21>(logstride, poly, offsets_mult);
  else if(n==22) mg_reverse_core<22>(logstride, poly, offsets_mult);
  else if(n==23) mg_reverse_core<23>(logstride, poly, offsets_mult);
  else if(n==24) mg_reverse_core<24>(logstride, poly, offsets_mult);
  else if(n==25) mg_reverse_core<25>(logstride, poly, offsets_mult);
  else if(n==26) mg_reverse_core<26>(logstride, poly, offsets_mult);
  else if(n==27) mg_reverse_core<27>(logstride, poly, offsets_mult);
  else if(n==28) mg_reverse_core<28>(logstride, poly, offsets_mult);
  else if(n==29) mg_reverse_core<29>(logstride, poly, offsets_mult);
  else if(n==30) mg_reverse_core<30>(logstride, poly, offsets_mult);
  else if(n==31) mg_reverse_core<31>(logstride, poly, offsets_mult);
  else if(n==32) mg_reverse_core<32>(logstride, poly, offsets_mult);
  else if(n==33) mg_reverse_core<33>(logstride, poly, offsets_mult);
  else if(n==34) mg_reverse_core<34>(logstride, poly, offsets_mult);
  else if(n==35) mg_reverse_core<35>(logstride, poly, offsets_mult);
  else if(n==36) mg_reverse_core<36>(logstride, poly, offsets_mult);
  else if(n==37) mg_reverse_core<37>(logstride, poly, offsets_mult);
  else if(n==38) mg_reverse_core<38>(logstride, poly, offsets_mult);
  else if(n==39) mg_reverse_core<39>(logstride, poly, offsets_mult);
  else if(n==40) mg_reverse_core<40>(logstride, poly, offsets_mult);
}

void mg_binary_polynomial_multiply(uint64_t* p1, uint64_t* p2, uint64_t* result, uint64_t d1, uint64_t d2)
{
  constexpr unsigned int s = 6;
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
  uint64_t offsets_mult[1 << s];
  for(int i = 0; i < (1 << s); i++) offsets_mult[i] = 0;
  constify_mgr(logsize_result, 0, b1, offsets_mult);
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
  constify_mgr(logsize_result, 0, p1, offsets_mult);
  contract(p1, p1, d1 + d2);
}
