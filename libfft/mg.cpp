#include "mg.h"

#include "helpers.hpp"
#include "decompose_taylor.h"

#include <immintrin.h>

static constexpr uint64_t m_beta_over_mult[]={
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

static inline uint64_t beta_to_mult(uint64_t w, const uint64_t* t)
{
  uint64_t res = 0;
  for(unsigned int byte_idx = 0; byte_idx < 8; byte_idx++)
  {
    res ^= t[(byte_idx << 8) ^ (w & 0xFF)];
    w >>= 8;
    if(w==0) break;
  }
  return res;
}
/**
 * computes the 2**logsize first values of 2**logstride interleaved polynomials on input
 * with <= 2**logsize coefficients
 * i.e. performes 2**logstride interleaved partial DFTS
 * => acts on an array of size 2**(logstride+logsize)
 */
template <int s, int logstride>
inline void mg_aux(
    uint64_t* beta_table,
    uint64_t* poly,
    uint64_t offset,
    unsigned int logsize,
    bool first_taylor_done = false)
// log2 size of each stride to be processed;
// should be <= 2**s, the input is assumed to be of size 2**logsize and only the 2**logsize first output values are computed
{
  if constexpr(s == 0)
  {
    const uint64_t val = beta_to_mult(offset, beta_table);
#if 1
    eval_degree1<false, logstride>(val, poly);
#else
    for(uint64_t i = 0; i < stride; i++)
    {
      // computes (u,v) where
      // u = f_0 + f1 * w_{2.j},
      // v = f_0 + f1 * w_{2.j+1} = f_0 + f1 *(w_{2.j} + 1) = u + f_1
      // f_0 = poly[i], f_1 = poly[i + stride]
      poly[i]          ^= product(poly[i + stride], val);
      poly[i + stride] ^= poly[i];
    }
#endif
  }
  else
  {
    constexpr unsigned int t = 1 << (s - 1);
    if(logsize <= t)
    {
      mg_aux<s - 1, logstride>(beta_table, poly, offset, logsize, first_taylor_done);
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
      mg_aux<s - 1, logstride + t>(beta_table, poly, offset >> t, logsize - t, false);
      const uint64_t row_size = 1uLL << (t + logstride);
      for(uint64_t i = 0; i < tau; i++)
      {
        mg_aux<s - 1, logstride>(beta_table, poly, offset + (i << t), t, false);
        poly += row_size;
      }
    }
  }
}

void mg_smalldegree(
    uint64_t* beta_table,
    uint64_t* poly, // of degree < 2**logsize_prime
    unsigned int logsizeprime, // <= logsize
    unsigned int logsize
    )
{
  constexpr unsigned int s = 6;
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
    mg_aux<s, 0>(beta_table, poly + (i << logsizeprime), i << logsizeprime, logsizeprime, true);
  }
}

template <int s, int logstride>
void mg_reverse_aux(
    uint64_t* beta_table,
    uint64_t* poly,
    uint64_t offset,
    unsigned int logsize)
{
  if constexpr(s == 0)
  {
    const uint64_t val = beta_to_mult(offset, beta_table);
#if 1
    eval_degree1<true, logstride>(val, poly);
#else
    for(uint64_t i = 0; i < stride; i++)
    {
      poly[i + stride] ^= poly[i];
      poly[i]          ^= product(poly[i + stride], val);
    }
#endif
  }
  else
  {
    constexpr unsigned int t = 1 << (s - 1);
    if(logsize <= t)
    {
      mg_reverse_aux<s - 1, logstride>(beta_table, poly, offset, logsize);
    }
    else
    {
      const uint64_t tau   = 1uLL <<  (logsize - t);
      const uint64_t eta   = 1uLL << logsize;
      const uint64_t row_size = 1uLL << (t + logstride);
      uint64_t* poly_loc = poly;
      for(uint64_t i = 0; i < tau; i++)
      {
        mg_reverse_aux<s - 1, logstride>(beta_table, poly_loc, offset + (i << t), t);
        poly_loc += row_size;
      }
      // reverse fft on columns
      mg_reverse_aux<s - 1, logstride + t>(beta_table, poly, offset >> t, logsize - t);
      decompose_taylor_reverse_iterative_alt(logstride, 2 * t, t, eta, poly);
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
  constexpr unsigned int bits_per_half_uint64_t  = c_b_t<uint64_t>::n >> 1;
  const size_t bound = d / bits_per_half_uint64_t + 1;
  uint64_t* dest_even = reinterpret_cast<uint64_t*>(p);
  uint64_t* dest_odd  = reinterpret_cast<uint64_t*>(p + (bits_per_half_uint64_t >> 3));
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

void mateer_gao_polynomial_product::binary_polynomial_multiply(
    uint8_t* p1, uint8_t* p2, uint8_t* result, uint64_t d1, uint64_t d2)
{
  constexpr unsigned int s = 6;
  unsigned int logsize = 1;
  while((1uLL<<logsize) * 32 < (d1 + d2 + 1)) logsize++;
  uint64_t* b1 = new uint64_t[1uLL<<logsize];
  unique_ptr<uint64_t[]> _1(b1);
  uint64_t* b2 = new uint64_t[1uLL<<logsize];
  unique_ptr<uint64_t[]> _2(b2);
  const size_t sz = (1uLL << logsize);
  uint64_t w1 = expand(p1, b1, d1, sz);
  uint64_t w2 = expand(p2, b2, d2, sz);
  int logsizeprime = logsize;
  while(1uLL << logsizeprime >= 2 * w1) logsizeprime--;
  mg_smalldegree(m_beta_to_mult_table, b1, logsizeprime, logsize);
  logsizeprime = logsize;
  while(1uLL << logsizeprime >= 2 * w2) logsizeprime--;
  mg_smalldegree(m_beta_to_mult_table, b2, logsizeprime, logsize);
  for(size_t i = 0; i < sz; i++) b1[i] = product(b1[i], b2[i]);
  mg_reverse_aux<s,0>(m_beta_to_mult_table, b1, 0, logsize);
  contract(b1, result, d1 + d2);
}

mateer_gao_polynomial_product::mateer_gao_polynomial_product():
  m_beta_to_mult_table(new uint64_t[256*8])
{
  for(unsigned int byte_idx = 0; byte_idx < 8; byte_idx++)
  {
    for(unsigned int b = 0; b < 256; b++)
    {
      uint64_t w = static_cast<uint64_t>(b) << (8*byte_idx);
      uint64_t im_w;
      transpose_matrix_vector_product(m_beta_over_mult, w, im_w);
      m_beta_to_mult_table[256*byte_idx + b] = im_w;
    }
  }
}

mateer_gao_polynomial_product::~mateer_gao_polynomial_product()
{
  delete[] m_beta_to_mult_table;
  m_beta_to_mult_table = nullptr;
}
