#pragma once

#include <cstdint> // for uint64_t and other types
#include <cstddef>
#include <cassert> // for assert

#include <immintrin.h> // for x86_64 intrinsics

/**
 * @brief binary_polynomial_multiply
 * multiplication of binary polynomials using Mateer-Gao DFT.
 * In this version, the required buffers are allocated internally.
 * @param p1
 * buffer with 1st polynomial, of degree d1. Buffer should be readable up to index i = d1 / 8.
 * @param p2
 * same for second polynomial, of degree d2. Buffer should be readable up to index i = d2 / 8.
 * @param result
 * Buffer for result, of byte size at least (d1+d2) / 8 + 1.
 * @param d1
 * degree of 1st polynomial.
 * @param d2
 * degree of 2nd polynomial.
 */
void mg_binary_polynomial_multiply(uint8_t *p1, uint8_t *p2, uint8_t *result, uint64_t d1, uint64_t d2);

#if 0
/**
 beta_over_mult[i] = beta_i in multplicative representation, i.e. in GF(2**64) as described in
 function 'product'. beta_i refers to the beta basis in Mateer-Gao algorithm.
*/
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

/** beta_over_mult_cumulative[i] = sum_j=0 ... i - 1 beta_over_mult[i]
 *
 * produced with
 *
  cout << hex;
  for(int i = 1; i < 65; i++)
  {
    uint64_t val = mult_beta_pow_table[i-1]^beta_over_mult[i-1];
    cout << "0x" << val << ",";
    if(((i+1)&0x3) == 0) cout << endl;
  }
  cout << endl << dec;
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
