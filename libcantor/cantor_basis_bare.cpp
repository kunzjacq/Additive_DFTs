#include <algorithm>
#include <iomanip>
#include <type_traits>

#include <cassert>
#include <cstring> // for memcpy/memset/memcmp
#include <immintrin.h>

#include "helpers.hpp"
#include "cantor.h"

static int depth = 0;
static constexpr int debug = false;

// enable a more efficient multiplication in multiplicative representation thanks
// to a specially chosen minimal polynomial of the generator
static constexpr bool use_new_gen = true;

template<class word>
cantor_basis_bare<word>::cantor_basis_bare():
  m_beta_over_gamma(new word[c_b_t<word>::n]),
  m_gamma_over_beta(new word[c_b_t<word>::n]),
  m_beta_to_gamma_table(new word[256*sizeof(word)]),
  m_gamma_to_beta_table(new word[256*sizeof(word)]),
  m_mult_over_gamma(new word[c_b_t<word>::n]),
  m_gamma_over_mult(new word[c_b_t<word>::n]),
  m_beta_over_mult(new word[c_b_t<word>::n]),
  m_mult_to_gamma_table(new word[256*sizeof(word)]),
  m_gamma_to_mult_table(new word[256*sizeof(word)]),
  m_mult_mul_table(nullptr),
  m_beta_to_mult_table(new word[256*sizeof(word)]),
  m_masks(new word[c_b_t<word>::word_logsize]),
  m_gamma_squares(new word[c_b_t<word>::n]),
  m_log_table_sizes(1 << min(c_b_t<word>::n, 16u)),
  m_order(m_log_table_sizes - 1),
  m_log_computed(false),
  m_log(nullptr),
  m_exp(nullptr),
  m_log_beta(nullptr),
  m_exp_beta(nullptr),
  m_A(new word[c_b_t<word>::n]),
  m_AA(new word[c_b_t<word>::n]),
  m_error(0)
{
  for(size_t i = 0; i< c_b_t<word>::n; i++)
  {
    m_beta_over_gamma[i] = 0;
    m_gamma_over_beta[i] = 0;
  }
  build();
  if(m_error) free_memory();
}

template<class word>
cantor_basis_bare<word>::~cantor_basis_bare()
{
  free_memory();
}

template<class word>
void cantor_basis_bare<word>::free_memory()
{
  delete[] m_beta_over_gamma;
  m_beta_over_gamma = nullptr;
  delete[] m_gamma_over_beta;
  m_gamma_over_beta = nullptr;
  delete[] m_beta_to_gamma_table;
  m_beta_to_gamma_table = nullptr;
  delete[] m_gamma_to_beta_table;
  m_gamma_to_beta_table = nullptr;

  delete[] m_mult_over_gamma;
  m_mult_over_gamma = nullptr;
  delete[] m_gamma_over_mult;
  m_gamma_over_mult = nullptr;
  delete[] m_beta_over_mult;
  m_beta_over_mult = nullptr;
  delete[] m_mult_to_gamma_table;
  m_mult_to_gamma_table = nullptr;
  delete[] m_beta_to_mult_table;
  m_beta_to_mult_table = nullptr;
  delete[] m_mult_mul_table;
  m_mult_mul_table = nullptr;
  delete[] m_gamma_to_mult_table;
  m_gamma_to_mult_table = nullptr;

  delete[] m_masks;
  m_masks = nullptr;
  delete[] m_gamma_squares;
  m_gamma_squares = nullptr;
  delete[] m_log;
  m_log = nullptr;
  delete[] m_exp;
  m_exp = nullptr;
  delete[] m_log_beta;
  m_log_beta = nullptr;
  delete[] m_exp_beta;
  m_exp_beta = nullptr;
  delete [] m_A;
  m_A = nullptr;
  delete [] m_AA;
  m_AA = nullptr;
}

template<class word>
int cantor_basis_bare<word>::error() const
{
  return m_error;
}

/**
 * constructs the main data structures of cantor_basis_bare, using the same data structures for
 * a subfield of degree 2.
 * if n is the dimension of the field over f2,
 * - m_beta_over_gamma[j] for j < n/2 is copied from the subfield;
 * - m_gamma_squares[j]  for j < n/2 is copied from the subfield;
 * - m_gamma_squares[j]  for j >= n/2 is computed with the formula
 *     gamma_j**2 = gamma_(j-n/2)**2 *  beta_(n/2)**2
 *                = gamma_(j-n/2)**2 * (beta_(n/2-1) + beta_(n/2))
 * - logs and exp tables are stolen from the subfield.
 */
template<class word>
void cantor_basis_bare<word>::build_from_subfield()
{
  // hypotheses:
  // - m_gamma_products[sq + n * nonsq] = g1*g2, sq = g1 & g2, nonsq = g1^g2,
  //    is computed for g1, g2 < n1, n1 = 2**i
  // - m_beta_over_gamma[j] is computed for j < n1
  // - m_beta_gamma_products[i' * n + k] = beta_{2**i'-1} * gamma_k
  //    is computed for i' < i and k < n1

  time_point<high_resolution_clock> begin, end;
  auto subfield = new cantor_basis<typename c_b_t<word>::half_type>;
  unique_ptr<remove_reference_t<decltype(*subfield)>> _(subfield);
  begin = high_resolution_clock::now();
  static_assert(c_b_t<typename c_b_t<word>::half_type>::n < c_b_t<word>::n, "template instatiation error");
  static_assert(is_same<typename c_b_t<typename c_b_t<word>::half_type>::log_type, typename c_b_t<word>::log_type>::value, "template instatiation error");
  const unsigned int np = c_b_t<typename c_b_t<word>::half_type>::n;

  for(unsigned int k = 0; k < np; k++)
  {
    m_beta_over_gamma[k] = static_cast<word>(subfield->beta_over_gamma(k));
  }
  m_t.m_linear_algebra_time += subfield->times().linalg_time();
  m_t.m_generator_decomposition_time += subfield->times().gendec_time();

  for(unsigned int k = 0; k < np; k++)
  {
    m_gamma_squares[k] = static_cast<word>(subfield->gamma_square(k));
    m_gamma_squares[k + np] =
        subfield->multiply(subfield->gamma_square(k), subfield->beta_over_gamma(np-1)) ^
        m_gamma_squares[k] << (np);
  }

  if(debug)
  {
    cout << "squares: " << endl;
    print_matrix(m_gamma_squares);
  }

  // steal log / exp tables from subfield
  m_log = subfield->log_table();
  subfield->log_table() = nullptr;
  m_exp = subfield->exp_table();
  subfield->exp_table() = nullptr;
  m_log_beta = subfield->beta_log_table();
  subfield->beta_log_table() = nullptr;
  m_exp_beta = subfield->beta_exp_table();
  subfield->beta_exp_table() = nullptr;
  m_log_computed = true;
  m_t.m_log_computation_time  = subfield->times().logcomp_time();
  end = high_resolution_clock::now();
  m_t.m_reuse_time += static_cast<double>(duration_cast<nanoseconds>(end-begin).count()) / pow(10, 9);
}



template<class word>
void cantor_basis_bare<word>::build()
{
  time_point<high_resolution_clock> begin, end, begin_linalg, end_linalg;
  static const unsigned int n = c_b_t<word>::n;
  depth = 0;
  m_beta_over_gamma[0] = 1;
  m_gamma_squares[0] = 1;
  for(unsigned int i = 0; i < c_b_t<word>::word_logsize; i++) m_masks[i] = mask<word>(i);
  build_from_subfield();
  begin = high_resolution_clock::now();
  // all steps are reused from subfield except last beta step
  for(unsigned int j = 0; j < (1u << c_b_t<word>::word_logsize); j++) m_A[j] = gamma_square(j);
  beta_step(c_b_t<word>::word_logsize - 1, m_A, m_AA, m_beta_over_gamma, m_t);

  // initialize multiplicative base
  word u = 1;
  m_mult_over_gamma[0] = u;
  if constexpr(c_b_t<word>::n == 64 && use_new_gen)
  {
    // for n=64: use 0x50c3389433a2b6cc, which has minimal polynomial x^64 + x^4 + x^3 + x + 1
    m_mult_over_gamma[1] = 0x50c3389433a2b6cc;
  }
  else
  {
    m_mult_over_gamma[1] = u << (n/2);
  }
  for(unsigned int i = 2 ; i < n; i++)
  {
    m_mult_over_gamma[i] = multiply(m_mult_over_gamma[i-1], m_mult_over_gamma[1]);
  }
  word mult_generator_minimal_polynomial_gamma = multiply(m_mult_over_gamma[n-1], m_mult_over_gamma[1]);

  if(debug)
  {
    cout << endl << "beta/gamma:" << endl;
    print_matrix(m_beta_over_gamma);
    cout << endl << "mult/gamma:" << endl;
    print_matrix(m_mult_over_gamma);
  }
  // invert matrix of beta over gamma to obtain the expression of {gamma_i} in terms of {beta_j}
  begin_linalg = high_resolution_clock::now();
  for(size_t i = 0; i< c_b_t<word>::n; i++) m_A[i] = m_beta_over_gamma[i];
  int res = invert_matrix(m_A, m_gamma_over_beta, 0);

  for(size_t i = 0; i< c_b_t<word>::n; i++) m_A[i] = m_mult_over_gamma[i];
  res |= invert_matrix(m_A, m_gamma_over_mult, 0);

  end_linalg = high_resolution_clock::now();
  m_t.m_linear_algebra_time += static_cast<double>(duration_cast<nanoseconds>(end_linalg-begin_linalg).count()) / pow(10, 9);
  end = high_resolution_clock::now();
  m_t.m_generator_decomposition_time += static_cast<double>(duration_cast<nanoseconds>(end-begin).count()) / pow(10, 9);
  if(res)
  {
    if(debug) cout << "ERROR : no solution" << endl;
    m_error = 6;
    return;
  }

  for(unsigned int byte_idx = 0; byte_idx < sizeof(word); byte_idx++)
  {
    for(unsigned int b = 0; b < 256; b++)
    {
      word vec = 0;
      word w = static_cast<word>(b) << (8*byte_idx);
      transpose_matrix_vector_product(m_beta_over_gamma, w, vec);
      m_beta_to_gamma_table[256*byte_idx + b] = vec;
      transpose_matrix_vector_product(m_gamma_over_beta, w, vec);
      m_gamma_to_beta_table[256*byte_idx + b] = vec;
      transpose_matrix_vector_product(m_mult_over_gamma, w, vec);
      m_mult_to_gamma_table[256*byte_idx + b] = vec;
      transpose_matrix_vector_product(m_gamma_over_mult, w, vec);
      m_gamma_to_mult_table[256*byte_idx + b] = vec;
    }
  }

  mult_generator_minimal_polynomial = gamma_to_mult(mult_generator_minimal_polynomial_gamma);

  if constexpr(c_b_t<word>::n == 64 && use_new_gen && debug)
  {
    cout << "Multiplicative generator minimal polynomial:" << hex << mult_generator_minimal_polynomial << endl;
  }


  if constexpr(c_b_t<word>::n == 32 || c_b_t<word>::n == 64)
  {
    m_mult_mul_table = new word[(1uLL<<mult_mul_table_logsize)*c_b_t<word>::n/mult_mul_table_logsize];
    for(unsigned int sub_idx = 0; sub_idx < c_b_t<word>::n/mult_mul_table_logsize; sub_idx++)
    {
      unsigned int k = (1uLL << mult_mul_table_logsize);
      for(word w = 0; w < k; w++)
      {
        m_mult_mul_table[sub_idx * k + w] =
            gamma_to_mult(
              multiply(
                mult_generator_minimal_polynomial_gamma,
                mult_to_gamma(w<<(sub_idx*mult_mul_table_logsize))
                )
              );
      }
    }
  }

  if constexpr(c_b_t<word>::n == 64 && !use_new_gen)
  {
    // a root of x^64 + x^4 + x^3 + x + 1, computed with sage
    uint64_t new_mult_gen = 0x23ec4049a11d7802;
    uint64_t new_mult_gen_gamma = mult_to_gamma(new_mult_gen);
    if(debug) cout << "Multiplicative generator in gamma representation w: " << hex << new_mult_gen_gamma << dec << endl;
    uint64_t acc = 1, val = new_mult_gen_gamma;
    for(int i=1; i <= 64; i++)
    {
      if(i==1 || i==3 || i==4 || i==64) acc ^= val;
      val = multiply(val,new_mult_gen_gamma);
    }
    if(debug) cout << "Minimal polynomial of w evaluated at w: " << acc << endl;
  }

  word v = 1;
  for(unsigned int i = 0; i < c_b_t<word>::n; i++)
  {
    m_beta_over_mult[i] = gamma_to_mult(beta_to_gamma(v));
    v <<= 1;
  }

  if(debug)
  {
    cout << endl << "beta/mult:" << endl;
    print_matrix(m_beta_over_mult);
  }


  for(unsigned int byte_idx = 0; byte_idx < sizeof(word); byte_idx++)
  {
    for(unsigned int b = 0; b < 256; b++)
    {
      word w = static_cast<word>(b) << (8*byte_idx);
      word im_w = gamma_to_mult(beta_to_gamma(w));
      m_beta_to_mult_table[256*byte_idx + b] = im_w;
    }
  }

  if(debug)
  {
    if(!matrix_product_is_identity(m_gamma_over_beta, m_beta_over_gamma))
    {
      cout << "ERROR : product is not equal to identity" << endl;
      m_error = 7;
      return;
    }
    cout << endl << "gamma / beta:" << endl;
    print_matrix(m_gamma_over_beta);
  }
}

template<class word>
word cantor_basis_bare<word>::beta_over_gamma(unsigned int i) const
{
  return m_beta_over_gamma[i];
}

template<class word>
word cantor_basis_bare<word>::gamma_over_beta(unsigned int i) const
{
  return m_gamma_over_beta[i];
}

template<class word>
word cantor_basis_bare<word>::mult_over_gamma(unsigned int i) const
{
  return m_mult_over_gamma[i];
}

template<class word>
word cantor_basis_bare<word>::gamma_over_mult(unsigned int i) const
{
  return m_gamma_over_mult[i];
}

template<class word>
word cantor_basis_bare<word>::gamma_to_beta(const word& w) const
{
  word res = 0;
#if 0
  transpose_matrix_vector_product(m_gamma_over_beta, w, res);
#else
  word wp = w;
  for(unsigned int byte_idx = 0; byte_idx < sizeof(word); byte_idx++)
  {
    res ^= m_gamma_to_beta_table[256*byte_idx + static_cast<unsigned int>(wp & 0xFF)];
    wp >>= 8;
    if(wp == 0) break;
  }
#endif
  return res;
}

template<class word>
word cantor_basis_bare<word>::beta_to_mult(const word& w, unsigned int num_bytes) const
{
  word res = 0;
  word wp = w;
  unsigned int nb = num_bytes ? num_bytes : sizeof(word);
  for(unsigned int byte_idx = 0; byte_idx < nb; byte_idx++)
  {
    unsigned int bp = static_cast<unsigned int>(wp & 0xFF);
    if(bp) res ^= m_beta_to_mult_table[256*byte_idx + bp];
    wp >>= 8;
    if(wp == 0) break;
  }
  return res;
}

template<class word>
word cantor_basis_bare<word>::beta_to_gamma(const word& w, unsigned int num_bytes) const
{
  word res = 0;
#if 0
  transpose_matrix_vector_product(m_beta_over_gamma, w, res);
#else
  word wp = w;
  unsigned int nb = num_bytes ? num_bytes : sizeof(word);
  for(unsigned int byte_idx = 0; byte_idx < nb; byte_idx++)
  {
    unsigned int bp = static_cast<unsigned int>(wp & 0xFF);
    if(bp) res ^= m_beta_to_gamma_table[256*byte_idx + bp];
    wp >>= 8;
    if(wp == 0) break;
  }
#endif
  return res;
}

template<class word>
word cantor_basis_bare<word>::gamma_to_mult(const word& w) const
{
  word res = 0;
#if 0
  transpose_matrix_vector_product(m_gamma_over_mult, w, res);
#else
  word wp = w;
  for(unsigned int byte_idx = 0; byte_idx < sizeof(word); byte_idx++)
  {
    unsigned int bp = static_cast<unsigned int>(wp & 0xFF);
    if(bp) res ^= m_gamma_to_mult_table[256*byte_idx + bp];
    wp >>= 8;
    if(wp == 0) break;
  }
#endif
  return res;
}

template<class word>
word cantor_basis_bare<word>::mult_to_gamma(const word& w, unsigned int num_bytes) const
{
  word res = 0;
#if 0
  transpose_matrix_vector_product(m_mult_over_gamma, w, res);
#else
  word wp = w;
  unsigned int nb = num_bytes ? num_bytes : sizeof(word);
  for(unsigned int byte_idx = 0; byte_idx < nb; byte_idx++)
  {
    res ^= m_mult_to_gamma_table[256*byte_idx + static_cast<unsigned int>(wp & 0xFF)];
    wp >>= 8;
    if(wp == 0) break;
  }
#endif
  return res;
}

template<class word>
template<int byte_idx>
word cantor_basis_bare<word>::beta_to_gamma_byte(uint32_t v) const
{
  return m_beta_to_gamma_table[256*byte_idx + v];
}

template<class word>
template<int byte_idx>
word cantor_basis_bare<word>::beta_to_mult_byte(uint32_t v) const
{
  return m_beta_to_mult_table[256*byte_idx + v];
}

template<class word>
word cantor_basis_bare<word>::trace(const word& w) const
{
  word res = 0;
  word wp = w;
  for(unsigned int i=0;i< c_b_t<word>::n;i++)
  {
    res ^= wp;
    wp = square(wp);
  }
  return res;
}

template <class word>
word cantor_basis_bare<word>::multiply_beta_repr_ref(const word &a, const word &b) const
{
  if(a == 0 || b == 0) return 0;
  if(a == 1) return b;
  if(b == 1) return a;
  word a_g = beta_to_gamma(a);
  word b_g = beta_to_gamma(b);
  return gamma_to_beta(multiply(a_g, b_g));
}

// ---------------------------- core product/square methods -------------------------------------

// helper functions

uint16_t multiply_w_log_u16(uint16_t a, uint16_t b, const uint16_t* log, const uint16_t* exp, const uint32_t order)
{
  if(a == 0 || b == 0) return 0;
  else if (a == 1) return b;
  else if (b == 1) return a;
  uint32_t l = log[a] + log[b]; // uint32_t to prevent overflows
  if(l >= order) l-= order;
  return exp[l];
}

uint8_t multiply_w_log_u8(uint8_t a, uint8_t b, const uint8_t* log, const uint8_t* exp, const uint32_t order)
{
  if(a == 0 || b == 0) return 0;
  else if (a == 1) return b;
  else if (b == 1) return a;
  uint16_t l = log[a] + log[b]; // uint16_t to prevent overflows
  if(l >= order) l-= order;
  return exp[l];
}

uint16_t square_w_log_u16(uint16_t a, const uint16_t* log, const uint16_t* exp, const uint32_t order)
{
  if(a == 0) return 0;
  else if(a == 1) return 1;
  uint32_t l = ((uint32_t) log[a]) << 1; // uint32_t to prevent overflows
  if(l >= order) l-= order;
  return exp[l];
}

uint16_t inverse_w_log_u16(uint16_t a, const uint16_t* log, const uint16_t* exp, const uint32_t order)
{
  if(a == 0) throw(0);
  else if(a == 1) return 1;
  uint16_t l = order - log[a];
  return exp[l];
}

template <class beta_word>
uint32_t multiply_t(const uint32_t &a, const uint32_t &b, const beta_word* beta, const uint16_t* log, const uint16_t* exp, const uint32_t order)
{
  if(a == 0 || b == 0) return 0;
  else if (a == 1) return b;
  else if (b == 1) return a;
  const uint16_t beta15 = (uint16_t) beta[15];
  static const uint32_t m1 = (((uint32_t) 1) << 16) - 1;
  const uint16_t a0 = a & m1;
  const uint16_t a1 = a >> 16;
  const uint16_t b0 = b & m1;
  const uint16_t b1 = b >> 16;
  const uint16_t u = multiply_w_log_u16(a0, b0, log, exp, order);
  const uint16_t v = multiply_w_log_u16(a1, b1, log, exp, order);
  const uint16_t w = multiply_w_log_u16(a0^a1, b0^b1, log, exp, order);
  const uint16_t z = multiply_w_log_u16(beta15, v, log, exp, order);
  const uint32_t res = (u^z) | (((uint32_t) (u^w)) << 16);
  return res;
}

template <class word, class beta_word>
word multiply_t(const word &a, const word &b, const beta_word* beta, const uint16_t* log, const uint16_t* exp, const uint32_t order)
{
  if(a == 0 || b == 0) return 0;
  else if (a == 1) return b;
  else if (b == 1) return a;
  static_assert(sizeof(word) >= 8, "error in template instantiation: this template should not be used for sizes below 64 bits");
  static constexpr unsigned int hn = c_b_t<word>::n/2;
  static const word m1 = (c_b_t<word>::u << hn) - c_b_t<word>::u;
  const typename c_b_t<word>::half_type a0 = static_cast<typename c_b_t<word>::half_type>(a & m1);
  const typename c_b_t<word>::half_type a1 = static_cast<typename c_b_t<word>::half_type>(a >> hn);
  const typename c_b_t<word>::half_type b0 = static_cast<typename c_b_t<word>::half_type>(b & m1);
  const typename c_b_t<word>::half_type b1 = static_cast<typename c_b_t<word>::half_type>(b >> hn);
  const typename c_b_t<word>::half_type a0xa1 = a0^a1;
  const typename c_b_t<word>::half_type b0xb1 = b0^b1;
  const typename c_b_t<word>::half_type bn = static_cast<typename c_b_t<word>::half_type>(beta[hn - 1]);
  const typename c_b_t<word>::half_type u = multiply_t(a0, b0, beta, log, exp, order);
  const typename c_b_t<word>::half_type v = multiply_t(a1, b1, beta, log, exp, order);
  const typename c_b_t<word>::half_type w = multiply_t(a0xa1, b0xb1, beta, log, exp, order);
  const typename c_b_t<word>::half_type z = v==0?0:multiply_t(bn, v, beta, log, exp, order);
  const word uxw = u^w;
  const word res = (u^z) | (uxw << hn);
  return res;
}

template <class beta_word>
uint32_t square_t(const uint32_t &a, const beta_word* beta, const uint16_t* log, const uint16_t* exp, const uint32_t order)
{
  if(a == 0) return 0;
  else if(a == 1) return 1;
  static const uint32_t m1 = (((uint32_t) 1) << 16) - 1;
  const uint16_t a0 = a & m1;
  const uint16_t a1 = a >> 16;
  const uint16_t beta15 = (uint16_t) beta[15];
  const uint16_t u = square_w_log_u16(a0, log, exp, order);
  const uint16_t v = square_w_log_u16(a1, log, exp, order);
  const uint16_t z = multiply_w_log_u16(beta15, v, log, exp, order);
  const uint32_t ev = v;
  const uint32_t res = (u^z) | (ev << 16);
  return res;
}

template <class word, class beta_word>
word square_t(const word &a, const beta_word* beta, const uint16_t* log, const uint16_t* exp, const uint32_t order)
{
  static_assert(sizeof(word) >= 8, "error in template instantiation: this template should not be used for sizes below 64 bits");
  if(a == 0) return 0;
  else if(a == 1) return 1;
  static const word unt = 1;
  static const unsigned int hn = c_b_t<word>::n/2;
  static const word m1 = (unt << hn) - 1;
  const typename c_b_t<word>::half_type a0 = static_cast<typename c_b_t<word>::half_type>(a & m1);
  const typename c_b_t<word>::half_type a1 = static_cast<typename c_b_t<word>::half_type>(a >> hn);
  const typename c_b_t<word>::half_type bn = static_cast<typename c_b_t<word>::half_type>(beta[hn - 1]);
  const typename c_b_t<word>::half_type u = square_t(a0, beta, log, exp, order);
  const typename c_b_t<word>::half_type v = square_t(a1, beta, log, exp, order);
  const typename c_b_t<word>::half_type z = v==0?0:multiply_t(bn, v, beta, log, exp, order);
  const word res = (u^z) | (((word) v) << hn);
  return res;
}

template <class beta_word>
uint32_t inverse_t(const uint32_t &a, const beta_word* beta, const uint16_t* log, const uint16_t* exp, const uint32_t order)
{
  if(a == 0) return 0;
  else if(a == 1) return 1;
  static const uint32_t m1 = (((uint32_t) 1) << 16) - 1;
  const uint16_t a0 = a & m1;
  const uint16_t a1 = a >> 16;
  const uint16_t beta15 = (uint16_t) beta[15];
  const uint16_t u = multiply_w_log_u16(a0, a0^a1, log, exp, order);
  const uint16_t v = square_w_log_u16(a1, log, exp, order);
  const uint16_t z = multiply_w_log_u16(beta15, v, log, exp, order);
  const uint16_t inv_den = inverse_w_log_u16(u^z, log, exp, order);
  const uint32_t b0 = static_cast<uint32_t>(multiply_w_log_u16(a0^a1, inv_den, log, exp, order));
  const uint32_t b1 = static_cast<uint32_t>(multiply_w_log_u16(a1, inv_den, log, exp, order));
  const uint32_t res = b0 | (b1 << 16);
  return res;
}

template <class word, class beta_word>
word inverse_t(const word &a, const beta_word* beta, const uint16_t* log, const uint16_t* exp, const uint32_t order)
{
  static_assert(sizeof(word) >= 8, "error in template instantiation: this template should not be used for sizes below 64 bits");
  if(a == 0) return 0;
  else if(a == 1) return 1;
  static const word unt = 1;
  static const unsigned int hn = c_b_t<word>::n/2;
  static const word m1 = (unt << hn) - 1;
  const typename c_b_t<word>::half_type a0 = static_cast<typename c_b_t<word>::half_type>(a & m1);
  const typename c_b_t<word>::half_type a1 = static_cast<typename c_b_t<word>::half_type>(a >> hn);
  const typename c_b_t<word>::half_type bn = static_cast<typename c_b_t<word>::half_type>(beta[hn - 1]);
  const typename c_b_t<word>::half_type u = multiply_t(a0, a0^a1, beta, log, exp, order);
  const typename c_b_t<word>::half_type v = square_t(a1, beta, log, exp, order);
  const typename c_b_t<word>::half_type z = v==0?0:multiply_t(bn, v, beta, log, exp, order);
  const typename c_b_t<word>::half_type inv_den = inverse_t(u^z, beta, log, exp, order);
  const word b0 = static_cast<word>(multiply_t(a0^a1, inv_den, beta, log, exp, order));
  const word b1 = static_cast<word>(multiply_t(a1, inv_den, beta, log, exp, order));
  const word res = b0 | (b1 << hn);
  return res;
}

template<class word>
word cantor_basis_bare<word>::inverse(const word &a) const
{
  return inverse_t(a, m_beta_over_gamma, m_log, m_exp, m_order);
}

template<class word>
word cantor_basis_bare<word>::multiply(const word &a, const word &b) const
{
  return multiply_t(a, b, m_beta_over_gamma, m_log, m_exp, m_order);
}

template<class word>
word cantor_basis_bare<word>::square(const word &a) const
{
  return square_t(a, m_beta_over_gamma, m_log, m_exp, m_order);
}

template<class word>
word cantor_basis_bare<word>::multiply_safe(const word &a, const word &b) const
{
  return multiply(a,b);
}

template<class word>
word cantor_basis_bare<word>::square_safe(const word &a) const
{
  return square(a);
}

template<class word>
word cantor_basis_bare<word>::multiply_mult_repr_ref(const word &a, const word &b) const
{
  word a_gamma = mult_to_gamma(a);
  word b_gamma = mult_to_gamma(b);
  return gamma_to_mult(multiply(a_gamma, b_gamma));
}

template<class word>
inline word cantor_basis_bare<word>::multiply_mult_repr(const word &a, const word &b) const
{
  if constexpr(c_b_t<word>::n == 32)
  {
    __m128i aa = _mm_set1_epi64x(a);
    __m128i bb = _mm_set1_epi64x(b);
    __m128i cc = _mm_clmulepi64_si128(aa, bb, 0);
    uint64_t c = _mm_cvtsi128_si64(cc); // lower 64 bits
    uint32_t ab_high = c >> 32;
    uint32_t ab_low  = (uint32_t) c;
    uint32_t* t = m_mult_mul_table;
    constexpr uint32_t mask = (1uLL << mult_mul_table_logsize) - 1;
    uint32_t m = t[ab_high&mask];
    for(unsigned int i = 0; i < c_b_t<uint32_t>::n/mult_mul_table_logsize - 1; i++)
    {
      t+= 1uLL << mult_mul_table_logsize;
      ab_high >>= mult_mul_table_logsize;
      m ^= t[ab_high&mask];
    }
    return ab_low^m;
  }
  else if constexpr(c_b_t<word>::n == 64)
  {
    word res;
    if constexpr(use_new_gen)
    {
      constexpr uint64_t poly = 0x1b; // x**64 = x**4 + x**3 + x + 1
      // x**64 + x**4 + x**3 + x + 1 is primitive over GF(2)
      // it is the minimal polynomial of the multiplicative generator

      __m128i xa = _mm_set_epi64x(poly, a);
      __m128i xb = _mm_set1_epi64x(b);
      __m128i xc = _mm_clmulepi64_si128(xa, xb, 0x00);
      __m128i xd = _mm_clmulepi64_si128(xa, xc, 0xff);
      __m128i xe = _mm_clmulepi64_si128(xa, xd, 0xff);
      __m128i xres = _mm_xor_si128(xc, xd);
      xres = _mm_xor_si128(xres, xe);
      res = xres[0];
    }
    else
    {
      __m128i xa = _mm_set1_epi64x(a);
      __m128i xb = _mm_set1_epi64x(b);
      __m128i xc = _mm_clmulepi64_si128(xa,xb, 0);

      uint64_t ab_low = _mm_cvtsi128_si64(xc);     // lower 64 bits
      uint64_t ab_high = _mm_extract_epi64(xc, 1); // upper 64 bits
      uint64_t* t = m_mult_mul_table;
      constexpr uint64_t mask = (1uLL << mult_mul_table_logsize) - 1;
      uint64_t m = t[ab_high&mask];
      for(unsigned int i = 0; i < c_b_t<uint64_t>::n/mult_mul_table_logsize - 1; i++)
      {
        t+= 1uLL << mult_mul_table_logsize;
        ab_high >>= mult_mul_table_logsize;
        m ^= t[ab_high&mask];
      }
      res = ab_low^m;
    }
    return res;
  }
  else
  {
    return multiply_mult_repr_ref(a, b);
  }
}

#ifdef HAS_UINT2048
template class cantor_basis_bare<uint2048_t>;
template class cantor_basis<uint2048_t>;
const unsigned int c_b_t<uint2048_t>::word_logsize;
const unsigned int c_b_t<uint2048_t>::n;
template uint2048_t cantor_basis_bare<uint2048_t>::beta_to_gamma_byte<0>(uint32_t v) const;
template uint2048_t cantor_basis_bare<uint2048_t>::beta_to_gamma_byte<1>(uint32_t v) const;
#endif

#ifdef HAS_UINT1024
template class cantor_basis_bare<uint1024_t>;
template class cantor_basis<uint1024_t>;
const unsigned int c_b_t<uint1024_t>::word_logsize;
const unsigned int c_b_t<uint1024_t>::n;
template uint1024_t cantor_basis_bare<uint1024_t>::beta_to_gamma_byte<0>(uint32_t v) const;
template uint1024_t cantor_basis_bare<uint1024_t>::beta_to_gamma_byte<1>(uint32_t v) const;
#endif

#ifdef HAS_UINT512
template class cantor_basis_bare<uint512_t>;
template class cantor_basis<uint512_t>;
const unsigned int c_b_t<uint512_t>::word_logsize;
const unsigned int c_b_t<uint512_t>::n;
template uint512_t cantor_basis_bare<uint512_t>::beta_to_gamma_byte<0>(uint32_t v) const;
template uint512_t cantor_basis_bare<uint512_t>::beta_to_gamma_byte<1>(uint32_t v) const;
#endif

#ifdef HAS_UINT256
template class cantor_basis_bare<uint256_t>;
template class cantor_basis<uint256_t>;
const unsigned int c_b_t<uint256_t>::word_logsize;
const unsigned int c_b_t<uint256_t>::n;
template uint256_t cantor_basis_bare<uint256_t>::beta_to_gamma_byte<0>(uint32_t v) const;
template uint256_t cantor_basis_bare<uint256_t>::beta_to_gamma_byte<1>(uint32_t v) const;
#endif

#ifdef HAS_UINT128
template class cantor_basis_bare<uint128_t>;
template class cantor_basis<uint128_t>;
const unsigned int c_b_t<uint128_t>::word_logsize;
const unsigned int c_b_t<uint128_t>::n;
template uint128_t cantor_basis_bare<uint128_t>::beta_to_gamma_byte<0>(uint32_t v) const;
template uint128_t cantor_basis_bare<uint128_t>::beta_to_gamma_byte<1>(uint32_t v) const;
#endif

template class cantor_basis_bare<uint64_t>;
template class cantor_basis<uint64_t>;
template uint64_t cantor_basis_bare<uint64_t>::beta_to_gamma_byte<0>(uint32_t v) const;
template uint64_t cantor_basis_bare<uint64_t>::beta_to_gamma_byte<1>(uint32_t v) const;
template uint64_t cantor_basis_bare<uint64_t>::beta_to_mult_byte<0>(uint32_t v) const;
template uint64_t cantor_basis_bare<uint64_t>::beta_to_mult_byte<1>(uint32_t v) const;

template class cantor_basis_bare<uint32_t>;
template class cantor_basis<uint32_t>;
template uint32_t cantor_basis_bare<uint32_t>::beta_to_gamma_byte<0>(uint32_t v) const;
template uint32_t cantor_basis_bare<uint32_t>::beta_to_gamma_byte<1>(uint32_t v) const;
template uint32_t cantor_basis_bare<uint32_t>::beta_to_mult_byte<0>(uint32_t v) const;
template uint32_t cantor_basis_bare<uint32_t>::beta_to_mult_byte<1>(uint32_t v) const;

template class cantor_basis<uint16_t>;

template class cantor_basis<uint8_t>;
