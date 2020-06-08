#include <algorithm>
#include <iomanip>
#include <type_traits>

#include <cassert>
#include <cstring> // for memcpy/memset/memcmp

#include "helpers.hpp"

#include "cantor.h"

static int depth = 0;
static constexpr int debug = false;

template<class word>
cantor_basis_bare<word>::cantor_basis_bare():
  m_beta_over_gamma(new word[c_b_t<word>::n]),
  m_gamma_over_beta(new word[c_b_t<word>::n]),
  m_beta_to_gamma_table(new word[256*sizeof(word)]),
  m_gamma_to_beta_table(new word[256*sizeof(word)]),
  m_mult_over_gamma(new word[c_b_t<word>::n]),
  m_gamma_over_mult(new word[c_b_t<word>::n]),
  m_mult_to_gamma_table(new word[256*sizeof(word)]),
  m_gamma_to_mult_table(new word[256*sizeof(word)]),
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
  //memset(m_beta_over_gamma, 0, c_b_t<word>::n * sizeof(word));
  //memset(m_gamma_over_beta, 0, c_b_t<word>::n * sizeof(word));
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
  delete[] m_mult_to_gamma_table;
  m_mult_to_gamma_table = nullptr;
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
  // - m_gamma_products[sq + n * nonsq] = g1*g2, sq = g1 & g2, nonsq = g1^g2,
  //    is computed for g1, g2 < n1, n1 = 2**i
  // - m_beta_over_gamma[j] is computed for j < n1
  // - m_beta_gamma_products[i' * n + k] = beta_{2**i'-1} * gamma_k
  //    is computed for i' < i and k < n1

  time_point<high_resolution_clock> begin, end;
  auto subfield = new cantor_basis<typename c_b_t<word>::half_type>;
  unique_ptr<typeof(*subfield)> _(subfield);
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
  m_mult_over_gamma[0] = 1;
  m_mult_over_gamma[1] = u << (n/2);
  for(unsigned int i = 2 ; i < n;i++)
  {
    m_mult_over_gamma[i] = multiply(m_mult_over_gamma[i-1], m_mult_over_gamma[1]);
  }

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
  //memcpy(m_A, m_beta_over_gamma, n * sizeof(word));
  int res = invert_matrix(m_A, m_gamma_over_beta, 0);

  for(size_t i = 0; i< c_b_t<word>::n; i++) m_A[i] = m_mult_over_gamma[i];
  //memcpy(m_A, m_mult_over_gamma, n * sizeof(word));
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
      word res = 0;
      word w = static_cast<word>(b) << (8*byte_idx);
      transpose_matrix_vector_product(m_beta_over_gamma, w, res);
      m_beta_to_gamma_table[256*byte_idx + b] = res;
      transpose_matrix_vector_product(m_gamma_over_beta, w, res);
      m_gamma_to_beta_table[256*byte_idx + b] = res;
      transpose_matrix_vector_product(m_mult_over_gamma, w, res);
      m_mult_to_gamma_table[256*byte_idx + b] = res;
      transpose_matrix_vector_product(m_gamma_over_mult, w, res);
      m_gamma_to_mult_table[256*byte_idx + b] = res;
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
  }
#endif
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
    res ^= m_beta_to_gamma_table[256*byte_idx + static_cast<unsigned int>(wp & 0xFF)];
    wp >>= 8;
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
    res ^= m_gamma_to_mult_table[256*byte_idx + static_cast<unsigned int>(wp & 0xFF)];
    wp >>= 8;
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
word cantor_basis_bare<word>::square_safe(const word &a) const {
  return square(a);
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
const unsigned int c_b_t<uint64_t>::word_logsize;
const unsigned int c_b_t<uint64_t>::n;
template uint64_t cantor_basis_bare<uint64_t>::beta_to_gamma_byte<0>(uint32_t v) const;
template uint64_t cantor_basis_bare<uint64_t>::beta_to_gamma_byte<1>(uint32_t v) const;

template class cantor_basis_bare<uint32_t>;
template class cantor_basis<uint32_t>;
const unsigned int c_b_t<uint32_t>::word_logsize;
const unsigned int c_b_t<uint32_t>::n;
template uint32_t cantor_basis_bare<uint32_t>::beta_to_gamma_byte<0>(uint32_t v) const;
template uint32_t cantor_basis_bare<uint32_t>::beta_to_gamma_byte<1>(uint32_t v) const;

template class cantor_basis<uint16_t>;
const unsigned int c_b_t<uint16_t>::word_logsize;
const unsigned int c_b_t<uint16_t>::n;

template class cantor_basis<uint8_t>;
const unsigned int c_b_t<uint8_t>::word_logsize;
const unsigned int c_b_t<uint8_t>::n;
