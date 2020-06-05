#include <algorithm>
#include <iomanip>
#include <type_traits>

#include <cassert>
#include <cstring> // for memcpy/memset/memcmp
#include <cmath>

#include "helpers.hpp"

#include "cantor.h"

static int depth = 0;
static constexpr int debug = false;

template<class word>
cantor_basis_full<word>::cantor_basis_full():
  m_beta_over_gamma(new word[c_b_t<word>::n]),
  m_gamma_over_beta(new word[c_b_t<word>::n]),
  m_beta_to_gamma_table(new word[256*sizeof(word)]),
  m_gamma_to_beta_table(new word[256*sizeof(word)]),
  m_mult_over_gamma(new word[c_b_t<word>::n]),
  m_gamma_over_mult(new word[c_b_t<word>::n]),
  m_mult_to_gamma_table(new word[256*sizeof(word)]),
  m_gamma_to_mult_table(new word[256*sizeof(word)]),
  m_masks(new word[c_b_t<word>::word_logsize]),
  m_A(new word[c_b_t<word>::n]),
  m_AA(new word[c_b_t<word>::n]),
  m_log_computed(false),
  m_log_table_sizes(1 << min(c_b_t<word>::n, 16u)),
  m_order(m_log_table_sizes - 1),
  m_log(new typename c_b_t<word>::log_type[m_log_table_sizes]),
  m_exp(new typename c_b_t<word>::log_type[m_log_table_sizes]),
  m_log_beta(new typename c_b_t<word>::log_type[m_log_table_sizes]),
  m_exp_beta(new typename c_b_t<word>::log_type[m_log_table_sizes]),
  m_gamma_products(nullptr),
  m_beta_gamma_products(new word[c_b_t<word>::n*c_b_t<word>::word_logsize]),
  m_product_unique_index(new int[c_b_t<word>::n*c_b_t<word>::n]),
  m_num_distinct_products(c_b_t<word>::word_logsize),
  m_error(0)
{
  static_assert(sizeof(word) < 4, "this class can only be instatiated with word size < 4 bytes");
  static_assert(is_same<typename c_b_t<word>::log_type, word>::value, "this class uses logs for computations hence the word log type must be equal to the word type");
  static const unsigned int n = c_b_t<word>::n;

  time_point<high_resolution_clock> begin, end, begin_linalg, end_linalg;
  begin = high_resolution_clock::now();
  memset(m_beta_gamma_products, 0, c_b_t<word>::n*c_b_t<word>::word_logsize * sizeof(word));

  // assign to each pair of basis element indexes i, j a unique index for the product
  // gamma_i * gamma_j. This enables to speed up the reference product method multiply_ref
  // for large word sizes.
  // walk through each pair (i, j),  2**s <= i, j <= 2**s
  // s increasing, 0 <= s <= word_logsize.
  // This particular enumeration order enables to efficiently restrict ourselves to pairs (i,j)
  // s.t. i,j <= 2**s', s' < word_logsize.

  int* index_aux = new int[c_b_t<word>::n*c_b_t<word>::n];
  unique_ptr<int[]> _(index_aux);
  for(unsigned int i = 0; i < c_b_t<word>::n*c_b_t<word>::n; i++) index_aux[i] = -1;
  int ctr = 0;
  auto record_pair = [&](unsigned int i, unsigned int j)
  {
    // record pair(i, j)
    // the product gamma_i * gamma_j is uniquely determined by 'squares' and 'nonsquares' below
    // if the pair (squares, nonsquares) has a unique index index_aux[idx], use it;
    // otherwise assign a new index
    unsigned int squares = i & j;
    unsigned int nonsquares = i ^ j;
    unsigned int idx = squares + c_b_t<word>::n * nonsquares;
    if(index_aux[idx] == -1) index_aux[idx] = ctr++;
    m_product_unique_index[i * c_b_t<word>::n + j] = index_aux[idx];
  };
  record_pair(0, 0);
  for(unsigned int s = 1; s < c_b_t<word>::word_logsize + 1; s++)
  {
    for(unsigned int i = 0; i < (1u << s); i++)
    {
      for(unsigned int j = i < (1u << (s-1)) ? (1u << (s-1)) : 0; j < (1u << s); j++)
      {
        record_pair(i, j);
      }
    }
    m_num_distinct_products[s - 1] = ctr;
  }
  if (debug) cout << "Number of distinct base products: " << dec << ctr << "/" << c_b_t<word>::n*c_b_t<word>::n << endl;
  m_ref_product_bits.resize(ctr, false);
  m_gamma_products = new word[ctr]; // will be indexed by the unique product indexes
  for(int i = 0; i < ctr; i++) m_gamma_products[i] = 0;
  depth = 0;
  m_beta_over_gamma[0] = 1;
  m_gamma_products[0] = 1;
  for(unsigned int i = 0; i < c_b_t<word>::word_logsize; i++) m_masks[i] = mask<word>(i);
  for(unsigned int i = 0; i < c_b_t<word>::word_logsize; i++)
  {
    gamma_step(i);
    for(unsigned int j = 0; j < (1u << (i + 1)); j++) m_A[j] = gamma_square(j);
    beta_step(i, m_A, m_AA, m_beta_over_gamma, m_t);
  }
  end = high_resolution_clock::now();
  m_t.m_generator_decomposition_time += static_cast<double>(duration_cast<nanoseconds>(end-begin).count()) / pow(10, 9);

  // initialize multiplicative base
  word u = 1;
  m_mult_over_gamma[0] = 1;
  m_mult_over_gamma[1] = u << (n/2);
  for(unsigned int i = 2 ; i < n;i++)
  {
    m_mult_over_gamma[i] = multiply_safe(m_mult_over_gamma[i-1], m_mult_over_gamma[1]);
  }

  begin_linalg = high_resolution_clock::now();
  memcpy(m_A, m_beta_over_gamma, c_b_t<word>::n * sizeof(word));
  int res = invert_matrix(m_A, m_gamma_over_beta, 0);
  memcpy(m_A, m_mult_over_gamma, n * sizeof(word));
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

  if(debug)
  {
    if(!matrix_product_is_identity(m_gamma_over_beta, m_beta_over_gamma))
    {
      cout << "ERROR : product is not equal to identity" << endl;
      m_error = 7;
      return;
    }
    cout << endl << "beta / gamma:" << endl;
    print_matrix(m_beta_over_gamma);
    cout << endl << "gamma / beta:" << endl;
    print_matrix(m_gamma_over_beta);
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

  begin = high_resolution_clock::now();
  create_log_exp_tables();
  end = high_resolution_clock::now();
  m_t.m_log_computation_time = static_cast<double>(duration_cast<nanoseconds>(end-begin).count()) / pow(10, 9);
}

template<class word>
cantor_basis_full<word>::~cantor_basis_full()
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

  delete[] m_gamma_products;
  m_gamma_products = nullptr;
  delete[] m_beta_gamma_products;
  m_beta_gamma_products = nullptr;
  delete[] m_product_unique_index;
  m_product_unique_index = nullptr;
  delete[] m_masks;
  m_masks = nullptr;
  delete [] m_A;
  m_A = nullptr;
  delete [] m_AA;
  m_AA = nullptr;
  delete [] m_log;
  m_log = nullptr;
  delete [] m_exp;
  m_exp = nullptr;
  delete [] m_log_beta;
  m_log_beta = nullptr;
  delete [] m_exp_beta;
  m_exp_beta = nullptr;
}

/** multiplies v by base element 2**i, i.e by 1 << (1 << i)
 * top_sq_bit_index has the following meaning : v was obtained as a product of (gamma_i)**2 gamma_j
 * with i & j == 0 and i = sum 2**i_k with for all k, 2**i_k < top_sq_bit_index.
 * (j may contain powers 2**j_k s.t. 2**j_k > top_sq_bit_index (but not equal since i & j == 0)).
 * since each product (gamma_(2**i_k))**2 is a sum of products of gamma_{2**l} with l <= i_k,
 */

template<class word>
word cantor_basis_full<word>::mult_gen(unsigned int i, const word& v, int top_sq_bit_index) const
{
  static const unsigned int n = c_b_t<word>::n;
  if(v == 0) return 0;
  if(debug)
  {
    depth++;
    indent(depth);
    cout << "mult_gen(" << hex << (unsigned int) i << ", " << v << ")" << endl;
  }
  const word m = m_masks[i];
  int dec = 1 << i;
  const word v1 = v & (~m); // terms not multiple of beta_{2**i}
  const word v2p = v & m;    // terms multiple of beta_{2**i}
  const word v2 = v2p >> dec;        // = v2p divided by beta_{2**i}
  const word v1p = static_cast<word> (v1 << dec); // = v1 multiplied by beta_{2**i}
  word res = v1p ^ v2p;
  // if i == 0, v1 is sent to v1p, v2p = beta_{0} * v2 is sent to
  // beta_{0} * v2 + v2, hence the result is v1p ^ v2p ^ v2
  if (i == 0) res ^= v2;
  else
  {
    // return v1p ^ v2p ^ beta_{2**i - 1} * v2
    for(int k = 0; k < top_sq_bit_index - dec; k++)
    {
      if(bit(v2, k))
      {
        res ^= m_beta_gamma_products[i * n + k];
      }
    }
  }
  if(debug)
  {
    indent(depth);
    cout << "mult_gen = " << res << endl;
    depth--;
  }
  return res;
}

// on input a base element index gamma < n, outputs gamma**2 as a field element
// assumes gamma'**2 are already computed for gamma' < gamma
template<class word>
word cantor_basis_full<word>::g_square(base_index base_square, int topmost_bit) const
{
  if(debug)
  {
    depth++;
    indent(depth);
    cout << "g_square(" << hex << (unsigned int) base_square  << ")" << endl;
  }

  // find topmost bit k in j=base_square to compute gamma_j**2 as
  // gamma_2**k * (gamma_2**k * gamma_jprime**2)
  // where jprime =  j & (~(1<<k))
  // uses the fact that gamma_jprime**2 has no component on any multiple of gamma_2**k
  // hence gamma_2**k * gamma_jprime**2 is computed by a shift
  // this is why k has to be the topmost bit index
  int k = topmost_bit - 1;
  while((base_square & (1 << k)) == 0) k--;
  base_index jprime = base_square & (~(1 << k));
  word w = m_gamma_products[m_product_unique_index[jprime*(1+c_b_t<word>::n)]];
  if(debug)
  {
    assert((w & m_masks[k]) == 0);
  }
  word v = mult_gen(k, w << (1 << k), 1 << topmost_bit);
  m_gamma_products[m_product_unique_index[base_square*(1+c_b_t<word>::n)]] = v;
  if(debug)
  {
    indent(depth);
    cout << "g_square = " << v  << endl;
    depth--;
  }
  return v;
}

// on input g1, g2 base element indexes < n, outputs g1 * g2 as a field element
template<class word>
word cantor_basis_full<word>::g_product(base_index squares, base_index nonsquares, int top_bit_square_index) const
{
  const int idx = m_product_unique_index[squares + ((squares|nonsquares) << c_b_t<word>::word_logsize)];
  if(debug)
  {
    depth++;
    indent(depth);
    cout << "g_product(" << hex << (unsigned int) squares << ", " << (unsigned int) nonsquares << ")" << endl;
  }
  word w = 1;
  if (squares == 0)
  {
    w <<= nonsquares;
    m_gamma_products[idx] = w;
  }
  else
  {
    if(nonsquares == 0) return g_square(squares, top_bit_square_index);
    // there are two cases here.
    // either the largest k s.t. gamma_{2**k} divides nonsquare is s.t. 2**k < top_bit_square_index
    // (which is a power of 2).
    // In that case, g1.g2 only has terms gamma j with j < top_bit_square_index.
    // or (1 << k) >= top_bit_square_index.
    // in that case, the computation of gamma_{2**k} * ((nonsquare/gamma_{2**k}) * (gamma_squares)**2)
    // can be simplified as explained below.
    // note that for the simplification to work, one has to process the bits if nonsquares in
    // increasing order: we look for the topmost bit below, but since our method is recursive,
    // this bit is processed last.

    int k = c_b_t<word>::word_logsize - 1;
    while((nonsquares & (1 << k)) == 0) k--;
    base_index nonsqprime = nonsquares & (~(1 << k));
    const int idx2 = m_product_unique_index[squares + ((squares|nonsqprime) << c_b_t<word>::word_logsize)];
    w = m_gamma_products[idx2];
    if(k >= top_bit_square_index)
    {
      if(debug) assert((w & m_masks[k]) == 0);
      // the condition above implies w&m_masks[k]==0, which simplifies the call to mult_gen
      // indeed, initally gamma_squares**2 only has nonzero coefficients on
      // gamma_0, ..., gamma_{top_bit_square_index -1}.
      // product by elements of nonsqprime can introduce nonzero coefficients for higher terms,
      // but since k is the topmost bit of nonsquare, all indexes of terms that appear have
      // bit k equal to 0.
      // hence if 2**k > top_bit_square_index, there is no multiple of gamma_2**k in w.
      w = (w & (~m_masks[k])) << (1 << k);
    }
    else
    {
      w = mult_gen(k, w, 1 << top_bit_square_index);
    }
    m_gamma_products[idx] = w;
  }
  if(debug)
  {
    indent(depth);
    cout << "g_product = " << w << endl;
    depth--;
  }

  return w;
}

template<class word>
void cantor_basis_full<word>::gamma_step(unsigned int i)
{
  static const word u = 1;
  static const unsigned int n = c_b_t<word>::n;

  // here,
  // - m_gamma_products[sq + n * nonsq] = g1*g2, sq = g1 & g2, nonsq = g1^g2,
  //    is computed for g1, g2 < n1, n1 = 2**i
  // - m_beta_over_gamma[j] is computed for j < n1
  // - m_beta_gamma_products[i' * n + k] = beta_{2**i'-1} * gamma_k
  //    is computed for i' < i and k < n1

  const base_index n1 =  1u << i;
  const base_index n2 =  1u << (i + 1);
  // gamma_2**i * 1 = 1 * gamma_2**i = gamma_2**i
  m_gamma_products[m_product_unique_index[n1]] = u << n1;


  // 1)
  // compute the beta_{2**i-1} * gamma_k for 0 <= k < n1
  {
    word& beta_prev = m_beta_over_gamma[n1 - 1];
    for(base_index j = 0; j < n1; j++)
    {
      const int* p = m_product_unique_index + (j << c_b_t<word>::word_logsize);
      word* q = m_beta_gamma_products + i * n;
      if(bit(beta_prev, j))
      {
        for(base_index k = 0; k < n1; k++)
        {
          word w = m_gamma_products[p[k]];
          q[k] ^= w;
        }
      }
    }
  }

  //2)
  // compute the required gamma products for step 3), i.e.
  // gamma_j * gamma_k for j < 2**i, 2**i <= k < 2**(i+1). For sq = j&k and nonsq = j^k,
  // this is equivalent to sq < 2**i and 2**i <= nonsq < 2**(i+1).
  // since n1 <= nonsq < n2, it seems that we need the beta_gamma products of step 3)
  // (not yet computed), but it is not the case; see last argument of g_product.
  for(unsigned int ip = 0; ip <= i; ip++)
  {
    for(base_index sq = ip == 0 ? 0 : (1u << (ip-1)); sq < (1u << ip); sq++)
    {
      for(base_index nonsq = n1; nonsq < n2; nonsq++)
      {
        if((sq & nonsq) == 0) g_product(sq, nonsq, ip);
      }
    }
  }

  // 3)
  // compute the beta_{2**i'-1} * gamma_k for
  // n1 <= k < n2
  // i' <= i
  for(unsigned int ip = 0; ip <= i; ip++)
  {
    word& beta_prev = m_beta_over_gamma[(1u << ip) - 1];
    for(base_index j = 0; j < (1u << ip); j++)
    {
      if(bit(beta_prev, j))
      {
        const int* p = m_product_unique_index + (j << c_b_t<word>::word_logsize);
        word* q = m_beta_gamma_products + ip * n;
        for(base_index k = n1; k < n2; k++)
        {
          word w = m_gamma_products[p[k]];
          q[k] ^= w;
        }
      }
    }
  }

  // 4)
  // compute the gamma_j * gamma_k products for n1 <= j,k < n2
  // which is equivalent to  n1 <= sq = j&k < n2 and 0 <= nonsq = j^k < n1
  // these computations use the beta_gamma products of step 3)
  for(unsigned int sq = n1; sq < n2; sq++)
  {
    for(unsigned int nonsq = 0; nonsq < n1; nonsq++)
    {
      if((sq & nonsq) == 0) g_product(sq, nonsq, i + 1);
    }
  }
}

template<class word>
word cantor_basis_full<word>::beta_over_gamma(unsigned int i) const
{
  return m_beta_over_gamma[i];
}

template<class word>
word cantor_basis_full<word>::gamma_over_beta(unsigned int i) const
{
  return m_gamma_over_beta[i];
}

template<class word>
word cantor_basis_full<word>::mult_over_gamma(unsigned int i) const
{
  return m_mult_over_gamma[i];
}

template<class word>
word cantor_basis_full<word>::gamma_over_mult(unsigned int i) const
{
  return m_gamma_over_mult[i];
}

template<class word>
void cantor_basis_full<word>::create_log_exp_tables()
{
  static constexpr uint32_t np     = c_b_t<typename c_b_t<word>::log_type>::n;
  static constexpr uint32_t np_log = c_b_t<typename c_b_t<word>::log_type>::word_logsize;
  // find a primitive element
  typename c_b_t<word>::log_type p = 0;
  for(unsigned int i = np / 2; i < np; i++)
  {
    if(is_primitive(m_beta_over_gamma[i], *this))
    {
      p = static_cast<typename c_b_t<word>::log_type>(m_beta_over_gamma[i]);
      if(debug)
        cout << "Element chosen: " << i << endl;
      break;
    }
  }
  if (p == 0)
  {
    m_error = 4;
    return;
  }
  if(debug) for (uint32_t i = 0; i < m_log_table_sizes; i++) m_log[i] = m_log_table_sizes - 1;
  typename c_b_t<word>::log_type curr = 1;
  typename c_b_t<word>::log_type curr_beta = 1;
  for (uint16_t e = 0; e < m_log_table_sizes - 1; e++)
  {
    curr_beta = static_cast<typename c_b_t<word>::log_type>(gamma_to_beta(curr));
    m_exp[e] = curr;
    m_log[curr] = e;
    m_exp_beta[e] = curr_beta;
    m_log_beta[curr_beta] = e;
    curr = (uint16_t) multiply_ref(p, curr, np_log);
  }

  if(debug)
  {
    for (uint32_t i = 1; i < m_log_table_sizes; i++)
    {
      if(m_log[i] == m_log_table_sizes - 1) m_error = 5;
    }
  }
  m_exp[m_log_table_sizes - 1] = 1;
  m_exp_beta[m_log_table_sizes - 1] = 1;
  m_log_computed = true;
}

template<class word>
word cantor_basis_full<word>::multiply_ref(const word &a, const word &b, const unsigned int log_nprime) const
{
  const unsigned int log_n = log_nprime == 0 ? c_b_t<word>::word_logsize : log_nprime;
  const unsigned int n = 1u << log_n;
  word res = 0;
  // both product methods below work for any word size, but the optimization brought by
  // m_ref_product_bits is only useful for large words.
  if(c_b_t<word>::n < 256)
  {
    for(base_index k = 0; k < n; k++)
    {
      if(bit(b, k))
      {
        for(base_index j = 0; j < n; j++)
        {
          if(bit(a, j))
          {
            word w = m_gamma_products[m_product_unique_index[j + (k << c_b_t<word>::word_logsize)]];
            res ^= w;
          }
        }
      }
    }
  }
  else
  {
    for(base_index i = 0; i < n; i++)
    {
      if(bit(b, i))
      {
        for(base_index j = 0; j < n; j++)
        {
          int k = m_product_unique_index[i * c_b_t<word>::n + j];
          if(bit(a, j))
          {
            m_ref_product_bits[k] = !m_ref_product_bits[k];
          }
        }
      }
    }
    for(int idx = 0; idx < m_num_distinct_products[max(log_n,1u)-1]; idx++)
    {
      if(m_ref_product_bits[idx])
      {
        res ^= m_gamma_products[idx];
        m_ref_product_bits[idx] = false;
      }
    }
  }
  return res;
}


template<class word>
word cantor_basis_full<word>::square_ref(const word &a) const
{
  static const unsigned int n = c_b_t<word>::n;
  word res = 0;
  for(base_index j = 0; j < n; j++)
  {
    if(bit(a, j))
    {
      res ^= m_gamma_products[m_product_unique_index[j*(1+c_b_t<word>::n)]]; // = gamma_j**2
    }
  }
  return res;
}

template<class word>
word cantor_basis_full<word>::multiply_safe(const word &a, const word &b) const
{
  if(m_log_computed) return multiply(a,b); else return multiply_ref(a,b);
}

template<class word>
word cantor_basis_full<word>::square_safe(const word &a) const {
  if(m_log_computed) return square(a); else return square_ref(a);
}

template<class word>
word cantor_basis_full<word>::gamma_to_beta(const word& w) const
{
  word res = 0;
#if 0
  transpose_matrix_vector_product(m_gamma_over_beta, w, res);
#else
  word wp = w;
  for(unsigned int byte_idx = 0; byte_idx < sizeof(word); byte_idx++)
  {
    res ^= m_gamma_to_beta_table[256*byte_idx + (wp & 0xFF)];
    wp >>= 8;
  }
#endif
  return res;
}

template<class word>
inline word cantor_basis_full<word>::beta_to_gamma(const word& w, unsigned int num_bytes) const
{
  word res = 0;
#if 0
  transpose_matrix_vector_product(m_beta_over_gamma, w, res);
#else
  word wp = w;
  unsigned int nb = num_bytes ? num_bytes : sizeof(word);
  for(unsigned int byte_idx = 0; byte_idx < nb; byte_idx++)
  {
    res ^= m_beta_to_gamma_table[256*byte_idx + (wp & 0xFF)];
    wp >>= 8;
  }
#endif
  return res;
}

template<class word>
word cantor_basis_full<word>::gamma_to_mult(const word& w) const
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
word cantor_basis_full<word>::mult_to_gamma(const word& w, unsigned int num_bytes) const
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
word cantor_basis_full<word>::beta_to_gamma_byte(uint32_t v) const
{
  return m_beta_to_gamma_table[256*byte_idx + v];
}


template<class word>
word cantor_basis_full<word>::trace(const word& w) const
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


template<class word>
word cantor_basis_full<word>::gamma_square(unsigned int i) const
{
  return m_gamma_products[m_product_unique_index[i*(1+c_b_t<word>::n)]];
}

template<class word>
word cantor_basis_full<word>::inverse(const word& a) const
{
  if(a == 0) throw(0);
  else if(a == 1) return 1;
  uint32_t l = m_order - m_log[a];
  return m_exp[l];
}
template<class word>
word cantor_basis_full<word>::multiply(const word& a, const word& b) const
{
  if(a == 0 || b == 0) return 0;
  else if(a == 1) return b;
  else if(b == 1) return a;
  uint32_t l = m_log[a] + m_log[b];
  if(l >= m_order) l-= m_order;
  return m_exp[l];
}

template<class word>
word cantor_basis_full<word>::multiply_beta_repr(const word& a, const word& b) const
{
  if(a == 0 || b == 0) return 0;
  else if(a == 1) return b;
  else if(b == 1) return a;
  uint32_t l = m_log_beta[a] + m_log_beta[b];
  if(l >= m_order) l-= m_order;
  return m_exp_beta[l];
}

template<class word>
word cantor_basis_full<word>::square(const word& a) const
{
  if(a == 0) return 0;
  else if(a == 1) return 1;
  uint32_t l = ((uint32_t) m_log[a]) << 1; // uint32_t to prevent overflows
  if(l >= m_order) l-= m_order;
  return m_exp[l];
}

template <class word>
word cantor_basis_full<word>::multiply_beta_repr_ref(const word &a, const word &b) const
{
  if(a == 0 || b == 0) return 0;
  else if(a == 1) return b;
  else if(b == 1) return a;
  word a_g = beta_to_gamma(a);
  word b_g = beta_to_gamma(b);
  return gamma_to_beta(multiply(a_g, b_g));
}


template class cantor_basis_full<uint8_t>;
template class cantor_basis_full<uint16_t>;

template uint8_t cantor_basis_full<uint8_t>::beta_to_gamma_byte<0>(uint32_t v) const;
template uint8_t cantor_basis_full<uint8_t>::beta_to_gamma_byte<1>(uint32_t v) const;

template uint16_t cantor_basis_full<uint16_t>::beta_to_gamma_byte<0>(uint32_t v) const;
template uint16_t cantor_basis_full<uint16_t>::beta_to_gamma_byte<1>(uint32_t v) const;


