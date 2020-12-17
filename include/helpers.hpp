#pragma once

#include "cantor.h"

#include <iomanip>

void indent(int depth);

/**
 * returns a word with all bits i s.t. bit k in i is 1, set to 1.
 */
template<class word>
word mask(unsigned int k)
{
  static const unsigned int n = c_b_t<word>::n;
  word m = 0;
  for(unsigned int i = 0; i < n ; i++) if( (i >> k) & 1) set_bit(m, i, 1);
  return m;
}

template<class word>
unsigned int bit(const word &w, unsigned int idx)
{
  return static_cast<unsigned int>((w >> idx) & 1);
}

template<class word>
void set_bit(word &w, unsigned int idx, int bit_val)
{
  if(bit_val) w |= ((word) 1) << idx;
  else w &= ~(((word) 1) << idx);
}

template<class word>
void xor_bit(word &w, unsigned int idx, int bit_val)
{
  if(bit_val) w ^= ((word) 1) << idx;
}

template<class word>
unsigned int popcnt_mod_2(const word &w, unsigned int)
{
  return (bitset<numeric_limits<word>::digits>(w).count() & 1);
}

#ifdef HAS_UINT2048
template<> unsigned int bit<uint2048_t>(const uint2048_t& w, unsigned int idx);
template<> void set_bit<uint2048_t>(uint2048_t& w, unsigned int idx, int bit_val);
template<> void xor_bit<uint2048_t>(uint2048_t& w, unsigned int idx, int bit_val);
template<> unsigned int popcnt_mod_2<uint2048_t>(const uint2048_t &w, unsigned int n);
#endif

#ifdef HAS_UINT1024
template<> unsigned int bit<uint1024_t>(const uint1024_t& w, unsigned int idx);
template<> void set_bit<uint1024_t>(uint1024_t& w, unsigned int idx, int bit_val);
template<> void xor_bit<uint1024_t>(uint1024_t& w, unsigned int idx, int bit_val);
template<> unsigned int popcnt_mod_2<uint1024_t>(const uint1024_t &w, unsigned int n);
#endif

#ifdef HAS_UINT512
template<> unsigned int bit<uint512_t>(const uint512_t& w, unsigned int idx);
template<> void set_bit<uint512_t>(uint512_t& w, unsigned int idx, int bit_val);
template<> void xor_bit<uint512_t>(uint512_t& w, unsigned int idx, int bit_val);
template<> unsigned int popcnt_mod_2<uint512_t>(const uint512_t &w, unsigned int n);
#endif

#ifdef HAS_UINT256
template<> unsigned int bit<uint256_t>(const uint256_t& w, unsigned int idx);
template<> void set_bit<uint256_t>(uint256_t& w, unsigned int idx, int bit_val);
template<> void xor_bit<uint256_t>(uint256_t& w, unsigned int idx, int bit_val);
template<> unsigned int popcnt_mod_2<uint256_t>(const uint256_t &w, unsigned int n);
#endif

#ifdef HAS_UINT128
template<> unsigned int bit<uint128_t>(const uint128_t& w, unsigned int idx);
template<> void set_bit<uint128_t>(uint128_t& w, unsigned int idx, int bit_val);
template<> void xor_bit<uint128_t>(uint128_t& w, unsigned int idx, int bit_val);
template<> unsigned int popcnt_mod_2<uint128_t>(const uint128_t &w, unsigned int n);
#endif


template<class word>
unsigned int bit_cppinteger(const word &w, unsigned int idx)
{
  // not all boost::multiprecision::cpp_int types use the same limb size
  // all sizes >= 256 use 64-bit limp sizes, size 128 uses one 128-bit limp mapped to a
  // (gcc native) __uint128
  // see
  // "Backend for unsigned fixed precision (i.e. no allocator) type which will fit entirely inside a "double_limb_type""
  // in cpp_int.hpp
  // the line below ensures the correct limb size is always used
  static const unsigned int s = 8 * sizeof(*w.backend().limbs());
  const auto a = idx / s;
  if(a >= w.backend().size()) return 0;
  const auto b = idx - a * s;
  return ((w.backend().limbs()[a]) >> b) & 1;
}

template<class word>
void set_bit_cppinteger(word &w, unsigned int idx, int bit_val)
{
  static const unsigned int s = 8 * sizeof(*w.backend().limbs());
  typeof(*w.backend().limbs()) v = 1;
  const auto a = idx / s;
  if(a >= w.backend().size()  && bit_val) {
    auto p = w.backend().size();
    w.backend().resize(a + 1, 0);
    for(auto i = p; i < a + 1; i++) w.backend().limbs()[i] = 0;
  }
  const auto b = idx - a * s;
  if(bit_val) w.backend().limbs()[a] |=   v << b;
  else        w.backend().limbs()[a] &= ~(v << b);
}

template<class word>
void xor_bit_cppinteger(word &w, unsigned int idx, int bit_val)
{
  static const unsigned int s = 8 * sizeof(*w.backend().limbs());
  static typeof(*w.backend().limbs()) v = 1;
  if(bit_val)
  {
    const auto a = idx / s;
    if(a >= w.backend().size()) {
      auto p = w.backend().size();
      w.backend().resize(a + 1, 0);
      for(auto i = p; i < a + 1; i++) w.backend().limbs()[i] = 0;
    }
    const auto b = idx - a * s;
    w.backend().limbs()[a] ^= v << b;
  }
}

template<class T>
class popcnt_mod_2_cppinteger_helper{};

template<>
class popcnt_mod_2_cppinteger_helper<unsigned long long>
{
  public:
  static const unsigned int word_logsize = 6;
};

template<>
class popcnt_mod_2_cppinteger_helper<__uint128_t>
{
  public:
  static const unsigned int word_logsize = 7;
};


template<class word>
unsigned int popcnt_mod_2_cppinteger(const word &w, unsigned int n)
{
  decltype(w.backend().limbs()) l = w.backend().limbs();
  typename remove_const<typename remove_reference<decltype(*l)>::type>::type v = 0;
  static const unsigned int dec = popcnt_mod_2_cppinteger_helper<typeof(v)>::word_logsize;
  static_assert(sizeof(v) <= 16);
  unsigned int imax =  w.backend().size();
  if(n > 0 && ((imax << dec) > n)) imax = (n + (1u << dec) - 1) >> dec;
  for(unsigned int i = 0; i < imax ; i++) v ^= l[i];
  // bitset.count() below fails silently with __uint128_t limbs
  if(sizeof(v) > 8) v^= (v >> 64);
  uint64_t vt = v;
  return (bitset<numeric_limits<uint64_t>::digits>(vt).count() & 1);
}

/**
 * solves a binary system AX = B with Gauss pivoting
 * returns 0 on success, 1 on absence of solution, 2 if there is an internal error
 * (which should never happen).
 * !! Both A and B are affected by this function !!
 */
template <class word>
int solve_system(word* A, word &B, word& res)
{
  static const int n = c_b_t<word>::n;
  int pivot[n];
  int is_pivot[n];
  res = 0;
  for(int row = 0; row < n; row++) is_pivot[row] = 0;
  for(int col = 0; col < n; col++)
  {
    pivot[col] = -1;
    for(int row = 0; row < n; row++)
    {
      if (!is_pivot[row] && (bit(A[row], col)))
      {
        if(pivot[col] == -1)
        {
          is_pivot[row] = 1;
          pivot[col] = row;
        }
        else
        {
          A[row] ^= A[pivot[col]];
          xor_bit(B, row, bit(B, pivot[col]));
        }
      }
    }
  }

  for(int row = 0; row < n; row++)
  {
    if(is_pivot[row] == 0)
    {
      if(A[row]) return 2; // internal error : non-pivot rows should be 0
      if(bit(B, row)) return 1; // no solution
    }
  }

  for(int i = 0; i < n; i++)
  {
    const int col = n - 1 - i;
    const int row = pivot[col];
    unsigned int bit_set = 0;
    if(row != -1)
    {
      bit_set = bit(B, row);
    }
    else
    {
      // we can set bit 'col' of res to whatever we want. here we choose 0.
      // (this is required for the usage of this function in cantor basis computation).
      bit_set = 0;
    }
    if(bit_set)
    {
      set_bit(res, col, 1);
      // propagate to other rows
      for(int rowp = 0; rowp < n; rowp++) xor_bit(B, rowp, bit(A[rowp], col));
    }
  }

  if(B != 0) return 2; // internal error
  return 0;
}

/**
 * invert matrix A with Gauss pivoting
 * returns 0 on success, 1 on absence of solution, 2 if there is an internal error
 * (which of course should never happen).
 * A is affected by this function: it is the identity on output.
 */
template <class word>
int invert_matrix(word* A, word* inv, unsigned int nprime)
{
  const int n = nprime == 0 ? c_b_t<word>::n : nprime;
  int pivot[n];
  int inv_pivot[n];
  bool is_pivot[n];
  word u = 1;
  for(int row = 0; row < n; row++)
  {
    is_pivot[row] = false;
    inv[row] = u << row; // inv is initialized with the identity matrix
  }
  for(int col = 0; col < n; col++)
  {
    // a pivot must be found for each column
    pivot[col] = -1;
    for(int row = 0; row < n; row++)
    {
      // first line which is not already a pivot and which has a 1 in position 'col'
      // will become pivot. subsequent lines satisfying the same conditions will be xored the
      // pivot row.
      if (!is_pivot[row] && (bit(A[row], col)))
      {
        if(pivot[col] == -1)
        {
          is_pivot[row] = true;
          pivot[col] = row;
        }
        else
        {
          // here row > pivot[col]
          A[row] ^= A[pivot[col]];
          inv[row] ^= inv[pivot[col]];
        }
      }
    }
  }

  // for the matrix to be invertible, every row must be a pivot
  for(int i = 0; i < n; i++)
  {
    if(!is_pivot[i])
    {
      cout << i << "  " << hex << setfill('0') << setw(c_b_t<word>::n/4) << A[i] << endl;
      return 1;
    }
  }

#if 0
  for(int i = 0; i < n; i++)
  {
    for (int j = 0; j < i; j++)
    {
      if(bit(A[pivot[i]], j)) return 3;
    }
    if(!bit(A[pivot[i]], i)) return 4;
  }
#endif

  // here A has one line starting at index i for each i (i.e one line with i-1 zeros followed by a 1),
  // but it is not triangular in general.
  // this line is A[pivot[i]].

  // if M is the initial value of A, i.e. the matrix that we are inverting,
  // inv. M = A and inv is lower triangular.
  // next, we build T s.t. T. A = id. then (T . inv) M = T . A = id, hence T. inv is the inverse
  // we are looking for.

  // first reorder lines of A s.t. line i starts at index i.
  for(int i = 0; i < n; i++)
  {
    inv_pivot[pivot[i]] = i;
    // inv_pivot[i] = start index of row i
  }

  for(int i = 0; i < n; i++)
  {
    swap(A[i],   A[pivot[i]]);
    swap(inv[i], inv[pivot[i]]);
    auto k = pivot[i];
    pivot[i] = i;
    pivot[inv_pivot[i]] = k;
    inv_pivot[k] = inv_pivot[i];
    inv_pivot[i] = i;
  }

#if 0
  for(int i = 0; i < n; i++)
  {
    for (int j = 0; j < i; j++)
    {
      if(bit(A[i], j)) return 5;
    }
    if(!bit(A[i], i)) return 6;
  }
#endif

  for(int i = 0; i < n; i++)
  {
    for (int j = i + 1; j < n; j++)
    {
      if(bit(A[n - 1 - j], n - 1 - i))
      {
        A  [n - 1 - j] ^= A  [n - 1 - i];
        inv[n - 1 - j] ^= inv[n - 1 - i];
      }
    }
  }

#if 0
  cout << endl;

  for(int i = 0; i < n; i++)
  {
    cout << i << "  " << hex << setfill('0') << setw(c_b_t<word>::n/4) << (A[i] ^ (u << i)) << endl;
  }
#endif


  // A should be equal to identity
  for(int i = 0; i < n; i++)
  {
    if(A[i] != (u << i))
    {
      return 2; // internal error
    }
  }
  return 0;
}

template<class word>
bool matrix_product_is_identity(word* A, word *B)
{
  word u = 1;
  static const int n = c_b_t<word>::n;
  for(unsigned int i = 0; i < n; i++)
  {
    word v = 0;
    for(unsigned int j = 0 ; j < n; j++) if(bit(A[i], j)) v ^= B[j];
    if(v != (u << i)) return false;
  }
  return true;
}

template<class word>
void matrix_vector_product(const word* A, const word&B, word&C, unsigned int nprime = 0)
{
  // compute C = A . B
  const unsigned int n = nprime == 0 ? c_b_t<word>::n : nprime;
  C = 0;
#if 0
  word mask = 0;
  if(n < c_b_t<word>::n) {
    mask  = 1;
    mask <<= n;
    mask -= 1;
  }
  else
  {
    mask = ~mask;
  }
#endif

  for(unsigned int i = 0; i < n; i++)
  {
    const word w_i = B & A[i];
    unsigned int cprime = popcnt_mod_2(w_i, n);
    if(cprime) xor_bit(C, i, 1);
  }
}

template<class word>
void transpose_matrix_vector_product(const word* A, const word&B, word&C)
{
  // compute C = A . B
  C = 0;
  static const unsigned int n = c_b_t<word>::n;
  for(unsigned int i = 0 ; i < n; i++) if(bit(B, i)) C ^= A[i];
}

/**
 * Tells whether an element in gamma representation is primitive, for words of size <= 128 bits
 */
template<class word, class T>
bool is_primitive(word v, const T &c_b)
{
  static const vector<uint64_t> factors = {3, 5, 17, 257, 65537, 641, 6700417, 274177, 67280421310721};
  static const vector<unsigned int> num_factors = {3, 4, 5, 7, 9};
  constexpr unsigned int sz_log = c_b_t<word>::word_logsize;
  static_assert (sz_log >= 3 && sz_log <= 7);
  bool is_primitive = true;
  for(unsigned int j = 0; j < num_factors[sz_log - 3]; j++)
  {
    word e = 1;
    for(unsigned int k = 0; k < num_factors[sz_log - 3]; k++) if(k != j) e *= factors[k];
    word m1 = power(v, e, c_b);
    if(m1 == 1) is_primitive = false;
  }
  return is_primitive;
}

template<class word, class T>
word power(const word & v, const word & e, const T &c_b)
{
  static const unsigned int n = c_b_t<word>::n;
  word result = 1;
  for(int i = n - 1; i >= 0; i--)
  {
    result = c_b.square_safe(result);
    if(bit(e, i)) result = c_b.multiply_safe(result, v);
  }
  return result;
}

/**
 * compute beta_{2**i}, ...., beta_{2**(i+1)-1}
 * takes as input square_matrix which contains gamma_j**2 for j = 0 ... 2**(i+1)
 * tmp_matrix is a buffer of 2**(i+1) words
 * the result is stored in beta_over_gamma
 * **square_matrix is affected during the computations**
 */

template<class word>
int beta_step(
    unsigned int i,
    word* square_matrix,
    word* tmp_matrix,
    word* beta_over_gamma,
    time_measurements& t)
{
  time_point < high_resolution_clock > begin, end;
  static const word u = 1;
  const base_index n1 =  1u << i;
  const base_index n2 =  1u << (i + 1);
  int retcode = 0;
  // compute beta_{2**i}, ...., beta_{2**(i+1)-1} by interatively solving the linear relations
  // beta_j**2 + beta_j = beta_(j-1)  (remember that squaring is linear)
  for(unsigned int j = 0; j < n2; j++) tmp_matrix[j] = 0;

  for(unsigned int j = 0; j < n2; j++)
  {
    word v = square_matrix[j] ^ (u << j);  // = gamma_j^2 + gamma_j
    for(unsigned int k = 0 ; k < n2; k++) if(bit(v, k)) set_bit(tmp_matrix[k], j, 1);
  }

  // instead of performing n1 system resolutions with the same matrix, we invert once the matrix
  // and then perform n1 matrix - vector products.
  // Change a bit in tmp_matrix to make the it invertible.
  // This does not change the solutions we are looking for.
  set_bit(tmp_matrix[n2 - 1], 0, 1);

  //cout << "Artin-Schreier matrix, step " << i << endl;
  //print_matrix(tmp_matrix, 1 << (i+1));

  begin = high_resolution_clock::now();
  int res = invert_matrix(tmp_matrix, square_matrix, n2);

  //cout << "Artin-Schreier inverse matrix, step " << i << endl;
  //print_matrix(square_matrix, 1 << (i+1));

  if(res) retcode = 3;
  else
  {
    word B = u << n1, C;
    beta_over_gamma[n1] = B;
    for(unsigned int j = n1 + 1; j < n2; j++)
    {
      matrix_vector_product(square_matrix, B, C, n2);
      if(C & 1)
      {
        // solutions should not have this bit set because it involves the bit that is
        // artificially set to 1 in the matrix to make it invertible
        retcode = 4;
        break;
      }
      // C^=1; // choose the other solution
      beta_over_gamma[j] = C;
      B = C;
    }
  }
  end = high_resolution_clock::now();
  t.m_linear_algebra_time += static_cast<double>(duration_cast<nanoseconds>(end-begin).count()) / pow(10, 9);
  return retcode;
}

template <class word>
int test_solve(bool verbose)
{
  static const unsigned int n = c_b_t<word>::n;
  for(int t = 0; t < 10; t++)
  {
    word A[n], AA[n], B = 0, C;
    for(unsigned int i = 0; i < n; i++)
    {
      A[i] = 0;
      if(rand() & 1) set_bit(B, i, 1);
      for(unsigned int j = 0; j < n; j++)
      {
        if(rand() & 1) set_bit(A[i], j , 1);
      }
    }
    memcpy(AA, A, n * sizeof(word));
    word B_save = B;
    int res = solve_system(A, B, C);
    B = B_save;
    if(res == 0)
    {
      cout << "a solution was found" << endl;
      word Bprime = 0;
      matrix_vector_product(AA, C, Bprime);
      if(verbose)
      {
        if(Bprime == B)
        {
          cout << " solution "<< hex << C << " is correct" << endl;
        }
        else
        {
          cout << " ** solution is wrong" << endl;
          return 1;
        }
      }
    }
    else
    {
      if(verbose) cout << "no solution was found" << endl;
    }
  }
  return 0;
}
