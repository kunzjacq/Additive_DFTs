#include <cassert>
#include <cstring>
#include <memory>


#include "additive_fft.h"
#include "utils.h"


constexpr bool debug = false;
constexpr bool manual_align = false;

// #define MANUAL_ALIGN
// there seems to be no point in aligning arrays to 16-byte boundaries
// on x86_64 as the values returned by new are already 16-byte aligned.
// see
// cout << __STDCPP_DEFAULT_NEW_ALIGNMENT__; // returns 16 on x86_64 linux

template <class word>
additive_fft<word>::additive_fft(cantor_basis<word> *c_b, unsigned int p_m):
  buf(nullptr),
  cst_coefficients(nullptr),
  coeffs_to_cancel(nullptr),
  m(min(p_m, c_b_t<word>::n)),
  packed_degrees(new uint64_t[static_cast<size_t>(c_b_t<word>::n*c_b_t<word>::n)]),
  num_coefficients(new unsigned int[static_cast<size_t>(c_b_t<word>::n)]),
  m_c_b(c_b)
{
  // all loop indexes will fit in 64-bit values
  if(m == 0)  m = c_b_t<word>::n;
  if(m >= 40) cout << "FFT size requested too large!" << endl;
  const uint64_t sz = 1uLL << static_cast<size_t>(max(1u, m) - 1);
  if constexpr(manual_align)
  {
    const size_t alignment = 0x10;
    cst_coefficients = (word*) std::aligned_alloc(alignment, sizeof(word) * sz);
    coeffs_to_cancel = (word*) std::aligned_alloc(alignment, sizeof(word) * sz);
    buf              = (word*) std::aligned_alloc(alignment, sizeof(word) * (1uLL << m));
  }
  else
  {
    cst_coefficients = new word[sz];
    coeffs_to_cancel = new word[sz];
    buf              = new  word[1uLL << m];
  }
  prepare_polynomials();
  if(debug) print_fft_polynomials();
}

template <class word>
additive_fft<word>::~additive_fft()
{
  delete[] packed_degrees;
  packed_degrees = nullptr;
  delete[] num_coefficients;
  num_coefficients = nullptr;
  if constexpr(manual_align)
  {
    std::free(cst_coefficients);
    cst_coefficients = nullptr;
    std::free(coeffs_to_cancel);
    coeffs_to_cancel = nullptr;
    std::free(buf);
    buf = nullptr;
  }
  else
  {
    delete[] cst_coefficients;
    cst_coefficients = nullptr;
    delete[] coeffs_to_cancel;
    coeffs_to_cancel = nullptr;
    delete[] buf;
    buf = nullptr;
  }
}

// po_result[j] = P(j)
template <class word>
void additive_fft<word>::fft_direct(
    const word* p_poly,
    uint64_t p_poly_degree,
    word* po_result,
    uint64_t blk_offset) const
{
  const uint64_t imax = 1uLL << m;
  for (uint64_t i = 0; i < imax; i++)
  {
    po_result[i] = 0;
    word x_pow_j = 1; // evaluate polynomial at i; start at i**0
    // word x = static_cast < word > (i + (1uLL << m) * blk_offset);
    word x = m_c_b->beta_to_gamma(static_cast<word>(i + (1uLL << m) * blk_offset));
    for (uint64_t j = 0; j < p_poly_degree + 1; j++)
    {
      word c = p_poly[j];
      if(c != 0)
      {
        po_result[i] ^= m_c_b->multiply(c, x_pow_j);
      }
      x_pow_j = m_c_b->multiply(x_pow_j, x);
    }
  }
}

// po_result[i] = P(x**i)
template <class word>
void additive_fft<word>::fft_direct_exp(
    const word* p_poly,
    uint64_t p_poly_degree,
    const word& x,
    word* po_result) const
{
  assert(m == c_b_t<word>::n && m <= 32);
  const uint64_t imax = (1uLL << m) - 1;
  for (uint64_t i = 0; i < imax; i++)
  {
    // compute p[x**i]
    po_result[i] = 0;
    word x_pow_i = power(x, static_cast<word>(i), *m_c_b);
    word x_pow_ij = 1; // x_pow_i**j; start at x_pow_i**0
    for (uint64_t j = 0; j < p_poly_degree + 1; j++)
    {
      word c = p_poly[j];
      if(c != 0)
      {
        po_result[i] ^= m_c_b->multiply(c, x_pow_ij);
      }
      x_pow_ij = m_c_b->multiply(x_pow_ij, x_pow_i);
    }
  }
  //po_result[imax] = 0;
}

/**
 * reduces in-place a polynomial by X**multiplicative order - 1,
 * where multiplicative_order refers to the multiplicative order of the binary field
 * with elements of type word, i.e., multiplicative order = 2**n - 1 if word has n bits.
 */
template<class word>
uint64_t fold_polynomial(word* p_poly, uint64_t p_poly_degree)
{
  // deal with cases where the polynomial degree exceeds the multiplicative order of elements in
  // the finite field : fold the polynomial by reducing it by X**multiplicative order - 1.
  // this can only occur if c_b_t<word>::n <= 64, since the degree is a 64-bit value.
  // Degrees around or beyond 2**64 are unrealistic anyway.
  word mult_order = 0;
  mult_order = ~mult_order; // 2**n-1
  if constexpr(c_b_t<word>::n == 64)
  {
    if(p_poly_degree == mult_order)
    {
      p_poly[0]^=p_poly[mult_order];
      p_poly_degree = mult_order - 1;
    }
  }
  else if constexpr(c_b_t<word>::n < 64)
  {
    if(p_poly_degree >= mult_order)
    {
      uint64_t k = p_poly_degree / mult_order;

      for(uint64_t j = 0; j + k*mult_order < p_poly_degree + 1; j++)
      {
        p_poly[j] ^= p_poly[j+k*mult_order];
      }

      for(uint64_t i = 1; i < k; i++)
      {
        for(uint64_t j = 0; j < mult_order; j++)
        {
          // triggers warning: iteration 1073741825 invokes undefined behavior [-Waggressive-loop-optimizations]
          p_poly[j] ^= p_poly[j+i*mult_order];
        }
      }
      p_poly_degree = mult_order - 1;
    }
  }
  return p_poly_degree;
}

template<class word>
void additive_fft<word>::additive_fft_ref_in_place(
    word* p_poly,
    uint64_t p_poly_degree,
    uint64_t blk_index
    )  const
{
  const uint64_t blk_size = 1uLL << m;
  unsigned int step, first_step;
  uint64_t j, k;
  if(p_poly_degree == 0)
  {
    for(j = 1; j < blk_size; j++) p_poly[j] = p_poly[0];
    return;
  }

  // deal with cases where the polynomial degree exceeds the field size
  if constexpr(c_b_t<word>::n <= 64)
  {
    p_poly_degree = fold_polynomial<word>(p_poly, p_poly_degree);
  }
  for(uint64_t i = p_poly_degree + 1; i < blk_size; i++) p_poly[i] = 0;

  // we can skip euclidean divisions of size larger than 2**m, but don't have to
  //first_step =  0;
  first_step =  max(1u, c_b_t<word>::n - m) - 1;
  // we also still skip the ones that would do nothing because reductor degree > target poly degree
  while((1uLL << (c_b_t<word>::n - first_step - 1)) > p_poly_degree) first_step++;
  // for first step, reductor_degree = ho <= p_poly_degree in all cases (since p_poly_degree > 0)

  for(step = first_step; step < c_b_t<word>::n; step++)
  {
    unsigned int i = c_b_t<word>::n - 1 - step; // reductor index
    const uint64_t o  = 1uLL << (i + 1);
    const uint64_t ho = 1uLL << i;
    // log2 of number of blocks of size o covering the interval 0 ... 2**m - 1
    const int log_num_blocks = max(0, static_cast<int>(m - (i + 1)));
    const uint64_t num_blocks = 1uLL << log_num_blocks;
    //reductor = P_i
    const uint64_t reductee_d = step == first_step ? p_poly_degree : o - 1;
    assert(reductee_d >= reductor_d);
    const bool compute_dividend = ho < blk_size;
    if(debug)
    {
      cout << endl << dec << "step " << step << endl;
      cout << " reductee degree = " << reductee_d << endl;
      cout << " reducing by P_" << i << ", of degree " << ho << endl;
      cout << " performing 2**" << log_num_blocks << " = " << num_blocks << " reductions" << endl;
      if(!compute_dividend) cout << " not";
      cout << " computing dividend" << endl;
      fflush(stdout);
    }

    if(step == first_step)
    {
      for(j = 1; j < num_blocks; j++)
      {
        for(k = 0; k < p_poly_degree + 1; k++)
        {
          p_poly[j*o+k] = p_poly[k];
        }
      }
    }

    const uint64_t* p_reductor_degrees = packed_degrees + c_b_t<word>::n * i;
    const int nc = num_coefficients[i] - 1;
    for(j = 0; j < num_blocks; j++)
    {
      word* poly_ij = p_poly + j * o;
      // Compute constant term to add to P_i, i = m - 1 - step, so that P_i(u) = 0,
      // u = j * o + blk_index * blk_size
      // because of the properties of Cantor bases, in beta representation, P_i(u) = u >> i
      const word cst_term_beta_repr = static_cast<word>((j * o + blk_index * blk_size) >> i);
      const word cst_term = m_c_b->beta_to_gamma(cst_term_beta_repr);
      //cst_coefficients[j] = m_c_b->beta_to_gamma(cst_term_beta_repr);

      // do euclidean division in-line
      for (uint64_t ii = reductee_d; ii > ho - 1; ii--)
      {
        // introduce a new monomial of the reductee that must be cancelled
        // this is also the resulting coefficient in dividend since reductor is monic
        const word c = poly_ij[ii];
        // add (c * X**(i - reductor_degree) * reductor) to reductee
        word* base = poly_ij + ii - ho;
        base[0] ^= m_c_b->multiply(cst_term, c);
        for (int ik = 0; ik < nc; ik++) base[p_reductor_degrees[ik]] ^= c;
      }
      // end of euclidean division
      if(debug && j <= 1)
      {
        cout << "r:" << endl;
        print_series<word>(poly_ij, ho, min(ho, 8uL));
        if(compute_dividend)
        {
          cout << "q:" << endl;
          print_series<word>(poly_ij+ho, ho, min(ho, 8uL));

        }
      }

      if(compute_dividend)
      {
        for(k = 0; k < ho; k++) poly_ij[k + ho] ^= poly_ij[k]; // r + q
      }
    }
  }
}

// fast version of fft_ref_in_place (approx. 25% faster)
template<class word>
void additive_fft<word>::additive_fft_fast_in_place(
    word* p_poly,
    uint64_t p_poly_degree,
    uint64_t blk_index
    )  const
{
  const uint64_t blk_size = 1uLL << m;
  unsigned int step, first_step;
  uint64_t i, j, k;
  if(p_poly_degree == 0)
  {
    for(j = 1; j < blk_size; j++) p_poly[j] = p_poly[0];
    return;
  }
  // deal with cases where the polynomial degree exceeds the field size
  if constexpr(c_b_t<word>::n <= 64)
  {
    p_poly_degree = fold_polynomial<word>(p_poly, p_poly_degree);
  }
  for(uint64_t i = p_poly_degree + 1; i < blk_size; i++) p_poly[i] = 0;

  first_step =  max(1u, c_b_t<word>::n - m) - 1;
  while((1uLL << (c_b_t<word>::n - first_step - 1)) > p_poly_degree) first_step++;
  // for first step, reductor_degree = ho <= p_poly_degree in all cases (since p_poly_degree > 0)
  for(step = first_step; step < c_b_t<word>::n; step++)
  {
    i = c_b_t<word>::n - 1 - step; // reductor index
    const uint64_t o  = 1uLL << (c_b_t<word>::n - step);
    const uint64_t ho = 1uLL << (c_b_t<word>::n - step - 1);
    // log2 of number of blocks of size o covering the interval 0 ... 2**m - 1
    const int log_num_blocks = max(0, static_cast<int>(m - c_b_t<word>::n + step));
    const uint64_t num_blocks = 1uLL << log_num_blocks;
    //reductor = P_i
    const uint64_t reductee_d = step == first_step ? p_poly_degree : o - 1;
    const uint64_t reductor_d = ho;
    assert(reductee_d >= reductor_d);
    if(debug)
    {
      cout << "step " << step << endl;
      cout << " reductee degree = 0x" << hex << reductee_d << endl;
      cout << " reducing by P_" << dec << i << ", of degree 0x" << hex << reductor_d << endl;
      cout << " performing 2**0x" << hex << log_num_blocks << " = 0x" << num_blocks <<
              " reductions" << dec << endl;
      fflush(stdout);
    }

    const bool compute_dividend = ho < blk_size;
    if(compute_dividend)
    {
      assert(reductee_d - reductor_d + 1 <= ho);
    }
    else
    {
      assert(num_blocks == 1);
    }

    if(step == first_step)
    {
      // create num_blocks interlaced copies of the initial polynomial by copying each coefficient
      // num_block times, starting at the end.
      for(k = p_poly_degree ; k != -1uLL; k--)
      {
        for(j = 0; j < num_blocks; j++)
        {
          p_poly[k * num_blocks + j] = p_poly[k];
        }
      }
    }

    for(j = 0; j < num_blocks; j++)
    {
      // Compute constant term of P_i, so that P_i(u) = 0, u = j * o + blk_index * blk_size.
      // because of the properties of Cantor bases, in beta representation, P_i(u) = u >> i.
      const word pt_beta_repr = static_cast<word>((j * o + blk_index * blk_size) >> i);
      cst_coefficients[j] = m_c_b->beta_to_gamma(pt_beta_repr);
    }

    // divide source by sparse polynomial reductor, for num_blocks interleaved polynomials
    // simultaneously.
    // on input, coefficent ii of polynomial j is in p_poly[j + num_blocks * ii], ii < o.
    // at the end of euclidean division,
    // remainder coefficient ii of polynomial j is in p_poly[j + num_blocks * ii], ii < ho
    // and (if computed)
    // dividend coefficient ii of polynomial j is in p_poly[j + num_blocks * (ii + ho)], ii < ho.
    uint64_t* reductor_packed_degrees = packed_degrees + c_b_t<word>::n * i;
    for (uint64_t ii = reductee_d; ii > reductor_d - 1; ii--)
    {
      // ho = degree of reductor
      // position of the first reductee coefficient affected by current addition of reductor
      word* p = p_poly + ((ii - ho) << log_num_blocks);
      word* q = p_poly + (ii << log_num_blocks);
      for(j = 0; j < num_blocks; j++)
      {
        // the coeffs to cancel of the num_blocks polynomials reduced simultaneously are the q[j]
        // the corresponding constant terms of the reductors are the cst_coefficients[j]
        p[j] ^= m_c_b->multiply(cst_coefficients[j], q[j]);
      }
      for (unsigned int ik = 0; ik < num_coefficients[i] - 1; ik++)
      {
        word* r = p + (reductor_packed_degrees[ik] << log_num_blocks);
        for(j = 0; j < num_blocks; j++) r[j] ^= q[j];
      }
    }
    // end of euclidean division

    if(compute_dividend)
    {
      // compute results r and r+q for all interleaved polynomials
      const uint64_t hos  = num_blocks * ho;
      for(k = 0; k < hos; k++) p_poly[hos + k] ^= p_poly[k]; // q <- r + q
      // to put data into the correct location for next round,
      // each pair of polynomials (r, r+q) created must be interleaved again
      // experimentally the fastest method, see utils.h for variants

      //it is probably ok to put m as the logsize below.
      // with the formula
      // c_b_t<word>::n - step + log_num_blocks,
      // it can be m+1 at the first step if poly_degre >= 2**m.
      // But then ho = blk_size and compute_dividend = false, and we do not end up here.
      if(m != c_b_t<word>::n - step + log_num_blocks)
      {
        cout << "Internal Error!" << endl;
        exit(1);
      }
      // with interleave_in_place, which does not use any buffer, this function
      // becomes slower than additive_fft_ref_in_place...
      interleave_quarter_buffer<word>(buf, p_poly, m);
    }
  }
}

template<class word>
void additive_fft<word>::additive_fft_rev_ref_in_place(
    word* p_values,
    uint64_t blk_index
    )  const
{
  const uint64_t blk_size = 1uLL << m;
  unsigned int step, first_step, last_step;
  uint64_t j, k;

  first_step = c_b_t<word>::n - 1;
  last_step =  c_b_t<word>::n - m;

  if(debug)
  {
    print_series<word>(p_values, 1uLL << m, 1uLL << m);
  }

  for(step = first_step; step != last_step - 1 ; step--)
  {
    unsigned int i = c_b_t<word>::n - 1 - step; // polynomial index
    const uint64_t o  = 1uLL << (i + 1);
    const uint64_t ho = 1uLL << i;
    // log2 of number of blocks of size o covering the interval 0 ... 2**m - 1
    const int log_num_blocks = max(0, static_cast<int>(m - (i + 1)));
    const uint64_t num_blocks = 1uLL << log_num_blocks;
    //reductor = P_i
    const uint64_t multiplier_d = ho;
    if(debug)
    {
      cout << endl << dec << "step " << step << endl;
      cout << " output degree = " << o - 1 << endl;
      cout << " multiplying by P_" << i << ", of degree " << multiplier_d << endl;
      cout << " performing 2**" << log_num_blocks << " = " << num_blocks << " multiplications" << endl;
      fflush(stdout);
    }

    uint64_t* packed_degrees_i = packed_degrees + c_b_t<word>::n * i;
    for(j = 0; j < num_blocks; j++)
    {
      word* poly_ij = p_values + j * o;
      // Compute constant term to add to P_i, i = m - 1 - step, so that P_i(u) = 0,
      // u = j * o + blk_index * blk_size
      // because of the properties of Cantor bases, in beta representation, P_i(u) = u >> i
      const word cst_term_beta_repr = static_cast<word>((j * o + blk_index * blk_size) >> i);
      const word cst_term = m_c_b->beta_to_gamma(cst_term_beta_repr);
      // as arrays of size ho,
      // poly_ij = remainder
      // poly_ij + ho  = remainder + dividend
      // at the end of the loop iteration, poly_ij = remainder + dividend * (P_i + cst_term)
      for(k = 0; k < ho; k++) poly_ij[k + ho] ^= poly_ij[k];
      // now
      // poly_ij + ho = dividend (degree < ho)
      // poly_ij =  remainder    (degree < ho)
      for(k = 0; k < ho; k++)
      {
        word* p = poly_ij+k;
        word c = p[ho];
        p[0] ^= m_c_b->multiply(cst_term, c);
        for(uint64_t ii = 0; ii < num_coefficients[i] - 1; ii++)
        {
          p[packed_degrees_i[ii]] ^= c;
        }
      }
    }
  }
}

// fast version of fft_rev_ref_in_place (approx. 25% faster on large instances)
template<class word>
void additive_fft<word>::additive_fft_rev_fast_in_place(
    word* p_values,
    uint64_t blk_index
    )  const
{
  const uint64_t blk_size = 1uLL << m;
  uint64_t j, k;

  if(debug)
  {
    print_series<word>(p_values, 1uLL << m, 1uLL << m);
  }

  for(unsigned int i = 0; i < m; i++)
  {
    const uint64_t o  = 1uLL << (i + 1);
    const uint64_t ho = 1uLL << i;
    // log2 of number of blocks of size o covering the interval 0 ... 2**m - 1
    const int log_num_blocks = max(0, static_cast<int>(m - (i + 1)));
    const uint64_t num_blocks = 1uLL << log_num_blocks;
    //reductor = P_i
    const uint64_t multiplier_d = ho;
    if(debug)
    {
      cout << endl << dec << "step " << c_b_t<word>::n - 1 - i << endl;
      cout << " output degree = " << o - 1 << endl;
      cout << " multiplying by P_" << i << ", of degree " << multiplier_d << endl;
      cout << " performing 2**" << log_num_blocks << " = " << num_blocks << " multiplications" << endl;
      fflush(stdout);
    }

    uint64_t* packed_degrees_i = packed_degrees + c_b_t<word>::n * i;
    for(j = 0; j < num_blocks; j++)
    {
      // Compute constant term to add to P_i, i = m - 1 - step, so that P_i(u) = 0,
      // u = j * o + blk_index * blk_size
      // because of the properties of Cantor bases, in beta representation, P_i(u) = u >> i
      const word cst_term_beta_repr = static_cast<word>((j * o + blk_index * blk_size) >> i);
      cst_coefficients[j] = m_c_b->beta_to_gamma(cst_term_beta_repr);
    }

    deinterleave_quarter_buffer<word>(buf, p_values, m);

    const uint64_t hos  = num_blocks * ho;
    const uint64_t mask = num_blocks - 1;
    // here k is the coefficient index inside a polynomial, j is the polynomial index,
    // and ii is the reductor coefficient index.
    // in this first fused loop below jk = j + k * num_blocks.
    // j is equal to jk % num_blocks, i.e. (jk & mask).
    for(uint64_t jk = 0; jk < hos; jk++)
    {
      p_values[jk + hos] ^= p_values[jk];
      p_values[jk] ^= m_c_b->multiply(cst_coefficients[jk&mask], p_values[jk + hos]);

    }
    for(k = 0; k < ho; k++)
    {
      for(uint64_t ii = 0; ii < num_coefficients[i] - 1; ii++)
      {
        word* p = p_values + ((k + packed_degrees_i[ii]) << log_num_blocks);
        word* q = p_values + hos + (k << log_num_blocks);
        for(j = 0; j < num_blocks; j++) p[j] ^= q[j];
      }
    }
  }
}


template <class word>
void additive_fft<word>::prepare_polynomials()
{
  unsigned int i, j;
  word* div = new word[2*c_b_t<word>::n];
  // recursive computation of the p_i which are linear polynomials
  // p_i vanishes on 0 ... 2**i - 1 (included) in beta representation
  // y_ i = p_i(beta_i) = 1
  // p_0 = X
  // p_{i+1}(X) = p_i(X) * p_i(X - beta_i)
  //          = p_i(X) * (p_i(X) - 1)
  //          = p_i**2(X) - p_i(X)
  // therefore all the nonzero coefficients of p_i are equal to 1.

  // initialize p_0
  for (j = 0; j < c_b_t<word>::n; j++) div[j] = 0;
  div[0] = 1;
  // initialize packed representation of p_0
  packed_degrees[0] = 1;
  num_coefficients[0] = 1;
  word* p_i_minus_1 = div;
  word* p_i = div + c_b_t<word>::n;
  for(i = 1; i < c_b_t<word>::n; i++)
  {
    // p_i = p_{i-1} + p_{i-1}**2
    p_i[0] = p_i_minus_1[0];
    for(j = 1; j < c_b_t<word>::n; j++) p_i[j] = p_i_minus_1[j] ^ p_i_minus_1[j - 1];
    // build packed representation of p_i
    unsigned int num_coeffs = 0;
    for(j = 0; j < c_b_t<word>::n; j++)
    {
      if(p_i[j])
      {
        packed_degrees[c_b_t<word>::n * i + num_coeffs] = 1uLL << j;
        num_coeffs++;
      }
    }
    num_coefficients[i] = num_coeffs;
    swap(p_i_minus_1, p_i);
  }
  delete[] div;
}

template<class word>
void additive_fft<word>::print_fft_polynomials()
{
  unsigned int i, k;
  for(i = 0; i < c_b_t<word>::n; i++)
  {
    cout <<"P_" << dec << i << " = ";
    for(k = 0; k < num_coefficients[i]; k++)
    {
      if(k > 0) cout << " + ";
      cout << "X^0x" << hex << packed_degrees[c_b_t<word>::n * i + k];
    }
    cout << dec << endl;
  }
}

#ifdef Boost_FOUND
template class additive_fft<uint2048_t>;
template class additive_fft<uint1024_t>;
template class additive_fft<uint512_t>;
template class additive_fft<uint256_t>;
template class additive_fft<uint128_t>;
#endif

template class additive_fft<uint64_t>;
template class additive_fft<uint32_t>;
template class additive_fft<uint16_t>;
template class additive_fft<uint8_t>;
