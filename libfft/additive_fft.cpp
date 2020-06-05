#include <cassert>
#include <cstring>
#include <memory>

#include "additive_fft.h"
#include "utils.h"


constexpr bool debug = false;

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
  packed_logdegrees(new unsigned int[static_cast<size_t>(c_b_t<word>::n*c_b_t<word>::n)]),
  num_coefficients(new unsigned int[static_cast<size_t>(c_b_t<word>::n)]),
  m_c_b(c_b)
{
  // all loop indexes will fit in 64-bit values
  if(m == 0)  m = c_b_t<word>::n;
  if(m >= 40) cout << "FFT size requested too large!" << endl;
  const uint64_t sz = 1uLL << static_cast<size_t>(max(1u, m) - 1);
#ifdef MANUAL_ALIGN
  const size_t alignment = 0x10;
  cst_coefficients = (word*) std::aligned_alloc(alignment, sizeof(word) * sz);
  coeffs_to_cancel = (word*) std::aligned_alloc(alignment, sizeof(word) * sz);
  buf              = (word*) std::aligned_alloc(alignment, sizeof(word) * (1uLL << m));
#else
  cst_coefficients = new word[sz];
  coeffs_to_cancel = new word[sz];
  buf              = new  word[1uLL << m];
#endif
  prepare_polynomials();
  if(debug) print_fft_polynomials();
}

template <class word>
additive_fft<word>::~additive_fft()
{
  delete[] packed_logdegrees;
  packed_logdegrees = nullptr;
  delete[] num_coefficients;
  num_coefficients = nullptr;
#ifdef MANUAL_ALIGN
  std::free(cst_coefficients);
  cst_coefficients = nullptr;
  std::free(coeffs_to_cancel);
  coeffs_to_cancel = nullptr;
  std::free(buf);
  buf = nullptr;
#else
  delete[] cst_coefficients;
  cst_coefficients = nullptr;
  delete[] coeffs_to_cancel;
  coeffs_to_cancel = nullptr;
  delete[] buf;
  buf = nullptr;
#endif
}

// out[j] = P(j)

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

// out[i] = P(x**i)

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

template<class word>
void additive_fft<word>::additive_fft_ref(
    word* p_poly,
    uint64_t p_poly_degree,
    word* po_result,
    uint64_t blk_index
    )  const
{
  const uint64_t blk_size = 1uLL << m;
  unsigned int step, first_step, v;
  uint64_t j, k;
  if(p_poly_degree == 0)
  {
    for(j = 0; j < blk_size; j++) po_result[j] = p_poly[0];
    return;
  }

  if constexpr(c_b_t<word>::n < 64)
  {
    word mult_order = static_cast<word>(0xFFFFFFFFuL); // assumes complement-2 representation
    if(m == c_b_t<word>::n && p_poly_degree >= mult_order) assert(0);
  }
  first_step =  max(1u, c_b_t<word>::n - m) - 1;
  while((1uLL << (c_b_t<word>::n - first_step - 1)) > p_poly_degree) first_step++;
  // for first step, reductor_degree = ho <= p_poly_degree in all cases (since p_poly_degree > 0)
  for(step = first_step; step < c_b_t<word>::n; step++)
  {
    v = c_b_t<word>::n - 1 - step; // reductor index
    const uint64_t o  = 1uLL << (v + 1);
    const uint64_t ho = 1uLL << v;
    // log2 of number of blocks of size o covering the interval 0 ... 2**m - 1
    const int log_num_blocks = max(0, static_cast<int>(m - (v + 1)));
    const uint64_t num_blocks = 1uLL << log_num_blocks;
    //reductor = P_i
    const uint64_t reductee_d = step == first_step ? p_poly_degree : o - 1;
    const uint64_t reductor_d = ho;
    assert(reductee_d >= reductor_d);
    const bool compute_dividend = o <= blk_size;
    if(debug)
    {
      cout << endl << dec << "step " << step << endl;
      cout << " reductee degree = " << reductee_d << endl;
      cout << " reducing by P_" << v << ", of degree " << reductor_d << endl;
      cout << " performing 2**" << log_num_blocks << " = " << num_blocks << " reductions" << endl;
      if(!compute_dividend) cout << " not";
      cout << " computing dividend" << endl;
      fflush(stdout);
    }

    if(compute_dividend)
    {
      // euclidean_division will write in dividend buffer from index 0 to reductee_d - reductor_d,
      // included. set coefficients at other indexes to 0.
      // this is only needed once per step.
      assert(reductee_d - reductor_d + 1 <= ho);
      for(k = reductee_d - reductor_d + 1LL; k < ho; k++) buf[ho + k] = 0;
    }

    for(j = 0; j < num_blocks; j++)
    {
      word* poly_ij = po_result + j * o;
      // Compute constant term to add to P_v, v = m - 1 - step, so that P_v(u) = 0,
      // u = j * o + blk_index * blk_size
      // because of the properties of Cantor bases, P_v(u) = u >> v in beta representation
      const word cst_term_beta_repr = static_cast<word>((j * o + blk_index * blk_size) >> v);
      const word cst_term = m_c_b->beta_to_gamma(cst_term_beta_repr);
      const word* source = step == first_step ? p_poly : poly_ij;
      word* dividend = compute_dividend ? buf + ho : nullptr;
      // divide source by sparse polynomial reductor.
      if(debug)
      {
        cout << "j = " << j << endl;
        cout << "r+q*(P_v+cst):" << endl;
        print_series<word>(source, o, o);
      }
      euclidean_division(
            m_c_b, source, reductee_d, packed_logdegrees + c_b_t<word>::n * v, cst_term, num_coefficients[v],
            reductor_d, buf, dividend);
      if(debug)
      {
        cout << "r:" << endl;
        print_series<word>(buf, ho, ho);
        if(compute_dividend)
        {
          cout << "q:" << endl;
          print_series<word>(buf+ho, ho, ho);

        }
      }

      for(k = 0; k < ho; k++)   poly_ij[k]      = buf[k]; // r
      if(compute_dividend)
      {
        for(k = 0; k < ho; k++) poly_ij[k + ho] = buf[k] ^ buf[k + ho]; // r + q
        if(debug)
        {
          cout << "r+q:" << endl;
          print_series<word>(poly_ij + ho, ho, ho);
        }
      }
    }
  }
}

template<class word>
void additive_fft<word>::additive_fft_rev_ref(
    word* p_values,
    word* po_result,
    uint64_t blk_index
    )  const
{
  const uint64_t blk_size = 1uLL << m;
  unsigned int step, first_step, last_step, v;
  uint64_t i, j, k;

  first_step = c_b_t<word>::n - 1;
  last_step =  c_b_t<word>::n - m;
  // for first step, reductor_degree = ho <= p_poly_degree in all cases (since p_poly_degree > 0)

  if(debug)
  {
    print_series<word>(p_values, 1uLL << m, 1uLL << m);
  }

  for(k = 0; k < (1uLL << m); k++) po_result[k] = p_values[k];

  for(step = first_step; step != last_step - 1 ; step--)
  {
    v = c_b_t<word>::n - 1 - step; // polynomial index
    const uint64_t o  = 1uLL << (v + 1);
    const uint64_t ho = 1uLL << v;
    // log2 of number of blocks of size o covering the interval 0 ... 2**m - 1
    const int log_num_blocks = max(0, static_cast<int>(m - (v + 1)));
    const uint64_t num_blocks = 1uLL << log_num_blocks;
    //reductor = P_i
    const uint64_t multiplier_d = ho;
    if(debug)
    {
      cout << endl << dec << "step " << step << endl;
      cout << " output degree = " << o - 1 << endl;
      cout << " multiplying by P_" << v << ", of degree " << multiplier_d << endl;
      cout << " performing 2**" << log_num_blocks << " = " << num_blocks << " multiplications" << endl;
      fflush(stdout);
    }

    for(j = 0; j < num_blocks; j++)
    {
      word* poly_ij = po_result + j * o;
      // Compute constant term to add to P_v, v = m - 1 - step, so that P_v(u) = 0,
      // u = j * o + blk_index * blk_size
      // because of the properties of Cantor bases, P_v(u) = u >> v in beta representation
      const word cst_term_beta_repr = static_cast<word>((j * o + blk_index * blk_size) >> v);
      const word cst_term = m_c_b->beta_to_gamma(cst_term_beta_repr);

      // as arrays of size ho,
      // poly_ij = remainder
      // poly_ij + ho  = remainder + dividend
      // at the end of the loop iteration, poly_ij = remainder + dividend * (P_i + cst_term)

      if(debug)
      {
        cout << "j = " << j << endl;
        cout << "r:" << endl;
        print_series < word > (poly_ij, ho, ho);
      }
      for(k = 0; k < ho; k++)
      {
        buf[k] = poly_ij[k] ^ poly_ij[k + ho];
        poly_ij[k] ^= m_c_b->multiply(cst_term, buf[k]);
        poly_ij[k + ho] = 0;
      }
      if(debug)
      {
        cout << "q:" << endl;
        print_series < word > (buf, ho, ho);
      }
      // now
      // buf = dividend
      // poly_ij =  remainder + dividend * cst_term
      for(i = 0; i < num_coefficients[v]; i++)
      {
        uint64_t degree = 1uLL << packed_logdegrees[c_b_t<word>::n * v + i];
        assert(degree <= ho);
        for(k = 0; k < ho; k++)
        {
          poly_ij[k + degree] ^= buf[k];
        }
      }
      if(debug)
      {
        cout << "r+q*(P_v+cst):" << endl;
        print_series < word > (poly_ij, o, o);
      }
    }
  }
}

template<class word>
void additive_fft<word>::additive_fft_fast(
    word* p_poly,
    uint64_t p_poly_degree,
    word* po_result,
    uint64_t blk_index
    )  const
{
  const uint64_t blk_size = 1uLL << m;
  unsigned int step, first_step;
  uint64_t i, j, k;
  if(p_poly_degree == 0)
  {
    for(j = 0; j < blk_size; j++) po_result[j] = p_poly[0];
    return;
  }
  if constexpr(c_b_t<word>::n < 64)
  {
    word mult_order = static_cast<word>(0xFFFFFFFFuL); // assumes complement-2 representation
    if(c_b_t<word>::n == m && p_poly_degree >= mult_order) assert(0);
  }
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
      cout << " performing 2**0x" << hex << log_num_blocks << " = 0x" << num_blocks << " reductions" << dec << endl;
      fflush(stdout);
    }

    const bool compute_dividend = o <= blk_size;
    if(compute_dividend)
    {
      assert(reductee_d - reductor_d + 1 <= ho);
    }
    else
    {
      assert(num_blocks == 1);
    }

    for(j = 0; j < num_blocks; j++)
    {
      // Change constant term of P_i, so that P_i(u) = 0, u = j * o + blk_index * blk_size.
      // because of the properties of Cantor bases, P_i(u) = u >> i in beta representation
      const word pt_beta_repr = static_cast<word>((j * o + blk_index * blk_size) >> i);
      cst_coefficients[j] = m_c_b->beta_to_gamma(pt_beta_repr);
    }

    // divide source by sparse polynomial reductor, for num_blocks interleaved polynomials
    // simultaneously.
    // on input, except for first step,
    // coefficent ii of polynomial j is in p_poly[j + num_blocks * ii], ii < o.
    // at the end of euclidean division,
    // remainder coefficient ii of polynomial j is in buf[j + num_blocks * ii]
    // and (if computed)
    // dividend coefficient ii of polynomial j is in buf[j + num_blocks * (ii + ho)], ii < ho.
    uint64_t ii, ik;
    const uint64_t hos  = num_blocks * ho;
    const uint64_t kmax = num_blocks * (compute_dividend? o : ho);
    for(k = 0; k < kmax; k++) buf[k] = 0;

    for (ii = reductee_d; ii > reductor_d - 1; ii--)
    {
      const uint64_t ii_mod_reductor_degree = ii & (reductor_d - 1); // reductor_d is a power of 2
      // here, and in the whole loop, ii_mod_reductor_degree = ii mod reductor_d;
      // introduce a new monomial of the reductee
      word* buf_off1 = buf + (num_blocks * ii_mod_reductor_degree);
      if(step == first_step)
      {
        for(j = 0; j < num_blocks; j++)
        {
          coeffs_to_cancel[j] = buf_off1[j] ^ p_poly[ii];
          //add constant term
          buf_off1[j] = m_c_b->multiply(cst_coefficients[j], coeffs_to_cancel[j]);
        }
      }
      else
      {
        for(j = 0; j < num_blocks; j++)
        {
          coeffs_to_cancel[j] = buf_off1[j] ^ po_result[ii * num_blocks + j];
          //add constant term
          buf_off1[j] = m_c_b->multiply(cst_coefficients[j], coeffs_to_cancel[j]);
        }
      }

      if (compute_dividend)
      {
        word* buf_off2 = buf + ((ii - reductor_d)*num_blocks) + num_blocks*ho;
        // above formula could be simplified since reductor_d = ho, but would be even more cryptic
        for(j = 0; j < num_blocks; j++) buf_off2[j] = coeffs_to_cancel[j];
      }
      // non-constant terms
      for (ik = 0; ik < num_coefficients[i] - 1; ik++)
      {
        uint64_t idx = ((1uLL << (packed_logdegrees[c_b_t<word>::n * i + ik])) + ii_mod_reductor_degree) & (reductor_d-1);
        // idx = (packed_logdegrees[m * i + ik] + i) % reductor_d
        word* buf_off3 = buf + num_blocks*idx;
        for(j = 0; j < num_blocks; j++) buf_off3[j] ^= coeffs_to_cancel[j];
      }
    }

    if(step == first_step)
    {
      for (ii = 0; ii < reductor_d; ii++)
      {
        word* buf_off4 = buf + num_blocks * ii;
        for(j = 0; j < num_blocks; j++) buf_off4[j] ^= p_poly[ii];
      }
    }
    else
    {
      // reductor_d = ho ; num_blocks * reductor_d = hos
      for (k = 0; k < hos; k++) buf[k] ^= po_result[k];
    }
    // end of euclidean division

    if(compute_dividend)
    {
      // compute results r and r+q for all interleaved polynomials and re-interleave them
      for(k = 0; k < hos; k++)
      {
        po_result[    2 * k] = buf[k]; // r
        po_result[1 + 2 * k] = buf[k] ^ buf[hos + k]; // r + q
      }
    }
    else
    {
      for(k = 0; k < ho; k++) po_result[k] = buf[k]; // r
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
  packed_logdegrees[0] = 0;
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
      if(p_i[j]) packed_logdegrees[c_b_t<word>::n * i + num_coeffs++] = j;
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
      cout << "X^0x" << hex << (1uLL << packed_logdegrees[c_b_t<word>::n * i + k]);
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
