#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>
#include <bitset>
#include <functional>
#include <cstdint>

#include "cantor.h"
#include "helpers.hpp"

using namespace std;
using namespace std::chrono;

template<class word> class target{};

template<>
class target<uint8_t>{
public:
  static constexpr uint8_t result = 0x83u;
};

template<>
class target<uint16_t>{
public:
  static constexpr uint16_t result = 0xc95du;
};

template<>
class target<uint32_t>{
public:
  static constexpr uint32_t result = 0xea86b890uL;
};

template<>
class target<uint64_t>{
public:
  static constexpr uint64_t result = 0x5692c93ee7863cc1uLL;
};

#ifdef Boost_FOUND
template<>
class target<uint128_t>{
public:
  static constexpr uint128_t result = 0xC30575050DB3E3EEC2D23C2F7C9086E1_cppui128;
};
#else
template<>
class target<uint128_t>{
public:
  static constexpr uint128_t result = ((uint128_t) 0xC30575050DB3E3EEuLL) << 64 |
                                      ((uint128_t) 0xC2D23C2F7C9086E1uLL);
};
#endif

#ifdef HAS_UINT256
template<>
class target<uint256_t>{
public:
  static constexpr uint256_t result = 0x4DE0E79972B36BD4F26E0AD3B1FEEC291988FC1A69E60FCA4529FFC055175F30_cppui256;
};
constexpr uint256_t target<uint256_t>::result;
#endif

#ifdef HAS_UINT512
template<>
class target<uint512_t>{
public:
  static constexpr uint512_t result = 0xD2356CAF26417158A44F36DD0080A18764DA672A0ABCE14E70DA5662B8B52DF091AD8A090B539A98CC2F8BF82DEA2FB4E0B586AC5A9D5CA97EF36883E5DB87AF_cppui512;
};
constexpr uint512_t target<uint512_t>::result;
#endif

#ifdef HAS_UINT1024
template<>
class target<uint1024_t>{
public:
  static constexpr uint1024_t result = 0x66BF8106A7E4FC88C4C0DD0B78ADE2208C44DDCB8EFECF38A593A899AF45B54C7D68DF39DC934A948747E09C3D6FC53D16244C30051AA625484B4FD2D8820010D463745F801D91A5C56921A33FCB2ADC6C27384B44F0999FA0CEBEC6E71F654F780867D57747F93302873E251433C3083E913F254FE703ED281DDAE8578CB794_cppui1024;
};
constexpr uint1024_t target<uint1024_t>::result;
#endif

#ifdef HAS_UINT2048
template<>
class target<uint2048_t>{
public:
  static constexpr uint2048_t result = 0xF1CDBBCE9F34D315419BCC18D8D45846E887EFBDCF3219455972E1EE8476CF79F1D620E40FDFBA2CBE5D91B2F9EE498C5B2FD142F2EC5BE6E4C9F539A95F9C1C6A449AA0E13D8B141126E96F39A86B9F13DE7350F2048E727C3B58DCDB1309F8191993328E4FEF83C2392DF6082CBABB2414DA57B960B2F343C7FF6ACA3C9D2A31BE403892F0DB9B4798D62E236D270170756B4867B734127AAA715413A593FC3757806A43A59AC356B796D6E0681F1E642843CB56C385D2535F32ADD01954D1DB7996C1216DD479BF9839E97E9AA5F9120FB1FF5EF3F3813AA79AEFA355D658457DB08BDA1EB200C0730DFE9077C5C663816BAC8CE6F05F4A30CB1614342949_cppui2048;
};
constexpr uint2048_t target<uint2048_t>::result;
#endif


constexpr uint8_t target<uint8_t>::result;
constexpr uint16_t target<uint16_t>::result;
constexpr uint32_t target<uint32_t>::result;
constexpr uint64_t target<uint64_t>::result;
constexpr uint128_t target<uint128_t>::result;

static const size_t n_vals = 256;

template <class word> cantor_basis<word>* create_basis();
template <class word> word* create_values(function<word()> draw);
template <class word> void compute_primitive_beta(cantor_basis<word>& c);
template <class word> double test_ref(cantor_basis<word>* c, word* values);
template <class word> double test(cantor_basis<word>* c, word* values);
template <class word> void full_test();

void multiprecision_playground();

constexpr bool reduced_test = true;
constexpr bool print_matrices = false;
constexpr bool use_random_test_values = false;
constexpr bool compute_primitive_elements = false;
int main()
{
#if 1
  multiprecision_playground();
#endif

  if constexpr(reduced_test)
  {
    cout << "Reduced test" << endl;
    cout << "Testing f[2**8]" << endl;
    full_test<uint8_t>();
    cout << "Testing f[2**16]" << endl;
    full_test<uint16_t>();
    cout << "Testing f[2**32]" << endl;
    full_test<uint32_t>();
    cout << "Testing f[2**64]" << endl;
    full_test<uint64_t>();
#ifdef HAS_UINT2048
    cout << "Testing f[2**2048]" << endl;
    full_test<uint2048_t>();
#endif
  }
  else
  {
    // test of all basis sizes
    cout << "Testing f[2**8]" << endl;
    full_test<uint8_t>();

    cout << endl << "Testing f[2**16]" << endl;
    full_test<uint16_t>();

    cout << endl << "Testing f[2**32]" << endl;
    full_test<uint32_t>();

    cout << endl << "Testing f[2**64]" << endl;
    full_test<uint64_t>();

    cout << endl << "Testing f[2**128]" << endl;
    full_test<uint128_t>();

  #ifdef HAS_UINT256
    cout << endl << "Testing f[2**256]" << endl;
    full_test<uint256_t>();
  #endif

  #ifdef HAS_UINT512
    cout << endl << "Testing f[2**512]" << endl;
    full_test<uint512_t>();
  #endif

  #ifdef HAS_UINT1024
    cout << endl << "Testing f[2**1024]" << endl;
    full_test<uint1024_t>();
  #endif

  #ifdef HAS_UINT2048
    cout << endl << "Testing f[2**2048]" << endl;
    full_test<uint2048_t>();
  #endif
  }

  return 0;
}


template <class word> cantor_basis<word>* create_basis()
{
  time_point<high_resolution_clock> begin, end;
#if 0
  int res = test_solve<word>(true);
#endif

  begin = high_resolution_clock::now();
  auto c = new cantor_basis<word>;
  end = high_resolution_clock::now();
  double t = static_cast<double>(duration_cast<chrono::nanoseconds>(end-begin).count()) / pow(10, 9);

  auto ts = c->times();
  cout << "Total linear system solving time: " << ts.linalg_time() << " sec" << endl;
  cout << "Generator decomposition time (w/o linear algebra): " << ts.gendec_time() - ts.linalg_time() << " sec" << endl;
  cout << "Log/exp tables computation time: " << ts.logcomp_time() << " sec" << endl;
  cout << "data reuse time: " << ts.reuse_time() << " sec" << endl;
  cout << "total construction time: " << t << " sec" << endl;

  if(c->error())
  {
    cout << "Error " << c->error() << " building basis" << endl;
    return nullptr;
  }

  cout << "basis built" << endl;
  return c;
}

template <class word> void compute_primitive_beta(cantor_basis<word>& c)
{
  static const int n = c_b_t<word>::n; // not equal to **sizeof(word) for boost types which are larger than the underlying bitfields
  if(n > 128) return;
  cout << "Primitive elements among beta family:" << endl;
  // primitive elements generate the field multiplicatively.
  // the beta_i for i < n/2 generate a strict subfield and are therefore not primitive.
  for(unsigned int idx = n / 2; idx < n ; idx++)
  {
    cout << idx << ": " << is_primitive<word>(c.beta_over_gamma(idx), c) << endl;
  }
}

template <class word> double test_ref(cantor_basis<word>& c, word* values)
{
  time_point<high_resolution_clock> begin, end;
  int error_count = 0;
  bool errors = false;
  // tests whether multiply and multiply_ref return identical results.
  for(unsigned int i = 0; i < n_vals; i++)
  {
    word a = values[i];
    word b = values[i + 1];
    word m1 = c.multiply(a, b);
    word m2 = c.multiply_ref(a, b);
    if(m1 != m2)
    {
      if (error_count < 5) {
        cout << "Difference between multiply and multiply_ref for i=" << i << endl;
        cout << hex << m1 << endl;
        cout << hex << m2 << endl;
        cout << endl;
      }
      else if(error_count == 5)
      {
        cout << "..." << endl;
      }
      error_count++;
    }
  }
  if(error_count > 0) errors = true;
  cout << "number of errors in multiplication in gamma representation: " << dec << error_count << endl;
  // tests whether multiply_beta_repr and multiply_beta_repr_ref return identical results.
  error_count = 0;
  for(unsigned int i = 0; i < n_vals; i++)
  {
    word a = values[i];
    word b = values[i + 1];
    word m3 = c.multiply_beta_repr(a, b);
    word m4 = c.multiply_beta_repr_ref(a, b);
    if(m3 != m4)
    {
      if (error_count < 5) {
        cout << "Difference between multiply_beta_repr and multiply_beta_repr_ref for i=" << i << endl;
        cout << hex << m3 << endl;
        cout << hex << m4 << endl;
        cout << endl;
      }
      else if(error_count == 5)
      {
        cout << "..." << endl;
      }
      error_count++;
    }
  }
  if(error_count > 0) errors = true;
  cout << "number of errors in multiplication in beta representation: " << dec << error_count << endl;

  error_count = 0;
  for(size_t i = 0; i < n_vals; i++)
  {
    if(c.square(values[i]) != c.square_ref(values[i])) error_count++;
  }
  if(error_count > 0) errors = true;
  cout << "Differences between reference square and optimized square: " << error_count << endl;

  const word u = 0;
  const word orderminus1 = static_cast<word>(~u);
  error_count = 0;
  // tests that for any a != 0, a**(2**n - 1) == 1
  // where the field has 2**n elements
  for(int i = 0; i < 10; i++)
  {
    word a = values[i];
    word m1 = power(a, orderminus1, c);
    if(m1 != 1) error_count++;
  }
  if(error_count > 0) errors = true;
  cout << "number of errors in power: " << dec << error_count << endl;

  int repeat = 1024 / c_b_t<word>::n + 1;
  word sum_1 = 0, sum3 = 0;
  begin = high_resolution_clock::now();
  for(size_t i = 0; i < n_vals; i++) sum_1 += c.multiply_ref(values[i], values[i+1]);
  for(int j = 0; j < repeat - 1; j++)
  {
    for(size_t i = 0; i < n_vals; i++) sum3 += c.multiply_ref(values[i], values[i+1]);
  }
  end = high_resolution_clock::now();
  double t1 = static_cast<double>(duration_cast<chrono::nanoseconds>(end-begin).count()) / pow(10, 9);
  if(sum_1 != target<word>::result)
  {
    cout << "** ERROR: sum_1 differs from target **" << endl;
    cout << hex << sum_1 << endl;
    errors = true;
  }
  if(sum3 == 0) cout << "the impossible happened!" << endl; // to prevent the compiler from optimizing computations
  cout << "reference multiply time: " << t1 << endl;
  //cout << hex << setfill('0') << setw(c_b_t<word>::n/4) << sum1 << endl;

  if(errors)
  {
    cout << "Some tests failed" << endl;
  }
  else
  {
    cout << "All tests succeded" << endl;
  }
  return t1;
}

template <class word>
double test(cantor_basis<word>* c_b, word* values)
{
  time_point<high_resolution_clock> begin, end;

  int errors = 0;
  // check that a * (1/a) = 1 for all test values a != 0
  for(size_t i = 0; i < n_vals; i++)
  {
    if(values[i] != 0 && c_b->multiply(values[i], c_b->inverse(values[i]))!=1) errors++;
  }
  cout << "inverse errors: " << errors << endl;

  errors = 0;
  word w = 2;
  word wres = 3;
  // compute beta_i**2 using gamma representation, convert it back to beta representation,
  // and check that beta_i**2 = beta_{i-1} + beta_i
  for(size_t i = 1; i < c_b_t<word>::n; i++)
  {
    // w = beta_i in beta representation (i.e. 1 << i)
    word w_g = c_b->beta_to_gamma(w); // beta_i in gamma representation
    word res = c_b->gamma_to_beta(c_b->square(w_g));
    if(res != wres) errors++; // res should be equal to beta_i + beta_{i-1} in beta reprentation
    w <<= 1;
    wres <<= 1;
  }
  cout << "beta relation errors: " << errors << endl;

  if constexpr(c_b_t<word>::n ==32 || c_b_t<word>::n == 64)
  {
    errors = 0;
    for(size_t i = 0; i < n_vals - 1; i++)
    {
      word a = values[i];
      word b = values[i+1];
      word am = c_b->gamma_to_mult(a);
      word bm = c_b->gamma_to_mult(b);
      word pr = c_b->multiply_mult_repr(am, bm);
      if(c_b->mult_to_gamma(pr) != c_b->multiply(a, b)) errors++;
      if(pr != c_b->multiply_mult_repr_ref(am,bm)) errors++;
    }
    cout << "mult representation errors: " << errors << endl;
  }

  int repeat = 1024 / c_b_t<word>::n + 1;
  word sum_2 = 0, sum3 = 0;
  int opt_repeat = 128;
  begin = high_resolution_clock::now();
  for(size_t i = 0; i < n_vals; i++) sum_2 += c_b->multiply(values[i], values[i+1]);
  for(int j = 0; j < opt_repeat * repeat - 1; j++)
  {
    for(size_t i = 0; i < n_vals; i++) sum3 += c_b->multiply(values[i], values[i+1]);
  }
  end = high_resolution_clock::now();
  double t2 = static_cast<double>(duration_cast<chrono::nanoseconds>(end-begin).count()) / pow(10, 9);
  if(sum_2 != target<word>::result) cout << "** ERROR: sum_2 differs from target **" << endl;
  if(sum3 == 0) cout << "the impossible happened twice!" << endl; // to prevent the compiler from optimizing computations
  cout << "optimized multiply time: " << t2 << "; per run: " << t2 / opt_repeat << endl;

  if constexpr(print_matrices) print_beta_dec_matrix(c_b);

  return t2 / opt_repeat;
}


/**
 * Returns a word array with n_vals + 1 pseudorandom or random values
 */
template <class word> word* create_values(function<word()> draw)
{
  word* values = new word[n_vals + 1];
  for(size_t i = 0; i < n_vals + 1; i++) values[i] = draw();
  return values;
}

template<class word> void full_test()
{
  cout << endl << endl << "Full test for words of size " << c_b_t<word>::n << " bits" << endl;
  cantor_basis<word>* c_b = create_basis<word>();
  //FIXME: put engine, dist and draw in a templated class used to generate random or pseudorandom word values.

  // use a lambda evaluated immediately as an equivalent of a ternary operator
  auto engine = [](){
    if constexpr(use_random_test_values)
    {
      // non-deterministic random numbers
      random_device rd;
      mt19937_64 eng(rd());
      return eng;
    }
    else
    {
      // deterministic random numbers
      mt19937_64 eng;
      return eng;
    }
  }();
  uniform_int_distribution<typename c_b_t<word>::random_type > distr;

  // a lambda that returns a random word drawn following distribution 'distr' and using random source 'eng'
  auto draw = [&distr, &engine]()
  {
    static constexpr auto s = sizeof(typename c_b_t<word>::random_type);
    constexpr auto num_parts = c_b_t<word>::n / (8 * s);
    if constexpr(num_parts == 1) return static_cast<word>(distr(engine));
    else
    {
      word res = 0;
      for(size_t i = 0 ; i < num_parts; i++)
      {
        word r = distr(engine);
        res |= r << (i * 8 * s);
      }
      return res;
    }
  };

  if(!c_b)
  {
      cout << "** Basis creation failed, aborting test" << endl;
      return;
  }
  word* v = new word[n_vals + 1];
  for(size_t i = 0; i < n_vals + 1; i++) v[i] = draw();

  cout << "** Tests on multiplication and inverse" << endl;
  double t1 = test(c_b, v);
  if constexpr(cantor_basis<word>::has_ref_product())
  {
    cout << "** Basis has a reference multication implementation: test it and compare it to the optimized implementation" << endl;
    double t2 = test_ref(*c_b, v);
    cout <<"** Optimized / reference speedup: " << t2 / t1 << endl;
  }
  if constexpr(compute_primitive_elements) compute_primitive_beta(*c_b);

  delete c_b;
  delete[] v;
}
void multiprecision_playground()
{
#ifdef Boost_FOUND
  uint256_t x =   0x600000000000000540000000000000032000000000000001_cppui256;
  cout << "sizeof(uint256_t): " << sizeof(uint256_t) << " (can differ from 32)" << endl;
  cout << "sizeof(boost::multiprecision::limb_type): " << sizeof(boost::multiprecision::limb_type) << endl;

  cout << "limb 0 of uint256_t 0x600000000000000540000000000000032000000000000001: " << hex << x.backend().limbs()[0] << endl;
  cout << "limb 1 of uint256_t 0x600000000000000540000000000000032000000000000001: " << hex << x.backend().limbs()[1] << endl;
  cout << "limb 2 of uint256_t 0x600000000000000540000000000000032000000000000001: " << hex << x.backend().limbs()[2] << endl;

  uint128_t y = 0x30000000000000000_cppui128;
  cout << "sizeof(uint128_t): " << sizeof(uint128_t) << endl;
  cout << "sizeof(boost::multiprecision::uint128_t): " << sizeof(boost::multiprecision::uint128_t) << endl;
  cout << "sizeof(uint128_t limb): " << sizeof(*y.backend().limbs()) << endl;

#else
  uint128_t y = 1;
  y <<= 64;
  cout << bitset<numeric_limits<typeof(y)>::digits>(y).count();
#endif
}

template <class word>
void print_beta_dec_matrix(cantor_basis<word>* c_b)
{
  const unsigned int n = c_b_t<word>::n;
  word u = 1;
  word m1 = static_cast<word>((u << (n/2)) - 1);
  cout << "decomposition of beta_i * beta_n/2 on gamma basis, i = 0 ... n/2-1:" << endl;
  for(unsigned int i = n / 2; i < n; i++)
  {
    word z_g = c_b->beta_over_gamma(i);
    word p1 = c_b->gamma_to_beta(z_g & m1);
    word p2 = c_b->gamma_to_beta((z_g >> (n/2)));
    p1 |= p2 << (n/2);
    print_row(p1);
  }

  cout << endl;
  for(unsigned int i = 1; i < n-1; i++)
  {
    word w = c_b->beta_to_gamma(c_b->gamma_to_beta(u<<i)<< 1);
    print_row(w);
  }

#if 0
  for(unsigned int i = n / 2; i < n; i++)
  {
    word p1 = c_b->multiply(c_b->beta_over_gamma(i-n/2), c_b->beta_over_gamma(n/2)) ^ c_b->beta_over_gamma(i);
    cout << ((p1 & m2) == 0) << endl;
  }
#endif
}
