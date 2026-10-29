#include <cstdint>
#include <immintrin.h>
#include <iostream>
#include <random> // for mt19937_64, uniform_int_distribution<>
#include <memory> // for unique_ptr
#include <vector>
#include <string>
#include <algorithm> // for copy, fill, equal, max

#include "timer.h"
#include "mg.h"
#include "mg_fused.h"
#include "mg_templates.h"
#include "mg_taylor_opt.h"
#include "utils.h"

#include "gf2x.h"

using namespace std;

// test results against gf2x (unavailable with MSVC)
#ifdef __GNUC__
constexpr bool do_gf2x = true;
#else
#ifdef _MSC_VER
constexpr bool do_gf2x = false;
#else
#error "unsupported compiler"
#endif
#endif
// choose between the plain Mateer-Gao product (mg.h) and the variant with fused small stages (mg_fused.h)
// (default; see command line options in main)
static bool use_fused_variant = false;
// choose between in-place or out-of-place product variants of Mateer-Gao product
constexpr bool test_in_place_variant = true;
constexpr bool test_naive_product = false; // test or benchmark a naive implementation of polynomial
// product instead of the DFT-based one. Forces an out-of-place computation.

// largest size at which the result is compared with gf2x's (default; see command line options in main)
static unsigned int gf2x_max_logsize = 25;
constexpr unsigned int naive_max_logsize = 18;


// benchmark Mateer-Gao product
// (if false, only one run is done for each tested size; if true, many runs are done
// when runs are quick, as controlled by variables below)
constexpr bool benchmark = true;
// maximum execution time for each test (except if 1 run exceeds this time)
constexpr double min_time = 2.;
// maximum number of runs for each test
constexpr uint64_t max_runs = 1000000000;
//constexpr uint64_t max_runs = 1;
// minimum duration of a batch of runs (see time_calls)
constexpr double batch_min_time = 1e-3;

// compile-time test that platform is little endian
// (required by the implementation of 'contract' and 'expand' in mg.cpp)
// gcc only
// static_assert( __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__);

// Calls f repeatedly and returns the average duration of a call in seconds.
// The first call is made alone; once it is done, after_first is called (to extract its result) and,
// if 'benchmark' is set and the first call lasted less than min_time, more calls are made in batches,
// until min_time seconds have elapsed or max_runs calls have been made. Calls are timed by batch
// of at least batch_min_time seconds (batches get larger as long as they are shorter), because the
// cost of reading the clock is not negligible with respect to the duration of a small product
// (about 25 ns). The first call, which is slower than the others (cold caches), is only used if
// there are no other calls.
// run_count is set to the number of calls.
template<class F, class G> static double time_calls(F f, G after_first, uint64_t& run_count)
{
  timer tm;
  tm.set_start();
  f();
  const double t_first = tm.measure();
  after_first();
  run_count = 1;
  if(!benchmark || t_first > min_time) return t_first;
  double total = 0;
  uint64_t batch = 1, calls = 0;
  while(total <= min_time && run_count < max_runs)
  {
    tm.set_start();
    for(uint64_t i = 0; i < batch; i++) f();
    const double t_batch = tm.measure();
    total += t_batch;
    calls += batch;
    run_count += batch;
    if(t_batch < batch_min_time) batch *= 2;
  }
  return calls == 0 ? t_first : total / (double) calls;
}

// compares the optimized Taylor decompositions (mg_taylor_opt.h) with the original ones
// (mg_templates.h) on random arrays, for the sizes allowed (t < logsize <= 2t) and for several strides
template<unsigned int logstride, unsigned int t>
static uint64_t taylor_test_one(mt19937_64& engine, uint64_t& errors)
{
  uint64_t cases = 0;
  for(unsigned int logsize = t + 1; logsize <= 2 * t; logsize++)
  {
    const uint64_t n = 1uLL << (logstride + logsize);
    vector<uint64_t> a(n), r1, r2;
    for(auto& x: a) x = engine();
    r1 = a; r2 = a;
    mg_decompose_taylor_recursive<logstride, t, uint64_t>(logsize, r1.data());
    mg_decompose_taylor_opt<logstride, t, uint64_t>(logsize, r2.data());
    bool ok = (r1 == r2);
    r1 = a; r2 = a;
    mg_decompose_taylor_reverse_recursive<logstride, t>(logsize, r1.data());
    mg_decompose_taylor_reverse_opt<logstride, t, uint64_t>(logsize, r2.data());
    ok = ok && (r1 == r2);
    // 32-bit words
    vector<uint32_t> a32(n), s1, s2;
    for(auto& x: a32) x = (uint32_t) engine();
    s1 = a32; s2 = a32;
    mg_decompose_taylor_recursive<logstride, t, uint32_t>(logsize, s1.data());
    mg_decompose_taylor_opt<logstride, t, uint32_t>(logsize, s2.data());
    ok = ok && (s1 == s2);
    cases++;
    if(!ok)
    {
      errors++;
      cout << " ** Wrong Taylor decomposition ** logstride = " << logstride << ", t = " << t << ", logsize = " << logsize << endl;
    }
  }
  return cases;
}

static bool taylor_test()
{
  cout << "Testing the optimized Taylor decompositions" << endl;
  mt19937_64 engine(2);
  uint64_t cases = 0, errors = 0;
  for(int rep = 0; rep < 20; rep++)
  {
    cases += taylor_test_one<0, 1>(engine, errors) + taylor_test_one<0, 2>(engine, errors)
        + taylor_test_one<0, 3>(engine, errors) + taylor_test_one<0, 4>(engine, errors)
        + taylor_test_one<0, 5>(engine, errors);
    cases += taylor_test_one<1, 1>(engine, errors) + taylor_test_one<2, 2>(engine, errors)
        + taylor_test_one<4, 2>(engine, errors) + taylor_test_one<3, 3>(engine, errors)
        + taylor_test_one<8, 4>(engine, errors) + taylor_test_one<5, 4>(engine, errors)
        + taylor_test_one<7, 5>(engine, errors);
  }
  cout << " " << cases << " cases, " << errors << " errors" << endl;
  return errors != 0;
}

// naive product of binary polynomials, for the degree tests below
static void naive_bit_product(const uint64_t* a, uint64_t d1, const uint64_t* b, uint64_t d2, uint64_t* r)
{
  for(uint64_t i = 0; i <= d1; i++)
  {
    if(((a[i / 64] >> (i % 64)) & 1) == 0) continue;
    for(uint64_t j = 0; j <= d2; j++)
    {
      if((b[j / 64] >> (j % 64)) & 1) r[(i + j) / 64] ^= 1uLL << ((i + j) % 64);
    }
  }
}

// same, 64 times faster for large degrees (schoolbook product on words, with carry-less multiplications)
static void naive_word_product(const uint64_t* a, uint64_t d1, const uint64_t* b, uint64_t d2, uint64_t* r)
{
  for(uint64_t i = 0; i <= d1 / 64; i++)
  {
    for(uint64_t j = 0; j <= d2 / 64; j++)
    {
      const __m128i x = _mm_clmulepi64_si128(_mm_cvtsi64_si128((long long) a[i]), _mm_cvtsi64_si128((long long) b[j]), 0);
      r[i + j] ^= (uint64_t) _mm_cvtsi128_si64(x);
      r[i + j + 1] ^= (uint64_t) _mm_extract_epi64(x, 1);
    }
  }
}

// tests the in-place and out-of-place products of both variants against a naive product,
// for all pairs d1 >= d2 of degrees up to max_degree (small degrees only, in particular
// operands of at most 64 bits, and degrees that are not of the form 2**k - 1), with
// minimal buffer sizes. Bits of the result above its degree must be zero.
static bool degree_test(uint64_t max_degree)
{
  cout << "Testing products of polynomials of all degrees d1 >= d2, d1 <= " << max_degree << endl;
  mt19937_64 engine(1);
  uint64_t cases = 0, errors = 0;
  vector<pair<uint64_t, uint64_t>> pairs;
  for(uint64_t d1 = 0; d1 <= max_degree; d1++)
    for(uint64_t d2 = 0; d2 <= d1; d2 += (d1 < 140) ? 1 : 7) pairs.push_back({d1, d2});
  // larger degrees, around the sizes at which the base case (small products) stops being used
  for(uint64_t d1: {511, 512, 1023, 1024, 2047, 2048, 4095, 4096, 8191, 8192, 9000, 16383, 32767, 32768, 33000})
    for(uint64_t d2: {uint64_t(0), uint64_t(1), uint64_t(63), uint64_t(64), uint64_t(255), d1 / 3, d1}) pairs.push_back({d1, d2});
  for(const auto& dd: pairs)
  {
    const uint64_t d1 = dd.first, d2 = dd.second;
    {
      // sizes in 32-bit words, as in the product functions
      unsigned int lg = 0, l2 = 0;
      while((1uLL << lg) * 32 < (d1 + d2 + 1)) lg++;
      while((1uLL << l2) * 32 < (d2 + 1)) l2++;
      const uint64_t nw = (d1 + d2) / 64 + 1;
      vector<uint64_t> a(d1 / 64 + 1, 0), b(d2 / 64 + 1, 0), expected(nw + 1, 0);
      for(uint64_t k = 0; k <= d1; k++) if(engine() & 1) a[k / 64] |= 1uLL << (k % 64);
      for(uint64_t k = 0; k <= d2; k++) if(engine() & 1) b[k / 64] |= 1uLL << (k % 64);
      if(d1 * d2 < 20000000) naive_bit_product(a.data(), d1, b.data(), d2, expected.data());
      else naive_word_product(a.data(), d1, b.data(), d2, expected.data());
      expected.resize(nw);
      for(int variant = 0; variant < 2; variant++)
      {
        // out of place: result buffer filled with garbage
        vector<uint64_t> p1 = a, p2 = b, r(nw, 0xaaaaaaaaaaaaaaaauLL);
        if(variant == 0) mg_binary_polynomial_multiply(p1.data(), p2.data(), r.data(), d1, d2);
        else mg_fused::mg_binary_polynomial_multiply(p1.data(), p2.data(), r.data(), d1, d2);
        bool ok = (r == expected);
        // in place: p1 has 2**lg words, p2 as few as allowed
        vector<uint64_t> q1(1uLL << lg, 0xaaaaaaaaaaaaaaaauLL), q2(max<uint64_t>(max<uint64_t>(1, b.size()), (1uLL << l2) / 2), 0);
        copy(a.begin(), a.end(), q1.begin());
        copy(b.begin(), b.end(), q2.begin());
        // bits above the degree of the input polynomials must be zero
        fill(q1.begin() + (long) a.size(), q1.end(), 0);
        if(variant == 0) mg_binary_polynomial_multiply_in_place(q1.data(), q2.data(), d1, d2);
        else mg_fused::mg_binary_polynomial_multiply_in_place(q1.data(), q2.data(), d1, d2);
        ok = ok && equal(expected.begin(), expected.end(), q1.begin());
        cases++;
        if(!ok)
        {
          errors++;
          if(errors <= 10) cout << " ** Wrong result ** (" << (variant == 0 ? "plain" : "fused") << ") d1 = " << d1 << ", d2 = " << d2 << endl;
        }
      }
    }
  }
  cout << " " << cases << " cases, " << errors << " errors" << endl;
  return errors != 0;
}

static bool mateer_gao_product_test(
    unsigned int* log_sz,
    unsigned int num_sz)
{
  uint64_t run_count;
  double t1 = 0, t2 = 0;
  bool local_error, error = false;
  cout << "Testing F2 polynomial product through Mateer-Gao DFT in GF(2**64)" << endl;
  cout << "Minimum execution time per test: " << min_time << " sec." << endl;
  cout << "Maximum number of runs per test: " << max_runs << endl;
  mt19937_64 engine;
  uniform_int_distribution<uint64_t> distr;
  auto draw = [&distr, &engine]() {return distr(engine);};
  constexpr uint32_t extract_size = 6;
  uint64_t e1[extract_size];
  uint64_t e2[extract_size];
  // compute products of polynomials over F2 with Mateer-Gao DFT
  // optionally check the result with gf2x
  for(unsigned int j = 0; j < num_sz; j++)
  {
    const uint64_t lsz = log_sz[j];
    const uint64_t sz = 1uLL << lsz;
    const bool do_gf2x_local = do_gf2x && lsz <= gf2x_max_logsize;
    const bool do_naive_local = do_gf2x && lsz <= naive_max_logsize;
    // delay large allocation needed by in-place variant of MG product when comparing to gf2x,
    // in order for gf2x to have as much available space as possible
    constexpr bool compute_in_place = test_in_place_variant && !test_naive_product;
    const bool large_alloc_first = !do_gf2x_local && compute_in_place;
    const bool delayed_large_alloc = do_gf2x_local && compute_in_place;
    bool compare = do_gf2x_local && (!test_naive_product || do_naive_local);

    // buffer size needed, in bytes
    cout << endl;
    cout << "Multiplying 2 polynomials of degree (2**" << lsz-1 << ")-1 = " << sz/2 - 1 << endl;
    uint64_t* p1 = nullptr;
    uint64_t* p2 = nullptr;
    uint64_t* p3 = nullptr;
    try
    {
      p1 = new uint64_t[large_alloc_first?sz/32:sz/128];
      p2 = new uint64_t[sz/128];
      if(do_gf2x_local) p3 = new uint64_t[sz/64];
    }
    catch(bad_alloc&)
    {
      cout << "Not enough memory for current test" << endl;
      continue;
    }
    unique_ptr<uint64_t[]> _1(p1);
    unique_ptr<uint64_t[]> _2(p2);
    unique_ptr<uint64_t[]> _3(p3);
    // create two random polynomials with sz/2 coefficients
    engine.seed(lsz);
    for(uint64_t i = 0; i < sz / 128; i++)
    {
      p1[i] = draw();
      p2[i] = draw();
    }
    uint32_t e_sz = 0;
    if constexpr(do_gf2x)
    {
      run_count = 0;
      if(do_gf2x_local)
      {
        cout << " Performing product with gf2x" << endl;
        const unsigned long ulsz = (unsigned long) (sz / (16*sizeof(unsigned long)));
        t1 = time_calls(
            [&]{gf2x_mul((unsigned long *) p3,(unsigned long *) p1, ulsz, (unsigned long *) p2, ulsz);},
            [&]{e_sz =  extract<uint64_t>(p3, sz/64, e1, extract_size);},
            run_count);
        cout << " gf2x runs: " << run_count << endl;
        cout << " gf2x time per run: " << t1 << " sec." << endl;
        // reset result
        for(uint64_t i = 0; i < sz/64; i++) p3[i] = 0;
      }
    }
    if(delayed_large_alloc)
    {
      try
      {
        uint64_t* tmp = new uint64_t[sz/32];
        {
          uint64_t i = 0;
          for(; i < sz/128; i++) tmp[i] = p1[i];
          for(; i < sz/32; i++) tmp[i] = 0;
        }
        p1 = tmp;
        _1.reset(tmp);
      }
      catch(bad_alloc&)
      {
        cout << "Not enough memory for current test" << endl;
        continue;
      }
    }

    cout << " Performing product with MG DFT" << endl;
    auto mg_product = [&]
    {
      if constexpr(compute_in_place)
      {
        if(use_fused_variant) mg_fused::mg_binary_polynomial_multiply_in_place(p1, p2, sz/2 - 1, sz/2 - 1);
        else mg_binary_polynomial_multiply_in_place(p1, p2, sz/2 - 1, sz/2 - 1);
      }
      else
      {
        if constexpr(test_naive_product)
        {
          if(do_naive_local) naive_product(p1, sz/128, p2, sz/128, p3);
        }
        else
        {
          if(use_fused_variant) mg_fused::mg_binary_polynomial_multiply(p1, p2, p3, sz/2 - 1, sz/2 - 1);
          else mg_binary_polynomial_multiply(p1, p2, p3, sz/2 - 1, sz/2 - 1);
        }
      }
    };
    t2 = time_calls(
        mg_product,
        [&]
        {
          uint64_t* result = compute_in_place ? p1 : p3;
          if(compare) extract<uint64_t>(result, sz/64, e2, extract_size);
        },
        run_count);
    cout << " Mateer-Gao runs: " << run_count << endl;
    cout << " Mateer-Gao product time per run: " << t2 << " sec." << endl;
    // line meant to be parsed by scripts (benchmark.sh): size, gf2x time (0 if not run), Mateer-Gao time
    cout << "RESULT " << lsz << " " << (do_gf2x_local ? t1 : 0.) << " " << t2 << endl;
    if(compare)
    {
      cout << " MG / gf2x speed ratio: " << t1 / t2 << endl;
      // compare results
      local_error = false;
      for(uint32_t i = 0; i < e_sz; i++)
      {
        if(e1[i] != e2[i])
        {
          local_error = true;
          cout << i << ": " << hex << e1[i] << "!=" << e2[i] << dec << endl;
        }
      }
      cout << "Checking result against gf2x: ";
      if(!local_error) cout << " ok" << endl;
      else cout << " ** Wrong result **" << endl;
      error |= local_error;
    }
  }
  return error;
}

static void usage(const char* name)
{
  cout << "usage: " << name << " [options]" << endl
       << "  --fused            use the fused variant of the product (mg_fused.h) instead of the plain one" << endl
       << "  --sizes LIST       output sizes (log2 of the number of bits) to test, e.g. 8-19,25,30,34 (default)" << endl
       << "  --gf2x-max N       compare with gf2x (and time it) up to size N only (default " << gf2x_max_logsize << ")" << endl
       << "  --no-degree-test   do not run the tests on small degrees" << endl
       << "  --help             this help" << endl
       << "The in-place product of size 2**u bits needs 3.5 * 2**(u-3) bytes of RAM (56 GiB for u = 37)." << endl;
}

// parse a list of sizes such as "8-19,25,30"
static bool parse_sizes(const string& s, vector<unsigned int>& sizes)
{
  sizes.clear();
  size_t pos = 0;
  while(pos <= s.size())
  {
    size_t end = s.find(',', pos);
    if(end == string::npos) end = s.size();
    const string item = s.substr(pos, end - pos);
    try
    {
      const size_t dash = item.find('-');
      const unsigned long lo = stoul(item.substr(0, dash));
      const unsigned long hi = (dash == string::npos) ? lo : stoul(item.substr(dash + 1));
      if(lo < 7 || hi < lo || hi > 60) return false;
      for(unsigned long i = lo; i <= hi; i++) sizes.push_back((unsigned int) i);
    }
    catch(const exception&) {return false;}
    pos = end + 1;
  }
  return !sizes.empty();
}

int main(int argc, char** argv)
{
  // log_sz = 7 (product of 2 64-bit polynomials) is covered by degree_test
  vector<unsigned int> log_sz = {8,9,10,11,12,13,14,15,16,17,18,19,25,30,34};
  bool run_degree_test = true;
  for(int i = 1; i < argc; i++)
  {
    const string a = argv[i];
    if(a == "--fused") use_fused_variant = true;
    else if(a == "--no-degree-test") run_degree_test = false;
    else if(a == "--gf2x-max" && i + 1 < argc) gf2x_max_logsize = (unsigned int) stoul(argv[++i]);
    else if(a == "--sizes" && i + 1 < argc)
    {
      if(!parse_sizes(argv[++i], log_sz)) {cout << "invalid list of sizes" << endl; return 1;}
    }
    else {usage(argv[0]); return a == "--help" ? 0 : 1;}
  }
  cpu_features f;
  bool cpu_has_required_flags = f.has_popcnt && f.has_bmi1 && f.has_avx2 && f.has_pclmul;
#if defined(__VPCLMULQDQ__) && defined(__AVX2__)
  // the build targets VPCLMULQDQ (see MG_USE_YMM_CLMUL in mg_templates.h)
  cpu_has_required_flags = cpu_has_required_flags && f.has_vpclmul;
#endif
  if(!cpu_has_required_flags)
  {
    cout << "cpu flags required not present (AVX2, PCMULQDQ, POPCNT, BMI1, and VPCLMULQDQ if the build targets it). Exiting" << endl;
    exit(2);
  }
  cout << "Variant: " << (use_fused_variant ? "fused" : "plain") << endl;
  bool mateer_gao_error = run_degree_test ? (taylor_test() | degree_test(300)) : false;
  mateer_gao_error |= mateer_gao_product_test(log_sz.data(), (unsigned int) log_sz.size());
  if(mateer_gao_error) cout << "Mateer-Gao product test failed" << endl;
  return mateer_gao_error? EXIT_FAILURE : EXIT_SUCCESS;
}
