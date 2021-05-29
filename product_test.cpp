#include <cstdint>
#include <iostream>
#include <random> // for mt19937_64, uniform_int_distribution<>
#include <memory> // for unique_ptr

#include "timer.h"
#include "mg.h"
#include "utils.h"
#include "sha256.h"

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
// choose between in-place or out-of-place product variants of Mateer-Gao product
constexpr bool test_in_place_variant = true;
constexpr bool test_naive_product = false; // test or benchmark a naive implementation of polynomial
// product instead of the DFT-based one. Forces an out-of-place computation.

constexpr unsigned int gf2x_max_logsize = 35;
constexpr unsigned int naive_max_logsize = 18;

constexpr bool compare_with_sha2 = true;


// benchmark Mateer-Gao product
// (if false, only one run is done for each tested size; if true, many runs are done
// when runs are quick, as controlled by variables below)
constexpr bool benchmark = true;
// maximum execution time for each test (except if 1 run exceeds this time)
constexpr double min_time = 2.;
// maximum number of runs for each test
constexpr uint64_t max_runs = 100;
//constexpr uint64_t max_runs = 1;

// compile-time test that platform is little endian
// (required by the implementation of 'contract' and 'expand' in mg.cpp)
// gcc only
// static_assert( __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__);

static bool mateer_gao_product_test(
    unsigned int* log_sz,
    unsigned int num_sz,
    bool benchmark)
{
  uint64_t run_count;
  double t1 = 0, t2 = 0;
  timer tm;
  bool local_error, error = false;
  cout << "Testing F2 polynomial product through Mateer-Gao DFT in GF(2**64)" << endl;
  cout << "Minimum execution time per test: " << min_time << " sec." << endl;
  cout << "Maximum number of runs per test: " << max_runs << endl;
  mt19937_64 engine;
  uniform_int_distribution<uint64_t> distr;
  auto draw = [&distr, &engine]() {return distr(engine);};
  constexpr uint32_t extract_size = compare_with_sha2 ? 4 : 6;
  uint64_t e1[extract_size];
  uint64_t e2[extract_size];
  for(int i = 0; i < extract_size; i++)
  {
    e1[i] = 0;
    e2[i] = 0;
  }
  // compute products of polynomials over F2 with Mateer-Gao DFT
  // optionally check the result with gf2x
  for(unsigned int j = 0; j < num_sz; j++)
  {
    const uint64_t lsz = log_sz[j];
    // result bit size
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
        do
        {
          tm.set_start();
          unsigned long ulsz = (unsigned long) (sz / (16*sizeof(unsigned long)));
          gf2x_mul((unsigned long *) p3,(unsigned long *) p1, ulsz, (unsigned long *) p2, ulsz);
          t1 += tm.measure();
          if(run_count == 0)
          {
            if constexpr(compare_with_sha2)
            {
              SHA256_CTX c;
              sha256_init(&c);
              sha256_update(&c, (BYTE*) p3, sz/8);
              sha256_final(&c, (BYTE*) e1);
              e_sz = 4;
            }
            else
            {
              e_sz =  extract<uint64_t>(p3, sz/64, e1, extract_size);
            }
          }
          run_count++;
        }
        while(benchmark && (t1 <= min_time) && (run_count < max_runs));
        t1 /= (double) run_count;
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
    run_count = 0;
    do
    {
      tm.set_start();
      if constexpr(compute_in_place)
      {
        mg_binary_polynomial_multiply_in_place(p1, p2, sz/2 - 1, sz/2 - 1);
      }
      else
      {
        if constexpr(test_naive_product)
        {
          if(do_naive_local) naive_product(p1, sz/128, p2, sz/128, p3);
        }
        else
        {
          mg_binary_polynomial_multiply(p1, p2, p3, sz/2 - 1, sz/2 - 1);
        }
      }
      t2 += tm.measure();
      uint64_t* result = compute_in_place ? p1 : p3;
      if(compare && (run_count == 0))
      {
        if constexpr(compare_with_sha2)
        {
          SHA256_CTX c;
          sha256_init(&c);
          sha256_update(&c, (BYTE*) result, sz/8);
          sha256_final(&c, (BYTE*) e2);
        }
        else
        {
          extract<uint64_t>(result, sz/64, e2, extract_size);
        }
      }
      run_count++;
    }
    while(benchmark && (t2 <= min_time) && (run_count < max_runs));
    t2 /= (double) run_count;
    cout << " Mateer-Gao runs: " << run_count << endl;
    cout << " Mateer-Gao product time per run: " << t2 << " sec." << endl;
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

int main(int UNUSED(argc), char** UNUSED(argv))
{
  // FIXME log_sz = 7 (product of 2 64-bit polynomials) shows a difference between both
  // the naive product implementation and the DFT implentation differ from gf2x.
  unsigned int log_sz[] = {10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37};
  //unsigned int log_sz[] = {8,9,10,11,12,13,14,15,16,17,18};
  cpu_features f;
  bool cpu_has_required_flags = f.has_popcnt && f.has_bmi1 && f.has_avx2 && f.has_pclmul;
  if(!cpu_has_required_flags)
  {
    cout << "cpu flags required not present (AVX2, PCMULQDQ, POPCNT, BMI1). Exiting" << endl;
    exit(2);
  }
  unsigned int num_sz = sizeof(log_sz) / sizeof(unsigned int);
  bool mateer_gao_error = mateer_gao_product_test(log_sz, num_sz, benchmark);
  if(mateer_gao_error) cout << "Mateer-Gao product test failed" << endl;
  return mateer_gao_error? EXIT_FAILURE : EXIT_SUCCESS;
}
