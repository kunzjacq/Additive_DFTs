#include <cstdint>
#include <iostream>
#include <chrono>
#include <cmath>
#include <random>
#include <memory>

#include "timer.h"
#include "mg.h"
#include "utils.h"

#include "gf2x.h"

using namespace std;

constexpr bool benchmark = true;
constexpr bool do_gf2x = false;

unsigned int extract(uint8_t* tbl, uint64_t sz, uint8_t* extract, uint32_t extract_sz)
{
  if(sz <= extract_sz)
  {
    for(unsigned int i = 0; i < sz; i++) extract[i] = tbl[i];
    return (uint32_t) sz;
  }
  else
  {
    uint32_t a = extract_sz / 3;
    uint32_t c = extract_sz - 2*a;
    uint32_t i = 0;
    uint64_t j = 0;
    for(; i < a; i++, j++) extract[i] = tbl[j];
    j = sz/2;
    for(; i < 2*a; i++, j++) extract[i] = tbl[j];
    j = sz - c;
    for(; i < extract_sz; i++, j++) extract[i] = tbl[j];
    return extract_sz;
  }
}

static bool mateer_gao_product_test(
    unsigned int* log_sz,
    unsigned int num_sz,
    bool benchmark,
    bool do_gf2x)
{
  uint64_t i;
  double t1 = 0, t2 = 0, min_time = 5.;
  uint64_t max_runs = 1000;

  timer tm;
  bool local_error, error = false;
  unsigned int lsz;
  uint64_t sz;
  cout << "Testing F2 polynomial product through Mateer-Gao DFT in GF(2**64)" << endl;
  cout << "Minimum execution time per test: " << min_time << " sec." << endl;
  cout << "Maximum number of runs per test: " << max_runs << endl;

  mt19937_64 engine;
  uniform_int_distribution<uint8_t> distr;
  auto draw = [&distr, &engine]() {return distr(engine);};

  constexpr uint32_t extract_size = 48;
  uint8_t e1[extract_size];
  uint8_t e2[extract_size];

  // computing products of polynomials over f2 with Mateer-Gao DFT and
  // checking the result with gf2x.
  for(unsigned int j = 0; j < num_sz; j++)
  {
    lsz = log_sz[j];
    if(lsz > 64) continue;
    sz = 1uLL << lsz;
    uint64_t needed_buf_bytesize = sz >> 2; // buffer size needed, in bits
    cout << endl << "Multiplying 2 polynomials of degree (2**" << lsz-1 << ")-1 = " <<
            sz/2 - 1 << endl;
    uint64_t* buf = nullptr;
    try
    {
      buf = new uint64_t[needed_buf_bytesize >> 3];
    }
    catch(bad_alloc&)
    {
      cout << "Not enough memory for current test" << endl;
      continue;
    }
    unique_ptr<uint64_t[]>_(buf);

    uint8_t* p1 = (uint8_t*) buf; // byte size sz/16
    uint8_t* p2 = p1 + sz/16; // byte size sz/16
    uint8_t* p3 = p2 + sz/16; // byte size sz/8

    // create two random polynomials with sz/2 coefficients
    for(size_t i = 0; i < sz / 16; i++)
    {
      p1[i] = draw();
      p2[i] = draw();
    }
    tm.set_start();
    i = 0;
    uint32_t e_sz = 0;
    if(do_gf2x)
    {
      cout << " Performing product with gf2x" << endl;
      do
      {
        gf2x_mul((unsigned long *) p3,(unsigned long *) p1, sz/(16*sizeof(unsigned long)),
                 (unsigned long *) p2, sz/(16*sizeof(unsigned long)));
        if(i == 0) e_sz =  extract(p3, sz/8, e1, extract_size);
        i++;
        t1 = tm.measure();
      }
      while(benchmark && (t1 <= min_time) && (i < max_runs));
      t1 /= i;
      cout << " gf2x iterations: " << i << endl;
      cout << " gf2x time per iteration: " << t1 << " sec." << endl;
      // reset result
      for(uint64_t i = 0; i < sz/8; i++) p3[i] = 0;
    }

    cout << " Performing product with MG DFT" << endl;
    tm.set_start();
    i = 0;
    do
    {
      mg_binary_polynomial_multiply(p1, p2, p3, sz/2 - 1, sz/2 - 1);
      if(i == 0) extract(p3, sz/8, e2, extract_size);
      i++;
      t2 = tm.measure();

    }
    while(benchmark && (t2 <= min_time) && (i < max_runs));
    t2 /= i;

    cout << " Mateer-Gao iterations: " << i << endl;
    cout << " Mateer-Gao product time per iteration: " << t2 << " sec." << endl;

    if(do_gf2x)
    {
      cout << " MG / gf2x speed ratio: " << t1 / t2 << endl;
      // compare results
      local_error = false;
      for(uint32_t i = 0; i < e_sz; i++)
      {
        if(e1[i] != e2[i])
        {
          local_error = true;
          break;
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
  unsigned int log_sz[] = {16, 20, 24, 29};
  bool cpu_has_SSE2_and_PCMUL = detect_cpu_features();
  if(!cpu_has_SSE2_and_PCMUL)
  {
    cout << "SSE2 and PCMULQDQ are required. Exiting" << endl;
    exit(2);
  }
  unsigned int num_sz = sizeof(log_sz) / sizeof(unsigned int);
  bool mateer_gao_error = mateer_gao_product_test(log_sz, num_sz, benchmark, do_gf2x);
  if(mateer_gao_error) cout << "Mateer-Gao product test failed" << endl;
  return mateer_gao_error? EXIT_FAILURE : EXIT_SUCCESS;
}

