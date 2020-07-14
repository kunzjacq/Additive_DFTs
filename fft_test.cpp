#include "finite_field.h"
#include "additive_fft.h"
#include "mateer_gao.h"
#include "mateer_gao_alt.h"

#include "mg.h"

#include "utils.h"
#include "polynomial_product.h"

#include "gf2x.h"

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

// can be uint32_t or uint64_t currently, because cantor basis multiplicative representation functions
// are only available in these sizes
typedef uint64_t word;

constexpr uint64_t allocated_buf_size = (1uLL<<(33 - c_b_t<word>::word_logsize))*1; //1 GB

constexpr bool verbose = false;
constexpr bool check_correctness = true;
constexpr bool benchmark = true;
constexpr int full_dft_test_size_limit = 16; // do not perform full DFTs above this size

static int test_relations(const cantor_basis<word>& c_b, const word& x);


static bool additive_dft_test(
    word* p_buffers,
    cantor_basis<word>& c,
    unsigned int* log_sz,
    unsigned int num_sz,
    bool check_correctness,
    bool benchmark);
static bool reverse_dft_test(
    word* p_buffers,
    cantor_basis<word> &c,
    unsigned int* log_sz,
    unsigned int num_sz,
    bool check_correctness,
    bool benchmark);
static bool full_mateer_gao_test(
    word* p_buffers,
    cantor_basis<word>& c,
    bool check_correctness,
    bool benchmark);
static bool truncated_mateer_gao_dft_test(
    word* p_buffers,
    cantor_basis<word>& c,
    unsigned int* log_sz,
    unsigned int num_sz,
    double* truncated_times,
    bool check_correctness,
    bool benchmark,
    bool full);
static bool reverse_truncated_mateer_gao_dft_test(
    word* p_buffers,
    cantor_basis<word>& c,
    unsigned int* log_sz,
    unsigned int num_sz,
    double* truncated_times,
    bool has_truncated_times,
    bool check_correctness,
    bool benchmark,
    bool full);
static bool mateer_gao_product_test(word* p_buffers,
    unsigned int* log_sz,
    unsigned int num_sz,
    bool check_correctness,
    bool benchmark);
static bool dft_inversion_test(
    word* p_buffers,
    cantor_basis<word>& c);
static bool decompose_taylor_test(
    word* p_buffers,
    bool check_correctness,
    bool benchmark
    );

int main(int UNUSED(argc), char** UNUSED(argv))
{
  word* buffers = new word[allocated_buf_size];
  unsigned int log_sz[] = {8, 16, 24, 26};
  unsigned int num_sz = sizeof(log_sz) / sizeof(unsigned int);
  bool error = false;
  double truncated_times[sizeof(log_sz) / 4];
  bool has_truncated_times = false;
  cantor_basis<word> c;


  init_time();

  bool run_all_tests[]    = {true,  true,  true,  true, true, true,  true,  true};
  bool run_custom_tests[] = {false, false, false, true, true, false, false, false};
  bool* run_tests = run_all_tests;

  if(run_tests[0])
  {
    bool additive_fft_error = additive_dft_test(buffers, c, log_sz, num_sz, check_correctness, benchmark);
    if(additive_fft_error) cout << "additive_fft_test failed" << endl;
    error |= additive_fft_error;
  }
  if(run_tests[1])
  {
    bool reverse_fft_error = reverse_dft_test(buffers, c, log_sz, num_sz, check_correctness, benchmark);
    if(reverse_fft_error) cout << "reverse_dft_test failed" << endl;
    error |= reverse_fft_error;
  }
  bool mateer_gao_error = false;
  if(run_tests[2])
  {
    mateer_gao_error |= full_mateer_gao_test(buffers, c, check_correctness, benchmark);
  }
  if(run_tests[3])
  {
    mateer_gao_error |= truncated_mateer_gao_dft_test(
          buffers, c, log_sz, num_sz, truncated_times, check_correctness, benchmark, false);
    has_truncated_times = true;
  }
  if(run_tests[4])
  {
    mateer_gao_error |= reverse_truncated_mateer_gao_dft_test(
          buffers, c, log_sz, num_sz, truncated_times, has_truncated_times,
          check_correctness, benchmark, false);
  }
  if(run_tests[5])
  {
    mateer_gao_error |= mateer_gao_product_test(
          buffers, log_sz, num_sz, check_correctness, benchmark);
  }
  if(mateer_gao_error) cout << "Some Mateer-Gao tests failed" << endl;
  error |= mateer_gao_error;
  if(run_tests[6])
  {
    bool decompose_taylor_error = decompose_taylor_test(buffers, check_correctness, benchmark);
    error |= decompose_taylor_error;
  }
  if(run_tests[7])
  {
    bool inverse_fft_error = dft_inversion_test(buffers, c);
    if(inverse_fft_error) cout << "dft_inversion_test failed" << endl;
    error |= inverse_fft_error;
  }
  delete [] buffers;
  if (!error)
  {
    cout << endl << "All tests were successful!" << endl;
    return EXIT_SUCCESS;
  }
  else
  {
    cout << endl << "* Some tests failed! *" << endl;
    return EXIT_FAILURE;
  }
}

bool additive_dft_test(
    word* p_buffers,
    cantor_basis<word>& c,
    unsigned int* log_sz,
    unsigned int num_sz,
    bool check_correctness,
    bool benchmark)
{
  uint32_t i;
  double t1 = 0, t2 = 0;
  bool local_error, error = false;
  surand(5);
  additive_fft<word> fft(&c);
  cout << endl << endl << "Von zur Gathen - Gerhard additive DFT test" << endl;
  for(unsigned int j = 0; j < num_sz; j++)
  {
    const unsigned int lsz = log_sz[j];
    uint64_t degree = 0xFFFE;
    uint64_t buf_lsz = lsz;
    while((1uLL << buf_lsz) < degree + 1) buf_lsz++;
    uint64_t buf_size = 1uLL << buf_lsz;
    uint32_t word_sz = c_b_t<word>::n;
    cout << endl << "Testing VzGG Additive DFT of polynomial of degree " << degree <<
            ", on 2**" <<  log_sz[j] << " points, in field of size 2**" << word_sz << endl;
    if(buf_lsz > c_b_t<word>::n)
    {
      cout << "degree or DFT size exceeds finite field size" << endl;
    }

    int check_logsize = min(lsz, 10u);
    if(2*buf_size + (buf_size >> 2) + (1uLL << check_logsize)  > allocated_buf_size)
    {
      cout << "degree or DFT size too large to fit into buffers for this test" << endl;
      continue;
    }

    const uint64_t blk_offset = urand() & ((1uLL << (c_b_t<word>::n - lsz)) - 1); // must be < (field size) / (sz)
    word* refIn, *refOut, *buffer1;
    uint64_t buf_sz = 1uLL << buf_lsz;
    refIn   = p_buffers;
    buffer1 = refIn + buf_sz;
    refOut  = buffer1 + (buf_sz >> 2);
    if(check_correctness)
    {
      for (i = 0; i <= degree; i++) refIn[i] = static_cast<word>(urand());
      for(; i < buf_sz; i++) refIn[i] = 0;
      cout << "Evaluating polynomial with direct method on 2**" << check_logsize << " points..." << endl;
      t1 = absolute_time();
      fft.fft_direct(refIn, degree, check_logsize, refOut, blk_offset << (lsz - check_logsize));
      t1 = (absolute_time() - t1) * (1u <<  (lsz - check_logsize));
      fft.additive_fft_fast_in_place(refIn, degree, lsz, buffer1, blk_offset);
      local_error = compare_results < word > (refOut, refIn, 1uL << check_logsize, 8, verbose);
      error |= local_error;
    }
    if(benchmark)
    {
      cout << "extrapolated time of full DFT:  " << t1  << " sec." << endl;
      cout << "Doing additive DFT..." << endl;
      t2 = absolute_time();
      i = 0;
      do
      {
        fft.additive_fft_fast_in_place(refIn, degree, lsz, buffer1, blk_offset);
        i++;
      }
      while(absolute_time() <= t2 + 1);
      t2 = (absolute_time() - t2) / i;
      cout << "time of additive DFT:  " <<  t2 << " sec." << endl;
      cout << "Additive DFT/direct evaluation speed ratio : " << t1 / t2 << endl;
    }
  }
  if(check_correctness)
  {
    if(error) cout << "* Additive DFT Failed *" << endl;
    else      cout << "additive DFT succeeded" << endl;
  }
  return error;
}


bool reverse_dft_test(
    word* p_buffers,
    cantor_basis<word>& c,
    unsigned int* log_sz,
    unsigned int num_sz,
    bool check_correctness,
    bool benchmark)
{
  uint32_t i;
  int local_error, error = false;
  surand(5);
  cout << endl << endl << "Reverse Von zur Gathen - Gerhard additive DFT test" << endl;
  for(unsigned int j = 0; j < num_sz; j++)
  {
    unsigned int lsz = log_sz[j];
    if(lsz > c_b_t<word>::n) continue;
    const uint64_t sz = 1uLL << lsz;
    // for this test, the degree must be below the fft subblock size
    uint64_t degree = sz - 2;
    const uint64_t blk_offset = urand() & ((1uLL << (c_b_t<word>::n - lsz)) - 1); // must be < (field size) / (sz)
    uint32_t word_sz = c_b_t<word>::n;
    additive_fft<word> fft(&c);
    cout << endl << "Testing reverse additive DFT of polynomial of degree " << degree <<
            ", on 2**" <<  lsz << " points, in field of size 2**" << word_sz << endl;

    if(2 * sz + (sz >> 2)  > allocated_buf_size)
    {
      cout << "DFT size too large to fit into buffers for this test" << endl;
      continue;
    }

    word* refIn, *buffer1, *buffer2;

    refIn   = p_buffers;    //size sz
    buffer1 = refIn + sz;   // size sz
    buffer2 = buffer1 + sz; // size sz >> 2
    for (i = 0; i <= degree; i++) refIn[i] = static_cast<word>(urand());
    for(; i < sz; i++) refIn[i] = 0;

    if(check_correctness)
    {
      cout << "Doing additive DFT..." << endl;
      memcpy(buffer1, refIn, sz * sizeof(word));
      fft.additive_fft_fast_in_place(buffer1, degree, lsz, buffer2, blk_offset);
      cout << "Applying reverse function..." << endl;
      fft.additive_fft_rev_fast_in_place(buffer1, lsz, buffer2, blk_offset);
      cout << "Comparing with initial polynomial..." << endl;
      local_error = compare_results<word>(refIn, buffer1, sz);
      error |= local_error;
    }

    if(benchmark)
    {
      cout << "relative speed measurement" << endl;
      double t1 = absolute_time();
      i = 0;
      do
      {
        fft.additive_fft_rev_ref_in_place(buffer1, lsz, blk_offset);
        i++;
      }
      while(absolute_time() <= t1 + 1);
      t1 = absolute_time() - t1;
      double t2 = absolute_time();
      i = 0;
      do
      {
        fft.additive_fft_rev_fast_in_place(buffer1, lsz, buffer2, blk_offset);
      }
      while(absolute_time() <= t2 + 1);
      t2 = absolute_time() - t2;
      cout << "time of reference implementation " << t1 << endl;
      cout << "time of fast implementation " << t2 << endl;
      cout << "rev fast / rev ref speed ratio:" << t1 / t2 << endl;
    }
  }
  if(error)
  {
    cout << "* Inverse DFT Failed *" << endl;
  }
  else
  {
    cout << "Inverse DFT succeeded" << endl;
  }
  return error;
}

bool full_mateer_gao_test(word* p_buffers, cantor_basis<word>& c, bool check_correctness, bool benchmark)
{
  bool error = false;
  cout << dec;
  cout << endl << "Full Mateer-Gao (M-G) DFT test/benchmark" << endl;
  if constexpr(c_b_t<word>::n >= full_dft_test_size_limit)
  {
    cout << endl << "Size too large, skipping test" << endl;
    return false;
  }
  else
  {
    uint64_t i;
    constexpr uint32_t field_sz = c_b_t<word>::n;
    uint64_t sz = 1uLL << field_sz;
    constexpr unsigned int s = c_b_t<word>::word_logsize;
    additive_fft<word> fft(&c);
    uint64_t check_sz = min<uint64_t>(1uLL << 12, sz);
    word* refIn   = p_buffers;
    word* buffer1 = refIn + sz; // buffer of size sz >> 2 for additive_fft_fast_in_place
    word* refOut  = buffer1 + (sz >> 2); // buffer of size check_sz to verify result

    if(sz + (sz>>2) + check_sz > allocated_buf_size)
    {
      cout << "Size too large to fit into allocated buffer for this test" << endl;
      return true;
    }
    uint64_t degree = sz - 2;
    cout << "Input is a polynomial of degree " << degree << " in field of size 2**" << field_sz << endl;
    if(check_correctness)
    {
      cout << "Evaluating polynomial with full regular additive DFT..." << endl;
      surand(5);
      for (i = 0; i <= degree; i++) refIn[i] = static_cast<word>(urand());
      for (; i < sz; i++) refIn[i] = 0;
      fft.additive_fft_fast_in_place(refIn, sz - 2, field_sz, buffer1, 0);
      memcpy(refOut, refIn, check_sz * sizeof(word));
      cout << "Evaluating polynomial with M-G DFT..." << endl;
      surand(5);
      for (i = 0; i <= degree; i++) refIn[i] = static_cast<word>(urand());
      for (; i < sz; i++) refIn[i] = 0;
      fft_mateer<word ,s>(&c, refIn);
      error = compare_results<word>(refIn, refOut, check_sz);
    }
    if(benchmark)
    {
      double t1 = 0, t2 = 0;
      cout << "Benchmarking: full regular additive DFT..." << endl;
      t1 = absolute_time();
      i = 0;
      do
      {
        for (i = 0; i <= min<uint64_t>(degree,16); i++) refIn[i] = static_cast<word>(urand());
        for (;i <= degree; i++) refIn[i] = refIn[i&0xF];
        for (; i < sz; i++) refIn[i] = 0;
        fft.additive_fft_fast_in_place(refIn, sz - 2, field_sz, buffer1, 0);
        i++;
      }
      while(absolute_time() <= t1 + 1);
      t1 = (absolute_time() - t1) / i;
      cout << "time for full regular additive DFT per iteration:  " << t1 << " sec." << endl;

      t2 = absolute_time();
      i = 0;
      do
      {
        for (i = 0; i <= min<uint64_t>(degree,16); i++) refIn[i] = static_cast<word>(urand());
        for (;i <= degree; i++) refIn[i] = refIn[i&0xF];
        for (; i < sz; i++) refIn[i] = 0;
        fft_mateer<word ,s>(&c, refIn);
        i++;
      }
      while(absolute_time() <= t2 + 1);
      t2 = (absolute_time() - t2) / i;
      cout << "time of FMG DFT per iteration:  " <<  t2 << " sec." << endl;
      cout << "FMG/regular additive DFT speed ratio : " <<  t1 / t2 << endl;
    }
  }
  return error;
}

bool truncated_mateer_gao_dft_test(
    word* p_buffers,
    cantor_basis<word>& c,
    unsigned int* log_sz,
    unsigned int num_sz,
    double* truncated_times,
    bool check_correctness,
    bool benchmark,
    bool full)
{
  uint64_t i;
  bool local_error, error = false;
  uint32_t field_logsz = c_b_t<word>::n;
  constexpr unsigned int s = c_b_t<word>::word_logsize;

  word* refIn = nullptr, *refOut = nullptr, *buffer1 = nullptr, *buffer2 = nullptr;
  unsigned int lsz;
  uint64_t degree;
  uint64_t sz;
  lsz = field_logsz;
  sz = 1uLL << lsz;
  additive_fft<word> fft(&c);

  cout << endl << endl << "Truncated Mateer-Gao DFT test" << endl;
  // test truncated mateer-gao fft, and compare it to truncated regular additive fft
  for(unsigned int j = 0; j < num_sz; j++)
  {
    lsz = log_sz[j];
    if(lsz > c_b_t<word>::n) continue;
    sz = 1uLL << lsz;
    degree = sz - 1;//1uLL<< min(20u, c_b_t<word>::n);
    unsigned int buf_lsz = lsz;
    while ((1uLL << buf_lsz) <= degree) buf_lsz++;
    uint64_t buf_sz = 1uLL << buf_lsz;
    unsigned int check_lsz = min(12u, lsz);
    uint64_t check_sz = 1uLL << check_lsz;
    if(2 * buf_sz + (buf_sz >> 2) + check_sz > allocated_buf_size)
    {
      cout << "Degree too large to fit into allocated buffer for this test" << endl;
      continue;
    }
    refIn   = p_buffers;
    buffer1 = refIn + buf_sz;
    buffer2 = buffer1 + buf_sz;
    refOut  = buffer2 + (buf_sz >> 2);
    surand(5);
    for (i = 0; i <= degree; i++) refIn[i] = static_cast<word>(urand());
    for(; i < sz; i++) refIn[i] = 0;
    if(check_correctness)
    {
      cout << endl << "Testing truncated Mateer-Gao Fourier Transform of polynomial of degree " <<
              degree << ", on 2**" <<  lsz << " points, in field of size 2**" << field_logsz << endl;

      if(verbose) cout << "Computing reference values with truncated VzGG DFT..." << endl;
      for(i = 0; i < buf_sz; i++) buffer1[i] = refIn[i];
      uint64_t block_offset = 0; //(1uLL<<12)-1;
      fft.additive_fft_fast_in_place(buffer1, degree, check_lsz, buffer2, block_offset);
      for( i = 0; i < check_sz; i++) refOut[i] = buffer1[i];

      if(full)
      {
        if(verbose) cout << "Evaluating polynomial with truncated VzGG DFT combined with MG DFT..." << endl;
        for(i = 0; i < buf_sz; i++) buffer1[i] = refIn[i];
        fft.vzgg_mateer_gao_combination(buffer1, degree, lsz);
        local_error = compare_results<word>(buffer1, refOut, check_sz, 8, verbose);
        error |= local_error;

        if(verbose) cout << "Evaluating polynomial with truncated MG DFT..." << endl;
        for(i = 0; i < buf_sz; i++) buffer1[i] = refIn[i];
        fft_mateer_truncated<word, s>(&c, buffer1, lsz, block_offset);
        local_error = compare_results<word>(buffer1, refOut, check_sz, 8, verbose);
        error |= local_error;

        if(verbose) cout << "Evaluating polynomial with truncated MG DFT (\"fast\" variant)..." << endl;
        for(i = 0; i < buf_sz; i++) buffer1[i] = refIn[i];
        fft_mateer_truncated_fast<word, s>(&c, buffer1, lsz);
        local_error = compare_results<word>(buffer1, refOut, check_sz, 8, verbose);
        error |= local_error;
      }
      for(i = 0; i < check_sz; i++) refOut[i] = c.gamma_to_mult(refOut[i]);

      if(verbose) cout << "Evaluating polynomial with truncated MG DFT in multiplicative representation (\"fast\" variant)..." << endl;
      for(i = 0; i < buf_sz; i++) buffer1[i] = c.gamma_to_mult(refIn[i]);
      fft_mateer_truncated_fast_mult<word, s>(&c, buffer1, lsz);
      local_error = compare_results<word>(buffer1, refOut, check_sz, 8, verbose);
      error |= local_error;

      if(verbose) cout << "Evaluating polynomial with truncated MG DFT in multiplicative representation..." << endl;
      for(i = 0; i < buf_sz; i++) buffer1[i] = c.gamma_to_mult(refIn[i]);
      fft_mateer_truncated_mult<word, s>(&c, buffer1, lsz);
      local_error = compare_results<word>(buffer1, refOut, check_sz, 8, verbose);
      error |= local_error;
    }
    if(benchmark)
    {
      uint32_t count;
      cout << endl << "Benchmarking truncated Mateer-Gao variants" << endl;
      double t1 = 0, t2 = 0, t3 = 0, t4 = 0, t6 = 0, t5 = 0, t7 = 0;
      if(full)
      {
        t1 = absolute_time();
        count = 0;
        do
        {
          for(i = 0; i < buf_sz; i++) buffer1[i] = refIn[i];
          fft.additive_fft_fast_in_place(buffer1, degree, lsz, buffer1, 0);
          count++;
        }
        while(absolute_time() <= t1 + 1);
        t1 = (absolute_time() - t1) / count;
        cout << "time of truncated fast VzGG additive DFT:  " << t1 << " sec." << endl;

        t2 = absolute_time();
        count = 0;
        do
        {
          for(i = 0; i < buf_sz; i++) buffer1[i] = refIn[i];
          fft.vzgg_mateer_gao_combination(buffer1, degree, lsz);
          count++;
        }
        while(absolute_time() <= t2 + 1);
        t2 = (absolute_time() - t2) / count;
        cout << "time of truncated VzGG - MG combination DFT:  " <<  t2 << " sec." << endl;

        t3 = absolute_time();
        count = 0;
        do
        {
          for(i = 0; i < buf_sz; i++) buffer1[i] = refIn[i];
          fft_mateer_truncated<word, s>(&c, buffer1, lsz);
          count++;
        }
        while(absolute_time() <= t3 + 1);
        t3 = (absolute_time() - t3) / count;
        truncated_times[j] = t3; // for later comparison with reverse truncated Mateer-Gao
        cout << "time of truncated MG DFT (reference):  " << t3 << " sec." << endl;

        t4 = absolute_time();
        count = 0;
        do
        {
          for(i = 0; i < buf_sz; i++) buffer1[i] = refIn[i];
          fft_mateer_truncated_fast<word, s>(&c, buffer1, lsz);
          count++;
        }
        while(absolute_time() <= t4 + 1);
        t4 = (absolute_time() - t4) / count;
        cout << "time of truncated MG DFT (\"fast\" variant):  " << t4 << " sec." << endl;
      }
      t5 = absolute_time();
      count= 0;
      do
      {
        memcpy(buffer1, refIn, sz * sizeof(word));
        fft_mateer_truncated_fast_mult<word, s>(&c, buffer1, lsz);
        count++;
      }
      while(absolute_time() <= t5 + 1);
      t5 = (absolute_time() - t5) / count;
      cout << "time of truncated MG DFT in multiplicative representation  (\"fast\" variant):  " << t5 << " sec." << endl;

      t6 = absolute_time();
      count= 0;
      do
      {
        for(i = 0; i < buf_sz; i++) buffer1[i] = refIn[i];
        fft_mateer_truncated_mult<word, s>(&c, buffer1, lsz);
        count++;
      }
      while(absolute_time() <= t6 + 1);
      t6 = (absolute_time() - t6) / count;
      cout << "time of truncated MG DFT in multiplicative representation (reference):  " << t6 << " sec." << endl;

      t7 = absolute_time();
      count= 0;
      do
      {
        for(i = 0; i < buf_sz; i++) buffer1[i] = c.gamma_to_mult(refIn[i]);
        fft_mateer_truncated_mult<word, s>(&c, buffer1, lsz);
        for(i = 0; i < buf_sz; i++) buffer1[i] = c.mult_to_gamma(buffer1[i]);
        count++;
      }
      while(absolute_time() <= t7 + 1);
      t7 = (absolute_time() - t7) / count;
      cout << "time of truncated MG DFT in multiplicative representation (reference) + representation conversion:  " << t7 << " sec." << endl;


      cout << endl;
      if(full)
      {
        cout << "(Truncated MG reference / truncated additive VzGG, fast) speed ratio : " << t1 / t2 << endl;
        cout << "Truncated MG (fast / reference) speed ratio : " << t3 / t4 << endl;
        cout << "Truncated MG, (multiplicative representation, reference / gamma representation, reference) speed ratio : " << t3 / t6 << endl;
        cout << "Truncated MG, ((multiplicative representation, reference + representation conversion) / gamma representation, reference) speed ratio : " << t3 / t7 << endl;
      }
      cout << "Truncated MG, multiplicative representation, (fast / reference) speed ratio : " << t6 / t5 << endl;
    }
  }
  return error;
}

bool reverse_truncated_mateer_gao_dft_test(
    word* p_buffers,
    cantor_basis<word>& c,
    unsigned int* log_sz,
    unsigned int num_sz,
    double* truncated_times,
    bool has_truncated_times,
    bool check_correctness,
    bool benchmark,
    bool full)
{
  uint64_t i;
  constexpr unsigned int s = c_b_t<word>::word_logsize;
  bool local_error, error = false;
  uint32_t field_logsz = c_b_t<word>::n;

  word* refIn = nullptr, *refOut = nullptr, *buffer1 = nullptr, *buffer2 = nullptr;
  unsigned int lsz;
  uint64_t sz;
  lsz = field_logsz;
  sz = 1uLL << lsz;
  additive_fft<word> fft(&c);
  cout << endl << endl << "Reverse truncated Mateer-Gao DFT test" << endl;
  // testing that fft_mateer_truncated_reverse \circ fft_mateer_truncated = id
  for(unsigned int j = 0; j < num_sz; j++)
  {
    lsz = log_sz[j];
    if(lsz > c_b_t<word>::n) continue;
    sz = 1uLL << lsz;
    unsigned int check_lsz = min(12u, lsz);
    uint64_t check_sz = 1uLL << check_lsz;
    if(2 * sz + (sz >> 2) + check_sz > allocated_buf_size)
    {
      cout << "degree too large to fit into allocated buffer for this test" << endl;
      continue;
    }
    refIn   = p_buffers;
    buffer1 = refIn + sz;
    buffer2 = buffer1 + sz;
    refOut  = buffer2 + (sz >> 2);

    surand(5);
    for(i = 0; i < sz; i++) refIn[i] = static_cast<word>(urand());
    if(check_correctness)
    {
      cout << endl << "Testing reverse truncated Mateer-Gao Fourier Transform " <<
              ", on 2**" <<  lsz << " points, in field of size 2**" << field_logsz << endl;

      if(verbose) cout << "Computing reference values with reverse truncated VzGG DFT..." << endl;
      for(i = 0; i < sz; i++) buffer1[i] = refIn[i];
      fft.additive_fft_rev_fast_in_place(buffer1, lsz, buffer2, 0);
      for( i = 0; i < check_sz; i++) refOut[i] = buffer1[i];

      if(full)
      {
        if(verbose) cout << "Applying reverse truncated Mateer-Gao DFT..." << endl;
        for(i = 0; i < sz; i++) buffer1[i] = refIn[i];
        fft_mateer_truncated_reverse<word,s>(&c, buffer1, lsz);
        local_error = compare_results < word > (buffer1, refOut, check_sz);
        error |= local_error;

        if(verbose) cout << "Applying reverse truncated Mateer-Gao DFT (\"fast\" variant)..." << endl;
        for(i = 0; i < sz; i++) buffer1[i] = refIn[i];
        fft_mateer_truncated_reverse_fast<word,s>(&c, buffer1, lsz);
        local_error = compare_results < word > (buffer1, refOut, check_sz);
        error |= local_error;
      }
      for(i = 0; i < check_sz; i++) refOut[i] = c.gamma_to_mult(refOut[i]);

      if(verbose) cout << "Evaluating polynomial with reverse truncated MG DFT in multiplicative representation (\"fast\" variant)..." << endl;
      for(i = 0; i < sz; i++) buffer1[i] = c.gamma_to_mult(refIn[i]);
      fft_mateer_truncated_reverse_mult_fast<word, c_b_t<word>::word_logsize>(&c, buffer1, lsz);
      local_error = compare_results<word>(buffer1, refOut, check_sz, 8, verbose);
      error |= local_error;

      if(verbose) cout << "Evaluating polynomial with reverse truncated MG DFT in multiplicative representation..." << endl;
      for(i = 0; i < sz; i++) buffer1[i] = c.gamma_to_mult(refIn[i]);
      fft_mateer_truncated_reverse_mult<word, c_b_t<word>::word_logsize>(&c, buffer1, lsz);
      local_error = compare_results<word>(buffer1, refOut, check_sz, 8, verbose);
      error |= local_error;
    }
    if(benchmark)
    {
      uint32_t count;
      cout << endl << "Benchmarking reverse truncated Mateer-Gao Fourier Transform " <<
              ", on 2**" <<  lsz << " points, in field of size 2**" << field_logsz << endl;

      double t1 = 0, t2 = 0, t3 = 0, t5 = 0, t4 = 0, t6 = 0;
      if(full)
      {
        t1 = absolute_time();
        count = 0;
        do
        {
          for(i = 0; i < sz; i++) buffer1[i] = refIn[i];
          fft.additive_fft_rev_fast_in_place(buffer1, lsz, buffer2, 0);
          count++;
        }
        while(absolute_time() <= t1 + 1);
        t1 = (absolute_time() - t1) / count;
        cout << "time of reverse truncated fast VzGG additive DFT:  " << t1 << " sec." << endl;

        t2 = absolute_time();
        count = 0;
        do
        {
          for(i = 0; i < sz; i++) buffer1[i] = refIn[i];
          fft_mateer_truncated_reverse<word,s>(&c, buffer1, lsz);
          count++;
        }
        while(absolute_time() <= t2 + 1);
        t2 = (absolute_time() - t2) / count;
        cout << "Time of reverse truncated Mateer-Gao DFT (reference): " <<  t2 << " sec." << endl;

        t3 = absolute_time();
        count = 0;
        do
        {
          for(i = 0; i < sz; i++) buffer1[i] = refIn[i];
          fft_mateer_truncated_reverse_fast<word,s>(&c, buffer1, lsz);
          count++;
        }
        while(absolute_time() <= t3 + 1);
        t3 = (absolute_time() - t3) / count;
        cout << "Time of reverse truncated Mateer-Gao DFT (\"fast\" variant): " <<  t3 << " sec." << endl;
      }

      t4 = absolute_time();
      count = 0;
      do
      {
        for(i = 0; i < sz; i++) buffer1[i] = refIn[i];
        fft_mateer_truncated_reverse_mult_fast<word,s>(&c, buffer1, lsz);
        count++;
      }
      while(absolute_time() <= t4 + 1);
      t4 = (absolute_time() - t4) / count;
      cout << "Time of reverse truncated Mateer-Gao DFT in multiplicative representation (\"fast\" variant):  " <<  t4 << " sec." << endl;

      t5 = absolute_time();
      count = 0;
      do
      {
        for(i = 0; i < sz; i++) buffer1[i] = refIn[i];
        fft_mateer_truncated_reverse_mult<word,s>(&c, buffer1, lsz);
        count++;
      }
      while(absolute_time() <= t5 + 1);
      t5 = (absolute_time() - t5) / count;
      cout << "Time of reverse truncated Mateer-Gao DFT in multiplicative representation (reference):  " <<  t5 << " sec." << endl;

      t6 = absolute_time();
      count= 0;
      do
      {
        for(i = 0; i < sz; i++) buffer1[i] = c.gamma_to_mult(refIn[i]);
        fft_mateer_truncated_reverse_mult<word, s>(&c, buffer1, lsz);
        for(i = 0; i < sz; i++) buffer1[i] = c.mult_to_gamma(buffer1[i]);
        count++;
      }
      while(absolute_time() <= t6 + 1);
      t6 = (absolute_time() - t6) / count;
      cout << "Time of reverse truncated MG DFT in multiplicative representation (reference) + representation conversion:  " << t6 << " sec." << endl;

      cout << endl;
      if(full)
      {
        cout << "(Reverse truncated MG / reverse truncated VzGG) speed ratio: " << t1 / t2 << endl;
        cout << "Reverse truncated MG, (fast / reference) speed ratio: " << t2 / t3 << endl;
        if(has_truncated_times)
        {
          cout << "(Reverse truncated MG, reference / Direct truncated MG, reference) speed ratio: " << truncated_times[j] / t2 << endl;
        }
        cout << "Reverse truncated MG (multiplicative representation, reference / gamma representation, reference) speed ratio: " << t2 / t5 << endl;
        cout << "Reverse truncated MG ((multiplicative representation, reference + representation conversion) / gamma representation, reference) speed ratio: " << t2 / t6 << endl;
      }
      cout << "Reverse truncated MG (multiplicative representation, fast / multiplicative representation, reference) speed ratio: " << t5 / t4 << endl;

    }
  }
  return error;
}

bool mateer_gao_product_test(
    word* p_buffers,
    unsigned int* log_sz,
    unsigned int num_sz,
    bool check_correctness,
    bool benchmark)
{
  uint64_t i;
  double t1 = 0, t2 = 0;
  bool local_error, error = false;
  uint32_t field_logsz = c_b_t<word>::n;

  word *buffer1 = nullptr, *buffer2 = nullptr;
  unsigned int lsz;
  uint64_t degree;
  uint64_t sz;
  mateer_gao_polynomial_product mp;
  cout << endl << endl << "Mateer-Gao polynomial product test" << endl;
  // computing products of polynomials over f2 with mateer-gao fft and checking the result
  // with gf2x
  for(unsigned int j = 0; j < num_sz; j++)
  {
    lsz = log_sz[j];
    if(lsz > c_b_t<word>::n) continue;
    sz = 1uLL << lsz;
    uint64_t dft_size = sz >> (c_b_t<word>::word_logsize - 1); // each word holds 2**(c_b_t<word>::logsize - 1) coefficients
    unsigned int dft_logsize = lsz - (c_b_t<word>::word_logsize - 1);
    uint64_t needed_buf_bitsize = sz*3 + 2*(dft_size << c_b_t<word>::word_logsize);
    if(needed_buf_bitsize > (allocated_buf_size << c_b_t<word>::word_logsize))
    {
      cout << "Allocated buffer is too small for this test" << endl;
      cout << "Needed buf size: " << (needed_buf_bitsize >> (23)) <<
              "MB; allocated: " << (allocated_buf_size >>(23-c_b_t<word>::word_logsize))<< "MB" << endl;
      break; // assume sizes are increasing, therefore remaining tests are skipped as well
    }
    // total buffer size needed : 2*sz words + 3/8*sz bytes =
    // ((sz*3) >> c_b_t<word>::word_logsize) words + 2*sz
    uint8_t* p1 = (uint8_t*) p_buffers; // size sz/16
    uint8_t* p2 = p1 + sz/16; // size sz/16
    uint8_t* p3 = p2 + sz/16; // size sz/8
    uint8_t* p4 = p3 + sz/8;  // size sz/8
    buffer1 = p_buffers + ((sz*3) >> c_b_t<word>::word_logsize); //size dft_size
    buffer2 = buffer1 + dft_size; //size dft_size
    // create two random polynomials with sz/2 coefficients
    surand(5);
    for(size_t i = 0; i < sz / 16; i++)
    {
      p1[i] = urand() & 0xFF;
      p2[i] = urand() & 0xFF;
    }
    if (check_correctness)
    {
      cout << endl << "Testing F2 polynomial product through FFTs. Multiplying 2 polynomials of degree " <<
              sz/2 - 1 << ", in field of size 2**" << field_logsz << endl;
      // multiply polynomials with gf2x, put result in p3
      gf2x_mul(
            (unsigned long *) p3, (unsigned long *) p1, sz/(16*sizeof(unsigned long)),
            (unsigned long *) p2, sz/(16*sizeof(unsigned long)));
      // multiply polynomials with additive fft, put result in p4
      mp.binary_polynomial_multiply(p1, p2, p4, buffer1, buffer2, sz/2 - 1, sz/2 - 1, dft_logsize);
      // compare results
      local_error = compare_results<uint8_t>(p3, p4, sz/8);
      error |= local_error;
    }
    if(benchmark)
    {
      degree = sz - 1; // degree of the product. Each polynomial will be of degree sz/2-1.
      cout << endl << "Benchmarking F2 polynomial product through FFTs with maximum total degree "
           << degree << ", in field of size 2**" << field_logsz << endl;
      t1 = absolute_time();
      i = 0;
#if 1
      cout << "Performing product with gf2x" << endl;
      do
      {
        gf2x_mul((unsigned long *) p3,(unsigned long *) p1, sz/(16*sizeof(unsigned long)),
                 (unsigned long *) p2, sz/(16*sizeof(unsigned long)));
        i++;
      }
      while(absolute_time() <= t1 + 1);
      t1 = (absolute_time() - t1) / i;
      cout << "gf2x time: " << t1 << endl;
      cout << "Performing product with MG DFT" << endl;
#endif
      t2 = absolute_time();
      i = 0;
      do
      {
        mp.binary_polynomial_multiply(p1, p2, p4, buffer1, buffer2, sz/2 - 1, sz/2 - 1, dft_logsize);
        i++;
      }
      while(absolute_time() <= t2 + 1);
      t2 = (absolute_time() - t2) / i;
      cout << "iterations: " << i << endl;
      cout << "Mateer-Gao time: " << t2 << endl;
      cout << "MG / gf2x speed ratio: " << t1 / t2 << endl;
    }
  }
  return error;
}


template <class word>
void reorder(cantor_basis<word>* c_b, word *in, word* out, size_t log_sz, const word& x)
{
  if constexpr(c_b_t<word>::n < 64)
  {
    word x_pow_i = 1;
    size_t order = (1uL << log_sz) - 1;
    for(size_t i = 0; i < order; i++)
    {
      // x_pow_i = x**i
      uint64_t idx = c_b->gamma_to_beta(x_pow_i);
      out[i] = in[idx];
      x_pow_i = c_b->multiply(x_pow_i, x);
    }
    out[order] = 0;
  }
}

bool dft_inversion_test(word* p_buffers, cantor_basis<word>& c)
{
  unsigned int log_sz = c_b_t<word>::n;
  // required to avoid hitting some static_asserts
  if constexpr(c_b_t<word>::n >= 64) return false;
  else
  {
    int local_error = 0, error = 0;
    const uint64_t sz = 1uLL << log_sz;
    uint64_t i;
    uint64_t degree = sz - 2;
    uint64_t buf_sz = sz;
    while(buf_sz < degree + 1) buf_sz <<= 1;
    cout << endl << endl << "Testing Inverse relation on Fourier Transform of polynomial of "
                            "degree " << degree << ", in field of size 2**" << log_sz << endl;
    if constexpr(c_b_t<word>::n >= full_dft_test_size_limit)
    {
      cout << "Size too large for this test, skipping it" << endl;
      return false;
    }
    if(3 * buf_sz > allocated_buf_size)
    {
      cout << "degree too large to fit into allocated buffer for this test" << endl;
      return false;
    }
    word* refIn, *buffer1, *buffer2;
    refIn   = p_buffers + 0 * buf_sz;
    buffer1 = p_buffers + 1 * buf_sz;
    buffer2 = p_buffers + 2 * buf_sz;
    surand(5);
    for (i = 0; i <= degree; i++) refIn[i] = static_cast<word>(urand());
    for(; i < sz; i++) refIn[i] = 0;

    // choose element according to which the fft will be re-ordered
    // i.e. a primitive element x s.t. after reordering, the output array contains
    // f(1), f(x), f(x**2), ..., f(x**order - 1)
    unsigned int idx = log_sz - 1;
    word x;
    while(!is_primitive((x = c.beta_over_gamma(idx)), c)) idx--;
    if(test_relations(c, x))
    {
      cout << "reording element is not primitive or inversion relations do not hold, aborting" << endl;
      return 1;
    }

    additive_fft<word> fft(&c);

    for(int a = 0; a < 2; a++)
    {
      for(int b = 0; b < 2; b++)
      {
        memcpy(buffer1, refIn, sz * sizeof(word));
        cout << "Doing 1st Fourier Transform ";
        if(a)
        {
          cout << "with direct method" << endl;
          fft.fft_direct_exp(buffer1, degree, log_sz, x, buffer2);
        }
        else
        {
          cout << "with additive method" << endl;
          fft.additive_fft_fast_in_place(buffer1, degree, log_sz, buffer2, 0);
          reorder<word>(&c, buffer1, buffer2, log_sz, x);
        }
        cout << "Doing 2nd Fourier Transform ";
        if(b)
        {
          cout << "with direct method" << endl;
          fft.fft_direct_exp(buffer2, sz - 2, log_sz, x, buffer1);
          memcpy(buffer2, buffer1, sz * sizeof(word));
        }
        else
        {
          cout << "with additive method" << endl;
          fft.additive_fft_fast_in_place(buffer2, sz - 2, log_sz, buffer1, 0);
          memcpy(buffer1, buffer2, sz * sizeof(word));
          reorder<word>(&c, buffer1, buffer2, log_sz, x);
        }
        for(size_t i = 1; i < sz/2; i++) swap(buffer2[i], buffer2[sz - 1 - i]);
        local_error = memcmp(buffer2, refIn, (sz - 1) * sizeof(word));
        if(local_error)
        {
          cout << "Double DFT and initial polynomial differ: " << endl;
          compare_results<word>(refIn, buffer2, (sz - 1), 16);
        }
        else
        {
          cout << "Double DFT is identity" << endl;
        }
        error |= local_error;
      }
    }
    return error;
  }
}


int test_relations(const cantor_basis<word>& c_b, const word& x)
{
  if(!is_primitive(x, c_b)) {
    cout << "test_relations: x is not primitive" << endl;
    return 2;
  }
  word k = 0;
  k =~k; // 2**n - 1;
  word bound = min(static_cast<word>(0xFF), k);
  for(word i = 0; i < bound; i++)
  {
    word sum = 0;
    word val = power(x, i, c_b);
    word curr = 1;
    for(word e = 0; e < k; e++)
    {
      sum ^= curr;
      curr = c_b.multiply(curr, val);
    }
    // print_value(sum);
    // cout << endl;
    word expected_sum = i==0 ? 1 : 0;
    if(sum != expected_sum) {
      cout << "i = ";
      print_value(i);
      cout << ", sum = ";
      print_value(sum);
      cout << endl;
      return 1;
    }
  }
  return 0;
}

bool decompose_taylor_test(
    word* p_buffers,
    bool check_correctness,
    bool benchmark
    )
{
  unsigned int logtau = 1;
  unsigned int logstride = 1;
  size_t max_sz = 1uLL << 10;
  size_t min_sz = 0;
  size_t large_sz = 1uLL << 24;
  size_t array_sz = max(large_sz, max_sz << logstride);
  word* copy = p_buffers + array_sz;
  word* ref  = p_buffers + 2*array_sz;
  uint64_t size_needed = 3*(array_sz<<logstride);
  if(size_needed > allocated_buf_size)
  {
    cout << "Allocated buffer too small for this test, skipping" << endl;
    return false;
  }

  unsigned int logblocksize = 0;
  bool error = false;
  if(check_correctness)
  {
    cout << endl << endl << "Test Taylor decomposition used in Mateer-Gao FFT" << endl;
    for(size_t sz = min_sz; sz < max_sz; sz++)
    {
      while((1uL << logblocksize) < sz) logblocksize++;
      memset(p_buffers, 0, (sz << logstride) * sizeof(word));
      for(size_t i = 0; i < (sz << logstride); i++)
      {
        p_buffers[i] = urand();
        copy[i] = p_buffers[i];
        ref[i]  = p_buffers[i];
      }

      decompose_taylor_recursive(logstride, logblocksize, logtau, sz, p_buffers);
      // compare output of decompose_taylor_iterative to output of decompose_taylor
      decompose_taylor(logstride, logblocksize, logtau, sz, copy);
      if(memcmp(p_buffers, copy, (sz << logstride) * sizeof(word))) error = true;

      decompose_taylor_reverse_recursive(logstride, logblocksize, logtau, sz, p_buffers);
      if(memcmp(p_buffers, ref, (sz << logstride) * sizeof(word))) error = true;
    }

    if(error == 0) cout << "Taylor decomposition succeeded" << endl;
    else cout << "Taylor decomposition failed" << endl;
  }
  if(benchmark)
  {
    cout << "Taylor decomposition benchmark" << endl;
    // process a large instance, for benchmarking purposes
    while((1uL << logblocksize) < large_sz) logblocksize++;
    for(size_t i = 0; i < large_sz; i++)
    {
      p_buffers[i] = urand();
      copy[i] = p_buffers[i];
    }
    init_time();
    double t1, t2;
    t1 = absolute_time();
    decompose_taylor_recursive(0, logblocksize, logtau, large_sz, copy);
    t2 = absolute_time();
    t1 = t2 - t1;
    decompose_taylor(0, logblocksize, logtau, large_sz, p_buffers);
    t2 = absolute_time() - t2;
    cout << "Iterative time: " << t2 << endl;
    cout << "Recursive time: " << t1 << endl;
    cout << "Taylor decomposition iterative / recursive speed ratio: " << t1 / t2 << endl;

    cout << "Reverse Taylor decomposition benchmark" << endl;
    t1 = absolute_time();
    decompose_taylor_reverse_recursive(0, logblocksize, logtau, large_sz, copy);
    t2 = absolute_time();
    t1 = t2 - t1;
    decompose_taylor_reverse(0, logblocksize, logtau, large_sz, p_buffers);
    t2 = absolute_time() - t2;
    cout << "Iterative time: " << t2 << endl;
    cout << "Recursive time: " << t1 << endl;
    cout << "Taylor decomposition iterative / recursive speed ratio: " << t1 / t2 << endl;
  }
  return error;
}
