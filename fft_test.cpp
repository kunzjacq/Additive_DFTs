#include "finite_field.h"
#include "additive_fft.h"
#include "mateer_gao.h"
#include "utils.h"
#include "polynomial_product.h"

#include "gf2x.h"

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

typedef uint32_t word;

constexpr unsigned int log_buf_size = 28;
constexpr unsigned int log_buf_bitsize = log_buf_size + c_b_t<word>::word_logsize;

constexpr bool verbose = false;
constexpr bool benchmark = true;

int additive_dft_test(word* p_buffers);
int reverse_dft_test(word* p_buffers, bool benchmark);
int mateer_gao_dft_test(word* p_buffers, bool benchmark);
int dft_inversion_test(word* p_buffers);


int main(int UNUSED(argc), char** UNUSED(argv))
{
  word* buffers = new word[4uLL * (1uLL << log_buf_size)];
  int error = 0;
  cout << "Default memory alignment: " << __STDCPP_DEFAULT_NEW_ALIGNMENT__ << endl;
  init_time();
  if constexpr(true)
  {
    int additive_fft_error = additive_dft_test(buffers);
    if(additive_fft_error) cout << "additive_fft_test failed" << endl;
    error |= additive_fft_error;
  }
  if constexpr(true)
  {
    int reverse_fft_error = reverse_dft_test(buffers, benchmark);
    if(reverse_fft_error) cout << "reverse_dft_test failed" << endl;
    error |= reverse_fft_error;
  }
  if constexpr(true)
  {
    int decompose_taylor_error = decompose_taylor_test<uint32_t>();
    error |= decompose_taylor_error;
  }
  if constexpr(true)
  {
    int mateer_gao_error = mateer_gao_dft_test(buffers, benchmark);
    error |= mateer_gao_error;
  }
  if constexpr(true)
  {
    int inverse_fft_error = dft_inversion_test(buffers);
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

int additive_dft_test(word* p_buffers)
{
  uint32_t i;
  //unsigned int log_sz[] = {8, 12, 16, 20};
  unsigned int log_sz[] = {20, 24};
  double t1 = 0, t2 = 0;
  int local_error, error = 0;
  surand(5);
  cantor_basis<word> c;
  cout  << endl << endl << "Comparing direct DFT computation with additive DFT" << endl;
  for(unsigned int j = 0; j < sizeof(log_sz) / 4; j++)
  {
    const unsigned int lsz = log_sz[j];
    uint64_t degree = 0xFFFE;
    uint64_t buf_lsz = lsz;
    while((1uLL << buf_lsz) < degree + 1) buf_lsz++;
    uint32_t word_sz = c_b_t<word>::n;
    cout << endl << "Testing Additive Fourier Transform of polynomial of degree " << degree <<
            ", on 2**" <<  log_sz[j] << " points, in field of size 2**" << word_sz << endl;
    if(buf_lsz > min(c_b_t<word>::n, log_buf_size))
    {
      cout << "degree or DFT size too large to fit into buffers for this test" << endl;
      continue;
    }

    int check_logsize = min(lsz, 10u);
    additive_fft<word> fft(&c);
    const uint64_t blk_offset = urand() & ((1uLL << (c_b_t<word>::n - lsz)) - 1); // must be < (field size) / (sz)
    word* refIn, *refOut, *buffer1, *buffer2;
    uint64_t buf_sz = 1uLL << buf_lsz;
    refIn   = p_buffers + 0 * buf_sz;
    refOut  = p_buffers + 1 * buf_sz;
    buffer1 = p_buffers + 2 * buf_sz;
    buffer2 = p_buffers + 3 * buf_sz;
    memset(p_buffers, 0, buf_sz * 4 * sizeof(word));
    for (i = 0; i <= degree; i++) refIn[i] = static_cast<word>(urand());
    uint32_t logrepeat = (lsz >= 20)? 0: 20 - lsz;
    uint32_t repeat = 1u << logrepeat;
    cout << "Evaluating polynomial with direct method on 2**" << check_logsize << " points..." << endl;
    t1 = absolute_time();
    fft.fft_direct(refIn, degree, check_logsize, refOut, blk_offset << (lsz - check_logsize));
    t1 = absolute_time() - t1;
    cout << "time of full DFT:  " << t1 * (1u <<  (lsz - check_logsize)) << " sec." << endl;
    cout << "Doing additive DFT..." << endl;
    t2 = absolute_time();
    for(i = 0 ; i < repeat; i++)
    {
      memcpy(buffer1, refIn, buf_sz * sizeof(word));
      fft.additive_fft_fast_in_place(buffer1, degree, lsz, buffer2, blk_offset);
    }
    t2 = absolute_time() - t2;
    cout << "time of additive DFT:  " <<  t2 / repeat << " sec." << endl;
    cout << "Additive DFT/direct evaluation speed ratio : " <<
            (repeat * t1) / t2 * (1u <<  (lsz - check_logsize)) << endl;
    local_error = compare_results < word > (refOut, buffer1, 1uL << check_logsize, 8, verbose);
    error |= local_error;
  }
  if(error)
  {
    cout << "* Additive DFT Failed *" << endl;
  }
  else
  {
    cout << "additive DFT succeeded" << endl;
  }
  return error;
}


int reverse_dft_test(word* p_buffers, bool benchmark)
{
  uint32_t i;
  //unsigned int log_sz[] = {8, 12, 16, 24};
  unsigned int log_sz[] = {24};
  int local_error, error = 0;
  surand(5);
  cantor_basis<word> c;
  cout << endl << endl << "Testing reverse additive DFT" << endl;
  for(unsigned int j = 0; j < sizeof(log_sz) / 4; j++)
  {
    unsigned int lsz = log_sz[j];
    if(lsz > c_b_t<word>::n) continue;
    const uint64_t sz = 1uLL << lsz;
    // for this test, the degree must be below the fft subblock size
    uint64_t degree = sz - 2;
    const uint64_t blk_offset = urand() & ((1uLL << (c_b_t<word>::n - log_sz[j])) - 1); // must be < (field size) / (sz)
    uint32_t word_sz = c_b_t<word>::n;
    additive_fft<word> fft(&c);
    cout << endl << "Testing reverse Fourier Transform of polynomial of degree " << degree <<
            ", on 2**" <<  log_sz[j] << " points, in field of size 2**" << word_sz << endl;
    word* refIn, *buffer1, *buffer2;

    refIn   = p_buffers + 0 * sz;
    buffer1 = p_buffers + 1 * sz;
    buffer2 = p_buffers + 2 * sz;
    memset(p_buffers, 0, sz * 3 * sizeof(word));
    for (i = 0; i <= degree; i++)
    {
      refIn[i] = static_cast<word>(urand());
    }

    if constexpr(true)
    {
      cout << "Doing additive DFT..." << endl;
      memcpy(buffer1, refIn, sz * sizeof(word));
      fft.additive_fft_fast_in_place(buffer1, degree, lsz, buffer2, blk_offset);
      cout << "Applying reverse function..." << endl;
      fft.additive_fft_rev_fast_in_place(buffer1, lsz, buffer2, blk_offset);
      cout << "Comparing with initial polynomial..." << endl;
      local_error = compare_results < word > (refIn, buffer1, sz);
      error |= local_error;
      if(local_error)
      {
        cout << "Wrong result" << endl;
      }
      else
      {
        cout << "Success!" << endl;
      }
    }

    if(benchmark)
    {
      cout << "relative speed measurement" << endl;
      uint32_t logrepeat = (lsz >= 24)? 0: 24 - lsz;
      uint32_t repeat = 1u << logrepeat;
      double t1 = absolute_time();
      for(unsigned int i = 0; i < repeat; i++) fft.additive_fft_rev_ref_in_place(buffer1, lsz, blk_offset);
      t1 = absolute_time() - t1;
      double t2 = absolute_time();
      for(unsigned int i = 0; i < repeat; i++) fft.additive_fft_rev_fast_in_place(buffer1, lsz, buffer2, blk_offset);
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

template <class word> void reorder(cantor_basis<word>* c_b, word *in, word* out, size_t log_sz, const word& x)
{
  word x_pow_i = 1;
  size_t order = (1uL << log_sz) - 1;
  for(size_t i = 0; i < order; i++)
  {
    // x_pow_i = x**i
    out[i] = in[c_b->gamma_to_beta(x_pow_i)];
    x_pow_i = c_b->multiply(x_pow_i, x);
  }
  out[order] = 0;
}

template <class word> int test_relations(const cantor_basis<word>& c_b, const word& x)
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

int dft_inversion_test(word* p_buffers)
{
  unsigned int log_sz = c_b_t<word>::n;
  surand(5);
  cantor_basis<word> c;
  int local_error = 0, error = 0;
  const uint64_t sz = 1uLL << log_sz;
  uint64_t degree = sz - 2;
  uint64_t buf_sz = sz;
  while(buf_sz < degree + 1)
  {
    buf_sz <<= 1;
  }
  cout << endl << endl << "Testing Inverse relation on Fourier Transform of polynomial of degree " <<
          degree << ", in field of size 2**" << log_sz << endl;
  if(buf_sz > (1uLL << log_buf_size))
  {
    cout << "degree too large to fit into buffers for this test" << endl;
    return 0;
  }
  word* refIn, *buffer1, *buffer2;
  refIn   = p_buffers + 0 * buf_sz;
  buffer1 = p_buffers + 1 * buf_sz;
  buffer2 = p_buffers + 2 * buf_sz;
  memset(p_buffers, 0, buf_sz * 4 * sizeof(word));
  for (uint64_t i = 0; i <= degree; i++)
  {
    refIn[i] = static_cast<word>(urand()) ;
  }

  // choose element according to which the fft will be re-ordered
  // i.e. a primitive element x s.t. after reordering, the output array contains
  // f(1), f(x), f(x**2), ..., f(x**order - 1)
  unsigned int idx = log_sz - 1;
  word x;
  while(!is_primitive((x = c.beta_over_gamma(idx)), c)) idx--;
  if(test_relations<word>(c, x))
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
        reorder(&c, buffer1, buffer2, log_sz, x);
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
        reorder(&c, buffer1, buffer2, log_sz, x);
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

bool mateer_gao_full(word* p_buffers, cantor_basis<word>& c, bool benchmark)
{
  uint64_t i;
  constexpr unsigned int s = c_b_t<word>::word_logsize;
  double t1 = 0, t2 = 0;
  bool error = false;
  surand(5);
  uint32_t word_sz = c_b_t<word>::n;
  uint64_t sz = 1uLL << word_sz;
  // if word is uint64_t, sz will be 0, but in this case the test will not be run
  // since word_sz >= log_buf_size

  cout << dec;
  cout << endl << "Full Mateer-Gao (FMG) additive DFT test/benchmark" << endl;
  if(word_sz > log_buf_size) {
    cout << "Degree too large to fit into buffers for this test" << endl;
    return true;
  }

  if(!benchmark)
  {
    cout << "Testing FMG DFT of polynomial of degree " << sz - 2 <<
          " in field of size 2**" << word_sz << endl;
  }
  // testing full mateer-gao fft, and compare it to full regular additive dft
  //uint32_t fft_sz = min(word_sz, 16u);
  //additive_fft<word> fft(&c, fft_sz);
  additive_fft<word> fft(&c);
  word* refIn   = p_buffers + 0 * sz;
  word* refOut  = p_buffers + 1 * sz;
  word* buffer1 = p_buffers + 2 * sz;
  for (i = 0; i < sz - 1; i++)
  {
    refIn[i] = static_cast<word>(urand());
  }
  refIn[sz - 1] = 0; // for full additive DFT, polynomial degree must be < to field multiplicative order
  uint64_t repeat = benchmark? 16:1;
  t1 = absolute_time();
  if(benchmark) {
    cout << "Benchmarking: full regular additive DFT..." << endl;
  } else {
    cout << "Evaluating polynomial with full regular additive DFT..." << endl;
  }
  for(i = 0 ; i < repeat; i++)
  {
    memcpy(refOut, refIn, sz * sizeof(word));
    fft.additive_fft_fast_in_place(refOut, sz - 2, word_sz, buffer1, 0);
  }
  t1 = (absolute_time() - t1) / repeat;
  if(benchmark) {
    cout << "time for full regular additive DFT per iteration:  " << t1 / repeat << " sec." << endl;
  } else {
    cout << "Evaluating polynomial with FMG DFT..." << endl;
  }
  t2 = absolute_time();
  for(i = 0 ; i < repeat; i++)
  {
    memcpy(buffer1, refIn, sz * sizeof(word));
    fft_mateer(&c, buffer1, s);
  }
  t2 = (absolute_time() - t2) / repeat;
  if(benchmark)
  {
    cout << "time of FMG DFT per iteration:  " <<  t2 / repeat << " sec." << endl;
    cout << "FMG/regular additive DFT speed ratio : " <<  t1 / t2 << endl;
  }
  else
  {
    error = compare_results<word>(refOut, buffer1, sz);
  }
  return error;
}

int mateer_gao_dft_test(word* p_buffers, bool benchmark)
{
  uint64_t i;
  constexpr unsigned int s = c_b_t<word>::word_logsize;
  unsigned int log_sz[] = {8, 12, 16, 20, 24};
  double t1 = 0, t2 = 0;
  int local_error, error = 0;
  surand(5);
  cantor_basis<word> c;
  uint32_t word_sz = c_b_t<word>::n;

  word* refIn = nullptr, *refOut = nullptr, *buffer1 = nullptr, *buffer2 = nullptr;
  unsigned int lsz;
  uint64_t degree;
  uint64_t sz;
  uint64_t repeat;
  lsz = word_sz;
  sz = 1uLL << lsz;
  // if word is uint64_t, sz will be 0, but in this case the test will not be run
  // since word_sz >= log_buf_size
  degree = sz - 1;

  cout << dec;
#if 1
#if 1
  cout << endl << endl << "Comparing regular additive DFT computation with Mateer-Gao additive DFT" << endl;
  cout << endl << "Testing full Mateer-Gao Fourier Transform of polynomial of degree " << degree <<
          " in field of size 2**" << word_sz << endl;
  if(word_sz > log_buf_size)
  {
    cout << "Degree too large to fit into buffers for this test" << endl;
  }
  else
  {
    // testing full mateer-gao fft, and compare it to full regular additive dft
    additive_fft<word> fft(&c);
    refIn   = p_buffers + 0 * sz;
    refOut  = p_buffers + 1 * sz;
    buffer1 = p_buffers + 2 * sz;
    buffer2 = p_buffers + 3 * sz;
    memset(refIn, 0, sz * sizeof(word));
    for (i = 0; i <= degree; i++)
    {
      refIn[i] = static_cast<word>(urand());
    }
    cout << "Evaluating polynomial with additive DFT..." << endl;
    memcpy(refOut, refIn, sz * sizeof(word));
    fft.additive_fft_fast_in_place(refOut, degree, lsz, buffer1, 0);

    cout << "Evaluating polynomial with Mateer-Gao DFT..." << endl;
    memcpy(buffer1, refIn, sz * sizeof(word));
    fft_mateer(&c, buffer1, s);
    local_error = compare_results < word > (refOut, buffer1, sz);
    error |= local_error;

    if(benchmark)
    {
      repeat = 16;
      cout << "Benchmarking full Mateer-Gao DFT..." << endl;
      t1 = absolute_time();
      for(i = 0 ; i < repeat; i++)
      {
        memcpy(refOut, refIn, sz * sizeof(word));
        fft.additive_fft_fast_in_place(refOut, degree, lsz, buffer1, 0);
      }
      t1 = (absolute_time() - t1) / repeat;
      cout << "time for regular additive DFT:  " << t1 / repeat << " sec." << endl;
      t2 = absolute_time();
      // blk_offset != 0 not supported by this method
      for(i = 0 ; i < repeat; i++)
      {
        memcpy(buffer1, refIn, sz * sizeof(word));
        fft_mateer(&c, buffer1, s); // adjust last arg to log of field logsize
      }
      t2 = (absolute_time() - t2) / repeat;
      cout << "time of Mateer-Gao DFT:  " <<  t2 / repeat << " sec." << endl;
      cout << "speed ratio : " <<  t1 / t2 << endl;
    }
  }
#else
  mateer_gao_full(p_buffers, c, false);
  if(benchmark) mateer_gao_full(p_buffers, c, true);
#endif
#endif

#if 1
  double truncated_times[sizeof(log_sz) / 4];
  // test truncated mateer-gao fft, and compare it to truncated regular additive fft
  for(unsigned int j = 0; j < sizeof(log_sz) / 4; j++)
  {
    lsz = log_sz[j];
    if(lsz > c_b_t<word>::n) continue;
    sz = 1uLL << lsz;
    degree = sz - 1; // truncated mateer-gao of size sz needs a polynomial with at most sz coefficients
    cout << endl << "Testing truncated Mateer-Gao Fourier Transform of polynomial of degree " <<
            degree << ", on 2**" <<  lsz << " points, in field of size 2**" << word_sz << endl;
    if(lsz > log_buf_size)
    {
      cout << "Degree too large to fit into buffers for this test" << endl;
      continue;
    }

    additive_fft<word> fft(&c);
    refIn   = p_buffers + 0 * sz;
    refOut  = p_buffers + 1 * sz;
    buffer1 = p_buffers + 2 * sz;
    buffer2 = p_buffers + 3 * sz;
    memset(p_buffers, 0, 4 * sz * sizeof(word));
    for (i = 0; i <= degree; i++) refIn[i] = static_cast<word>(urand());
    if(verbose)
    {
      cout << "Polynomial constant term: " << hex << refIn[0] << dec << endl;
      cout << "Evaluating polynomial with truncated regular additive DFT..." << endl;
    }
    memcpy(buffer2, refIn, (degree+1) * sizeof(word));
    fft.additive_fft_fast_in_place(buffer2, degree, lsz, buffer1, 0);
    if(verbose) cout << "Evaluating polynomial with truncated Mateer-Gao DFT..." << endl;
    memcpy(buffer1, refIn, (degree+1) * sizeof(word));
    fft_mateer_truncated<word,s>(&c, buffer1, lsz);
    local_error = compare_results<word>(buffer1, buffer2, sz, 8, verbose);
    error |= local_error;
    if(benchmark)
    {
      uint64_t repeat_partial = max(1uLL, (1uLL << 24) >> lsz);
      cout << "Benchmarking truncated Mateer-Gao against truncated regular additive DFT" << endl;
      cout << "Performing " << repeat_partial << " operations" << endl;
      t1 = absolute_time();
      for(i = 0 ; i < repeat_partial; i++)
      {
        memcpy(buffer2, refIn, sz * sizeof(word));
        fft.additive_fft_fast_in_place(buffer2, degree, lsz, buffer1, 0);
      }
      t2 = absolute_time();
      t1 = (t2 - t1) / repeat_partial;
      cout << "time of truncated regular additive DFT:  " << t1 << " sec." << endl;
      for(i = 0 ; i < repeat_partial; i++)
      {
        memcpy(buffer1, refIn, sz * sizeof(word));
        fft_mateer_truncated<word,s>(&c, buffer1, lsz);
      }
      t2 = (absolute_time() - t2) / repeat_partial;
      truncated_times[j] = t2; // for later comparison with reverse truncated Mateer-Gao
      cout << "time of truncated Mateer-Gao DFT:  " <<  truncated_times[j] << " sec." << endl;
      cout << "MG/regular additive DFT speed ratio : " << t1 / t2 << endl;

    }
  }
#endif

#if 1
  // testing that fft_mateer_truncated_reverse \circ fft_mateer_truncated = id
  for(unsigned int j = 0; j < sizeof(log_sz) / 4; j++)
  {
    lsz = log_sz[j];
    if(lsz > c_b_t<word>::n) continue;
    sz = 1uLL << lsz;
    degree = sz - 1;
    cout << endl << "Testing reverse truncated Mateer-Gao Fourier Transform of polynomial of degree " <<
            degree << ", on 2**" <<  lsz << " points, in field of size 2**" << word_sz << endl;
    if(lsz > log_buf_size)
    {
      cout << "degree too large to fit into buffers for this test" << endl;
      continue;
    }
    refIn   = p_buffers + 0 * sz;
    refOut  = p_buffers + 1 * sz;
    buffer1 = p_buffers + 2 * sz;
    buffer2 = p_buffers + 3 * sz;
    for(i = 0; i <= degree; i++) refIn[i] = static_cast<word>(urand());
    memcpy(buffer1, refIn, (1 << lsz) * sizeof(word));
    fft_mateer_truncated<word,s>(&c, buffer1, lsz);
    fft_mateer_truncated_reverse<word,s>(&c, buffer1, lsz);
    local_error = compare_results < word > (refIn, buffer1, 1 << lsz);
    error |= local_error;
  }
#endif

#if 1
  // computing products of polynomials over f2 with mateer-gao fft and checking the result
  // with gf2x
  for(unsigned int j = 0; j < sizeof(log_sz) / 4; j++)
  {
    lsz = log_sz[j];
    sz = 1uLL << lsz;
    cout << endl << "Testing F2 polynomial product through FFTs. Multiplying 2 polynomials of degree " <<
            sz/2 - 1 << ", in field of size 2**" << word_sz << endl;
    if(lsz > min(log_buf_bitsize, c_b_t<word>::n))
    {
      if(lsz > log_buf_bitsize) cout << "Resulting degree too large to fit into buffers for this test" << endl;
      if(lsz > c_b_t<word>::n)  cout << "Resulting degree too large for the field size" << endl;
      continue;
    }
    uint8_t* p1 = (uint8_t*) p_buffers;//new uint8_t[4 * sz];
    uint8_t* p2 = p1 +     sz;
    uint8_t* p3 = p1 + 2 * sz;
    uint8_t* p4 = p1 + 3 * sz;

    buffer1 = p_buffers + 2 * sz; // FIXME this works by miracle!!
    buffer2 = p_buffers + 3 * sz; // FIXME this works by miracle!!

    memset(p1, 0, 4 * sz);
    // create two random polynomials with sz/2 coefficients
    for(size_t i = 0; i < sz / 16; i++)
    {
      p1[i] = rand() & 0xFF;
      p2[i] = rand() & 0xFF;
    }
    // multiply polynomials with additive fft, put result in p4
    binary_polynomial_multiply < word > ( &c, p1, p2, p4, buffer1, buffer2, sz/2 - 1, sz/2 - 1, lsz);
    // multiply polynomials with gf2x, put result in p3
    gf2x_mul(
          (unsigned long *) p3,
          (unsigned long *) p1,
          sz/(2*sizeof(unsigned long)),
          (unsigned long *) p2,
          sz/(2*sizeof(unsigned long)));
    // compare results
    local_error = compare_results < uint8_t > (p3, p4, sz/8+1);
    error |= local_error;
  }
#endif
  if(benchmark)
  {
#if 1
    for(unsigned int j = 0; j < sizeof(log_sz) / 4; j++)
    {
      lsz = log_sz[j];
      if(lsz > c_b_t<word>::n) continue;
      sz = 1uLL << lsz;
      degree = sz - 1;
      cout << endl << "Benchmarking reverse truncated Mateer-Gao Fourier Transform of polynomial of degree "
           << degree << ", on 2**" <<  lsz << " points, in field of size 2**" << word_sz << endl;
      if(sz + c_b_t<word>::word_logsize >= (1uLL << log_buf_size))
      {
        cout << "degree too large to fit into buffers for this test" << endl;
        continue;
      }
      refIn   = p_buffers + 0 * sz;
      refOut  = p_buffers + 1 * sz;
      buffer1 = p_buffers + 2 * sz;
      buffer2 = p_buffers + 3 * sz;
      uint64_t repeat_partial = max(1uLL, (1uLL << 24) >> lsz);
      cout << "Performing " << repeat_partial << " operations" << endl;
      t1 = absolute_time();
      for(i = 0 ; i < repeat_partial; i++) fft_mateer_truncated_reverse<word,s>(&c, refIn, lsz);
      t1 = (absolute_time() - t1) / repeat_partial;
      cout << "Time of reverse truncated Mateer-Gao DFT:  " <<  t1 << " sec." << endl;
      cout << "Reverse truncated MG / truncated MG speed ratio: " << truncated_times[j] / t1 << endl;
    }
#endif
    for(unsigned int j = 0; j < sizeof(log_sz) / 4; j++)
    {
      lsz = log_sz[j];
      sz = 1uLL << lsz;
      degree = sz - 1;
      cout << endl << "Benchmarking F2 polynomial product through FFTs with maximum total degree "
           << degree << ", in field of size 2**" << word_sz << endl;
      if(lsz > min(log_buf_bitsize, c_b_t<word>::n))
      {
        if(lsz > log_buf_bitsize) cout << "Resulting degree too large to fit into buffers for this test" << endl;
        if(lsz > c_b_t<word>::n)  cout << "Resulting degree too large for the field size" << endl;
        continue;
      }
      refIn   = p_buffers + 0 * sz;
      refOut  = p_buffers + 1 * sz;
      buffer1 = p_buffers + 2 * sz;
      buffer2 = p_buffers + 3 * sz;
      uint8_t* p1 = new uint8_t[4 * sz];
      uint8_t* p2 = p1 +     sz;
      uint8_t* p3 = p1 + 2 * sz;
      uint8_t* p4 = p1 + 3 * sz;
      unique_ptr<uint8_t[]> _(p1);

      memset(p1, 0, 3*sz);
      for(size_t i = 0; i < sz / 16; i++)
      {
        p1[i] = rand() & 0xFF;
        p2[i] = rand() & 0xFF;
      }
      repeat = max(1uLL, (1uLL << 24) >> lsz);
      cout << "Performing " << repeat << " operations" << endl;
      double t1 = absolute_time();
      for(size_t i = 0; i < repeat; i++)
      {
        gf2x_mul((unsigned long *) p3,(unsigned long *) p1, sz/(2*sizeof(unsigned long)),
                 (unsigned long *) p2, sz/(2*sizeof(unsigned long)));
      }

      t1 = (absolute_time() - t1) / repeat;
      cout << "gf2x time: " << t1 << endl;
      t2 = absolute_time();
      for(size_t i = 0; i < repeat; i++)
      {
        binary_polynomial_multiply<word>(&c, p1, p2, p4, buffer1, buffer2, sz/2 - 1, sz/2 - 1, lsz);
      }
      t2 = (absolute_time() - t2) / repeat;
      cout << "Mateer-Gao time: " << t2 << endl;
      cout << "MG / gf2x speed ratio: " << t1 / t2 << endl;
    }
  }

  if(error)
  {
    cout << "* Some Mateer-Gao DFT tests failed *" << endl;
  }
  else
  {
    cout << "Mateer-Gao DFT tests succeeded" << endl;
  }
  return error;
}

