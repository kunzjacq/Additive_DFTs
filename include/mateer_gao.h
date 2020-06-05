#pragma once

#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cassert>

#include "cantor.h"
#include "utils.h" // for absolute_time(), init_time()

using namespace std;

template<class word>
int check_decompose_taylor_one_step(
    word* ref,
    word* result,
    uint64_t num_coeffs,
    uint64_t m,
    uint64_t delta,
    unsigned int logstride);

// #define CHECK

/**
 * @brief decompose_taylor_one_step
 * computes a decomposition of several interleaved input polynomials.
 * For each input polynomial f, and given n, the smallest power of 2 >= num_coeffs,
 * and delta which divides m = n/2,
 * outputs u, v s.t. f = u * (x**tau -x)**delta + v, with tau = m / delta.
 * The function processes q = 2**logstride interleaved polynomials.
 * Processing is done in-place; v is written as the m lower-degree coefficients of
 * f, u as the upper part. degree(u) = degree(f) - m, and only num_coeffs coefficients of input
 * array are used during processing.
 *
 * Decomposition is done as below (see Mateer thesis):
 * f is written (f_a . x**(m - delta) + f_b) . x^m + f_c with
 * m = n / 2
 * d(f_a) <= delta
 * d(f_b) <  m - delta
 * d(f_c) <  m
 * then f = (f_a x**(m-delta) + f_b + f_a) * (x**tau - x)**delta + ((f_a + f_b) x**delta + f_c)
 * hence
 * u = f_a x**(m-delta) + f_b + f_a
 * v = (f_a + f_b) x**delta + f_c

 * @param logstride
 * log2 of number of polynomials processed. coefficient k of i-th polynomial is in position
 * poly[i + q * k], q = 2**logstride.
 * @param num_coeffs
 * the number of coefficients of polynomial f (i.e. degree(f) + 1)
 * @param delta
 * a number that divides n/2.
 * @param n
 * the smallest power of 2 >= num_coeffs.
 * (if n is not a power of 2, the output of the algorithm will be incorrect)
 * @param poly
 * a pointer to the coefficients of f.
 */

template<class word>
void decompose_taylor_one_step(
    unsigned int logstride,
    uint64_t num_coeffs,
    uint64_t delta,
    uint64_t n,
    word* poly)
{
  const uint64_t delta_s      = delta      << logstride;
  const uint64_t n_s          = n          << logstride;
  const uint64_t m_s          = (n >> 1)   << logstride;
  const uint64_t num_coeffs_s = num_coeffs << logstride;
#ifdef CHECK
  word* buf = new word[num_coeffs_s];
  unique_ptr<word[]> _(buf);
  memcpy(buf, poly, sizeof(word) * num_coeffs_s);
#endif

  /*
  The actual code does the operations of the commented code below in each stride
  n >= num_coeffs
  m = n/2 < num_coeffs
  delta = m / tau
  m, n, tau, delta are powers of 2
  // add f_a * x**delta
  // only impacts lower part (d < m)
  // n - 2 * delta >= 0 : write position is below read position
  for(uint64_t d = n - delta; d < num_coeffs; d++)
  {
    poly[delta + (d - (n - delta))] ^= poly[d];
  }
  // add f_b * x**delta
  // only impacts lower part (d < m)
  // delta <= m : write position is below read position
  for(uint64_t d = 0; d < min(num_coeffs - m, m - delta); d++)
  {
    poly[delta + d] ^= poly[m + d];
  }

  // add f_a * x**m
  // only impacts upper part (d >= m)
  // delta <= m : write position is below read position
  for(uint64_t d = n - delta; d < num_coeffs; d++)
  {
    // m + d - (2*m - delta) = d + delta - m
    poly[d - m + delta] ^= poly[d];
  }
  */

  // add f_a * x**delta
  // only impacts lower part (d < m)
  word* p1 = poly + delta_s;
  word* p2 = poly + (n_s - delta_s);
  word* p3 = poly + m_s;
  // if num_coeffs is a power of 2 (num_coeffs = n), line below reduces to num_iter = delta_s
  const size_t num_iter_a = max(n_s - delta_s, num_coeffs_s) - (n_s - delta_s);
  // if num_coeffs is a power of 2 (num_coeffs = n), line below reduces to num_iter = m_s
  const size_t num_iter_b = min(num_coeffs_s - m_s, m_s - delta_s);
  for(uint64_t i = 0; i < num_iter_a; i++) p1[i] ^= p2[i];
  for(uint64_t i = 0; i < num_iter_b; i++) p1[i] ^= p3[i];
  for(uint64_t i = 0; i < num_iter_a; i++) p3[i] ^= p2[i];

#ifdef CHECK
  const uint64_t m = (n >> 1);
  int error = check_decompose_taylor_one_step(buf, poly, num_coeffs, m, delta, logstride);
  if(error) cout << "Error at degree " << num_coeffs - 1 << endl;
#endif
}

template<class word>
int check_decompose_taylor_one_step(
    word* ref,
    word* result,
    uint64_t num_coeffs,
    uint64_t m,
    uint64_t delta,
    unsigned int logstride)
{
  int error = 0;
  const uint64_t delta_s      = delta      << logstride;
  const uint64_t m_s          = m          << logstride;
  const uint64_t num_coeffs_s = num_coeffs << logstride;
  for(size_t i = 0;   i < num_coeffs_s; i++) ref[i] ^= result[i];
  for(size_t i = m_s; i < num_coeffs_s; i++) ref[i - m_s + delta_s] ^= result[i];
  for(size_t i = 0; i < num_coeffs_s; i++)
  {
    if(ref[i])
    {
      error = 1;
      break;
    }
  }
  return error;
}

template <class word>
void decompose_taylor_one_step_reverse(
    unsigned int logstride,
    uint64_t num_coeffs,
    uint64_t delta,
    uint64_t n,
    word* poly)
{
  const uint64_t delta_s      = delta      << logstride;
  const uint64_t m_s          = (n >> 1)   << logstride;
  const uint64_t num_coeffs_s = num_coeffs << logstride;
  for(size_t i = m_s; i < num_coeffs_s; i++) poly[i - m_s + delta_s] ^= poly[i];
}

// tau >= 2
template<class word>
void decompose_taylor_recursive(
    unsigned int logstride,
    unsigned int logblocksize,
    unsigned int logtau,
    uint64_t num_coeffs,
    word* poly)
{
  uint64_t tau = 1uLL << logtau;
  if(num_coeffs <= tau) return; // nothing to do
  unsigned int logn = logblocksize;
  while((1uLL << logn) >= (2 * num_coeffs)) logn--;
  // we want 2**logn >= num_coeffs, 2**(logn-1) < num_coeffs
  uint64_t n = 1uLL << logn;
  uint64_t m = 1uLL << (logn - 1);
  assert(n >= num_coeffs);
  assert(m < num_coeffs);
  uint64_t delta = 1uLL << (logn - 1 - logtau);
  // the order of the additions below is chosen so that
  // values are used before they are modified
  decompose_taylor_one_step(logstride, num_coeffs, delta, n, poly);
  const uint64_t num_coeffs_high = num_coeffs - m;
  const uint64_t num_coeffs_low  = m;
  if(num_coeffs_high > tau)
  {
    decompose_taylor_recursive(logstride, logblocksize, logtau, num_coeffs_high, poly + (m<<logstride));
  }
  if(num_coeffs_low  > tau)
  {
    decompose_taylor_recursive(logstride, logblocksize, logtau, num_coeffs_low, poly);
  }
}

template<class word>
void decompose_taylor(
    unsigned int logstride,
    unsigned int logblocksize,
    unsigned int logtau,
    uint64_t num_coeffs,
    word* poly)
{
  if(num_coeffs <= (1uLL << logtau)) return; // nothing to do
  unsigned int logn = logblocksize;
  while((1uLL << logn) >= (2 * num_coeffs)) logn--;
  for(unsigned int interval_logsize = logn; interval_logsize > logtau; interval_logsize--)
  {
    const size_t n = 1uLL << interval_logsize;
    const uint64_t num_coeffs_rounded = num_coeffs & (~(n - 1));
    const uint64_t delta = 1uLL << (interval_logsize - 1 - logtau);
    for(size_t interval_start = 0; interval_start < num_coeffs_rounded; interval_start += n)
    {
      decompose_taylor_one_step<word>(
            logstride,
            n,
            delta,
            n,
            poly + (interval_start << logstride));
    }
    // last incomplete interval
    uint64_t interval_length = num_coeffs - num_coeffs_rounded;
    if(interval_length > (n >> 1))
    {
      decompose_taylor_one_step<word>(
            logstride,
            interval_length,
            delta,
            n,
            poly + (num_coeffs_rounded << logstride));
    }
  }
}

template<class word>
void decompose_taylor_reverse(
    unsigned int logstride,
    unsigned int logblocksize,
    unsigned int logtau,
    uint64_t num_coeffs,
    word* poly)
{
  if(num_coeffs <= (1uLL << logtau)) return; // nothing to do
  unsigned int logn = logblocksize;
  while((1uLL << logn) >= (2 * num_coeffs)) logn--;
  for(unsigned int interval_logsize = logtau + 1; interval_logsize <= logn; interval_logsize++)
  {
    const size_t n = 1uLL << interval_logsize;
    const uint64_t num_coeffs_rounded = num_coeffs & (~(n - 1));
    const uint64_t delta = 1uLL << (interval_logsize - 1 - logtau);
    for(size_t interval_start = 0; interval_start < num_coeffs_rounded; interval_start += n)
    {
      decompose_taylor_one_step_reverse <word> (
            logstride,
            n,
            delta,
            n,
            poly + (interval_start << logstride));
    }
    // last incomplete interval
    uint64_t interval_length = num_coeffs - num_coeffs_rounded;
    if(interval_length > (n >> 1))
    {
      decompose_taylor_one_step_reverse <word> (
            logstride,
            interval_length,
            delta,
            n,
            poly + (num_coeffs_rounded << logstride));
    }
  }
}

template<class word>
int decompose_taylor_test()
{
  uint64_t logtau = 1;
  unsigned int logstride = 1;
  size_t max_sz = 1uLL << 10;
  size_t min_sz = 0;
  size_t large_sz = 1uLL << 20;
  size_t array_sz = max(large_sz, max_sz << logstride);
  word* test = new word[3*array_sz];
  word* copy = test + array_sz;
  word* ref  = test + 2*array_sz;
  unique_ptr<word[]> _1(test);

  cout << "Test Taylor decomposition used in Mateer-Gao FFT" << endl;
  int error = 0;
  unsigned int logblocksize = 0;
  for(size_t sz = min_sz; sz < max_sz; sz++)
  {
    while((1uL << logblocksize) < sz) logblocksize++;
    memset(test, 0, (sz << logstride) * sizeof(word));
    for(size_t i = 0; i < (sz << logstride); i++)
    {
      test[i] = urand();
      copy[i] = test[i];
      ref[i]  = test[i];
    }

    decompose_taylor_recursive(logstride, logblocksize, logtau, sz, test);
    // compare output of decompose_taylor_iterative to output of decompose_taylor
    decompose_taylor(logstride, logblocksize, logtau, sz, copy);
    if(memcmp(test, copy, (sz << logstride) * sizeof(word))) error = 1;

#if 1
    decompose_taylor_reverse(logstride, logblocksize, logtau, sz, test);
#else
    // this algorithm transforms back the output of decompose_taylor or
    // decompose_taylor_iterative into its input (it is quadratic in sz/tau.)
    // this way, the obtained input can be compared to the initial value.
    uint64_t tau = 1uL << logtau;
    for(size_t i = (sz + tau - 1)/tau; i > 0; i--)
    {
      for(size_t j = i * tau; j < sz; j++)
      {
        for(size_t s = 0; s < (1uL << logstride); s++)
        {
          size_t i1 = s + ((j - tau + 1) << logstride);
          size_t i2 = s + (j << logstride);
          assert(i1 < (sz << logstride));
          assert(i2 < (sz << logstride));
          test[i1] ^= test[i2];
        }
      }
    }
#endif

    if(memcmp(test, ref, (sz << logstride) * sizeof(word))) error = 1;
  }

  if(error == 0) cout << "Taylor decomposition succeeded" << endl;
  else cout << "Taylor decomposition failure" << endl;

  cout << "Taylor decomposition benchmark" << endl;
  // process a large instance, for benchmarking purposes
  while((1uL << logblocksize) < large_sz) logblocksize++;
  for(size_t i = 0; i < large_sz; i++)
  {
    test[i] = urand();
    copy[i] = test[i];
  }
  init_time();
  double t1 = absolute_time();
  decompose_taylor_recursive(0, logblocksize, logtau, large_sz, copy);
  double t2 = absolute_time();
  decompose_taylor(0, logblocksize, logtau, large_sz, test);
  double t3 = absolute_time();
  cout << "Iterative time: " << (t3-t2) << endl;
  cout << "Recursive time: " << (t2-t1) << endl;
  cout << "Taylor decomposition iterative / recursive speed ratio: " << (t3-t2)/(t2-t1) << endl;
  return error ? 1 : 0;
}

// s = log log of block size (for recursion: initialized with field log-log size)
template <class word, int s>
void fft_aux_ref(
    cantor_basis<word>* c_b,
    word* poly,
    word j, // block index to process
    unsigned int logstride)
{
  if constexpr(s == 0)
  {
    const uint64_t stride = 1uL << logstride;
    const word val = c_b->beta_to_gamma(j << 1);
    for(uint64_t i = 0; i < stride; i++)
    {
      // computes (u,v) where
      // u = f_0 + f1 * w_{2.j},
      // v = f_0 + f1 * w_{2.j+1} = f_0 + f1 *(w_{2.j} + 1) = u + f_1
      // f_0 = poly[i], f_1 = poly[i + stride]
      poly[i]          ^= c_b->multiply(poly[i + stride], val);
      poly[i + stride] ^= poly[i];
    }
  }
  else
  {
    constexpr unsigned int t = 1   << (s - 1);
    const uint64_t tau   = 1uL <<  t;
    const uint64_t eta   = 1uL << (2 * t);
    decompose_taylor(logstride, 2 * t, t, eta, poly);
    // on input: there are 'stride' series of size 'eta' = 'tau' * 'tau';
    // elt k of series i is in poly[i + stride * k].
    // do fft on columns.
    // series i is split into 'tau' series of size 'tau', s.t. element a of series b is element of index
    // b + a tau in original series.
    // There are tau * stride subseries.
    // Consider element of index A=a < tau in subseries S of index B < tau * stride.
    // Series S is mapped as the b = (B - i) / stride subseries of the initial series of
    // index i = B mod stride. With this choice,
    // i = B mod stride
    // k = b + a tau =(B - i) / stride + A * tau; k * stride = B - i + A * tau * stride
    // => poly[i + k * stride] = poly[i + B - i + A * tau * stride] = poly[B + A * tau * stride]
    // => performing FFTs on all subseries can be performed as a recursive call with
    // with stride' = tau*stride, i.e. logstride' = logstride + t
    // hence the recursive call
    fft_aux_ref<word, s-1>(c_b, poly, j, logstride + t);
    const uint64_t row_size = 1uL << (t + logstride);
    for(uint64_t i = 0; i < tau; i++)
    {
      word j_loop = (j << t) | i;
      fft_aux_ref<word, s-1>(c_b, poly, j_loop, logstride);
      poly += row_size;
    }
  }
}

template <class word, int s>
void fft_aux_ref_truncated(
    cantor_basis<word>* c_b,
    word* poly,
    word j, // block index to process
    unsigned int logstride,
    unsigned int logsize)
// log2 size of each stride to be processed;
// should be <= 2**s, the input is assumed to be of size 2**logsize and only the 2**logsize first output values are computed
{
  if constexpr(s == 0)
  {
    const uint64_t stride = 1uL << logstride;
    const word val = c_b->beta_to_gamma(j << 1);
    for(uint64_t i = 0; i < stride; i++)
    {
      // computes (u,v) where
      // u = f_0 + f1 * w_{2.j},
      // v = f_0 + f1 * w_{2.j+1} = f_0 + f1 *(w_{2.j} + 1) = u + f_1
      // f_0 = poly[i], f_1 = poly[i + stride]
      poly[i]          ^= c_b->multiply(poly[i + stride], val);
      poly[i + stride] ^= poly[i];
    }
  }
  else
  {
    const unsigned int t = 1 << (s - 1);
    if(logsize <= t)
    {
      fft_aux_ref_truncated<word, s - 1>(c_b, poly, j << t, logstride, logsize);
    }
    else
    {
      const uint64_t tau   = 1uL <<  (logsize - t);
      const uint64_t eta   = 1uL << logsize;
      // on input: there are 2**'logstride' interleaved series of size 'eta' = 2**(2*logsize-t);
      decompose_taylor(logstride, 2 * t, t, eta, poly);
      // fft on columns
      // if logsize >= t, each fft should process 2**(logsize - t) values
      // i.e. logsize' = logsize - t
      fft_aux_ref_truncated<word, s - 1>(c_b, poly, j, logstride + t, logsize - t);
      const uint64_t row_size = 1uL << (t + logstride);
      for(uint64_t i = 0; i < tau; i++)
      {
        word j_loop = (j << t) | i;
        fft_aux_ref_truncated<word, s - 1>(c_b, poly, j_loop, logstride, t);
        poly += row_size;
      }
    }
  }
}

template <class word>
void fft_alt_truncated(
    cantor_basis<word>* c_b,
    word* poly,
    unsigned int s, // log log of block size (for recursion: initialized with field log-log size)
    word j, // block index to process
    unsigned int logsize,
    word* altbuf)
// log2 size of each stride to be processed;
// should be <= 2**s, the input is assumed to be of size 2**logsize and only the 2**logsize first output values are computed
{
  if(s == 0)
  {
    const word val = c_b->beta_to_gamma(j << 1);
    poly[0]          ^= c_b->multiply(poly[1], val);
    poly[1] ^= poly[0];
  }
  else
  {
    const unsigned int t = 1 << (s - 1);
    if(logsize <= t)
    {
      fft_alt_truncated<word>(c_b, poly, s - 1, j << t, logsize, altbuf);
    }
    else
    {
      const uint64_t tau   = 1uL <<  (logsize - t);
      const uint64_t eta   = 1uL << logsize;
      // on input: there is 1 series of size 'eta' = 2**(2*logsize-t);
      decompose_taylor(0, 2 * t, t, eta, poly);
      // fft on columns
      // if logsize >= t, each fft should process 2**(logsize - t) values
      // i.e. logsize' = logsize - t

      //input series is a matrix with  2**(logsize - t) rows and 2**t columns, in row order
      //transpose matrix in-place (write it in column order)
      const uint64_t column_size = tau;
      const uint64_t row_size = 1uL << t;
      //transpose(...);

      for(size_t j = 0; j < row_size; j++)
      {
        for(size_t i = 0; i < column_size; i++)
        {
          // elt en pos i,j : j+ i*row_size
          // echanger b*column_size + j, i et b*column_size+i,j
          altbuf[i + j * column_size] = poly[j + i * row_size];
        }
      }

      word* poly_loc = altbuf;
      for(uint64_t i = 0; i < row_size; i++)
      {
        fft_alt_truncated<word>(c_b, poly_loc, s - 1, j, logsize - t, poly);
        poly_loc += column_size;
      }
      // write the matrix in row order again
      for(size_t j = 0; j < row_size; j++)
      {
        for(size_t i = 0; i < column_size; i++)
        {
          // elt en pos i,j : j+ i*row_size
          // echanger b*column_size + j, i et b*column_size+i,j
          poly[j + i * row_size] = altbuf[i + j * column_size];
        }
      }

      poly_loc = poly;
      for(uint64_t i = 0; i < column_size; i++)
      {
        word j_loop = (j << t) | i;
        fft_alt_truncated<word>(c_b, poly_loc, s - 1, j_loop, t, altbuf);
        poly_loc += row_size;
      }
    }
  }
}

template <class word, int s>
void fft_aux_ref_truncated_reverse(
    cantor_basis<word>* c_b,
    word* poly,
    word j, // block index to process
    unsigned int logstride,
    unsigned int logsize)
// log2 size of each stride to be processed;
// should be <= 2**s, the input is assumed to be of size 2**logsize and only the 2**logsize first output values are computed
{
  if constexpr(s == 0)
  {
    const uint64_t stride = 1uL << logstride;
    const word val = c_b->beta_to_gamma(j << 1);
    for(uint64_t i = 0; i < stride; i++)
    {
      poly[i + stride] ^= poly[i];
      poly[i]          ^= c_b->multiply(poly[i + stride], val);
    }
  }
  else
  {
    constexpr unsigned int t = 1 << (s - 1);
    if(logsize <= t)
    {
      fft_aux_ref_truncated_reverse<word, s - 1>(c_b, poly, j << t, logstride, logsize);
    }
    else
    {
      const uint64_t tau   = 1uL <<  (logsize - t);
      const uint64_t eta   = 1uL << logsize;
      const uint64_t row_size = 1uL << (t + logstride);
      word* poly_loc = poly;
      for(uint64_t i = 0; i < tau; i++)
      {
        word j_loop = (j << t) | i;
        fft_aux_ref_truncated_reverse<word, s - 1>(c_b, poly_loc, j_loop, logstride, t);
        poly_loc += row_size;
      }
      // reverse fft on columns
      fft_aux_ref_truncated_reverse<word, s - 1>(c_b, poly, j, logstride + t, logsize - t);
      decompose_taylor_reverse(logstride, 2 * t, t, eta, poly);
    }
  }
}

template <class word>
void fft_aux_fast(
    cantor_basis<word>* c_b,
    word* poly,
    unsigned int s,
    word j,
    unsigned int logstride,
    unsigned int log_outer_parallelism)
{
  if(s == 0)
  {
    uint64_t stride = 1uL << logstride;
    word val = c_b->beta_to_gamma(j << 1);
    const uint64_t outer_parallelism = 1uL << (log_outer_parallelism + 1);
    const unsigned int num_bytes = (log_outer_parallelism >> 3) + 1;
    if (num_bytes == 1)
    {
      for(uint64_t k = 0; k < outer_parallelism; k += 2)
      {
        for(uint64_t i = 0; i < stride; i++)
        {
          poly[i]          ^= c_b->multiply(poly[i + stride], val);
          poly[i + stride] ^= poly[i];
        }
        val ^= c_b->template beta_to_gamma_byte<0> (k^(k + 2));
        poly += 2*stride;
      }
    }
    else
    {
      // k = (kr << 8)^kl
      for(uint64_t ku = 0; ku < outer_parallelism >> 8; ku++)
      {
        for(uint32_t kl = 0; kl < 0x100uL; kl += 2)
        {
          for(uint32_t i = 0; i < stride; i++)
          {
            poly[i]          ^= c_b->multiply(poly[i + stride], val);
            poly[i + stride] ^= poly[i];
          }
          val ^= c_b->template beta_to_gamma_byte<0> (kl^(kl + 2));
          poly += 2*stride;
        }
        val ^= c_b->beta_to_gamma((ku^(ku+1))<<8, num_bytes);
      }
    }
  }
  else
  {
    const unsigned int t = 1 << (s - 1);
    const uint64_t eta = 1uL << (2 * t);
    const uint64_t outer_block_size = 1uL << (2 * t + logstride);
    word* poly_loc = poly;
    for(uint64_t loop = 0; loop < (1uL << log_outer_parallelism); loop++)
    {
      // perform taylor decomposition in-line
      for(unsigned int interval_logsize = 2*t; interval_logsize > t; interval_logsize--)
      {
        const size_t n_s = 1uL << (interval_logsize + logstride);
        const uint64_t delta_s = 1uL << (interval_logsize - 1 - t + logstride);
        const uint64_t m_s = n_s >> 1;
        const size_t num_iter_a = delta_s;
        const size_t num_iter_b = m_s - delta_s;
        word *p1 = poly_loc + delta_s, *p2 = poly_loc + (n_s - delta_s), *p3 = poly_loc + m_s;
        for(size_t i = 0; i < (eta >> interval_logsize); i ++)
        {
          for(uint64_t p = 0; p < num_iter_a; p++) p1[p] ^= p2[p];
          for(uint64_t p = 0; p < num_iter_b; p++) p1[p] ^= p3[p];
          for(uint64_t p = 0; p < num_iter_a; p++) p3[p] ^= p2[p];
          p1 += n_s;
          p2 += n_s;
          p3 += n_s;
        }
      }
      uint64_t j_loop = j|loop;
      fft_aux_fast<word>(c_b, poly_loc, s-1, j_loop,        logstride + t, 0);
      fft_aux_fast<word>(c_b, poly_loc, s-1, (j_loop << t), logstride,     t);
      poly_loc += outer_block_size;
    }
  }
}

template <class word, int s>
void fft_aux_fast_truncated(
    cantor_basis<word>* c_b,
    word* poly,
    word j,
    unsigned int logstride,
    unsigned int log_outer_parallelism,
    unsigned int logsize)
{
  if constexpr(s == 0)
  {
    if(logsize == 0) return;
    uint64_t stride = 1uL << logstride;
    word val = c_b->beta_to_gamma(j << 1);
    const uint64_t outer_parallelism = 1uL << (log_outer_parallelism + 1);
    const unsigned int num_bytes = (log_outer_parallelism >> 3) + 1;
    if (num_bytes == 1)
    {
      for(uint64_t k = 0; k < outer_parallelism; k += 2)
      {
        for(uint64_t i = 0; i < stride; i++)
        {
          poly[i]          ^= c_b->multiply(poly[i + stride], val);
          poly[i + stride] ^= poly[i];
        }
        val ^= c_b->template beta_to_gamma_byte<0> (k^(k + 2));
        poly += 2*stride;
      }
    }
    else
    {
      // k = (kr << 8)^kl
      for(uint64_t ku = 0; ku < outer_parallelism >> 8; ku++)
      {
        for(uint32_t kl = 0; kl < 0x100uL; kl += 2)
        {
          for(uint32_t i = 0; i < stride; i++)
          {
            poly[i]          ^= c_b->multiply(poly[i + stride], val);
            poly[i + stride] ^= poly[i];
          }
          val ^= c_b->template beta_to_gamma_byte<0> (kl^(kl + 2));
          poly += 2*stride;
        }
        val ^= c_b->beta_to_gamma((ku^(ku+1))<<8, num_bytes);
      }
    }
  }
  else
  {
    const unsigned int t = 1 << (s - 1);
    if(logsize <= t)
    {
      fft_aux_fast_truncated<word, s-1>(c_b, poly, j << t, logstride, 0, logsize);
    }
    else
    {
      const uint64_t eta   = 1uL << logsize;
      const uint64_t outer_block_size = 1uL << (2 * t + logstride);
      for(uint64_t loop = 0; loop < (1uL << log_outer_parallelism); loop++)
      {
        // perform taylor decomposition in-line
        for(unsigned int interval_logsize = logsize; interval_logsize > t; interval_logsize--)
        {
          const size_t n_s = 1uL << (interval_logsize + logstride);
          const uint64_t delta_s = 1uL << (interval_logsize - 1 - t + logstride);
          const uint64_t m_s = n_s >> 1;
          const size_t num_iter_a = delta_s;
          const size_t num_iter_b = m_s - delta_s;
          word *p1 = poly + delta_s, *p2 = poly + (n_s - delta_s), *p3 = poly + m_s;
          for(size_t i = 0; i < (eta >> interval_logsize); i ++)
          {
            for(uint64_t p = 0; p < num_iter_a; p++) p1[p] ^= p2[p];
            for(uint64_t p = 0; p < num_iter_b; p++) p1[p] ^= p3[p];
            for(uint64_t p = 0; p < num_iter_a; p++) p3[p] ^= p2[p];
            p1 += n_s;
            p2 += n_s;
            p3 += n_s;
          }
        }
        uint64_t j_loop = j|loop;
        fft_aux_fast_truncated<word, s-1>(c_b, poly, j_loop,        logstride + t, 0, logsize - t);
        fft_aux_fast_truncated<word, s-1>(c_b, poly, (j_loop << t), logstride,     logsize - t, t);
        poly += outer_block_size;
      }
    }
  }
}

template <class word, int s>
void fft_aux_fast_truncated_reverse(
    cantor_basis<word>* c_b,
    word* poly,
    word j,
    unsigned int logstride,
    unsigned int log_outer_parallelism,
    unsigned int logsize)
{
  if constexpr(s == 0)
  {
    if(logsize == 0) return;
    uint64_t stride = 1uL << logstride;
    word val = c_b->beta_to_gamma(j << 1);
    const uint64_t outer_parallelism = 1uL << (log_outer_parallelism + 1);
    const unsigned int num_bytes = (log_outer_parallelism >> 3) + 1;
    if (num_bytes == 1)
    {
      for(uint64_t k = 0; k < outer_parallelism; k += 2)
      {
        for(uint64_t i = 0; i < stride; i++)
        {
          poly[i + stride] ^= poly[i];
          poly[i]          ^= c_b->multiply(poly[i + stride], val);
        }
        val ^= c_b->template beta_to_gamma_byte<0> (k^(k + 2));
        poly += 2*stride;
      }
    }
    else
    {
      // k = (kr << 8)^kl
      for(uint64_t ku = 0; ku < outer_parallelism >> 8; ku++)
      {
        for(uint32_t kl = 0; kl < 0x100uL; kl += 2)
        {
          for(uint32_t i = 0; i < stride; i++)
          {
            poly[i + stride] ^= poly[i];
            poly[i]          ^= c_b->multiply(poly[i + stride], val);
          }
          val ^= c_b->template beta_to_gamma_byte<0> (kl^(kl + 2));
          poly += 2*stride;
        }
        val ^= c_b->beta_to_gamma((ku^(ku+1))<<8, num_bytes);
      }
    }
  }
  else
  {
    const unsigned int t = 1 << (s - 1);
    if(logsize <= t)
    {
      fft_aux_fast_truncated_reverse<word, s-1>(c_b, poly, j, logstride, log_outer_parallelism, logsize);
    }
    else
    {
      const uint64_t eta   = 1uL << logsize;
      const uint64_t outer_block_size = 1uL << (2 * t + logstride);
      for(uint64_t loop = 0; loop < (1uL << log_outer_parallelism); loop++)
      {
        uint64_t j_loop = j|loop;
        fft_aux_fast_truncated_reverse<word, s-1>(c_b, poly, (j_loop << t), logstride,     logsize - t, t);
        fft_aux_fast_truncated_reverse<word, s-1>(c_b, poly, j_loop,        logstride + t, 0, logsize - t);
        // perform reverse taylor decomposition in-line
        for(unsigned int interval_logsize = t + 1; interval_logsize <= logsize; interval_logsize++)
        {
          const size_t n_s = 1uL << (interval_logsize + logstride);
          const uint64_t delta_s = 1uL << (interval_logsize - 1 - t + logstride);
          const uint64_t m_s = n_s >> 1;

          word *p1 = poly ;
          for(size_t i = 0; i < (eta >> interval_logsize); i ++)
          {
            for(size_t p = m_s; p < n_s; p++) p1[p - m_s + delta_s] ^= p1[p];
            p1 += n_s;
          }
        }
        poly += outer_block_size;
      }
    }
  }
}

// performs a fft of size 2**(2**s)
// s = 3 : 2**8
// s = 4 : 2**16
// s = 5 : 2**32
// s = 6 which should be 2**64 is not processed correctly (uint64_t eta = 1uL<<(1<<s) overflows)
template <class word>
void fft_mateer(cantor_basis<word>* c_b, word* poly, unsigned int s)
{
  if(s >= 6) throw 0;
#if 0
  fft_aux_ref<word>(c_b, poly, s, 0, 0);
#else
  fft_aux_fast<word>(c_b, poly, s, 0, 0, 0);
#endif
}

template <class word, int s>
void fft_mateer_truncated(cantor_basis<word>* c_b, word* poly, unsigned int logsize)
{
  if(s >= 6) throw 0;
  if(logsize > (1u << s)) logsize = 1 << s;
#if 0
  fft_aux_ref_truncated<word, s>(c_b, poly, 0, 0, logsize);
#else
#if 0
  word* buf = new word[1uLL << logsize];
  unique_ptr<word[]> _(buf);
  fft_alt_truncated<word>(c_b, poly, s, 0, logsize, buf);
#else
  fft_aux_fast_truncated<word, s>(c_b, poly, 0, 0, 0, logsize);
#endif
#endif
}

template <class word, int s>
void fft_mateer_truncated_reverse(cantor_basis<word>* c_b, word* poly, unsigned int logsize)
{
  if(s >= 6) throw 0;
  if(logsize > (1u << s)) logsize = 1 << s;
#if 1
  fft_aux_ref_truncated_reverse<word,s>(c_b, poly, 0, 0, logsize);
#else
  fft_aux_fast_truncated_reverse<word, s>(c_b, poly, 0, 0, 0, logsize);
#endif
}

