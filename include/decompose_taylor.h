#pragma once

#include <cstdint>
#include <cassert>

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
 * a number that divides n/2 (therefore a power of 2).
 * @param n
 * the smallest power of 2 >= num_coeffs.
 * (if n is not a power of 2, the output of the algorithm will be incorrect)
 * @param poly
 * a pointer to the coefficients of f.
 */

template<class word>
inline void decompose_taylor_one_step(
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

  word* __restrict__ p1 = poly + delta_s;
  word* __restrict__ p2 = poly + (n_s - delta_s);
  word* __restrict__ p3 = poly + m_s;
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
  bool error = false;
  const uint64_t delta_s      = delta      << logstride;
  const uint64_t m_s          = m          << logstride;
  const uint64_t num_coeffs_s = num_coeffs << logstride;
  for(size_t i = 0;   i < num_coeffs_s; i++) ref[i] ^= result[i];
  for(size_t i = m_s; i < num_coeffs_s; i++) ref[i - m_s + delta_s] ^= result[i];
  for(size_t i = 0; i < num_coeffs_s; i++)
  {
    if(ref[i])
    {
      error = true;
      break;
    }
  }
  return error;
}

template <class word>
inline void decompose_taylor_one_step_reverse(
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
  if(num_coeffs_low  > tau)
  {
    decompose_taylor_recursive(logstride, logblocksize, logtau, num_coeffs_low, poly);
  }
  if(num_coeffs_high > tau)
  {
    decompose_taylor_recursive(logstride, logblocksize, logtau, num_coeffs_high, poly + (m<<logstride));
  }

}

template<class word>
void decompose_taylor_iterative_alt(
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

  //state
  uint64_t current_position = 0;
  int previous_move = 2; // 2: descent; 0: left ascent; 1: right ascent
  int depth = 0;
  uint64_t left_bound = 0;
  uint64_t right_bound = num_coeffs;
  bool go_back_up = false;
  int is_full_lowest_depth = 1uLL << logn == num_coeffs ? 0 : -1;
  // lowest depth encountered at which the interval is full,
  // i.e. of length 2**(logn - depth)
  while(true)
  {
    go_back_up = false;
    assert(right_bound > left_bound + tau);
    if(previous_move == 2)
    {
      // decompose
      uint64_t local_logn = logn - depth;
      uint64_t local_num_coeffs = right_bound - left_bound;
      if(is_full_lowest_depth == -1) while((1uLL << local_logn) >= (2 * local_num_coeffs)) local_logn--;
      uint64_t local_delta = 1uLL << (local_logn - 1 - logtau); // 2**delta = m/tau
      decompose_taylor_one_step(logstride, local_num_coeffs, local_delta, 1uLL<<local_logn, poly+(left_bound<<logstride));

      //then descend in the left subtree if we are not already at a leaf
      // we check only the leaf criterion on the left subtree, since if that one is empty, so is the
      // right one
      if(logn - depth - 1 <= logtau) go_back_up = true;
      else
      {
        uint64_t middle = left_bound + (1uLL<< (logn-depth-1));
        right_bound = middle;
        if(is_full_lowest_depth == -1) // complex case: we may deal with a truncated interval
        {
          if(middle <= num_coeffs)
          {
            // interval at depth (depth+1) is full
            is_full_lowest_depth = depth + 1;
          }
          else
          {
            right_bound = num_coeffs;
            // the current interval, and the next, are [left_bound, num_coeffs[
            // hence the next interval cannot be too small to be processed
            assert(num_coeffs > left_bound + tau);
          }
        }
        current_position <<= 1;
        depth++;
      }
      // previous_move and left_bound are unchanged
    }
    else if(previous_move == 0)
    {
      // going back from left subtree, need to explore right subtree if not already at a leaf
      // we explored the left subtree hence its length at depth depth + 1 was not too small
      // hence the untruncated length is not too small
      assert(logn - depth - 1 > logtau);
      uint64_t middle = left_bound + (1uLL << (logn - depth - 1));
      if((is_full_lowest_depth == -1) &&(num_coeffs <= middle + tau))
      {
        // we hit the rightmost coefficient of the initial polynomial
        // and this makes our right subtree to small to be processed
        go_back_up = true;
      }
      else
      {
        current_position <<= 1;
        current_position |= 1;
        depth++;
        previous_move = 2;
        left_bound = middle;
        // right bound unchanged
      }
    }
    else // previous_move == 1
    {
      // went back from right subtree, need to go back up
      go_back_up = true;
    }

    if(go_back_up)
    {
      if(depth == 0) break;
      previous_move = current_position & 1;
      current_position >>= 1;
      depth--;
      if(previous_move == 0)
      {
        if(is_full_lowest_depth == depth + 1) is_full_lowest_depth = -1;

        if(is_full_lowest_depth == -1)
        {
          right_bound = num_coeffs;
        }
        else
        {
          right_bound = (current_position + 1) << (logn - depth);
        }

      }
      else
      {
        left_bound = current_position << (logn - depth);
      }
    }
  }
}


template<class word>
void decompose_taylor_reverse_iterative_alt(
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

  //state
  uint64_t current_position = 0;
  int previous_move = 2; // 2: descent; 0: left ascent; 1: right ascent
  int depth = 0;
  uint64_t left_bound = 0;
  uint64_t right_bound = num_coeffs;
  bool go_back_up = false;
  int is_full_lowest_depth = 1uLL << logn == num_coeffs ? 0 : -1;
  // lowest depth encountered at which the interval is full,
  // i.e. of length 2**(logn - depth)
  while(true)
  {
    go_back_up = false;
    assert(right_bound > left_bound + tau);
    if(previous_move == 2)
    {
      //descend in the left subtree if we are not already at a leaf
      // we check only the leaf criterion on the left subtree, since if that one is empty, so is the
      // right one
      if(logn - depth - 1 <= logtau) go_back_up = true;
      else
      {
        uint64_t middle = left_bound + (1uLL<< (logn-depth-1));
        right_bound = middle;
        if(is_full_lowest_depth == -1) // complex case: we may deal with a truncated interval
        {
          if(middle <= num_coeffs)
          {
            // interval at depth (depth+1) is full
            is_full_lowest_depth = depth + 1;
          }
          else
          {
            right_bound = num_coeffs;
            // the current interval, and the next, are [left_bound, num_coeffs[
            // hence the next interval cannot be too small to be processed
            assert(num_coeffs > left_bound + tau);
          }
        }
        current_position <<= 1;
        depth++;
      }
      // previous_move and left_bound are unchanged
    }
    else if(previous_move == 0)
    {
      // going back from left subtree, need to explore right subtree if not already at a leaf
      // we explored the left subtree hence its length at depth depth + 1 was not too small
      // hence the untruncated length is not too small
      assert(logn - depth - 1 > logtau);
      uint64_t middle = left_bound + (1uLL << (logn - depth - 1));
      if((is_full_lowest_depth == -1) &&(num_coeffs <= middle + tau))
      {
        // we hit the rightmost coefficient of the initial polynomial
        // and this makes our right subtree to small to be processed
        go_back_up = true;
      }
      else
      {
        current_position <<= 1;
        current_position |= 1;
        depth++;
        previous_move = 2;
        left_bound = middle;
        // right bound unchanged
      }
    }
    else // previous_move == 1
    {
      // went back from right subtree, need to go back up
      go_back_up = true;
    }

    if(go_back_up)
    {
      // recompose the initial polynomial
      uint64_t local_logn = logn - depth;
      uint64_t local_num_coeffs = right_bound - left_bound;
      if(is_full_lowest_depth == -1) while((1uLL << local_logn) >= (2 * local_num_coeffs)) local_logn--;
      uint64_t local_delta = 1uLL << (local_logn - 1 - logtau); // 2**delta = m/tau
      decompose_taylor_one_step_reverse(logstride, local_num_coeffs, local_delta, 1uLL<<local_logn, poly+(left_bound<<logstride));

      if(depth == 0) break;
      previous_move = current_position & 1;
      current_position >>= 1;
      depth--;
      if(previous_move == 0)
      {
        if(is_full_lowest_depth == depth + 1) is_full_lowest_depth = -1;

        if(is_full_lowest_depth == -1)
        {
          right_bound = num_coeffs;
        }
        else
        {
          right_bound = (current_position + 1) << (logn - depth);
        }

      }
      else
      {
        left_bound = current_position << (logn - depth);
      }
    }
  }
}

template<class word>
void decompose_taylor_iterative_alt2(
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

  //state
  uint64_t current_position = 0;
  int previous_move = 2; // 2: descent; 0: left ascent; 1: right ascent
  int depth = 0;
  while(true)
  {
    uint64_t left_bound = current_position << (logn - depth); // invariant: left_bound < num_coeffs
    uint64_t right_bound = min((current_position + 1) << (logn - depth), num_coeffs);
    // because left_bound < num_coeffs, left_bound < right_bound

    // FIXME do as little computation as possible to update left_bound, right_bound, rounded_logwidth
    // depending on their previous values and the last move.
    if(right_bound > left_bound + tau)
    {
      uint64_t local_logn = logn - depth;
      uint64_t local_num_coeffs = right_bound - left_bound;
      while((1uLL << local_logn) >= (2 * local_num_coeffs)) local_logn--;
      uint64_t local_delta = 1uLL << (local_logn - 1 - logtau); // 2**delta = m/tau
      if(previous_move == 2)
      {
        // decompose, then descend in the left subtree
        decompose_taylor_one_step(logstride, local_num_coeffs, local_delta, 1uLL<<local_logn, poly+(left_bound<<logstride));
        current_position <<=1;
        depth++;
        // previous_move is unchanged
      }
      else if(previous_move == 0)
      {
        // going back from left subtree, need to explore right subtree
        // no check done here to determine whether there is something to do in this subtree: will
        // be done when we are there
        current_position <<= 1;
        current_position |= 1;
        depth++;
        previous_move = 2;
      }
      else // previous_move == 1
      {
        if(depth == 0) break;
        // went back from right subtree, need to go back up if depth > 0
        previous_move = current_position & 1;
        current_position >>= 1;
        depth--;
      }
    }
    else { // right_bound <= left_bound + tau
      // we are in a leaf where there is nothing to do. go back up unless
      // we are already at the root.
      if(depth == 0) break;
      // go back up
      previous_move = current_position & 1;
      current_position >>= 1;
      depth--;
    }
  }
}


template<class word>
void decompose_taylor_reverse_recursive(
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
  const uint64_t num_coeffs_high = num_coeffs - m;
  const uint64_t num_coeffs_low  = m;
  if(num_coeffs_high > tau)
  {
    decompose_taylor_reverse_recursive(logstride, logblocksize, logtau, num_coeffs_high, poly + (m<<logstride));
  }
  if(num_coeffs_low  > tau)
  {
    decompose_taylor_reverse_recursive(logstride, logblocksize, logtau, num_coeffs_low, poly);
  }
  decompose_taylor_one_step_reverse<word>(logstride, num_coeffs, delta, n, poly);
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
