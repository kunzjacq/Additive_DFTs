#pragma once

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstdio>

#include <iostream>
#include <iomanip>

#if defined(__cplusplus)
#define UNUSED(x)
#elif defined(__GNUC__)
#define UNUSED(x)       x __attribute__((unused))
#else
#define UNUSED(x)       x
#endif

using namespace std;

template <class word>
void print_value(const word& p_data)
{
  constexpr size_t num_chars = 2* sizeof(word);
  cout << hex << setw(num_chars) << p_data;
}

template <>
void print_value(const uint8_t& p_data);

template <class word>
void print_series(const word* p_data, size_t p_group_size, size_t p_num_display)
{
  cout << hex << setfill('0');
  if(p_group_size > 2 * p_num_display)
  {
    for(size_t i = 0; i < p_num_display; i++)
    {
      print_value(p_data[i]);
      cout << " ";
    }
    cout << "... ";
    for(size_t i = 0; i < p_num_display; i++)
    {
      print_value(p_data[p_group_size - p_num_display + i]);
      cout << " ";
    }
    cout << endl;
  }
  else
  {
    for(size_t i = 0; i < p_group_size; i++)
    {
      print_value(p_data[i]);
      cout << " ";
    }
    cout << endl;
  }
  cout << dec << setfill(' ');
}

template<class word>
int compare_results(word* p_dataA, word* p_dataB, size_t p_group_size, size_t p_num_display = 8, bool always_display=false)
{
  size_t num_errors = 0;
  for (size_t i = 0; i < p_group_size; i++)
  {
    if (p_dataA[i] != p_dataB[i])
    {
      //cout << "Error at " << dec << i << ": "; print_value(p_dataA[i]);
      //cout << "/"; print_value(p_dataB[i]); cout << endl;
      num_errors++;
    }
  }

  if(always_display || (num_errors > 0))
  {
    cout << endl << "** " << num_errors << " errors **" << endl;
    print_series(p_dataA, p_group_size, p_num_display);
    print_series(p_dataB, p_group_size, p_num_display);
    cout << endl;
  }
  else
  {
    cout << "no error" << endl;
  }
  return num_errors > 0 ? 1 : 0;
}

static inline uint64_t a025480(uint64_t n){
    while (n&1) n=n>>1;
    return n>>1;
}

static inline uint64_t larger_half(uint64_t sz) {return sz - (sz >> 1); }
static inline uint64_t is_even(uint64_t i) { return ((i & 1) ^ 1); }

static inline uint64_t unshuffle_item(uint64_t j, uint64_t sz)
{
  uint64_t i = j;
  do {
    i = a025480((sz >> 1) + i);
  }
  while (i < j);
  return i;
}

/*
 * interleave two arrays with no extra buffer.
 * algorithm and code from
 * https://cs.stackexchange.com/questions/332/in-place-algorithm-for-interleaving-an-array
 * (with small adaptations)
 * practical performance is ok, but not stellar compared to simpler algorithms
 * using a buffer of size aN for a < 1. (see below)
*/
#if 0
template<class word>
void interleave_in_place(word *a, uint64_t log_n_items)
{
  uint64_t i = 0;
  uint64_t n_items = 1uLL << log_n_items;
  uint64_t midpt = larger_half(n_items);
  while (i < n_items - 1) {

    //for out-shuffle, the left item is at an even index
    if (is_even(i)) { i++; }
    uint64_t base = i;

    //emplace left half.
    for (; i < midpt; i++) {
      uint64_t j = a025480(i - base);
      std::swap<word>(a[i], a[midpt + j]);
    }

    //unscramble swapped items
    uint64_t swap_ct  = larger_half(i - base);
    for (uint64_t j = 0; j + 1 < swap_ct ; j++) {
      uint64_t k = unshuffle_item(j, i - base);
      if (j != k) {
        std::swap<word>(a[midpt + j], a[midpt + k]);
      }
    }
    midpt += swap_ct;
  }
}

#else
template<class word>
void interleave_in_place(word* a, uint64_t log_n_items)
{
  uint64_t i = 0;
  uint64_t n_items = 1uLL << log_n_items;
  uint64_t midpt = 1uLL << (log_n_items - 1);
  while (i < n_items - 1) {
    //for out-shuffle, the left item is at an even index
    //if (is_even(i)) { i++; }
    i |= 1;
    uint64_t base = i;
    //emplace left half.
    for (; i < midpt; i++)
    {
      uint64_t val = i - base;
      const int k = __builtin_ia32_tzcnt_u64(~val) + 1;
      uint64_t j = val >> k; // j = a025480(val);
      swap<word>(a[i], a[midpt + j]);
    }
    //unscramble swapped items
    uint64_t swap_ct  = ((i - base) + 1) >> 1;
    for (uint64_t j = 0; j + 1 < swap_ct ; j++)
    {
      const uint64_t sz = i - base;
      uint64_t k = j;
      do
      {
        k += sz >> 1;
        k >>= __builtin_ia32_tzcnt_u64(~k) + 1; // k = a025480(k);
      }
      while (k < j);
      //uint64_t k = unshuffle_item(j, i - base);
      if (j != k) swap<word>(a[midpt + j], a[midpt + k]);
    }
    midpt += swap_ct;
  }
}
#endif

// with a buffer as large as the result, trivial interleaving
template<class word>
void interleave_full_buffer(word*buf, word* p_poly, int logsz)
{
  uint64_t psz = 1uLL << logsz;
  uint64_t k;
  for(k = 0; k < psz>>1; k++)
  {
    buf[2 * k    ] = p_poly[k];
    buf[2 * k + 1] = p_poly[(psz>>1) + k];
  }
  for(k = 0; k < psz; k++)
  {
    p_poly[k] = buf[k];
  }
}

// interleaving with a buffer half the size of the result
template<class word>
void interleave_half_buffer(word* buf, word* p_poly, int logsz)
{
  // 'parts' refer to 1/4 of the output buffer of size 2**logsz.
  // buf can hold 2 parts.
  uint64_t psz = 1uLL << (logsz - 2); // part size
  uint64_t k;
  // copy parts 0 and 1 in buffer
  for(k = 0; k < (psz<<1); k++) buf[k] = p_poly[k];
  for(k = 0; k < psz; k++)
  {
    p_poly[2 * k]     = buf[k];               // part 0
    p_poly[2 * k + 1] = p_poly[(psz<<1) + k]; // part 2
  }

  for(k = 0; k < psz; k++) buf[k] = p_poly[psz * 3 + k];
  // buf now contains parts 3 and 1
  for(k = psz; k < (psz<<1); k++)
  {
    p_poly[2 * k]     = buf[k];       // part 1
    p_poly[2 * k + 1] = buf[k - psz]; // part 3
  }
}

// interleaving with a buffer 1/4 the size of the result
template<class word>
void interleave_quarter_buffer_alt(word* buf, word* p_poly, int logsz)
{
  // 'parts' refer to 1/8 of the output buffer of size 2**logsz.
  // buf can hold 2 parts.
  uint64_t psz = 1uLL << (logsz-3); // part size
  uint64_t k;
  // save parts 0, 1
  for(k = 0; k < (psz << 1); k++) buf[k] = p_poly[k];
  // buf = [0, 1]
  // result = [x, x, 2, 3, 4, 5, 6, 7] (x = don't care, R = result, number = part number)
  // write interleaved result in parts 0, 1 from parts 0 and 4
  for(k = 0; k < psz; k++)
  {
    p_poly[2*k]     = buf[k]; // part 0
    p_poly[2*k + 1] = p_poly[k + psz * 4]; // part 4
  }
  // save part 2 in part 0 of buf and part 3 in part 4 of result
  for(k = 0; k < psz; k++)
  {
    buf[k] = p_poly[k + psz * 2]; // part 2
    p_poly[k + psz * 4] = p_poly[k + psz * 3]; // part 3
  }
  // buf = [2, 1]
  // result = [R, R, x, x, 3, 5, 6, 7]
  // write interleaved result in parts 2, 3 from parts 1 and 5
  for(k = 0; k < psz; k++)
  {
    p_poly[psz * 2 + 2*k]     = buf[psz + k]; // part 1
    p_poly[psz * 2 + 2*k + 1] = p_poly[k + psz * 5]; // part 5
  }
  // save part 3 (which is now in result part 4) in buf part 1
  for(k = 0; k < psz; k++)
  {
    buf[psz + k] = p_poly[k + psz * 4]; // part 3
  }
  // buf = [2, 3]
  // result = [R, R, R, R, x, x, 6, 7]
  // write interleaved result in parts 4, 5 from parts 2 and 6
  for(k = 0; k < psz; k++)
  {
    p_poly[psz * 4 + 2*k]     = buf[k]; // part 2
    p_poly[psz * 4 + 2*k + 1] = p_poly[k + psz * 6]; // part 6
  }
  // save part 7 in buf part 0
  for(k = 0; k < psz; k++)
  {
    buf[k] = p_poly[k + psz * 7]; // part 7
  }
  // buf = [7, 3]
  // result = [R, R, R, R, R, R, x, x]
  // write interleaved result in parts 6, 7 from parts 3 and 7
  for(k = 0; k < psz; k++)
  {
    p_poly[psz * 6 + 2*k]     = buf[psz + k]; // part 3
    p_poly[psz * 6 + 2*k + 1] = buf[k]; // part 7
  }
  // buf = [x, x]
  // result = [R, R, R, R, R, R, R, R]
}

template<class word>
void interleave_quarter_buffer(word* buf, word* p_poly, int logsz)
{
  // with a buffer 1/4 the size of the result
  // 'parts' refer to 1/8 of the output buffer of size 2**logsz.
  // buf can hold 2 parts.
  uint64_t psz = 1uLL << (logsz-3); // part size
  uint64_t k;
  // save parts 1, 5; put part 0 in part 5; exchange part 7 and 2
  for(k = 0; k < psz; k++)
  {
    buf[k] = p_poly[k + psz];
    buf[k + psz] = p_poly[k + 5 * psz];
    p_poly[k + 5 * psz] = p_poly[k];
    swap<word>(p_poly[k + 2 * psz], p_poly[k + 7 * psz]);
  }
  // write interleaved result in parts 0, 1 from parts 0 and 4
  for(k = 0; k < psz; k++)
  {
    p_poly[2*k]     = p_poly[k + psz * 5]; // part 0
    p_poly[2*k + 1] = p_poly[k + psz * 4]; // part 4
  }
  // write interleaved result in parts 4, 5 from parts 2 and 6
  for(k = 0; k < psz; k++)
  {
    p_poly[psz * 4 + 2*k]     = p_poly[k + psz * 7]; // part 2
    p_poly[psz * 4 + 2*k + 1] = p_poly[k + psz * 6]; // part 6
  }
  // write interleaved result in parts 6, 7 from parts 3 and 7
  for(k = 0; k < psz; k++)
  {
    p_poly[psz * 6 + 2*k]     = p_poly[k + psz * 3]; // part 3
    p_poly[psz * 6 + 2*k + 1] = p_poly[k + psz * 2]; // part 7
  }
  // write interleaved result in parts 2, 3 from parts 1 and 5 in buffer
  for(k = 0; k < psz; k++)
  {
    p_poly[psz * 2 + 2*k]     = buf[k];       // part 1
    p_poly[psz * 2 + 2*k + 1] = buf[k + psz]; // part 5
  }
}

// mirror of interleave_quarter_buffer.
template<class word>
void deinterleave_quarter_buffer(word* buf, word* p_poly, int logsz)
{
  // with a buffer 1/4 the size of the result
  // 'parts' refer to 1/8 of the output buffer of size 2**logsz.
  // buf can hold 2 parts.
  uint64_t psz = 1uLL << (logsz-3); // part size
  uint64_t k;
  // retrieve parts 1 and 5 in buffer from parts 2, 3
  for(k = 0; k < psz; k++)
  {
    buf[k]       = p_poly[psz * 2 + 2*k];     // part 1
    buf[k + psz] = p_poly[psz * 2 + 2*k + 1]; // part 5
  }
  // retrieve parts 3 and 7 in parts 2 and 3 from parts 6 and 7
  for(k = 0; k < psz; k++)
  {
    p_poly[k + psz * 3] = p_poly[psz * 6 + 2*k];     // part 3
    p_poly[k + psz * 2] = p_poly[psz * 6 + 2*k + 1]; // part 7
  }
  // retrieve parts 2 and 6 in parts 7 and 6 from parts 4, 5
  for(k = 0; k < psz; k++)
  {
    p_poly[k + psz * 7] = p_poly[psz * 4 + 2*k];     // part 2
    p_poly[k + psz * 6] = p_poly[psz * 4 + 2*k + 1]; // part 6
  }
  // retrieve parts 0 and 4 in parts 5 and 4 from parts 0, 1
  for(k = 0; k < psz; k++)
  {
    p_poly[k + psz * 5] = p_poly[2*k];     // part 0
    p_poly[k + psz * 4] = p_poly[2*k + 1]; // part 4
  }

  // put part 5 in part 0; exchange part 7 and 2; restore parts 1, 5 from buf
  for(k = 0; k < psz; k++)
  {
    p_poly[k] = p_poly[k + 5 * psz];
    swap<word>(p_poly[k + 2 * psz], p_poly[k + 7 * psz]);
    p_poly[k + psz]     = buf[k];
    p_poly[k + 5 * psz] = buf[k + psz];
  }
}

template<class word>
void interleave_eigth_buffer(word* buf, word* p_poly, int logsz)
{
  // with a buffer 1/8 the size of the result
  // 'parts' refer to 1/16 of the output buffer of size 2**logsz.
  // buf can hold 2 parts.
  uint64_t psz = 1uLL << (logsz-4); // part size
  uint64_t k;
  // save parts 1, 9; put part 0 in part 9; exchange part 3 and 10, 4 and 13, 6 and 15.
  for(k = 0; k < psz; k++)
  {
    buf[k] = p_poly[k + psz];
    buf[k + psz] = p_poly[k + 9 * psz];
    p_poly[k + 9 * psz] = p_poly[k];
    swap<word>(p_poly[k + 3 * psz], p_poly[k + 10 * psz]);
    swap<word>(p_poly[k + 4 * psz], p_poly[k + 13 * psz]);
    swap<word>(p_poly[k + 6 * psz], p_poly[k + 15 * psz]);
  }
  // write interleaved result in parts 0, 1 from parts 0 and 8
  for(k = 0; k < psz; k++)
  {
    p_poly[2*k]     = p_poly[k + psz * 9]; // part 0
    p_poly[2*k + 1] = p_poly[k + psz * 8]; // part 8
  }
  // write interleaved result in parts 8, 9 from parts 4 and 12
  for(k = 0; k < psz; k++)
  {
    p_poly[psz * 8 + 2*k]     = p_poly[k + psz * 13]; // part 4
    p_poly[psz * 8 + 2*k + 1] = p_poly[k + psz * 12]; // part 12
  }
  // write interleaved result in parts 12, 13 from parts 6 and 14
  for(k = 0; k < psz; k++)
  {
    p_poly[psz * 12 + 2*k]     = p_poly[k + psz * 15]; // part 6
    p_poly[psz * 12 + 2*k + 1] = p_poly[k + psz * 14]; // part 14
  }

  // write interleaved result in parts 14, 15 from parts 7 and 15
  for(k = 0; k < psz; k++)
  {
    p_poly[psz * 14 + 2*k]     = p_poly[k + psz * 7]; // part 7
    p_poly[psz * 14 + 2*k + 1] = p_poly[k + psz * 6]; // part 15
  }

  // write interleaved result in parts 6, 7 from parts 3 and 11
  for(k = 0; k < psz; k++)
  {
    p_poly[psz * 6 + 2*k]     = p_poly[k + psz * 10]; // part 3
    p_poly[psz * 6 + 2*k + 1] = p_poly[k + psz * 11]; // part 11
  }

  // write interleaved result in parts 10, 11 from parts 5 and 13
  for(k = 0; k < psz; k++)
  {
    p_poly[psz * 10 + 2*k]     = p_poly[k + psz * 5]; // part 5
    p_poly[psz * 10 + 2*k + 1] = p_poly[k + psz * 4]; // part 13
  }

  // write interleaved result in parts 4, 5 from parts 2 and 10
  for(k = 0; k < psz; k++)
  {
    p_poly[psz * 4 + 2*k]     = p_poly[k + psz * 2]; // part 2
    p_poly[psz * 4 + 2*k + 1] = p_poly[k + psz * 3]; // part 10
  }

  // write interleaved result in parts 2, 3 from parts 1 and 9 in buffer
  for(k = 0; k < psz; k++)
  {
    p_poly[psz * 2 + 2*k]     = buf[k]; // part 1
    p_poly[psz * 2 + 2*k + 1] = buf[k + psz]; // part 9
  }
}


double absolute_time(bool reset_reference = false);
void init_time();

void surand(int32_t seed);
uint32_t urand();

