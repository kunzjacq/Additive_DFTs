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

double absolute_time(bool reset_reference = false);
void init_time();

void surand(int32_t seed);
uint32_t urand();

