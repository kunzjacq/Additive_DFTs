#pragma once

#include "base.h"

#include <iostream>
#include <iomanip>

class cpu_features{
public:
  bool has_mmx;
  bool has_sse;
  bool has_rdrand;
  bool has_avx;
  bool has_sse2;
  bool has_aesni;
  bool has_popcnt;
  bool has_pclmul;
  bool has_bmi1;
  bool has_bmi2;
  bool has_gfni;
  bool has_vaes;
  bool has_vpclmul;
  bool has_rdseed;
  bool has_avx2;
  bool has_abm;
  bool has_avx512f;
  bool has_avx512vl;

  cpu_features();
  void print();
};

/**
 * @brief extract
 * Extracts at most 'extract_sz' values from 'tbl' of size 'sz'  in front, middle and back
 * @param tbl
   source data
 * @param sz
 * source data size
 * @param extract
 * destination data
 * @param extract_sz
 * number of values to be copied
 * @return
 */
template <class T>
unsigned int extract(T* tbl, uint64_t sz, T* extract, uint32_t extract_sz)
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

template <class word>
void print_value(const word& p_data)
{
  constexpr size_t num_chars = 2* sizeof(word);
  std::cout << std::hex << std::setw(num_chars) << p_data;
}

template <>
void print_value(const uint8_t& p_data);

template <class word>
void print_series(const word* p_data, size_t p_group_size, size_t p_num_display)
{
  std::cout << std::hex << std::setfill('0');
  if(p_group_size > 2 * p_num_display)
  {
    for(size_t i = 0; i < p_num_display; i++)
    {
      print_value(p_data[i]);
      std::cout << " ";
    }
    std::cout << "... ";
    for(size_t i = 0; i < p_num_display; i++)
    {
      print_value(p_data[p_group_size - p_num_display + i]);
      std::cout << " ";
    }
    std::cout << std::endl;
  }
  else
  {
    for(size_t i = 0; i < p_group_size; i++)
    {
      print_value(p_data[i]);
      std::cout << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::dec << std::setfill(' ');
}

template<class word>
int compare_results(word* p_dataA, word* p_dataB, size_t p_group_size, size_t p_num_display = 8, bool always_display=false)
{
  size_t num_errors = 0;
  for (size_t i = 0; i < p_group_size; i++)
  {
    if (p_dataA[i] != p_dataB[i])
    {
      num_errors++;
    }
  }

  if(always_display || (num_errors > 0))
  {
    std::cout << std::endl << "** " << num_errors << " errors **" << std::endl;
    print_series(p_dataA, p_group_size, p_num_display);
    print_series(p_dataB, p_group_size, p_num_display);
    std::cout << std::endl;
  }
  else
  {
    std::cout << "no error" << std::endl;
  }
  return num_errors > 0 ? 1 : 0;
}
