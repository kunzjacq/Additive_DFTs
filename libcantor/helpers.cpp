#include <algorithm>
#include <cassert>
#include <iomanip>

#include <cstring> // for memcpy/memset/memcmp

#include "cantor.h"
#include "helpers.hpp"


const unsigned int popcnt_mod_2_cppinteger_helper<__uint128_t>::word_logsize;
const unsigned int popcnt_mod_2_cppinteger_helper<long long unsigned int>::word_logsize;

void indent(int depth)
{
  for(int i = 0; i < depth; i++) cout << " ";
}

#ifdef HAS_UINT2048
template<>
unsigned int bit<uint2048_t>(const uint2048_t& w, unsigned int idx)
{
  return bit_cppinteger<uint2048_t>(w, idx);
}

template<>
void set_bit<uint2048_t>(uint2048_t& w, unsigned int idx, int bit_val)
{
  set_bit_cppinteger<uint2048_t>(w, idx, bit_val);
}

template<>
void xor_bit<uint2048_t>(uint2048_t& w, unsigned int idx, int bit_val)
{
  xor_bit_cppinteger<uint2048_t>(w, idx, bit_val);
}

template<>
unsigned int popcnt_mod_2<uint2048_t>(const uint2048_t &w, unsigned int n)
{
  return popcnt_mod_2_cppinteger<uint2048_t>(w, n);
}
#endif

#ifdef HAS_UINT1024
template<>
unsigned int bit<uint1024_t>(const uint1024_t& w, unsigned int idx)
{
  return bit_cppinteger<uint1024_t>(w, idx);
}

template<>
void set_bit<uint1024_t>(uint1024_t& w, unsigned int idx, int bit_val)
{
  set_bit_cppinteger<uint1024_t>(w, idx, bit_val);
}

template<>
void xor_bit<uint1024_t>(uint1024_t& w, unsigned int idx, int bit_val)
{
  xor_bit_cppinteger<uint1024_t>(w, idx, bit_val);
}

template<>
unsigned int popcnt_mod_2<uint1024_t>(const uint1024_t &w, unsigned int n)
{
  return popcnt_mod_2_cppinteger<uint1024_t>(w, n);
}
#endif

#ifdef HAS_UINT512
template<>
unsigned int bit<uint512_t>(const uint512_t& w, unsigned int idx)
{
  return bit_cppinteger<uint512_t>(w, idx);
}

template<>
void set_bit<uint512_t>(uint512_t& w, unsigned int idx, int bit_val)
{
  set_bit_cppinteger<uint512_t>(w, idx, bit_val);
}

template<>
void xor_bit<uint512_t>(uint512_t& w, unsigned int idx, int bit_val)
{
  xor_bit_cppinteger<uint512_t>(w, idx, bit_val);
}

template<>
unsigned int popcnt_mod_2<uint512_t>(const uint512_t &w, unsigned int n)
{
  return popcnt_mod_2_cppinteger<uint512_t>(w, n);
}
#endif

#ifdef HAS_UINT256
template<>
unsigned int bit<uint256_t>(const uint256_t& w, unsigned int idx)
{
  return bit_cppinteger<uint256_t>(w, idx);
}

template<>
void set_bit<uint256_t>(uint256_t& w, unsigned int idx, int bit_val)
{
  set_bit_cppinteger<uint256_t>(w, idx, bit_val);
}

template<>
void xor_bit<uint256_t>(uint256_t& w, unsigned int idx, int bit_val)
{
  xor_bit_cppinteger<uint256_t>(w, idx, bit_val);
}

template<>
unsigned int popcnt_mod_2<uint256_t>(const uint256_t &w, unsigned int n)
{
  return popcnt_mod_2_cppinteger<uint256_t>(w, n);
}
#endif

#ifdef HAS_UINT128
template<>
unsigned int bit<uint128_t>(const uint128_t& w, unsigned int idx)
{
  return bit_cppinteger<uint128_t>(w, idx);
}

template<>
void set_bit<uint128_t>(uint128_t& w, unsigned int idx, int bit_val)
{
  set_bit_cppinteger<uint128_t>(w, idx, bit_val);
}

template<>
void xor_bit<uint128_t>(uint128_t& w, unsigned int idx, int bit_val)
{
  xor_bit_cppinteger<uint128_t>(w, idx, bit_val);
}

template<>
unsigned int popcnt_mod_2<uint128_t>(const uint128_t &w, unsigned int n)
{
  return popcnt_mod_2_cppinteger(w, n);
}
#endif

#ifndef Boost_FOUND
ostream& operator << (ostream& os, const uint128_t& v)
{
  uint64_t low = (uint64_t)v;
  uint64_t high = (v >> 64);
  os << "[" << hex << setfill('0') << setw(16) << high << "|" << setfill('0') << setw(16) << low << "]";
  return os;
}
#endif

