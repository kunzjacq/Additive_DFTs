#include "utils.h"

// see https://en.wikipedia.org/wiki/CPUID

//gcc (windows or linux)
#ifdef __GNUC__
#include <cpuid.h>
static void cpuid(uint32_t out[4], int32_t eax, int32_t ecx){
  __cpuid_count(eax, ecx, out[0], out[1], out[2], out[3]);
}

#else
// cl
#include <Windows.h>
#include <intrin.h>
static void cpuid(uint32_t out[4], int32_t eax, int32_t ecx){
  __cpuidex(out, eax, ecx);
}
#endif

/**
  Detects whether CPU has SSE2 and PCLMULQDQ, which are required for the fast GF2**64
  implementation used by DFT algorithms.
 */
bool detect_cpu_features()
{
  uint32_t info[4];
  cpuid(info, 0, 0);
  int n_ids = info[0];
  bool has_SSE2   = false;
  bool has_PCLMUL = false;
  if(n_ids >= 1)
  {
    cpuid(info, 1, 0);
    //has_MMX    = (info[3] & (1uLL << 23)) != 0;
    //has_SSE    = (info[3] & (1uLL << 25)) != 0;
    //has_RDRAND = (info[2] & (1uLL << 30)) != 0;
    has_SSE2   = (info[3] & (1uLL << 26)) != 0;
    has_PCLMUL = (info[2] & (1uLL << 1 )) != 0;
  }
  return has_SSE2 && has_PCLMUL;
}

template <>
void print_value(const uint8_t& p_data)
{
  if(p_data != 0)
  {
    cout << hex << setw(2) << static_cast<unsigned int>(p_data);
  }
  else
  {
    cout << "--";
  }
}
