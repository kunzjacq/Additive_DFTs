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
  __cpuidex((int*)out, eax, ecx);
}
#endif

/**
  CPU features detection
 */
bool detect_cpu_features()
{
  uint32_t info[4];
  cpuid(info, 0, 0);
  unsigned int n_ids = info[0];
  if (n_ids <= 7) return false;

  cpuid(info, 1, 0);
  //bool has_mmx    = (info[3] & (1uLL << 23)) != 0;
  //bool has_sse    = (info[3] & (1uLL << 25)) != 0;
  //bool has_rdrand = (info[2] & (1uLL << 30)) != 0;
  //bool has_avx    = (info[2] & (1uLL << 28)) != 0;
  //bool has_sse2   = (info[3] & (1uLL << 26)) != 0;
  //bool has_aesni  = (info[2] & (1uLL << 25)) != 0;
  bool has_popcnt = (info[2] & (1uLL << 23)) != 0;
  bool has_pclmul = (info[2] & (1uLL << 1 )) != 0;

  cpuid(info, 7, 0);
  //bool has_bmi2   = (info[1] & (1uLL << 8)) != 0;
  //bool has_gfni   = (info[2] & (1uLL << 3)) != 0;
  //bool has_rdseed = (info[1] & (1uLL << 18)) != 0;
  bool has_bmi1 = (info[1] & (1uLL << 3)) != 0;
  bool has_avx2 = (info[1] & (1uLL << 5)) != 0;

  // cpuid(info, 0x80000001, 0);
  // bool has_abm = (info[2] & (1uLL << 5)) != 0; // Advanced bit manipulation (lzcnt and popcnt)

  return has_popcnt && has_bmi1 && has_avx2 && has_pclmul;
}

template <>
void print_value(const uint8_t& p_data)
{
  if(p_data != 0)
  {
    std::cout << std::hex << std::setw(2) << static_cast<unsigned int>(p_data);
  }
  else
  {
    std::cout << "--";
  }
}
