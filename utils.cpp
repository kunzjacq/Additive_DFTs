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

cpu_features::cpu_features():
  has_mmx(false),
  has_sse(false),
  has_rdrand(false),
  has_avx(false),
  has_sse2(false),
  has_aesni(false),
  has_popcnt(false),
  has_pclmul(false),
  has_bmi1(false),
  has_bmi2(false),
  has_gfni(false),
  has_vaes(false),
  has_vpclmul(false),
  has_rdseed(false),
  has_avx2(false),
  has_abm(false),
  has_avx512f(false),
  has_avx512vl(false)
{
  uint32_t info[4];
  cpuid(info, 0, 0);
  unsigned int n_ids = info[0];
  if (n_ids <= 7) return;
  cpuid(info, 1, 0);
  has_mmx    = (info[3] & (1uLL << 23)) != 0;
  has_sse    = (info[3] & (1uLL << 25)) != 0;
  has_rdrand = (info[2] & (1uLL << 30)) != 0;
  has_avx    = (info[2] & (1uLL << 28)) != 0;
  has_sse2   = (info[3] & (1uLL << 26)) != 0;
  has_aesni  = (info[2] & (1uLL << 25)) != 0;
  has_popcnt = (info[2] & (1uLL << 23)) != 0;
  has_pclmul = (info[2] & (1uLL << 1 )) != 0;
  cpuid(info, 7, 0);
  has_bmi1    = (info[1] & (1uLL << 3)) != 0;
  has_bmi2    = (info[1] & (1uLL << 8)) != 0;
  has_gfni    = (info[2] & (1uLL << 8)) != 0;
  has_vaes    = (info[2] & (1uLL << 9)) != 0;
  has_vpclmul = (info[2] & (1uLL << 10)) != 0;
  has_rdseed  = (info[1] & (1uLL << 18)) != 0;
  has_avx2    = (info[1] & (1uLL << 5)) != 0;
  cpuid(info, 0x80000001, 0);
  has_abm = (info[2] & (1uLL << 5)) != 0;
  has_avx512f  = (info[1] & (1uLL << 16)) != 0;
  has_avx512vl = (info[1] & (1uLL << 31)) != 0;
}

void cpu_features::print()
{
  auto p = [](bool b){return b? "true":"false";};
  std::cout << "MMX: " << p(has_mmx) << std::endl;
  std::cout << "SSE: " << p(has_sse) << std::endl;
  std::cout << "RDRAND: " << p(has_rdrand) << std::endl;
  std::cout << "AVX: " << p(has_avx) << std::endl;
  std::cout << "SSE2: " << p(has_sse2) << std::endl;
  std::cout << "AES-NI: " << p(has_aesni) << std::endl;
  std::cout << "POPCNT: " << p(has_popcnt) << std::endl;
  std::cout << "PCLMUL: " << p(has_pclmul) << std::endl;
  std::cout << "Bit Manipulation Instructions 1 (BMI1): " << p(has_bmi1) << std::endl;
  std::cout << "Bit Manipulation Instructions 1 (BMI2): " << p(has_bmi2) << std::endl;
  std::cout << "GFNI: " << p(has_gfni) << std::endl;
  std::cout << "Vector AES: " << p(has_vaes) << std::endl;
  std::cout << "Vector PCLMUL: " << p(has_vpclmul) << std::endl; // for _mm256_clmulepi64_epi128
  std::cout << "RDSEED: " << p(has_rdseed) << std::endl;
  std::cout << "AVX2: " << p(has_avx2) << std::endl;
  std::cout << "Advanced Bit Manipulations (ABM): " << p(has_abm) << std::endl; // for lzcnt and popcnt
  std::cout << "AVX512 foundations: " << p(has_avx512f) << std::endl; // for basic _mm512_ intrinsics such as _mm512_set1_epi{16,32,64}
  std::cout << "AVX512 Vector Length Extensions: " << p(has_avx512vl) << std::endl; // for _mm512_clmulepi64_epi128, together with has_vpclmul
}
