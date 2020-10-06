#include "utils.h"

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

#if 0
void surand(uint32_t seed){
  srand(seed);
}

uint32_t urand(){
  return (uint32_t) rand();
}
#else
static int32_t state;

void surand(int32_t seed) {
    state = seed;
}

uint32_t urand() {
    int const a = 1103515245;
    int const c = 12345;
    state = a * state + c;
    return (uint32_t) ((state >> 16) & 0x7FFF);
}
#endif
