#include "utils.h"

#include <time.h>
#include <sys/time.h>


static long start_time_sec = 0;
static long start_time_usec = 0;

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

void init_time()
{
  struct timeval start;
  gettimeofday(&start, nullptr);
  start_time_sec = start.tv_sec;
  start_time_usec = start.tv_usec;
}

double absolute_time(bool reset_reference) {
  struct timeval t;
  gettimeofday(&t, nullptr);
  // subtracting start_time_sec aims at improving the precision of the conversion to double
  double sec = t.tv_sec - start_time_sec;
  double usec = t.tv_usec - start_time_usec;
  if(reset_reference)
  {
    start_time_sec = t.tv_sec;
    start_time_usec = t.tv_usec;
  }
  return sec + usec *1e-6;
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
