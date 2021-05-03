#pragma once

#include <chrono>
#include <cmath>

class timer
{
  std::chrono::time_point<std::chrono::high_resolution_clock> begin;
   std::chrono::time_point<std::chrono::high_resolution_clock> end;
public:
  void set_start()
  {
    begin = std::chrono::high_resolution_clock::now();
  }
  double measure()
  {
    end = std::chrono::high_resolution_clock::now();
    return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()) * pow(10, -9);
  }
};
