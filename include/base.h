#pragma once

#include <cstdint> // for uint64_t and other types

#ifndef restr
#define restr __restrict__
#endif

#if defined(__cplusplus)
#define UNUSED(x)
#elif defined(__GNUC__)
#define UNUSED(x)       x __attribute__((unused))
#else
#define UNUSED(x)       x
#endif
