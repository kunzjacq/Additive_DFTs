#pragma once

#include <cstdint> // for uint64_t and other types

#ifdef __GNUC__
#ifndef restr
#define restr __restrict__
#endif
#else
#ifdef _MSC_VER
#ifndef restr
#define restr __restrict
#endif
#else
#error "unsupported compiler"
#endif
#endif

#if defined(__cplusplus)
#define UNUSED(x)
#elif defined(__GNUC__)
#define UNUSED(x)       x __attribute__((unused))
#else
#define UNUSED(x)       x
#endif
