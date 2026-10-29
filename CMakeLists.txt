cmake_minimum_required(VERSION 3.12)

project(additive_dfts LANGUAGES CXX)

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang)$")
  message(FATAL_ERROR "Only gcc and clang are supported")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Link-time optimization
option(LTO "Link-time optimization" ON)
if(LTO)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

# Large integers from Boost (header-only part of Boost::multiprecision) enable Cantor bases in
# fields larger than 2**64. If CMake does not find Boost, add the option -D BOOST_ROOT=<boost path>
option(USE_BOOST "Use Boost for large integers" OFF)
if(USE_BOOST)
  find_package(Boost 1.65)
endif()
if(Boost_FOUND)
  message(STATUS "Using Boost for large integers")
else()
  message(STATUS "Boost large integers not available")
endif()
configure_file(defs.h.in defs.h) # defines Boost_FOUND if relevant

# Compiler settings shared by all targets.
# The code needs AVX2, PCLMULQDQ and BMI1 at least, and uses VPCLMULQDQ if available
# (-march=native selects all the features of the build machine).
add_library(common INTERFACE)
target_compile_features(common INTERFACE cxx_std_20) # for assume_aligned
target_compile_options(common INTERFACE -Wall -Wextra -march=native)
target_include_directories(common INTERFACE "${PROJECT_SOURCE_DIR}/include" "${PROJECT_BINARY_DIR}")
if(Boost_FOUND)
  target_link_libraries(common INTERFACE Boost::boost)
endif()

# Cantor bases
add_library(cantor STATIC
  libcantor/cantor_basis_bare.cpp
  libcantor/cantor_basis_full.cpp
  libcantor/helpers.cpp
)
target_link_libraries(cantor PUBLIC common)

# Mateer-Gao DFT in GF(2**64) and product of binary polynomials
add_library(mg STATIC
  libfft/mg.cpp
  libfft/mg_fused.cpp
  utils.cpp
)
target_link_libraries(mg PUBLIC common)

add_executable(cantor_basis_test cantor_test.cpp)
target_link_libraries(cantor_basis_test cantor)

add_executable(fft_test fft_test.cpp libfft/additive_fft.cpp)
target_link_libraries(fft_test cantor mg)

# product_bench only measures the run times of the products
add_executable(product_bench product_test.cpp)
target_compile_definitions(product_bench PRIVATE PRODUCT_BENCH)
target_link_libraries(product_bench mg)

# product_test also checks the products against gf2x (and times it); gf2x is searched for in
# <source>/lib, in $CUSTOM_LIBS/lib, and in the default library directories
find_library(GF2X_LIBRARY gf2x HINTS "${PROJECT_SOURCE_DIR}/lib" "$ENV{CUSTOM_LIBS}/lib")
if(GF2X_LIBRARY)
  add_executable(product_test product_test.cpp)
  target_link_libraries(product_test mg "${GF2X_LIBRARY}")
else()
  message(WARNING "gf2x library not found: product_test will not be built, only product_bench "
                  "(see README.md; set the environment variable CUSTOM_LIBS if needed)")
endif()
