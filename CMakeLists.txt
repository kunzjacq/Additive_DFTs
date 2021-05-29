cmake_minimum_required(VERSION 3.9)

project(additive_dfts)

include(TestBigEndian)

include_directories("${PROJECT_SOURCE_DIR}/include")

message("CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
#message("MINGW: ${MINGW}")
set(CMAKE_VERBOSE_MAKEFILE on)

option(LTO "Link-time optimization" ON)
option(USE_BOOST "Use Boost for large integers" OFF)

if(MINGW)
  message("Windows/MinGW detected")
  link_directories("${PROJECT_SOURCE_DIR}/lib_mingw")
else()
  link_directories("${PROJECT_SOURCE_DIR}/lib")
endif()

if("$ENV{CUSTOM_LIBS}" STREQUAL "")
else()
  link_directories("$ENV{CUSTOM_LIBS}/lib")
endif()

if(LTO)
  message("Turning on link-time optimization")
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
  # per-target setting:
  #  set_property(TARGET <target name> PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
endif()

set(CMAKE_CXX_STANDARD 20) #for assume_aligned
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

TEST_BIG_ENDIAN(BE)
if(BE EQUAL 1)
#if(NOT (CMAKE_CXX_BYTE_ORDER EQUAL "LITTLE_ENDIAN")) #alternative test, cmake 3.20 or later
#does not require module TestBigEndian
  message(FATAL_ERROR "This project only works on a little endian architecture")
endif()

if ((CMAKE_CXX_COMPILER_ID STREQUAL "GNU") OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
  set(WARN_FLAGS "-Wall -Wextra")
  # architecture flags strictly needed to compile the code: -mavx2 -mpclmul -mbmi
  set(CMAKE_C_FLAGS   "-march=native -mtune=native ${CMAKE_C_FLAGS}   ${WARN_FLAGS}")
  if(MINGW)
      set(CMAKE_C_FLAGS "-D__USE_MINGW_ANSI_STDIO=1 ${CMAKE_C_FLAGS}")
  endif()
  set(CMAKE_CXX_FLAGS "-march=native -mtune=native ${CMAKE_CXX_FLAGS} ${WARN_FLAGS}")
  # Additional optimization or debug options
  #set(OPT_FLAGS "-ftree-vectorize -ftree-vectorizer-verbose=4 -fno-rtti -Wl,--as-needed")
  #set(DEBUG_FLAGS "-D_DEBUG")
  #set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${OPT_FLAGS}")
  #set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OPT_FLAGS}")
  #set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${DEBUG_FLAGS}")
  #set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${DEBUG_FLAGS}")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  set(CMAKE_CXX_FLAGS "/arch:AVX2 /favor:AMD64 ${CMAKE_CXX_FLAGS}")
endif()

if(USE_BOOST)
  find_package(Boost 1.65)
  message("Boost librairies: ${Boost_LIBRARIES}")
endif()

configure_file("defs.h.in" "defs.h")

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_executable(product_test
 "product_test.cpp"
 "utils.cpp"
 "libfft/mg.cpp"
 "sha256.c"
 )

if ((CMAKE_CXX_COMPILER_ID STREQUAL "GNU") OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
add_executable(cantor_basis_test
  "cantor_test.cpp"
  "libcantor/cantor_basis_bare.cpp"
  "libcantor/cantor_basis_full.cpp"
  "libcantor/helpers.cpp")

add_executable(fft_test
 "fft_test.cpp"
 "utils.cpp"
 "libfft/additive_fft.cpp"
 "libfft/mg.cpp"
 "libcantor/cantor_basis_bare.cpp"
 "libcantor/cantor_basis_full.cpp"
 "libcantor/helpers.cpp")

#with MSVC, product_test does not support gf2x reference computations
target_link_libraries(product_test gf2x)
endif()



if(Boost_FOUND)
  message("Using Boost for large integers")
  target_link_libraries(cantor_basis_test ${Boost_LIBRARIES})
  target_link_libraries(fft_test ${Boost_LIBRARIES})
else()
  message("Boost large integers not available")
endif()

