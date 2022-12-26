cmake_minimum_required(VERSION 3.9)

project(additive_dfts)

include(TestBigEndian)

message("CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
set(CMAKE_VERBOSE_MAKEFILE on)

option(LTO "Link-time optimization" ON)

#option to use boost for large integers. To use boost: set the option,
# and if Cmake does not find Boost path (esp. under Windows), add to cmake the option
# -D BOOST_ROOT=<boost path>
#Windows example:
#-D BOOST_ROOT=C:\\boost_1_78_0

option(USE_BOOST "Use Boost for large integers" OFF)

if(LTO)
  message("Turning on link-time optimization")
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
  # per-target setting:
  #  set_property(TARGET <target name> PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
endif()

set(LOCAL_LIB_DIR "${PROJECT_SOURCE_DIR}/lib")

if(IS_DIRECTORY ${LOCAL_LIB_DIR})
  list(APPEND LIB_DIRS "${LOCAL_LIB_DIR}")
endif()

if("$ENV{CUSTOM_LIBS}" STREQUAL "")
else()
  list(APPEND LIB_DIRS "$ENV{CUSTOM_LIBS}/lib")
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
  add_compile_options(-Wall -Wextra)
  ## architecture flags strictly needed to compile the code: -mavx2 -mpclmul -mbmi
  add_compile_options(-march=native -mtune=native)
  if(MINGW)
      add_compile_definitions(__USE_MINGW_ANSI_STDIO=1)
  endif()
  ## Additional optimization or debug options
  #add_compile_options("$<$<CONFIG:RELEASE>:-ftree-vectorize;-ftree-vectorizer-verbose=4;-fno-rtti;-Wl,--as-needed>")
  #add_compile_definitions("$<$<CONFIG:DEBUG>:_DEBUG>")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  add_compile_options(/arch:AVX2 /favor:AMD64)
endif()

if(USE_BOOST)
  message("BOOST_ROOT: ${BOOST_ROOT}")
  find_package(Boost 1.65)
  message("Boost librairies: ${Boost_LIBRARIES}")
  message("Boost include dirs: ${Boost_INCLUDE_DIR}")
endif()

configure_file("defs.h.in" "defs.h")

add_executable(product_test
 "product_test.cpp"
 "utils.cpp"
 "libfft/mg.cpp"
)

target_include_directories(product_test PRIVATE ${CMAKE_CURRENT_BINARY_DIR} "${PROJECT_SOURCE_DIR}/include") 
# CMAKE_CURRENT_BINARY_DIR added for defs.h

if ((CMAKE_CXX_COMPILER_ID STREQUAL "GNU") OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
  target_link_libraries(product_test gf2x)
  target_link_directories(product_test PRIVATE "${LIB_DIRS}")

  add_executable(cantor_basis_test
    "cantor_test.cpp"
    "libcantor/cantor_basis_bare.cpp"
    "libcantor/cantor_basis_full.cpp"
    "libcantor/helpers.cpp"
  )
  target_include_directories(cantor_basis_test PRIVATE ${CMAKE_CURRENT_BINARY_DIR} "${PROJECT_SOURCE_DIR}/include")

  add_executable(fft_test
   "fft_test.cpp"
   "utils.cpp"
   "libfft/additive_fft.cpp"
   "libfft/mg.cpp"
   "libcantor/cantor_basis_bare.cpp"
   "libcantor/cantor_basis_full.cpp"
   "libcantor/helpers.cpp"
  )
  target_include_directories(fft_test PRIVATE ${CMAKE_CURRENT_BINARY_DIR} "${PROJECT_SOURCE_DIR}/include")
endif()

if(Boost_FOUND)
  message("Using Boost for large integers")
  target_link_libraries(cantor_basis_test ${Boost_LIBRARIES})
  target_include_directories(cantor_basis_test PRIVATE "${Boost_INCLUDE_DIR}")
  target_link_libraries(fft_test ${Boost_LIBRARIES})
  target_include_directories(fft_test PRIVATE "${Boost_INCLUDE_DIR}")
else()
  message("Boost large integers not available")
endif()

