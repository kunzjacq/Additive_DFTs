cmake_minimum_required(VERSION 3.9)

project(additive_dfts)

include_directories("${PROJECT_SOURCE_DIR}/include")

message("CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")
#message("MINGW: ${MINGW}")
set(CMAKE_VERBOSE_MAKEFILE on)

option(LTO "Link-time optimization" ON)
option(USE_BOOST "Link-time optimization" ON)

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
  # per-target use:
  #  set_property(TARGET <target name> PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
endif()

set(WARN_FLAGS "-Wall -Wextra")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99 -g -march=native -mtune=native ${WARN_FLAGS}")
set(CMAKE_CXX_FLAGS "-std=gnu++17 -g -march=native -mtune=native ${CMAKE_CXX_FLAGS} ${WARN_FLAGS}")

#set(OPT_FLAGS "-ftree-vectorize -ftree-vectorizer-verbose=4 -fno-rtti -Wl,--as-needed")
set(OPT_FLAGS "")
set(DEBUG_FLAGS "-D_DEBUG")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${OPT_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OPT_FLAGS}")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${DEBUG_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${DEBUG_FLAGS}")

if(MINGW)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D__USE_MINGW_ANSI_STDIO=1")
endif()

if(USE_BOOST)
  find_package(Boost 1.65)
  message("Boost librairies: ${Boost_LIBRARIES}")
endif()

configure_file("defs.h.in" "defs.h")

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_executable(cantor_basis_test
  "cantor_test.cpp"
  "libcantor/cantor_basis_bare.cpp"
  "libcantor/cantor_basis_full.cpp"
  "libcantor/helpers.cpp")

add_executable(fft_test
 "fft_test.cpp"
 "libfft/utils.cpp"
 "libfft/additive_fft.cpp"
 "libfft/mg.cpp"
 "libcantor/cantor_basis_bare.cpp"
 "libcantor/cantor_basis_full.cpp"
 "libcantor/helpers.cpp")

add_executable(product_test
 "product_test.cpp"
 "libfft/mg.cpp"
 "libfft/utils.cpp"
 )


if(Boost_FOUND)
  message("Using Boost for large integers")
  target_link_libraries(cantor_basis_test ${Boost_LIBRARIES})
  target_link_libraries(fft_test ${Boost_LIBRARIES})
else()
  message("Boost large integers not available")
endif()

target_link_libraries(product_test gf2x)

