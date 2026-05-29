cmake_minimum_required(VERSION 3.21)
project(neural_2lmc_wos_native LANGUAGES CXX)

option(N2WOS_ENABLE_FCPW_GPU "Enable FCPW GPU support through Slang/RHI" ON)
option(N2WOS_USE_ENOKI "Enable FCPW CPU vectorization through Enoki" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(FCPW_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/fcpw" CACHE PATH "Path to FCPW source tree")

if(NOT EXISTS "${FCPW_DIR}/CMakeLists.txt")
  message(FATAL_ERROR
    "FCPW was not found at ${FCPW_DIR}.\n"
    "Run one of:\n"
    "  bash scripts/bootstrap_external.sh\n"
    "or, manually:\n"
    "  git submodule add https://github.com/rohan-sawhney/fcpw external/fcpw\n"
    "  git -C external/fcpw submodule update --init --recursive\n")
endif()

set(FCPW_BUILD_DEMO OFF CACHE BOOL "" FORCE)
set(FCPW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FCPW_BUILD_BINDINGS OFF CACHE BOOL "" FORCE)
set(FCPW_USE_ENOKI ${N2WOS_USE_ENOKI} CACHE BOOL "" FORCE)
set(FCPW_ENABLE_GPU_SUPPORT ${N2WOS_ENABLE_FCPW_GPU} CACHE BOOL "" FORCE)

add_subdirectory("${FCPW_DIR}" external/fcpw-build)
add_subdirectory(native)
