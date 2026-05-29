cmake_minimum_required(VERSION 3.21)
project(neural_2lmc_wos_native LANGUAGES CXX)

option(N2WOS_ENABLE_FCPW_GPU "Enable FCPW GPU support through Slang/RHI" ON)
option(N2WOS_USE_ENOKI "Enable FCPW CPU vectorization through Enoki" OFF)
option(N2WOS_ENABLE_TCNN "Enable tiny-cuda-nn HashGrid/MLP support" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(FCPW_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/fcpw" CACHE PATH "Path to FCPW source tree")
set(TCNN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/tiny-cuda-nn" CACHE PATH "Path to tiny-cuda-nn source tree")

function(n2wos_require_path path hint)
  if(NOT EXISTS "${path}")
    message(FATAL_ERROR
      "Required dependency path was not found:\n"
      "  ${path}\n\n"
      "${hint}\n\n"
      "Recommended repair from the repository root:\n"
      "  bash scripts/bootstrap_external.sh\n"
      "  git submodule update --init --recursive\n")
  endif()
endfunction()

if(NOT EXISTS "${FCPW_DIR}/CMakeLists.txt")
  message(FATAL_ERROR
    "FCPW was not found at ${FCPW_DIR}.\n"
    "Run:\n"
    "  bash scripts/bootstrap_external.sh\n")
endif()

n2wos_require_path(
  "${FCPW_DIR}/deps/eigen/Eigen/Core"
  "FCPW's Eigen nested submodule is missing or incomplete.")

if(N2WOS_ENABLE_FCPW_GPU)
  n2wos_require_path(
    "${FCPW_DIR}/deps/slang-rhi/CMakeLists.txt"
    "FCPW GPU support is enabled, but FCPW's slang-rhi nested submodule is missing or incomplete.")
endif()

if(N2WOS_USE_ENOKI)
  n2wos_require_path(
    "${FCPW_DIR}/deps/enoki/include/enoki/array.h"
    "FCPW Enoki vectorization is enabled, but FCPW's Enoki dependency is missing or incomplete.")
endif()

set(FCPW_BUILD_DEMO OFF CACHE BOOL "" FORCE)
set(FCPW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FCPW_BUILD_BINDINGS OFF CACHE BOOL "" FORCE)
set(FCPW_USE_ENOKI ${N2WOS_USE_ENOKI} CACHE BOOL "" FORCE)
set(FCPW_ENABLE_GPU_SUPPORT ${N2WOS_ENABLE_FCPW_GPU} CACHE BOOL "" FORCE)

add_subdirectory("${FCPW_DIR}" external/fcpw-build)

if(N2WOS_ENABLE_TCNN)
  if(NOT EXISTS "${TCNN_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
      "tiny-cuda-nn was not found at ${TCNN_DIR}.\n"
      "Run:\n"
      "  bash scripts/bootstrap_tcnn.sh\n")
  endif()

  n2wos_require_path(
    "${TCNN_DIR}/dependencies/cutlass/CMakeLists.txt"
    "tiny-cuda-nn is present, but its nested submodules are missing or incomplete.")

  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  set(TCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
  set(TCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(TCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(TCNN_ALLOW_CUBLAS_CUSOLVER OFF CACHE BOOL "" FORCE)
  set(TCNN_BUILD_WITH_RTC ON CACHE BOOL "" FORCE)

  add_subdirectory("${TCNN_DIR}" external/tiny-cuda-nn-build)
endif()

add_subdirectory(native)
