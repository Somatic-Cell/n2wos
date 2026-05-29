add_executable(n2wos_probe_fcpw
  src/main_probe_fcpw.cpp
)

# FCPW is an INTERFACE target and should propagate these include directories.
# We still add them explicitly here because this executable directly includes
# FCPW public headers, and the first diagnostic build should fail as little as
# possible on dependency propagation details.
set(N2WOS_FCPW_INCLUDE_DIRS
  "${FCPW_DIR}/include"
  "${FCPW_DIR}/deps/eigen"
)

if(N2WOS_ENABLE_FCPW_GPU)
  list(APPEND N2WOS_FCPW_INCLUDE_DIRS
    "${FCPW_DIR}/deps/slang-rhi/include"
  )
endif()

if(N2WOS_USE_ENOKI)
  list(APPEND N2WOS_FCPW_INCLUDE_DIRS
    "${FCPW_DIR}/deps/enoki/include"
  )
endif()

target_include_directories(n2wos_probe_fcpw PRIVATE
  ${N2WOS_FCPW_INCLUDE_DIRS}
)

target_link_libraries(n2wos_probe_fcpw PRIVATE fcpw)

target_compile_definitions(n2wos_probe_fcpw
  PRIVATE
    N2WOS_ENABLE_FCPW_GPU=$<BOOL:${N2WOS_ENABLE_FCPW_GPU}>
    N2WOS_FCPW_SOURCE_DIR="${FCPW_DIR}"
    N2WOS_CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
)

if(MSVC)
  target_compile_options(n2wos_probe_fcpw PRIVATE /O2 /W4)
else()
  target_compile_options(n2wos_probe_fcpw PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()

# n2wos local fix:
# FCPW's vendored slang-rhi generates slang-rhi-config.h in the build tree.
# When FCPW is consumed through add_subdirectory, make this dependency explicit.
if(N2WOS_ENABLE_FCPW_GPU AND TARGET slang-rhi)
  target_link_libraries(n2wos_probe_fcpw PRIVATE slang-rhi)
  target_include_directories(n2wos_probe_fcpw PRIVATE
    "${FCPW_DIR}/deps/slang-rhi/include"
    "${CMAKE_BINARY_DIR}/external/fcpw-build/deps/slang-rhi/include"
  )
endif()
