add_executable(n2wos_probe_fcpw
  src/main_probe_fcpw.cpp
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
