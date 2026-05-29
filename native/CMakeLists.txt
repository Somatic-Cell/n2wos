set(N2WOS_FCPW_INCLUDE_DIRS
  "${FCPW_DIR}/include"
  "${FCPW_DIR}/deps/eigen"
)

if(N2WOS_USE_ENOKI)
  list(APPEND N2WOS_FCPW_INCLUDE_DIRS
    "${FCPW_DIR}/deps/enoki/include"
  )
endif()

function(n2wos_set_fcpw_target_options target)
  target_include_directories(${target} PRIVATE
    ${N2WOS_FCPW_INCLUDE_DIRS}
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
  )

  target_link_libraries(${target} PRIVATE fcpw)

  target_compile_definitions(${target}
    PRIVATE
      N2WOS_ENABLE_FCPW_GPU=$<BOOL:${N2WOS_ENABLE_FCPW_GPU}>
      N2WOS_ENABLE_TCNN=$<BOOL:${N2WOS_ENABLE_TCNN}>
      N2WOS_FCPW_SOURCE_DIR="${FCPW_DIR}"
      N2WOS_CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
  )

  if(MSVC)
    target_compile_options(${target} PRIVATE /O2 /W4)
  else()
    target_compile_options(${target} PRIVATE -O3 -Wall -Wextra -Wpedantic)
  endif()

  if(N2WOS_ENABLE_FCPW_GPU AND TARGET slang-rhi)
    target_link_libraries(${target} PRIVATE slang-rhi)
    target_include_directories(${target} PRIVATE
      "${FCPW_DIR}/deps/slang-rhi/include"
      "${CMAKE_BINARY_DIR}/external/fcpw-build/deps/slang-rhi/include"
    )
  endif()
endfunction()

add_executable(n2wos_probe_fcpw
  src/main_probe_fcpw.cpp
)
n2wos_set_fcpw_target_options(n2wos_probe_fcpw)

add_executable(n2wos_eval_2lmc
  src/main_eval_2lmc.cpp
)
n2wos_set_fcpw_target_options(n2wos_eval_2lmc)

if(N2WOS_ENABLE_TCNN)
  add_executable(n2wos_probe_tcnn
    src/main_probe_tcnn.cpp
  )

  target_link_libraries(n2wos_probe_tcnn PRIVATE tiny-cuda-nn CUDA::cudart)

  target_compile_definitions(n2wos_probe_tcnn PRIVATE
    N2WOS_ENABLE_TCNN=1
    N2WOS_CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
  )

  if(MSVC)
    target_compile_options(n2wos_probe_tcnn PRIVATE /O2 /W4)
  else()
    target_compile_options(n2wos_probe_tcnn PRIVATE -O3 -Wall -Wextra -Wpedantic)
  endif()
endif()
