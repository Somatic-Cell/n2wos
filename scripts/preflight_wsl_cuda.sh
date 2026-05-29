#!/usr/bin/env bash
set -u

section() {
  printf '\n## %s\n' "$1"
}

run() {
  printf '+ %s\n' "$*"
  "$@" 2>&1 || true
}

section "date"
run date -Is

section "WSL / kernel"
run uname -a
run cat /proc/version
if command -v wsl.exe >/dev/null 2>&1; then
  run wsl.exe --version
fi

section "OS"
run cat /etc/os-release

section "NVIDIA device visibility"
if command -v nvidia-smi >/dev/null 2>&1; then
  run nvidia-smi
elif [ -x /usr/lib/wsl/lib/nvidia-smi ]; then
  run /usr/lib/wsl/lib/nvidia-smi
else
  echo "nvidia-smi not found"
fi

section "CUDA compiler"
run which nvcc
run nvcc --version

section "CUDA driver libraries"
run sh -lc "ldconfig -p 2>/dev/null | grep -E 'libcuda\\.so|libnvidia'"
run ls -l /usr/lib/wsl/lib/libcuda.so /usr/lib/wsl/lib/libcuda.so.1

section "compiler / cmake"
run cmake --version
run c++ --version
run gcc --version
run clang++ --version

section "Vulkan"
run which vulkaninfo
run vulkaninfo --summary

section "selected environment variables"
for k in PATH LD_LIBRARY_PATH CUDA_HOME CUDA_PATH CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES VK_ICD_FILENAMES VK_LAYER_PATH; do
  printf '%s=%s\n' "$k" "${!k-}"
done
