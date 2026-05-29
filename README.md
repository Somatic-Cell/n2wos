# neural-2lmc-wos-native

Native C++ diagnostics for the Neural 2LMC / Walk-on-Spheres project.

The first patch deliberately avoids neural-cache inference. Its purpose is to isolate the geometry backend:

1. build a native C++ executable,
2. fetch FCPW as an external dependency or submodule,
3. generate a watertight bumpy-sphere triangle mesh in memory,
4. build an FCPW scene,
5. profile closest-point queries through CPU, Vulkan, and CUDA backends,
6. profile a host-controlled WoS-style wavefront loop without Python/PyTorch in the inner loop,
7. emit JSON metrics and logs that are useful for debugging WSL/CUDA/Slang/RHI failures.

This is not yet the final GPU-resident WoS implementation. In particular, FCPW's public GPU query API still receives query batches from the host and returns interactions to the host. That is acceptable for the first diagnostic step; it tells us whether the native FCPW CUDA backend can be created and how expensive the current public batched query path is.

## Repository setup

From an empty repository:

```bash
git init
git am /path/to/0001-bootstrap-native-fcpw-diagnostics.patch
```

Fetch FCPW:

```bash
bash scripts/bootstrap_external.sh
```

The script will use a git submodule when the directory is inside a git repository. Commit the resulting `.gitmodules` and submodule gitlink after it succeeds.

## WSL/CUDA preflight

Run this before building with GPU support:

```bash
bash scripts/preflight_wsl_cuda.sh | tee results/preflight_wsl_cuda.log
```

The output is meant to be pasted into an issue or a chat message when CUDA device creation fails.

## Build

GPU-enabled build:

```bash
python3 scripts/build.py --gpu --type Release --build-dir build-release-gpu
```

CPU-only fallback build:

```bash
python3 scripts/build.py --no-gpu --type Release --build-dir build-release-cpu
```

Equivalent CMake presets are also included:

```bash
cmake --preset release-gpu
cmake --build --preset release-gpu -j
```

## Minimal probes

CPU closest-point query benchmark:

```bash
./build-release-gpu/native/n2wos_probe_fcpw \
  --backend cpu \
  --mode cpq \
  --n-queries 262144 \
  --repeats 5 \
  --json results/cpq_cpu.json
```

Vulkan closest-point query benchmark:

```bash
./build-release-gpu/native/n2wos_probe_fcpw \
  --backend vulkan \
  --mode cpq \
  --n-queries 262144 \
  --repeats 5 \
  --print-logs \
  --json results/cpq_vulkan.json
```

CUDA closest-point query benchmark:

```bash
./build-release-gpu/native/n2wos_probe_fcpw \
  --backend cuda \
  --mode cpq \
  --n-queries 262144 \
  --repeats 5 \
  --print-logs \
  --json results/cpq_cuda.json
```

WoS-style profile without neural cache:

```bash
./build-release-gpu/native/n2wos_probe_fcpw \
  --backend cuda \
  --mode wos \
  --n-samples 65536 \
  --max-steps 512 \
  --eps 1e-4 \
  --safety 0.99 \
  --print-logs \
  --json results/wos_cuda.json
```

Run CPU/Vulkan/CUDA as a batch:

```bash
python3 scripts/run_probe_matrix.py \
  --build-dir build-release-gpu \
  --backends cpu vulkan cuda \
  --mode cpq \
  --n-queries 262144 \
  --repeats 5 \
  --print-logs
```

## JSON fields

The output contains, among other fields:

- `backend`
- `n_queries`
- `usec/query`
- `usec_per_query`
- `n_samples`
- `usec/sample`
- `usec_per_sample`
- `avg_steps`
- `p95_steps`
- `p99_steps`
- `max_steps`
- `query_count`
- `active_count_by_step`
- `active_remaining`
- `environment`
- `compiled`

## What to report after applying this patch

Send the repository URL, branch, commit hash, and these outputs:

```bash
git status --short
git log --oneline -5
bash scripts/preflight_wsl_cuda.sh | tee results/preflight_wsl_cuda.log
python3 scripts/build.py --gpu --type Release --build-dir build-release-gpu 2>&1 | tee results/build_gpu.log
```

Then run:

```bash
./build-release-gpu/native/n2wos_probe_fcpw --backend cpu --mode cpq --n-queries 262144 --repeats 5 --json results/cpq_cpu.json
./build-release-gpu/native/n2wos_probe_fcpw --backend vulkan --mode cpq --n-queries 262144 --repeats 5 --print-logs --json results/cpq_vulkan.json
./build-release-gpu/native/n2wos_probe_fcpw --backend cuda --mode cpq --n-queries 262144 --repeats 5 --print-logs --json results/cpq_cuda.json
```

If CUDA fails, include stdout/stderr exactly. The next patch should be based on the first failing command and its logs.
