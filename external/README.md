# external dependencies

The first patch intentionally does not vendor third-party source trees.

Required for phase 1:

```bash
git submodule add https://github.com/rohan-sawhney/fcpw external/fcpw
git -C external/fcpw submodule update --init --recursive
```

or:

```bash
bash scripts/bootstrap_external.sh
```

tiny-cuda-nn is not required for this patch. Add it only after FCPW CPU/Vulkan/CUDA timings are understood.
