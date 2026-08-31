*[繁體中文對照](ISSUE.zh-TW.md) — this file is the version to post; the header lines below are section markers, not part of the issue*

# Title

CUDA/ROCm presets miss LLAMA_INSTALL_FLAGS: CPU backend has no vector ISA

# Body

## Summary

While benchmarking a MoE model that does not fit in VRAM, we found `llama.app` running about
2.1x slower than llama.cpp's own release of the **same commit** on the same machine. Tracing
it back, `scripts/generate.py` sets `LLAMA_INSTALL_FLAGS` for the `cpu`, `vulkan` and
`metal` presets — so their CPU backend is compiled for a specific feature level and the
installer picks the matching variant via `featcode` — but the `cuda` and `rocm` presets do
not, so those binaries end up with the baseline x86-64 CPU backend and no vector ISA at all.

That is invisible while a model fits in VRAM. Once any weights are computed on the CPU
(partial offload, `--n-cpu-moe`, CPU-resident MoE experts), the CPU backend sits on the
per-token critical path and the difference shows up:

| | llama.cpp release | llama.app (`install.ps1`) |
| --- | --- | --- |
| Nemotron-30B-A3B-Q4_0, experts on CPU | **82.7 t/s** | **38.9 t/s** |

The same effect is visible with a much smaller reproducer, which may be more convenient to
check. `-p 0 -n 64 -ngl 0` benchmarks token generation on the CPU, so no large model and no
offload tuning is needed:

```
llama-bench -m qwen2.5-0.5b-instruct-q4_k_m.gguf -p 0 -n 64 -ngl 0 -t 6 -r 3
```

| Qwen2.5-0.5B-Instruct-Q4_K_M, `-ngl 0` | llama.cpp release | a source build with the CUDA preset's flags |
| --- | --- | --- |
| round 1 | 115.34 ± 9.02 | 23.72 ± 0.03 |
| round 2 | 120.32 ± 0.50 | 23.65 ± 0.15 |
| round 3 | 119.08 ± 1.25 | 23.11 ± 1.15 |
| round 4 | 119.60 ± 0.92 | 23.21 ± 0.71 |
| round 5 | 120.50 ± 0.69 | 22.22 ± 1.42 |
| **mean** | **118.97** | **23.18** |

**5.13x**, five alternating rounds, each binary warmed first, spreads of 4.3% and 6.5%. The 2.1x
above is the same penalty diluted by the part of the model that still runs on the GPU.

Everything below is on one machine, so please read the ratios rather than the absolutes. If
any of it does not reproduce for you, we are happy to run whatever additional measurement
would help.

## Reproduction of the symptom

Both the pinned `b10107` and `b10612`, on Windows and Linux, report:

```
system_info: ... | CUDA : ARCHS = 1200 | USE_GRAPHS = 1 | BLACKWELL_NATIVE_FP4 = 1 |
                   CPU : LLAMAFILE = 1 | REPACK = 1 |
```

No `AVX`, `AVX2`, `FMA`, `F16C` or `AVX_VNNI`, on a CPU that has all of them. llama.cpp's own
release of the same commit on the same machine reports:

```
CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 |
      BMI2 = 1 | LLAMAFILE = 1 | OPENMP = 1 | REPACK = 1 |
```

## Where it comes from

`CMakeLists.txt` disables native CPU detection globally, which is correct for a release
pipeline that ships per-featcode artifacts:

```cmake
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
```

`scripts/generate.py` re-adds an explicit feature level per preset — but only for some
backends:

| generator | sets `LLAMA_INSTALL_FLAGS` |
| --- | --- |
| `generate_cpu_presets` | yes |
| `generate_vulkan_presets` | yes |
| `generate_metal_presets` | yes (`-mcpu=`) |
| `generate_linux_cuda_presets` | **no** |
| `generate_windows_cuda_presets` | **no** |
| `generate_x86_64_linux_rocm_presets` | **no** |

The mechanism to fix it is already wired up for these presets: `exit.cmake` applies
`LLAMA_INSTALL_FLAGS` to `CMAKE_C_FLAGS_INIT` / `CMAKE_CXX_FLAGS_INIT`, and the CUDA presets
already use `toolchains/base.cmake`, which includes `exit.cmake`.

## Reproduction from source

Building llama.cpp `b10107` with this repo's forced settings reproduces `llama.app`'s
throughput, so the gap is fully attributable to the build configuration:

```
cmake -S llama.cpp -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DBUILD_SHARED_LIBS=OFF -DGGML_STATIC=ON -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF \
  -DGGML_LTO=OFF -DGGML_CCACHE=OFF -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
  -DGGML_SSE42=OFF -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_BMI2=OFF
```

| round | source replica | llama.app |
| --- | --- | --- |
| 1 | 42.99 | 40.83 |
| 2 | 43.78 | 44.20 |
| 3 | 44.88 | 44.84 |

Note that `GGML_NATIVE=OFF` alone does not reproduce it — ggml's `INS_ENB` leaves the
instruction-set options ON in that case. The ISA options have to be off as well, which is
the state the CUDA presets end up in.

## Which factor dominates

Each pair below was measured alternating within one batch, changing one variable only:

| factor | comparison | effect |
| --- | --- | --- |
| **vector ISA** | MSVC, no OpenMP, shared — ISA on vs off | **1.81x** |
| OpenMP | with AVX2 present | 1.27x |
| OpenMP | without SIMD | 1.06x |
| static vs shared linking | within the replica | none |

The vector ISA dominates. The two factors interact rather than compose, so they cannot be
multiplied together.

## Measurements

Hardware: RTX 5060 Ti 16 GB (sm_120), Core Ultra 5 225 (10c/10t, AVX2 + AVX-VNNI, no
AVX-512), 128 GB DDR5-5600, Windows 11. Model
`ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF`, `Q4_0` (17.59 GiB, larger than VRAM).
`llama-bench -p 0 -n 128 -ngl 99 -ncmoe 10 -t 10 -r 10`, binaries run alternately.

Two independent batches. Absolute throughput on this machine drifts 14–18% between runs
(see *Scope of the evidence*), so the two sets of absolutes differ — but the ratio
reproduces:

Batch A, llama.cpp release versus llama.app:

| round | llama.cpp release | llama.app |
| --- | --- | --- |
| 1 | 92.87 | 44.29 |
| 2 | 92.90 | 45.18 |
| 3 | 92.40 | 39.60 |
| 4 | 92.35 | 42.71 |

**ratio 2.16x**

Batch B, adding a local `GGML_NATIVE=ON` build as a reference ceiling:

| round | llama.cpp release | local build | llama.app |
| --- | --- | --- | --- |
| 1 | 78.33 | 83.43 | 39.09 |
| 2 | 84.51 | 80.60 | 39.54 |
| 3 | 83.34 | 84.03 | 38.57 |
| 4 | 84.67 | 84.39 | 38.55 |

**ratio 2.13x** — and the llama.cpp release matches a machine-specific local build, so the
release is not leaving performance on the table for reasons other than the CPU backend.

The gap tracks how much work the CPU backend does — the signature of a missing vector ISA
rather than a configuration difference:

| scenario | CPU involvement | ratio |
| --- | --- | --- |
| Qwen2.5-0.5B fully in VRAM | none | ~1.0x |
| Nemotron `-ncmoe 0` | low, PCIe-bound | ~1.1x |
| Nemotron `-ncmoe 10` | experts on CPU | **2.1x** |

Runtime parameters are identical (all 32 fields of `llama-bench -o json` match), and so is
the offload split:

```
load_tensors: offloaded 53/53 layers to GPU
load_tensors:   CPU_Mapped model buffer size =  3354.74 MiB
load_tensors:        CUDA0 model buffer size = 14917.64 MiB
sched_reserve: graph nodes  = 2187
```

## Ruled out

- **CUDA arch mismatch / PTX JIT** — `cuobjdump` shows native sm_120 cubins and zero PTX in both.
- **CUDA graphs** — enabled in both; `GGML_CUDA_DISABLE_GRAPHS=1` halves both equally.
- **BMI2** — toggling `GGML_BMI2`, and swapping llama.cpp's CPU variant DLLs, show no effect.
- **Static linking** — `BUILD_SHARED_LIBS` on/off makes no difference.
- **llama.cpp's variant dispatch** — AUTO lands in the top ISA tier on this CPU; forcing
  each variant by hand shows three tiers (AVX2 ~78-83, AVX ~69, none ~59) with `haswell` and
  `alderlake` within noise of each other.

## Why it may not have surfaced in CI

Context, not criticism — the affected configuration is not covered in CI:

- `bench.yml` and `bench-compare.yml` both run `runs-on: [self-hosted, gfx1151]` only. No CUDA
  backend in the performance suite, and gfx1151 is a unified-memory APU, so "discrete GPU +
  weights offloaded to system RAM" does not arise there.
- The bench invocation uses defaults (`llama bench -fa 0,1 -hf ...`), with no `-ngl` or
  `-ncmoe`.
- `test-release.yml` runs `llama bench -v -hf ggml-org/test-model-stories260K` — a 260 KB
  model, which fits in any VRAM.

## Note: builder CPU is not a constraint

The pipeline already emits the full 30-entry x86_64 featcode matrix — up to `krxzq`
(`avx512*` + `amx-*`) — from `runs-on: ubuntu-latest` in `build-any-cpu.yml` via zig cross
compilation. GitHub-hosted runners have no AMX, so the build machine's own ISA is already
irrelevant to what these artifacts target.

## Possible directions

1. **Give the x86_64 CUDA/ROCm presets a feature level**, reusing the existing featcode
   vocabulary. No new artifacts if a single level is chosen; any machine running a CUDA 12/13
   capable GPU almost certainly has AVX2, though raising the floor does drop pre-2013 Intel /
   pre-2015 AMD hosts.
2. **Add a CPU featcode dimension to CUDA**, as `vulkan` has. Best result, but multiplies the
   artifact matrix on top of the existing CUDA major x GPU arch.
3. **`GGML_BACKEND_DL` + `GGML_CPU_ALL_VARIANTS`**, which is what mainline llama.cpp's release
   does (one artifact, runtime CPU dispatch). Requires `BUILD_SHARED_LIBS`, conflicting with
   the single static binary this pipeline ships.
4. **Document it** in `PRESETS.md` if the current behaviour is intended — its CUDA table has
   only an `Architecture` column, with no indication that the CPU backend is baseline.

Happy to send a PR once you say which direction you prefer.

## Linux is affected too

The same `CMakeLists.txt` and the same generators cover Linux, and the Linux binary reports the
identical `CPU : LLAMAFILE = 1 | REPACK = 1`. Measured under WSL2 on the same machine, a local
`GGML_NATIVE=ON` gcc build is **1.69x** ahead. Also affected, though less so than Windows,
consistent with gcc still auto-vectorising at baseline where MSVC largely does not.

Mentioned so a fix does not land on the Windows presets alone.

## Scope of the evidence

Performance numbers are from one machine, and the Linux half under WSL2 rather than bare
metal, so the Linux ratio in particular would be worth re-measuring natively. The Windows
comparison is the solid one.

Absolute throughput drifts 14–18% between runs on this machine. That drift is CPU thread
placement on a hybrid P-core/E-core CPU, not thermal or memory: GPU clock, temperature, power
and free RAM stay flat across it, and pinning to the E-cores collapses the spread from 12.2%
to 2.4%. All conclusions here are drawn from ratios measured within a single alternating
batch, never across batches.

## Secondary observation

`CMakeLists.txt` disables LTO for Windows with a `# LTO is broken on windows for now`
comment. Given the "for now", it may be worth re-checking whether that still reproduces.

---

Build scripts, every measurement, and the full write-up (including the measurement
pitfalls that made naive benchmarking of this give wrong answers three times):
https://github.com/samhong5668/llama-bench-lab
