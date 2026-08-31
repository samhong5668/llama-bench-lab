*[繁體中文](RESULTS.zh-TW.md)* · *[Linux](../linux/RESULTS.md)*

# Windows: use the llama.cpp release, not `llama.app`

## Result

| option | t/s | needs installing |
|---|---|---|
| **llama.cpp release** (one zip) | **82.7** | nothing — download and unzip |
| self-compiled (`GGML_NATIVE=ON`) | 83.1 | VS Build Tools + CMake + CUDA Toolkit |
| `llama.app` (`llama-install.sh`) | **38.9** | one command |

All three need `cublas64_13.dll` (and, transitively, `cublasLt64_13.dll`) at run time. A project
that installs `torch==2.13.0+cu130` already ships both — see below — so none of them needs the
CUDA Toolkit just to run.

**The release matches a self-compiled build (82.7 vs 83.1, inside this machine's noise) and
needs no build toolchain at all.** `llama.app` is 2.1x behind.

Could the release be *faster* than self-compiling? Possibly — it reports `AVX_VNNI` and `BMI2`
which the MSVC self-compiled build did not. Measured, the two are indistinguishable, so this
data cannot separate them.

## What to download

**One asset, 137 MB.** Each backend zip is a superset of the CPU zip, so the CPU zip is not
a separate download. Verified by listing `llama-b10107-bin-win-cuda-13.3-x64.zip`:

| the CUDA zip contains | |
|---|---|
| 22 executables | including `llama-server.exe` |
| 14 `ggml-cpu-*.dll` variants | the runtime-dispatched CPU backend |
| `libomp140.x86_64.dll` | what those variants link against |
| `ggml-cuda.dll` | 133 MB of the 137 |

The `win-vulkan-x64` zip (31 MB) is self-contained the same way. Pin a `bXXXXX` build tag:
the semantic tags (`v0.3.0`) carry no binaries.

There are exactly two x64 CUDA variants, verified on `b10107`, `b10644` and `b10665`:

| variant | size | pair it with |
|---|---|---|
| `win-cuda-13.3-x64` | 137 MB | a CUDA 13 host, or PyTorch `+cu130` |
| `win-cuda-12.4-x64` | 235 MB | a CUDA 12 host, or PyTorch `+cu124` |

**There is no `13.0` asset** — `13.3` is the only CUDA 13 build, and it is the right one for a
`+cu130` PyTorch pin. Only the major version has to match, because the import is by DLL name
(`cublas64_13.dll`) and CUDA 13.x keeps minor-version compatibility. That pairing — a `13.3`
`ggml-cuda.dll` against `+cu130` cuBLAS — is the one measured below.

### The 372 MB `cudart` zip is not needed if PyTorch is already installed

`ggml-cuda.dll` has no `cudart64_*` dependency — the CUDA runtime is statically linked. What it
does need, per `dumpbin /dependents`, is a two-DLL chain:

```
ggml-cuda.dll  ->  cublas64_13.dll  ->  cublasLt64_13.dll   (50 MB + 478 MB)
```

**Both** are required. `cublasLt` is a transitive dependency, so copying `cublas64_13.dll` alone
fails to load. PyTorch's `+cu130` wheels ship both:

```
.venv/Lib/site-packages/torch/lib/cublas64_13.dll
.venv/Lib/site-packages/torch/lib/cublasLt64_13.dll
```

Verified by running `llama-bench` with a `PATH` rebuilt to contain only `system32` and
`torch/lib`, and `CUDA_PATH` / `CUDA_HOME` cleared — no CUDA Toolkit reachable:

```
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 16310 MiB)
load_backend: loaded CUDA backend from ...\ggml-cuda.dll
qwen2 1B Q4_K - Medium | CUDA | 99 |  tg32 |      503.86 ± 15.57
qwen2 1B Q4_K - Medium | CUDA | 99 | pp512 |  32266.02 ± 5888.94
```

Both token generation and prompt processing work, so the cuBLAS calls resolve — not just the
initial device probe. `llama.app`'s `llama.exe` imports `cublas64_13.dll` the same way (unlike
its Linux build, which statically links cuBLAS) and also runs against `torch/lib`.

**A project that already installs `torch==2.13.0+cu130` can point `PATH` at `torch/lib` and skip
the 372 MB download**, cutting the one-click install from 509 MB to 137 MB. Match the major
version: `+cu130` provides `cublas64_13`, which pairs with the `win-cuda-13.3` zip.

### The other run-time dependencies

None of these costs a download in this project, but all are real:

| dependency | needed by | where it comes from |
|---|---|---|
| `libomp140.x86_64.dll` | every `ggml-cpu-*.dll` | shipped inside the release zip — self-contained |
| `MSVCP140.dll`, `VCRUNTIME140.dll` | `ggml-cuda.dll` and every `ggml-cpu-*.dll` | the VC++ redistributable |
| `VCRUNTIME140_1.dll` | `ggml-cuda.dll` only | the VC++ redistributable |

The VC++ redistributable is **already a prerequisite of this project**: `torch_cpu.dll` imports
the same DLLs, so any machine that can run the backend's PyTorch can run the llama.cpp
release. It is not a new install step.

`llama.app`'s `llama.exe` is the exception — it links the CRT statically and needs no
redistributable. That is a real advantage of `llama.app` on a machine with no Python stack, and
irrelevant here.

#### Removing the dependency entirely, for 724 KB

Rather than rely on the redistributable being present, copy the three DLLs next to the
executables — app-local deployment, which Microsoft supports and which needs no administrator.
The whole VC++ surface of this distribution is exactly three files:

| file | size |
|---|---|
| `msvcp140.dll` | 545 KB |
| `vcruntime140.dll` | 122 KB |
| `vcruntime140_1.dll` | 49 KB |

Taken from any `VC/Redist/MSVC/<ver>/x64/Microsoft.VC143.CRT/` folder. Nothing else in the
closure comes from the redistributable — the 11 `api-ms-win-crt-*.dll` imports are the Universal
CRT, part of Windows itself since Windows 10.

Verified by copying the release into a clean directory with those three DLLs, then running it
with `PATH` set to that directory plus `system32` and `torch/lib`, and enumerating the running
process's loaded modules:

```
VCRUNTIME140.dll     <app dir>\VCRUNTIME140.dll
MSVCP140.dll         <app dir>\MSVCP140.dll
VCRUNTIME140_1.dll   <app dir>\VCRUNTIME140_1.dll
cublas64_13.dll      ...\torch\lib\cublas64_13.dll
cublasLt64_13.dll    ...\torch\lib\cublasLt64_13.dll
```

All five resolved outside the CUDA Toolkit and outside `system32`, and the run exited 0.

That the app-local copies are *listed* is not by itself proof that `system32` is out of the
picture, so it was confirmed the other way round — by breaking the app-local copy and leaving
`system32` intact:

| app-local `msvcp140.dll` | `system32\msvcp140.dll` | result |
|---|---|---|
| valid (545 KB) | intact | starts, loads every backend |
| **2048 zero bytes** | **intact** | **exit `0xC000012F`** (`STATUS_INVALID_IMAGE_NOT_MZ`) |
| restored | intact | starts again |

If the loader could fall back to `system32`, the middle row would have run. It did not, so the
app-local copy is what binds and `system32`'s copy is never consulted. **A host without the
redistributable installed therefore behaves identically** — the dependency is genuinely gone,
not merely satisfied twice.

(A clean-image check in Windows Sandbox was the other way to establish this. Sandbox is enabled
on this machine but fails to start with `0x800706d9` / `RPC_S_NO_MORE_ENDPOINTS`, with every
service it depends on running and no reboot pending; diagnosing further needs administrator
rights. The precedence test above is the stronger evidence anyway, since it demonstrates the
binding directly rather than inferring it from an absence.)

And it costs nothing. The same directory, still with no CUDA Toolkit reachable, on the full
deployment workload (Nemotron `-ncmoe 10`, `-t 10 -r 3`):

| round | t/s |
|---|---|
| 1 | 75.18 ± 4.92 |
| 2 | 80.77 ± 9.44 |
| 3 | 79.80 ± 10.86 |

Mean **78.58**, spread 7.1% — the same range as the release measured normally (92.87 in one
batch, 78.33 in another; see the drift note in the README). So the 372 MB saving is free.

## The problem

`llama.app`'s CUDA binary ships a CPU backend with **no vector instructions at all**. Its
`system_info`, on both the pinned `b10107` and `b10612`:

```
CPU : LLAMAFILE = 1 | REPACK = 1          <- no AVX / AVX2 / FMA / F16C / AVX_VNNI
```

The llama.cpp release of the same commit reports:

```
CPU : SSE3 | SSSE3 | AVX | AVX_VNNI | AVX2 | F16C | FMA | BMI2 | LLAMAFILE | OPENMP | REPACK
```

This is invisible while a model fits in VRAM. Once weights are computed on the CPU, the CPU
backend is on the per-token critical path:

| scenario | CPU involvement | `llama.app` behind by |
|---|---|---|
| 0.5B entirely in VRAM | none | ~1.0x |
| Nemotron `-ncmoe 0` | low, PCIe-bound | ~1.1x |
| Nemotron `-ncmoe 10` | experts on CPU | **2.1x** |

Root cause is in `ggml-org/llama-install.sh`: `CMakeLists.txt` forces `GGML_NATIVE=OFF` (correct
for a shipped binary), and `scripts/generate.py` compensates with `LLAMA_INSTALL_FLAGS` for the
`cpu`, `vulkan` and `metal` presets — but not for `cuda` or `rocm`. Filed as
[`../ISSUE.md`](../ISSUE.md).

## Numbers

Two independent batches. See the README for why only within-batch ratios count.

Batch A — release vs `llama.app`:

| round | llama.cpp release | `llama.app` |
|---|---|---|
| 1 | 92.87 | 44.29 |
| 2 | 92.90 | 45.18 |
| 3 | 92.40 | 39.60 |
| 4 | 92.35 | 42.71 |

**2.16x**

Batch B — adding a self-compiled build:

| round | llama.cpp release | self-compiled | `llama.app` |
|---|---|---|---|
| 1 | 78.33 | 83.43 | 39.09 |
| 2 | 84.51 | 80.60 | 39.54 |
| 3 | 83.34 | 84.03 | 38.57 |
| 4 | 84.67 | 84.39 | 38.55 |

**2.13x**, and release ≈ self-compiled.

## The CPU backend on its own: 5.13x

The cleanest measurement in this investigation. `-p 0 -n 64` benchmarks token generation only,
and token generation under `-ngl 0` runs entirely on the CPU, so the GPU, the offload split and
the thread-placement drift all drop out. (`-ngl 0` does *not* keep the GPU out of prompt
processing — see the note in the README — which is why `-p 0` is part of the command.) 0.5B model,
`-p 0 -n 64 -ngl 0 -t 6 -r 3`, every binary warmed, alternating five rounds
(`scripts/windows/bench.ps1`):

| round | llama.cpp release | self-compiled native | `replica` (no vector ISA) |
|---|---|---|---|
| 1 | 115.34 ± 9.02 | 121.88 ± 1.30 | 23.72 ± 0.03 |
| 2 | 120.32 ± 0.50 | 117.41 ± 5.53 | 23.65 ± 0.15 |
| 3 | 119.08 ± 1.25 | 122.10 ± 0.57 | 23.11 ± 1.15 |
| 4 | 119.60 ± 0.92 | 121.67 ± 0.84 | 23.21 ± 0.71 |
| 5 | 120.50 ± 0.69 | 122.90 ± 0.47 | 22.22 ± 1.42 |
| **mean** | **118.97** | **121.19** | **23.18** |
| spread | 4.3% | 4.5% | 6.5% |

**5.13x** release against replica, **5.23x** native against replica. No batching caveat needed at
this spread.

It also settles the release-versus-self-compiled question more sharply than the offload workload
could: **121.19 against 118.97 is 1.9% apart, inside a 4.3–4.5% spread.** Indistinguishable, in
the same direction and the same magnitude as the offload measurement (83.1 against 82.7).

This is the penalty on the CPU *portion* of the work, which is why it is larger than the 2.1x
end-to-end figure above: under `-ncmoe 10` only part of the model computes on the CPU, so the
5.13x is diluted by everything the GPU still does. It is also larger than the 1.81x isolated
below, because that pair held OpenMP off on both sides and ran the offload workload; this one
varies the whole CUDA-preset flag set on a pure-CPU workload.

It also double-checks the `replica` config after the fact — a build that reproduces
`llama.app`'s throughput on an offload workload should, and does, collapse on a pure-CPU one.

## Which factor, isolated

Each pair alternated within one batch, one variable changed:

| factor | comparison | effect |
|---|---|---|
| **vector ISA** | MSVC, no OpenMP, shared — ISA on vs off | **1.81x** |
| OpenMP | with AVX2 present | 1.27x |
| OpenMP | without SIMD | 1.06x |
| static vs shared linking | within the replica | **none** |

The vector ISA dominates. The two factors interact rather than compose, so they cannot be
multiplied.

## Reproducing `llama.app` from source

`scripts/windows/build.ps1 -Config replica` applies `llama-install.sh`'s forced settings and
lands on the same throughput, so the gap is fully attributable to that build configuration:

| round | replica | `llama.app` |
|---|---|---|
| 1 | 42.99 | 40.83 |
| 2 | 43.78 | 44.20 |
| 3 | 44.88 | 44.84 |

`GGML_NATIVE=OFF` alone does not reproduce it — ggml's `INS_ENB` leaves the ISA options ON.
They have to be disabled explicitly, which is the state the CUDA presets end up in.

## Ruled out

| hypothesis | how | result |
|---|---|---|
| `llama.app` installed the Vulkan or CPU build | `ggml_cuda_init` output | no, it is CUDA |
| GPU arch mismatch → PTX JIT | `cuobjdump -lelf` / `-lptx` | no — native sm_120 cubins, zero PTX in both |
| CUDA graphs on one side only | `GGML_CUDA_DISABLE_GRAPHS=1` | no, both have them |
| different runtime parameters | field diff of `llama-bench -o json` | no, all 32 identical |
| different offload split | `load_tensors` buffer sizes | no, byte-identical |
| BMI2 regression | `GGML_BMI2` toggle + swapping variant DLLs | no effect |
| static linking slower | `BUILD_SHARED_LIBS` toggle | no effect |
| llama.cpp's variant dispatch picking wrong | forcing each variant by hand | no — AUTO lands in the top ISA tier |
