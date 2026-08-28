*[繁體中文](RESULTS.zh-TW.md)* · *[Windows](../windows/RESULTS.md)*

# Linux: `llama.app` or self-compile — there is no release binary

## The constraint

**llama.cpp publishes no Linux CUDA binary.** Ubuntu assets exist — `ubuntu-x64` (CPU),
`ubuntu-vulkan-x64`, `ubuntu-rocm`, `ubuntu-sycl-fp16/fp32`, `ubuntu-openvino`, `arm64`,
`s390x` — but not one of them is CUDA. Every CUDA asset in the release is `win-`, verified on
`b10107`, `b10644` and the current `b10665`.

So the Windows answer (download one release asset) does not transfer. On Linux the CUDA options are:

| option | t/s | needs installing on each machine |
|---|---|---|
| self-compiled, `GGML_NATIVE=ON` | **83.5** | `build-essential` + `cmake` + CUDA Toolkit (~3 GB), then a 6–15 min compile |
| self-compiled, `-march=x86-64-v3` | ~82 (1.4% behind) | nothing — build once elsewhere, ship the binary |
| `llama.app` (`llama-install.sh`) | **49.4** | one command; **NVIDIA driver only** |

**1.69x apart** between `llama.app` and a self-compiled build — that is the solid finding, and
it reproduced across both Linux batches.

**`GGML_NATIVE=ON` versus `-march=x86-64-v3`: 1.4% apart.** The offload workload could not
separate them — one batch said 10% (83.5 vs 75.7), another said 0% — because three binaries in
one batch reintroduce the drift that pinning suppresses. Measuring the CPU backend directly with
`-ngl 0` settles it; see below. Since the ISA flags only affect the CPU backend, a 1.4%
difference there bounds the difference on the offload workload at 1.4% too, so the 10% reading
was noise.

**That makes the portable build viable.** `-march=x86-64-v3` costs about 1.4% and runs on any
2013-or-later x86 host, so one build can serve a fleet instead of compiling on every machine —
worth considering for an installer, since it removes the per-machine CUDA Toolkit and build time
while keeping essentially all of the 1.69x.

**AVX-VNNI is worth nothing measurable here: 0.04%.** `v3vnni` (`-march=x86-64-v3 -mavxvnni`,
verified to add exactly `AVX_VNNI` and nothing else) matched plain `v3`. AVX-VNNI accelerates
int8 dot products; for Q4_K token generation on this CPU, that path is not what is being
exercised.

83.5 lands in the same range as the Windows numbers (82.7–83.1), so **a self-compiled Linux
build is not slower than Windows**. The two platforms' figures were obtained under different
pinning, so treat that as "same range", not "equal".

## Which to pick

Three options for an installer, in increasing order of what you have to put on the target:

1. **Build `-march=x86-64-v3` once, ship the binary.** ~82 t/s, nothing on the target machine
   beyond the NVIDIA driver and the CUDA runtime libraries the binary links. Costs 1.4% against
   a native build and needs no toolchain, no 3 GB Toolkit, and no per-machine compile. **This is
   the option to reach for first** — it was not on the table while `native` vs `v3` looked like
   a 10% gap.
2. **Compile on each machine with `GGML_NATIVE=ON`.** 83.5 t/s, the fastest, but it puts
   `build-essential` + `cmake` + ~3 GB of CUDA Toolkit and a 6–15 minute build on every target.
   The Ubuntu side of that is a scriptable `apt install` with no GUI installers, so it is
   realistic here in a way it is not on Windows — it just buys only 1.4% over option 1.
3. **`llama.app`.** Genuinely one command, and no CUDA Toolkit: confirmed with `ldd` that the
   binary's only CUDA dependency is `libcuda.so.1`, the driver, because cuBLAS is statically
   linked — which is why it is 531 MB against Windows' 48 MB. Costs **1.69x**.

**Options 1 and 2 are within 1.4% of each other and both 1.69x ahead of option 3**, so the real
decision is only whether to ship a prebuilt binary or compile on site.

## Numbers

Ubuntu 24.04 under WSL2, `scripts/linux/bench3.sh`, five alternating rounds. Same machine,
same model, same commit `c0bc8591e`. gcc 13.3.

Pinned to the E-cores (`-t 6 -C 0x3F0 --cpu-strict 1`). Unpinned, this batch was unusable —
all three binaries spread 38–49% with a downward trend across rounds. Pinning brought it to
3-4%. See the README for the measurement rules.

| round | `llama.app` | self-compiled native | self-compiled v3 |
|---|---|---|---|
| 1 | 49.05 ± 7.99 | 82.07 ± 0.86 | 75.34 ± 15.09 |
| 2 | 49.70 ± 8.24 | 84.19 ± 1.57 | 74.46 ± 16.39 |
| 3 | 49.41 ± 8.56 | 82.92 ± 1.05 | 76.04 ± 15.91 |
| 4 | 48.70 ± 8.44 | 82.94 ± 1.45 | 75.11 ± 16.45 |
| 5 | 50.17 ± 9.15 | 85.29 ± 1.23 | 77.58 ± 16.99 |

means 49.4 / 83.5 / 75.7 — spreads 3.0% / 3.9% / 4.1% across rounds.

### Separating the build flags, on the CPU backend directly

The table above cannot separate `native` from `v3`, so this measures the CPU backend on its own:
0.5B model, `-p 0 -n 64 -ngl 0 -t 6 -r 5`, unpinned, two binaries per batch, eight rounds.

Two binaries rather than three, and unpinned rather than E-core pinned, both on purpose. WSL2
adds one-sided interference — occasional runs where the in-run stddev jumps to ±44 while the
mean collapses — and E-core pinning makes it *worse*, because Windows schedules its own
background work onto the E-cores. Rounds whose stddev exceeds ±15 are interference and are
excluded below; note the noise only ever slows a run down, never speeds one up.

| | clean rounds | mean | best round |
|---|---|---|---|
| `native` | 126.64 / 125.62 / 125.76 / 124.11 / 126.12 / 126.83 | **125.85** | 126.83 ± 0.81 |
| `v3` | 123.41 / 123.77 / 124.66 / 124.94 / 124.82 / 122.90 | **124.08** | 124.94 ± 0.83 |

`native` is **1.4%** ahead, and ahead in five of six rounds — small but consistent in sign.

| | clean rounds | mean | best round |
|---|---|---|---|
| `v3` | 126.39 / 125.58 / 123.46 / 119.17 / 125.36 / 122.10 / 125.28 / 125.42 | **124.10** | 126.39 ± 1.31 |
| `v3vnni` | 124.57 / 124.27 / 120.79 / 121.85 / 127.24 / 126.16 | **124.15** | 127.24 ± 1.27 |

**0.04%** apart on the means and 0.7% on the best rounds — AVX-VNNI does nothing here.

Both conclusions hold under either estimator, mean-of-clean-rounds or best-observation, which is
why they are reported rather than a single ratio.

## Same root cause as Windows

`llama.app`'s Linux CUDA binary reports the identical `CPU : LLAMAFILE = 1 | REPACK = 1` — no
vector ISA. `llama-install.sh`'s own `CMakeLists.txt` and `scripts/generate.py` generators serve both
platforms, so [`../ISSUE.md`](../ISSUE.md) covers Linux too. Fixing it upstream would remove
the reason to self-compile here at all.

## Known limits

These numbers are **WSL2, not native Linux** — the GPU is passed through. The ratios within a
batch are solid, but the absolutes are worth re-measuring on bare metal. That is the one thing
here that this machine cannot answer: a Docker container would not help, because Docker on
Windows runs on the same WSL2 kernel with the same virtualised GPU.

The Windows and Linux offload absolutes were obtained under different pinning (Windows unpinned,
Linux E-core pinned), because E-core pinning did not stabilise the Windows three-way batch the
way it did on Linux. So compare ratios within a platform, not absolutes across platforms.

On the `-ngl 0` CPU-backend measurement the pinning advice inverts: **E-core pinning makes WSL2
noisier**, since Windows puts its own background work on the E-cores and `--cpu-strict 1` stops
the guest scheduler moving away from it. Unpinned with stddev-based outlier rejection was the
usable configuration. Windows needed neither — the same measurement there ran at a 4–6% spread
with no rejection.
