*[繁體中文](README.zh-TW.md)*

# llama-bench-lab

`llama.app`'s CUDA binary runs **2.1x slower on Windows** and **1.7x slower on Linux** than the
alternatives, on models whose weights do not all fit in VRAM. Its CPU backend ships with no
vector instructions at all.

## Which binary is which

Both distributions ship ready-built executables, so "the prebuilt" is ambiguous:

| name used here | comes from | hosted on |
|---|---|---|
| **`llama.app`** | [`ggml-org/llama-install.sh`](https://github.com/ggml-org/llama-install.sh) — the one-line `install.ps1` / `install.sh` | Hugging Face |
| **llama.cpp release** | [`ggml-org/llama.cpp/releases`](https://github.com/ggml-org/llama.cpp/releases) | GitHub Releases |

Two separate build pipelines, which is why their CPU flags differ. The slow one is what the
`llama-install.sh` installer gives you.

## What to use

| platform | use | why |
|---|---|---|
| **Windows** | [llama.cpp release, one zip](docs/windows/RESULTS.md) — 137 MB | 82.7 t/s, matches self-compiled, no toolchain |
| **Linux** | [self-compile](docs/linux/RESULTS.md) — `-march=x86-64-v3` ships as one portable binary | no CUDA release binary exists for Linux; ~82-83.5 against `llama.app`'s 49.4 |

| document | |
|---|---|
| [`docs/windows/RESULTS.md`](docs/windows/RESULTS.md) | Windows findings, numbers, isolated factors |
| [`docs/linux/RESULTS.md`](docs/linux/RESULTS.md) | Linux findings and the install-cost trade |
| [`docs/ISSUE.md`](docs/ISSUE.md) | upstream issue draft for `ggml-org/llama-install.sh` |

Each has a `.zh-TW.md` counterpart.

## How to measure this without getting it wrong

Benchmarking this naively gives confidently wrong answers. Three pitfalls each produced a false
conclusion here, and each was only caught by re-measuring.

1. **Warm every binary separately before taking a number.** CPU-side weights are `mmap`ed, so
   the *first run of each binary* pays to read them. Warming one binary does not warm the next.
2. **Alternate binaries; never run them in blocks.** Whichever runs later is penalised.
3. **Never compare across batches.** Absolute throughput drifts 14–18% between runs on this
   machine — the same configuration has measured 33.7 and 45.8. Only within-batch ratios mean
   anything.

`-r N` catches none of the first two: the first iteration warms the rest.

**If you only need to compare CPU backends, use `-ngl 0` on a small model.** All compute lands
on the CPU, so the GPU and the offload split drop out and most of the drift below with them —
4–6% spread on Windows against 12–24% for the offload workload, with no pinning needed. It
answers a narrower question than the deployment numbers, but it answers it cleanly, and it is
what finally separated build flags that the offload workload could not.

Under WSL2 the same measurement is noisier, and differently so: interference is **one-sided**,
showing up as occasional rounds where the in-run stddev jumps to ±44 while the mean collapses.
Reject rounds by stddev rather than averaging them in, and do **not** pin to the E-cores there —
Windows schedules its own background work onto them, so `--cpu-strict 1` makes it worse.

### Why the drift happens

Not thermal, not memory. Ten instrumented runs (`scripts/windows/drift_probe.ps1`) span
66.98 → 81.01 t/s while the GPU holds a flat 50 °C, 2857 MHz and ~34.7 W with 105 GB of RAM
free. It is **CPU thread placement on a hybrid P-core/E-core CPU**:

| configuration | four rounds | spread |
|---|---|---|
| default `-t 10`, unpinned | 75.66 / 80.06 / 70.89 / 73.51 | **12.2%** |
| 4 P-cores, `-t 4 -C 0x00F --cpu-strict 1` | 60.75 / 57.91 / 59.41 / 63.50 | 9.3% |
| 6 E-cores, `-t 6 -C 0x3F0 --cpu-strict 1` | 66.46 / 68.10 / 67.37 / 67.52 | **2.4%** |
| all 10 cores, `-t 10 -C 0x3FF --cpu-strict 1` | 55.92 / 58.16 / 66.13 / 51.84 | 24.4% |

`ggml-cpu.c` barriers after **every graph node** (2187 of them in this graph), so the slowest
thread gates every one. A pool spread across two core speeds pays at each barrier, and which
threads land where changes per process launch. Pinning to the E-cores collapses the spread to
2.4% for about 10% less throughput. Strict-pinning *every* core is worse than not pinning at
all — it stops the scheduler moving threads away from contention.

**For a stable absolute number, pin to the E-cores. For maximum throughput, leave it unpinned
and compare only within one alternating batch.**

## Test setup

RTX 5060 Ti 16 GB (sm_120, driver 610.62) / Core Ultra 5 225 (10c/10t, AVX2 + AVX-VNNI, no
AVX-512) / 128 GB DDR5-5600. Windows 11, plus Ubuntu 24.04 under WSL2 with **gcc 13.3** — check
`wsl -l -v` and `gcc --version`, because a newer default distro brings a different compiler and
the build-flag comparisons are only valid within one toolchain.

Model `ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF`, `Q4_0`, 17.59 GiB — larger than
VRAM, so `-ncmoe 10` puts 10 layers' MoE experts on the CPU, which is what makes the CPU
backend matter. That exact repository matters: other mirrors publish the same filename from a
different quantisation run.

Standard invocation, `CUDA_VISIBLE_DEVICES=0`:

```
llama-bench -m <model> -p 0 -n 128 -ngl 99 -ncmoe 10 -t 10 -r 10 -o md
```

Verified identical on both sides before comparing: all 32 fields of `llama-bench -o json`, and
the offload split (`CPU_Mapped 3354.74 MiB` / `CUDA0 14917.64 MiB`, `graph nodes = 2187`).

**Limits:** one machine. The Linux numbers are WSL2 with the GPU passed through, and were taken
E-core pinned while the Windows ones were not — so compare ratios within a platform, never
absolutes across platforms.

## Reproducing

```bash
uv run scripts/download_models.py --dest ./models --check-only   # sizes only, no download
uv run scripts/download_models.py --dest ./models
```

### Windows

```powershell
LLAMA_VERSION=b10107 powershell -File install.ps1        # llama.app

gh release download b10107 -R ggml-org/llama.cpp `       # llama.cpp release, one asset
  -p "llama-b10107-bin-win-cuda-13.3-x64.zip"

scripts/windows/build.ps1 -Config native                 # self-compiled
scripts/windows/build.ps1 -Config replica                # llama.app's own settings
```

### Linux

```bash
scripts/linux/build.sh v3                # -march=x86-64-v3 — portable, 1.4% behind native
scripts/linux/build.sh native             # GGML_NATIVE=ON — fastest, per-machine only
scripts/linux/build.sh replica            # llama.app's own settings
scripts/linux/bench3.sh <model.gguf> 5    # deployment workload

# CPU backends only, far less noisy — this is what separated native from v3
BENCH_ARGS="-p 0 -n 64 -ngl 0 -t 6 -r 5" scripts/linux/bench3.sh <small-model.gguf> 8
```

## Scripts

`scripts/` holds shared tooling; `scripts/windows/` and `scripts/linux/` the platform parts.

| script | purpose |
|---|---|
| `scripts/download_models.py` | fetches the two GGUF models (`uv run`; optional) |
| `scripts/windows/build.ps1` | builds `llama-bench` — `native`, `replica`, `replica-shared`, `noomp` |
| `scripts/windows/bench.ps1` | alternating N-way comparison; reproduces the Windows batches |
| `scripts/windows/run_variants.ps1` | forces each llama.cpp CPU variant DLL in turn |
| `scripts/windows/drift_probe.ps1` | repeats one measurement while sampling GPU and memory state |
| `scripts/linux/build.sh` | builds `llama-bench` — `native`, `v3`, `v3vnni`, `replica` |
| `scripts/linux/bench_linux.sh` | `llama.app` vs self-compiled, alternating |
| `scripts/linux/bench3.sh` | alternating three-way; any three binaries, labels and args overridable |

In both build scripts, **`replica` means "the settings `llama-install.sh` forces", so it
reproduces `llama.app`**; `native` is `GGML_NATIVE=ON`; `v3` is a fixed `-march=x86-64-v3`; `v3vnni` adds `-mavxvnni` to isolate that one flag.

Build directories, the llama.cpp source tree, downloaded archives and raw benchmark output are
not tracked — the scripts regenerate them, and the numbers live in the documents.
