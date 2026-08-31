*[English](ISSUE.md) — 實際送出時請用英文版，這份是給內部審閱的對照*

# 標題

CUDA/ROCm presets miss LLAMA_INSTALL_FLAGS: CPU backend has no vector ISA

# 內文

## 摘要

在測試一個放不進 VRAM 的 MoE 模型時，我們發現 `llama.app` 比 llama.cpp 自己
**同一顆 commit** 的 release 慢約 2.1 倍（同一台機器）。往回追，`scripts/generate.py`
會為 `cpu`、`vulkan`、`metal` preset 設定 `LLAMA_INSTALL_FLAGS` —— 所以它們的 CPU backend
會針對特定特性等級編譯，安裝時再由 `featcode` 挑選對應變體 —— 但 `cuda` 和 `rocm` preset
沒有設定，因此那些 binary 的 CPU backend 落到 baseline x86-64，完全沒有向量指令。

模型放得進 VRAM 時看不出來。一旦有權重在 CPU 上計算（部分 offload、`--n-cpu-moe`、
常駐 CPU 的 MoE expert），CPU backend 就進入每個 token 的關鍵路徑，差異才顯現：

| | llama.cpp release | llama.app (`install.ps1`) |
| --- | --- | --- |
| Nemotron-30B-A3B-Q4_0，expert 在 CPU | **82.7 t/s** | **38.9 t/s** |

同樣的效應用小得多的重現方式也看得到，可能更方便檢查。`-p 0 -n 64 -ngl 0` 量的是 CPU 上的
token generation，不需要大模型也不需要調 offload 參數：

```
llama-bench -m qwen2.5-0.5b-instruct-q4_k_m.gguf -p 0 -n 64 -ngl 0 -t 6 -r 3
```

| Qwen2.5-0.5B-Instruct-Q4_K_M，`-ngl 0` | llama.cpp release | 用 CUDA preset 那組旗標的原始碼建置 |
| --- | --- | --- |
| 第 1 輪 | 115.34 ± 9.02 | 23.72 ± 0.03 |
| 第 2 輪 | 120.32 ± 0.50 | 23.65 ± 0.15 |
| 第 3 輪 | 119.08 ± 1.25 | 23.11 ± 1.15 |
| 第 4 輪 | 119.60 ± 0.92 | 23.21 ± 0.71 |
| 第 5 輪 | 120.50 ± 0.69 | 22.22 ± 1.42 |
| **平均** | **118.97** | **23.18** |

**5.13 倍**，五輪交錯，每顆 binary 都先暖機，全距分別為 4.3% 與 6.5%。
上面的 2.1 倍就是同一個代價，被仍在 GPU 上執行的那部分稀釋後的結果。

以下所有數據都來自單一台機器，請以比值而非絕對值閱讀。若有任何一項在你們那邊無法重現，
我們很樂意補做任何有幫助的量測。

## 現象重現

已 pin 的 `b10107` 與 `b10612`，Windows 與 Linux 都回報：

```
system_info: ... | CUDA : ARCHS = 1200 | USE_GRAPHS = 1 | BLACKWELL_NATIVE_FP4 = 1 |
                   CPU : LLAMAFILE = 1 | REPACK = 1 |
```

在一顆具備全部這些指令的 CPU 上，卻沒有 `AVX`、`AVX2`、`FMA`、`F16C`、`AVX_VNNI`。
同一台機器上，llama.cpp 自己同一顆 commit 的 release 回報：

```
CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 |
      BMI2 = 1 | LLAMAFILE = 1 | OPENMP = 1 | REPACK = 1 |
```

## 來源

`CMakeLists.txt` 全域關閉原生 CPU 偵測。對於要發佈 per-featcode artifact 的管線，這是正確的：

```cmake
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
```

`scripts/generate.py` 接著為每個 preset 補回明確的特性等級 —— 但只有部分 backend 有：

| 產生函式 | 有設 `LLAMA_INSTALL_FLAGS` |
| --- | --- |
| `generate_cpu_presets` | 有 |
| `generate_vulkan_presets` | 有 |
| `generate_metal_presets` | 有（`-mcpu=`） |
| `generate_linux_cuda_presets` | **沒有** |
| `generate_windows_cuda_presets` | **沒有** |
| `generate_x86_64_linux_rocm_presets` | **沒有** |

修正所需的機制其實已經接好：`exit.cmake` 會把 `LLAMA_INSTALL_FLAGS` 套到
`CMAKE_C_FLAGS_INIT` / `CMAKE_CXX_FLAGS_INIT`，而 CUDA preset 用的 `toolchains/base.cmake`
本來就 include 了 `exit.cmake`。

## 從原始碼重現

用這個 repo 的 forced 設定編譯 llama.cpp `b10107`，可以重現 llama.app 的效能，
證明差距完全歸因於建置設定：

```
cmake -S llama.cpp -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DBUILD_SHARED_LIBS=OFF -DGGML_STATIC=ON -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF \
  -DGGML_LTO=OFF -DGGML_CCACHE=OFF -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
  -DGGML_SSE42=OFF -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_BMI2=OFF
```

| round | 原始碼複製版 | llama.app |
| --- | --- | --- |
| 1 | 42.99 | 40.83 |
| 2 | 43.78 | 44.20 |
| 3 | 44.88 | 44.84 |

注意：光是 `GGML_NATIVE=OFF` 不足以重現 —— 那種情況下 ggml 的 `INS_ENB` 會讓指令集選項維持開啟。
必須連 ISA 選項也關掉，而那正是 CUDA preset 最終落入的狀態。

## 哪個因素主導

以下每一組都在單一批次內交錯測量，只改一個變因：

| 因素 | 對照 | 影響 |
| --- | --- | --- |
| **向量指令** | MSVC、無 OpenMP、動態連結 —— 只差 ISA 開關 | **1.81 倍** |
| OpenMP | 有 AVX2 時 | 1.27 倍 |
| OpenMP | 無 SIMD 時 | 1.06 倍 |
| 靜態 vs 動態連結 | 複製版內部 | 無 |

向量指令主導。兩個因素會交互影響而非疊乘，不能相乘。

## 量測

硬體：RTX 5060 Ti 16 GB (sm_120)、Core Ultra 5 225 (10c/10t, AVX2 + AVX-VNNI, 無 AVX-512)、
128 GB DDR5-5600、Windows 11。模型 `ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF`
的 `Q4_0`（17.59 GiB，大於 VRAM）。`llama-bench -p 0 -n 128 -ngl 99 -ncmoe 10 -t 10 -r 10`，
交錯執行各 binary。

兩個獨立批次。這台機器的絕對吞吐量在不同執行之間會漂移 14～18%（見〈證據範圍〉），
所以兩組絕對值不同 —— 但比值可以複現：

批次 A，llama.cpp release 對 llama.app：

| round | llama.cpp release | llama.app |
| --- | --- | --- |
| 1 | 92.87 | 44.29 |
| 2 | 92.90 | 45.18 |
| 3 | 92.40 | 39.60 |
| 4 | 92.35 | 42.71 |

**比值 2.16 倍**

批次 B，加入本機 `GGML_NATIVE=ON` build 作為參考天花板：

| round | llama.cpp release | 本機 build | llama.app |
| --- | --- | --- | --- |
| 1 | 78.33 | 83.43 | 39.09 |
| 2 | 84.51 | 80.60 | 39.54 |
| 3 | 83.34 | 84.03 | 38.57 |
| 4 | 84.67 | 84.39 | 38.55 |

**比值 2.13 倍** —— 而llama.cpp release 與針對特定機器調校的本機 build 相當，
說明除了 CPU backend 之外，release 並沒有留下其他未榨取的效能。

差距隨 CPU backend 的工作量縮放 —— 這是缺少向量指令的特徵，而非設定差異：

| 情境 | CPU 參與 | 比值 |
| --- | --- | --- |
| Qwen2.5-0.5B 完全放進 VRAM | 無 | ~1.0 倍 |
| Nemotron `-ncmoe 0` | 低，PCIe 受限 | ~1.1 倍 |
| Nemotron `-ncmoe 10` | expert 在 CPU | **2.1 倍** |

執行期參數完全相同（`llama-bench -o json` 的 32 個欄位逐一相符），offload 切分也相同：

```
load_tensors: offloaded 53/53 layers to GPU
load_tensors:   CPU_Mapped model buffer size =  3354.74 MiB
load_tensors:        CUDA0 model buffer size = 14917.64 MiB
sched_reserve: graph nodes  = 2187
```

## 已排除

- **CUDA arch 不符 / PTX JIT** —— `cuobjdump` 顯示兩邊都是原生 sm_120 cubin、零 PTX。
- **CUDA graphs** —— 兩邊都啟用；`GGML_CUDA_DISABLE_GRAPHS=1` 讓兩者等比例腰斬。
- **BMI2** —— 切換 `GGML_BMI2`、以及抽換 llama.cpp 的 CPU 變體 DLL，都沒有影響。
- **靜態連結** —— `BUILD_SHARED_LIBS` 開關沒有差異。
- **llama.cpp 的變體 dispatch** —— AUTO 在這顆 CPU 上落在最高的 ISA 層級；
  手動強制指定各變體顯示三個層級（AVX2 ~78-83、AVX ~69、無 ~59），
  而 `haswell` 與 `alderlake` 彼此在雜訊內。

## 為什麼 CI 可能沒有顯現這件事

提供脈絡，不是批評 —— 受影響的組態在 CI 裡沒有覆蓋：

- `bench.yml` 和 `bench-compare.yml` 都只跑 `runs-on: [self-hosted, gfx1151]`。
  效能測試套件裡沒有 CUDA backend，而 gfx1151 是統一記憶體 APU，
  不會出現「獨立 GPU + 權重 offload 到系統記憶體」的情境。
- bench 指令使用預設值（`llama bench -fa 0,1 -hf ...`），沒有 `-ngl` 或 `-ncmoe`。
- `test-release.yml` 跑的是 `llama bench -v -hf ggml-org/test-model-stories260K` ——
  260 KB 的模型，任何 VRAM 都放得下。

## 附註：建置機器的 CPU 不是限制

這條管線已經在 `build-any-cpu.yml` 的 `runs-on: ubuntu-latest` 上，
透過 zig 交叉編譯產出完整的 30 個 x86_64 featcode 變體 —— 最高到 `krxzq`（`avx512*` + `amx-*`）。
GitHub 託管的 runner 沒有 AMX，所以建置機器自身的指令集，早就與這些 artifact 的目標無關。

## 可能的方向

1. **給 x86_64 的 CUDA/ROCm preset 一個特性等級**，沿用既有的 featcode 詞彙。
   若選單一等級則不增加任何 artifact；任何跑得動 CUDA 12/13 的 GPU，其主機幾乎必然有 AVX2 ——
   不過拉高底線確實會捨棄 2013 年前的 Intel / 2015 年前的 AMD 主機。
2. **為 CUDA 加上 CPU featcode 維度**，如同 `vulkan`。效果最好，但會在既有的
   CUDA 大版本 × GPU 架構之上再乘一層。
3. **`GGML_BACKEND_DL` + `GGML_CPU_ALL_VARIANTS`**，也就是 llama.cpp 主線 release 的做法
   （單一 artifact、執行期 CPU 分派）。它需要 `BUILD_SHARED_LIBS`，
   與這條管線發佈單一靜態 binary 的設計衝突。
4. 若目前行為是刻意的，**請在 `PRESETS.md` 說明** —— 它的 CUDA 表格只有 `Architecture` 一欄，
   沒有任何跡象顯示 CPU backend 是 baseline。

方向確定後我很樂意送 PR。

## Linux 同樣受影響

同一份 `CMakeLists.txt` 與同一組 generator 也涵蓋 Linux，且 Linux 的 binary 回報完全相同的
`CPU : LLAMAFILE = 1 | REPACK = 1`。在同一台機器的 WSL2 上實測，本機 `GGML_NATIVE=ON` 的
gcc build 領先 **1.69 倍**。同樣受影響，但程度比 Windows 輕，與「gcc 在 baseline 仍會自動向量化、
MSVC 幾乎不做」一致。

特別提出來，是為了避免修正只落在 Windows preset 上。

## 證據範圍

效能數據來自單一台機器，而 Linux 那一半是在 WSL2 而非裸機上做的，所以 Linux 的比值
特別值得在原生環境重測。Windows 那組比較才是紮實的。

這台機器上，絕對吞吐量在不同執行之間會漂移 14～18%。該漂移的來源是混合架構
P-core/E-core CPU 的執行緒配置，不是熱節流也不是記憶體：漂移期間 GPU 時脈、溫度、功耗與
可用記憶體全部持平，而綁定到 E-core 可將全距從 12.2% 壓到 2.4%。
本文所有結論都取自單一交錯批次內的比值，從不跨批次比較。

## 次要觀察

`CMakeLists.txt` 以 `# LTO is broken on windows for now` 註解關閉了 Windows 的 LTO。
既然寫的是 "for now"，或許值得重新確認該問題是否仍然重現。

---

建置腳本、全部量測數據、以及完整記錄（含讓這題天真量測三次得出錯誤答案的那些陷阱）：
https://github.com/samhong5668/llama-bench-lab
