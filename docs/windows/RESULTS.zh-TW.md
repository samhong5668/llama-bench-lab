*[English](RESULTS.md)* · *[Linux](../linux/RESULTS.zh-TW.md)*

# Windows：用 llama.cpp release，不要 `llama.app`

## 結果

| 方案 | t/s | 需要安裝 |
|---|---|---|
| **llama.cpp release**（兩個 zip） | **82.7** | 無 —— 下載解壓即可 |
| 自編（`GGML_NATIVE=ON`） | 83.1 | VS Build Tools + CMake + CUDA Toolkit |
| `llama.app`（`llama-install.sh`） | **38.9** | 一行指令 |

三者執行時都需要 `cublas64_13.dll`（以及它間接依賴的 `cublasLt64_13.dll`）。有安裝
`torch==2.13.0+cu130` 的專案本來就兩顆都有（見下節），所以三者都不需要為了「能跑」而裝 CUDA Toolkit。

**release 與自編相當（82.7 對 83.1，落在這台機器的雜訊內），而且完全不需要建置工具。**
`llama.app` 落後 2.1 倍。

release 有可能**比自編更快**嗎？理論上有 —— 它回報了 `AVX_VNNI` 和 `BMI2`，而 MSVC 自編版沒有。
但實測兩者無法區分，這組數據分辨不出來。

## 要下載什麼

兩個都要，解壓到**同一個**資料夾 —— 合計 154 MB：

| asset | 大小 | 內容 |
|---|---|---|
| `llama-bXXXXX-bin-win-cpu-x64.zip` | 17 MB | 執行檔 + 14 個 `ggml-cpu-*.dll` 變體 |
| `llama-bXXXXX-bin-win-cuda-13.3-x64.zip` | 137 MB | 只有 `ggml-cuda.dll` |

只下載 CUDA 那包沒有用 —— 裡面沒有任何執行檔。pin 版本要用 `bXXXXX` 這種 build tag：
語意化的 tag（`v0.3.0`）沒有附 binary。

x64 的 CUDA 變體**只有兩個**，在 `b10107`、`b10644`、`b10665` 上都確認過：

| 變體 | 大小 | 搭配 |
|---|---|---|
| `win-cuda-13.3-x64` | 137 MB | CUDA 13 的機器，或 PyTorch `+cu130` |
| `win-cuda-12.4-x64` | 235 MB | CUDA 12 的機器，或 PyTorch `+cu124` |

**沒有 `13.0` 這個 asset** —— `13.3` 是唯一的 CUDA 13 build，而它就是 `+cu130` PyTorch 該配的那個。
只有主版本需要對上，因為 import 是靠 DLL 檔名（`cublas64_13.dll`），而 CUDA 13.x 之間保有
minor version 相容性。下面量測的正是這個組合：`13.3` 的 `ggml-cuda.dll` 配 `+cu130` 的 cuBLAS。

### 已裝 PyTorch 的話，372 MB 的 `cudart` zip 不必下載

`ggml-cuda.dll` 沒有 `cudart64_*` 依賴 —— CUDA runtime 是靜態連結進去的。它真正需要的，
依 `dumpbin /dependents`，是一條兩層的 DLL 鏈：

```
ggml-cuda.dll  ->  cublas64_13.dll  ->  cublasLt64_13.dll   (50 MB + 478 MB)
```

**兩顆都要。** `cublasLt` 是間接依賴，所以只複製 `cublas64_13.dll` 一顆會載入失敗。
PyTorch 的 `+cu130` wheel 兩顆都帶：

```
.venv/Lib/site-packages/torch/lib/cublas64_13.dll
.venv/Lib/site-packages/torch/lib/cublasLt64_13.dll
```

驗證方式：把 `PATH` 重建成只有 `system32` 和 `torch/lib`，並清空 `CUDA_PATH` / `CUDA_HOME`
—— 完全碰不到 CUDA Toolkit：

```
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 16310 MiB)
load_backend: loaded CUDA backend from ...\ggml-cuda.dll
qwen2 1B Q4_K - Medium | CUDA | 99 |  tg32 |      503.86 ± 15.57
qwen2 1B Q4_K - Medium | CUDA | 99 | pp512 |  32266.02 ± 5888.94
```

token generation 和 prompt processing 都正常，所以 cuBLAS 的呼叫確實可用，不只是初始探測。
`llama.app` 的 `llama.exe` 也是同樣動態 import `cublas64_13.dll`（與它的 Linux 版不同 ——
那邊是靜態連結 cuBLAS），一樣能靠 `torch/lib` 跑起來。

**已經安裝 `torch==2.13.0+cu130` 的專案可以把 `PATH` 指向 `torch/lib`，省下那 372 MB**，
一鍵安裝的下載量從 526 MB 降到 154 MB。注意大版本要對上：`+cu130` 提供 `cublas64_13`，
搭配 `win-cuda-13.3` 那包。

### 其他執行期依賴

這幾項在本專案都不會多出下載量，但都是真的：

| 依賴 | 誰需要 | 從哪來 |
|---|---|---|
| `libomp140.x86_64.dll` | 每一顆 `ggml-cpu-*.dll` | 就在 `win-cpu-x64` 那包裡 —— 自足 |
| `MSVCP140.dll`、`VCRUNTIME140.dll` | `ggml-cuda.dll` 與每一顆 `ggml-cpu-*.dll` | VC++ redistributable |
| `VCRUNTIME140_1.dll` | 只有 `ggml-cuda.dll` | VC++ redistributable |

VC++ redistributable **本來就是這個專案的前置條件**：`torch_cpu.dll` import 同樣那些 DLL，
所以任何跑得動 backend 的 PyTorch 的機器就跑得動 llama.cpp release。這不是新增的安裝步驟。

`llama.app` 的 `llama.exe` 是例外 —— 它靜態連結 CRT，不需要 redistributable。那對於沒有
Python 環境的機器是實質優勢，在這裡則無關。

#### 用 724 KB 徹底移除這個依賴

與其賭目標機器裝了 redistributable，不如把那三顆 DLL 直接放在執行檔旁邊 —— 這是微軟支援的
app-local 部署，而且不需要管理員權限。整份發行版的 VC++ 依賴面剛好就是三個檔案：

| 檔案 | 大小 |
|---|---|
| `msvcp140.dll` | 545 KB |
| `vcruntime140.dll` | 122 KB |
| `vcruntime140_1.dll` | 49 KB |

從任何 `VC/Redist/MSVC/<ver>/x64/Microsoft.VC143.CRT/` 資料夾取得即可。閉包裡沒有別的東西
來自 redistributable —— 那 11 個 `api-ms-win-crt-*.dll` 是 Universal CRT，自 Windows 10
起就是作業系統本身的一部分。

驗證方式：把 release 複製到一個乾淨目錄、加上那三顆 DLL，然後把 `PATH` 設成該目錄加
`system32` 與 `torch/lib` 執行，並列舉執行中行程的已載入模組：

```
VCRUNTIME140.dll     <app 目錄>\VCRUNTIME140.dll
MSVCP140.dll         <app 目錄>\MSVCP140.dll
VCRUNTIME140_1.dll   <app 目錄>\VCRUNTIME140_1.dll
cublas64_13.dll      ...\torch\lib\cublas64_13.dll
cublasLt64_13.dll    ...\torch\lib\cublasLt64_13.dll
```

五顆全部在 CUDA Toolkit 之外、也在 `system32` 之外解析完成，執行結束碼為 0。

不過「app-local 那份被**列出來**」本身還不足以證明 `system32` 完全出局，所以又從反方向確認了
一次 —— 弄壞 app-local 那顆，並保持 `system32` 的完好：

| app-local `msvcp140.dll` | `system32\msvcp140.dll` | 結果 |
|---|---|---|
| 正常（545 KB） | 完好 | 正常啟動，載入所有 backend |
| **2048 個位元組的零** | **完好** | **exit `0xC000012F`**（`STATUS_INVALID_IMAGE_NOT_MZ`） |
| 還原 | 完好 | 又能啟動 |

如果載入器能退回 `system32`，中間那一列就會正常執行。它沒有 —— 所以綁定的是 app-local 那份，
`system32` 的副本根本不會被查詢。**因此一台沒有裝 redistributable 的主機行為完全相同** ——
這個依賴是真的消失了，不是被滿足了兩次。

（另一條路是用 Windows Sandbox 在乾淨映像上檢查。本機的 Sandbox 已啟用，但啟動時失敗於
`0x800706d9` / `RPC_S_NO_MORE_ENDPOINTS`，而它依賴的服務全部在跑、也沒有待處理的重開機；
再往下診斷需要管理員權限。無論如何上面那個優先順序測試是更強的證據 —— 它直接展示綁定，
而不是從「缺失」去推論。）

而且不付效能代價。同一個目錄、同樣碰不到 CUDA Toolkit，跑完整的部署工作負載
（Nemotron `-ncmoe 10`、`-t 10 -r 3`）：

| 輪次 | t/s |
|---|---|
| 1 | 75.18 ± 4.92 |
| 2 | 80.77 ± 9.44 |
| 3 | 79.80 ± 10.86 |

平均 **78.58**、全距 7.1% —— 與 release 正常量測的範圍相同（一批 92.87、另一批 78.33，
漂移說明見 README）。所以那 372 MB 是白省的。

## 問題

`llama.app` 的 CUDA binary 附帶的 CPU backend **完全沒有向量指令**。它的 `system_info`
（已 pin 的 `b10107` 與 `b10612` 皆同）：

```
CPU : LLAMAFILE = 1 | REPACK = 1          ← 沒有 AVX / AVX2 / FMA / F16C / AVX_VNNI
```

同一顆 commit 的 llama.cpp release 回報：

```
CPU : SSE3 | SSSE3 | AVX | AVX_VNNI | AVX2 | F16C | FMA | BMI2 | LLAMAFILE | OPENMP | REPACK
```

模型放得進 VRAM 時看不出來。一旦有權重在 CPU 上計算，CPU backend 就進入每個 token 的
關鍵路徑：

| 情境 | CPU 參與 | `llama.app` 落後 |
|---|---|---|
| 0.5B 全部放進 VRAM | 無 | ~1.0 倍 |
| Nemotron `-ncmoe 0` | 低（PCIe 受限） | ~1.1 倍 |
| Nemotron `-ncmoe 10` | expert 在 CPU | **2.1 倍** |

根因在 `ggml-org/llama-install.sh`：`CMakeLists.txt` 強制 `GGML_NATIVE=OFF`（對要發佈的
binary 是正確的），而 `scripts/generate.py` 用 `LLAMA_INSTALL_FLAGS` 為 `cpu`、`vulkan`、
`metal` preset 補回指令集 —— 但 `cuda` 和 `rocm` 沒有。已整理成
[`../ISSUE.zh-TW.md`](../ISSUE.zh-TW.md)。

## 數字

兩個獨立批次。為什麼只有同批次比值有意義，見 README。

批次 A —— release 對 `llama.app`：

| 輪次 | llama.cpp release | `llama.app` |
|---|---|---|
| 1 | 92.87 | 44.29 |
| 2 | 92.90 | 45.18 |
| 3 | 92.40 | 39.60 |
| 4 | 92.35 | 42.71 |

**2.16 倍**

批次 B —— 加入自編版：

| 輪次 | llama.cpp release | 自編 | `llama.app` |
|---|---|---|---|
| 1 | 78.33 | 83.43 | 39.09 |
| 2 | 84.51 | 80.60 | 39.54 |
| 3 | 83.34 | 84.03 | 38.57 |
| 4 | 84.67 | 84.39 | 38.55 |

**2.13 倍**，且 release ≈ 自編。

## 只看 CPU backend 本身：5.13 倍

這整個調查裡最乾淨的一次量測。`-ngl 0` 把**全部**計算放在 CPU 上，於是 GPU、offload 切分、
執行緒配置漂移全部退出。0.5B 模型，`-p 0 -n 64 -ngl 0 -t 6 -r 3`，每顆都先暖機，交錯五輪
（`scripts/windows/bench.ps1`）：

| 輪次 | llama.cpp release | 自編 native | `replica`（無向量指令） |
|---|---|---|---|
| 1 | 115.34 ± 9.02 | 121.88 ± 1.30 | 23.72 ± 0.03 |
| 2 | 120.32 ± 0.50 | 117.41 ± 5.53 | 23.65 ± 0.15 |
| 3 | 119.08 ± 1.25 | 122.10 ± 0.57 | 23.11 ± 1.15 |
| 4 | 119.60 ± 0.92 | 121.67 ± 0.84 | 23.21 ± 0.71 |
| 5 | 120.50 ± 0.69 | 122.90 ± 0.47 | 22.22 ± 1.42 |
| **平均** | **118.97** | **121.19** | **23.18** |
| 全距 | 4.3% | 4.5% | 6.5% |

release 對 replica **5.13 倍**，native 對 replica **5.23 倍**。在這個全距下不需要任何批次比較的
附帶條件。

它也比 offload 工作負載更銳利地回答了「release 對自編」這個問題：
**121.19 對 118.97 相差 1.9%，落在 4.3～4.5% 的全距內。** 無法區分 —— 方向與幅度都與 offload
那組（83.1 對 82.7）一致。

這是 CPU **那一部分**工作的代價，所以它比上面 2.1 倍的端到端數字大：`-ncmoe 10` 之下只有
一部分模型在 CPU 上算，5.13 倍被 GPU 仍在做的事稀釋掉了。它也比下面隔離出的 1.81 倍大，
因為那一組是兩邊都關掉 OpenMP、跑 offload 工作負載；這一組則是整組 CUDA preset 旗標的差異，
跑在純 CPU 工作負載上。

它同時也事後驗證了 `replica` 設定：一顆能在 offload 工作負載上重現 `llama.app` 吞吐的 build，
在純 CPU 工作負載上就應該崩掉 —— 而它確實崩了。

## 隔離出來的因素

每一組都在同一批次內交錯，只改一個變因：

| 因素 | 對照 | 影響 |
|---|---|---|
| **向量指令** | MSVC、無 OpenMP、動態連結 —— 只差 ISA 開關 | **1.81 倍** |
| OpenMP | 有 AVX2 時 | 1.27 倍 |
| OpenMP | 無 SIMD 時 | 1.06 倍 |
| 靜態 vs 動態連結 | 複製版內部 | **無影響** |

向量指令主導。兩個因素會交互影響而非疊乘，不能相乘。

## 從原始碼複製出 `llama.app`

`scripts/windows/build.ps1 -Config replica` 套用 `llama-install.sh` 的 forced 設定，
量到相同的吞吐 —— 所以差距完全歸因於那組建置設定：

| 輪次 | 複製版 | `llama.app` |
|---|---|---|
| 1 | 42.99 | 40.83 |
| 2 | 43.78 | 44.20 |
| 3 | 44.88 | 44.84 |

光是 `GGML_NATIVE=OFF` 不足以重現 —— ggml 的 `INS_ENB` 會讓指令集選項維持開啟，必須明確關掉，
而那正是 CUDA preset 最終落入的狀態。

## 已排除

| 假設 | 檢驗方式 | 結果 |
|---|---|---|
| `llama.app` 裝成 Vulkan 或 CPU 版 | `ggml_cuda_init` 輸出 | 否，是 CUDA |
| GPU arch 不符 → PTX JIT | `cuobjdump -lelf` / `-lptx` | 否 —— 兩邊都是原生 sm_120 cubin、零 PTX |
| CUDA graphs 只有一邊啟用 | `GGML_CUDA_DISABLE_GRAPHS=1` | 否，兩邊都有 |
| 執行期參數不同 | `llama-bench -o json` 逐欄比對 | 否，32 欄全同 |
| offload 切分不同 | `load_tensors` buffer size | 否，逐 byte 相同 |
| BMI2 造成倒退 | `GGML_BMI2` 開關 + 抽換變體 DLL | 無影響 |
| 靜態連結較慢 | `BUILD_SHARED_LIBS` 開關 | 無影響 |
| llama.cpp 的變體 dispatch 選錯 | 手動強制指定各變體 | 否 —— AUTO 落在最高的 ISA 層級 |
