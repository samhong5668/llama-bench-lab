*[English](RESULTS.md)* · *[Windows](../windows/RESULTS.zh-TW.md)*

# Linux：`llama.app` 或自編 —— 沒有 release binary 可用

## 前提

**llama.cpp 沒有發布 Linux 的 CUDA binary。** Ubuntu asset 確實存在 —— `ubuntu-x64`（純 CPU）、
`ubuntu-vulkan-x64`、`ubuntu-rocm`、`ubuntu-sycl-fp16/fp32`、`ubuntu-openvino`、`arm64`、
`s390x` —— 但**沒有任何一個是 CUDA**。release 裡所有 CUDA asset 都是 `win-`，`b10107`、`b10644` 與目前最新的 `b10665` 都查過。

所以 Windows 的答案（下載兩個 zip）在 Linux 上不成立。Linux 的 CUDA 選項是：

| 方案 | t/s | 每台機器要安裝 |
|---|---|---|
| 自編，`GGML_NATIVE=ON` | **83.5** | `build-essential` + `cmake` + CUDA Toolkit（約 3 GB），再編譯 6～15 分鐘 |
| 自編，`-march=x86-64-v3` | 約 82（落後 1.4%） | 無 —— 在別處編一次，直接散布 binary |
| `llama.app`（`llama-install.sh`） | **49.4** | 一行指令；**只要 NVIDIA 驅動** |

**`llama.app` 與自編相差 1.69 倍** —— 這是紮實的結論，兩個 Linux 批次都複現。

**`GGML_NATIVE=ON` 對 `-march=x86-64-v3`：相差 1.4%。** offload 工作負載分不出來 ——
一批說 10%（83.5 對 75.7），另一批說 0% —— 因為同一批裡放三顆 binary 會把綁定壓下去的漂移
重新引入。改用 `-ngl 0` 直接量 CPU backend 就定案了，見下節。既然 ISA 旗標只影響 CPU backend，
那裡差 1.4% 就把 offload 工作負載上的差異也上界在 1.4%，所以那個 10% 是雜訊。

**這讓可散布的 build 變成可行選項。** `-march=x86-64-v3` 只付出約 1.4%，而且能跑在任何
2013 年後的 x86 主機上 —— 所以可以編一次供整批機器使用，不必每台都編。對安裝流程值得考慮，
因為它同時免掉每台的 CUDA Toolkit 與編譯時間，卻幾乎保留全部的 1.69 倍。

**AVX-VNNI 在這裡沒有可量測的效益：0.04%。** `v3vnni`（`-march=x86-64-v3 -mavxvnni`，
已驗證只多加了 `AVX_VNNI`、沒有別的）與純 `v3` 打平。AVX-VNNI 加速的是 int8 點積；
對這顆 CPU 上的 Q4_K token generation，被操到的不是那條路徑。

83.5 落在 Windows 那組數字（82.7～83.1）的同一個範圍，所以**Linux 自編並不比 Windows 慢**。
兩個平台的數字取得方式不同（見〈已知限制〉），所以請理解為「同一範圍」，不是「相等」。

## 該選哪個

三個方案，依「目標機器上要放多少東西」遞增排列：

1. **用 `-march=x86-64-v3` 編一次，散布 binary。** 約 82 t/s，目標機器上除了 NVIDIA 驅動與
   binary 連結的 CUDA runtime 之外什麼都不需要。對比 native 只付出 1.4%，免工具鏈、免 3 GB
   Toolkit、免每台編譯。**這是該優先考慮的方案** —— 在 `native` 對 `v3` 看起來差 10% 的時候，
   它並不在選項裡。
2. **每台機器用 `GGML_NATIVE=ON` 編。** 83.5 t/s，最快，但每台目標機器都要放
   `build-essential` + `cmake` + 約 3 GB CUDA Toolkit，再編 6～15 分鐘。Ubuntu 這邊是可腳本化的
   `apt install`、沒有 GUI 安裝器，所以現實可行（Windows 上不是）—— 只是它比方案 1 只快 1.4%。
3. **`llama.app`。** 真的只有一行指令，而且不需要 CUDA Toolkit：用 `ldd` 確認過 binary 唯一的
   CUDA 依賴是 `libcuda.so.1`（驅動），因為 cuBLAS 是靜態連結 —— 這也是它有 531 MB 而
   Windows 那顆只有 48 MB 的原因。代價是 **1.69 倍**。

**方案 1 與 2 相差 1.4% 之內，而且都領先方案 3 達 1.69 倍**，所以真正要決定的只有一件事：
散布預編 binary，還是在現場編譯。

## 數字

WSL2 的 Ubuntu 24.04，用 `scripts/linux/bench3.sh`，五輪交錯。同一台機器、同一個模型、
同一顆 commit `c0bc8591e`。gcc 13.3。

綁定在 E-core 上（`-t 6 -C 0x3F0 --cpu-strict 1`）。不綁定的話這一批完全不能用 ——
三顆 binary 全距 38～49%，而且四輪呈一致下降趨勢。綁定後降到 3～4%。量測規則見 README。

| 輪次 | `llama.app` | 自編 native | 自編 v3 |
|---|---|---|---|
| 1 | 49.05 ± 7.99 | 82.07 ± 0.86 | 75.34 ± 15.09 |
| 2 | 49.70 ± 8.24 | 84.19 ± 1.57 | 74.46 ± 16.39 |
| 3 | 49.41 ± 8.56 | 82.92 ± 1.05 | 76.04 ± 15.91 |
| 4 | 48.70 ± 8.44 | 82.94 ± 1.45 | 75.11 ± 16.45 |
| 5 | 50.17 ± 9.15 | 85.29 ± 1.23 | 77.58 ± 16.99 |

平均 49.4 / 83.5 / 75.7 —— 輪次間全距 3.0% / 3.9% / 4.1%。

### 直接在 CPU backend 上分離建置旗標

上面那張表分不出 `native` 與 `v3`，所以這一節單獨量 CPU backend：0.5B 模型，
`-p 0 -n 64 -ngl 0 -t 6 -r 5`，不綁定，每批只放兩顆 binary，八輪。

「兩顆而非三顆」和「不綁定而非綁 E-core」都是刻意的。WSL2 有**單向**干擾 —— 偶爾某一輪的
行程內 stddev 跳到 ±44 而平均值塌掉 —— 而 E-core 綁定會讓它**更糟**，因為 Windows 把自己的
背景工作也排到 E-core 上。下面排除 stddev 超過 ±15 的輪次；注意這種雜訊只會讓執行變慢，
從不會讓它變快。

| | 乾淨輪次 | 平均 | 最佳輪次 |
|---|---|---|---|
| `native` | 126.64 / 125.62 / 125.76 / 124.11 / 126.12 / 126.83 | **125.85** | 126.83 ± 0.81 |
| `v3` | 123.41 / 123.77 / 124.66 / 124.94 / 124.82 / 122.90 | **124.08** | 124.94 ± 0.83 |

`native` 領先 **1.4%**，而且六輪中有五輪領先 —— 幅度小但方向一致。

| | 乾淨輪次 | 平均 | 最佳輪次 |
|---|---|---|---|
| `v3` | 126.39 / 125.58 / 123.46 / 119.17 / 125.36 / 122.10 / 125.28 / 125.42 | **124.10** | 126.39 ± 1.31 |
| `v3vnni` | 124.57 / 124.27 / 120.79 / 121.85 / 127.24 / 126.16 | **124.15** | 127.24 ± 1.27 |

平均相差 **0.04%**、最佳輪次相差 0.7% —— AVX-VNNI 在這裡毫無作用。

兩個結論在兩種估計法（乾淨輪次平均、最佳觀測值）之下都成立，所以兩者都列出而非只給一個比值。

## 根因與 Windows 相同

`llama.app` 的 Linux CUDA binary 回報完全相同的 `CPU : LLAMAFILE = 1 | REPACK = 1` ——
沒有向量指令。`llama-install.sh` 自己的 `CMakeLists.txt` 與 `scripts/generate.py` generator 服務兩個平台，
所以 [`../ISSUE.zh-TW.md`](../ISSUE.zh-TW.md) 也涵蓋 Linux。上游修好之後，這裡就沒有自編的
理由了。

## 已知限制

這些數字來自 **WSL2 而非原生 Linux** —— GPU 走 passthrough。批次內的比值紮實，但絕對值仍
值得在裸機上重測。這是本機唯一無法回答的一項：用 Docker 也沒用，因為 Windows 上的 Docker
跑在同一個 WSL2 kernel、同一套虛擬化 GPU 上。

Windows 與 Linux 的 offload 絕對值取得方式不同（Windows 未綁定、Linux 綁 E-core），因為
E-core 綁定在 Windows 的三方批次上沒有像 Linux 那樣穩定下來。**所以請比較各平台內部的比值，
不要跨平台比絕對值。**

在 `-ngl 0` 的 CPU backend 量測上，綁定的建議剛好反過來：**E-core 綁定會讓 WSL2 更吵**，
因為 Windows 把自己的背景工作放在 E-core 上，而 `--cpu-strict 1` 又阻止 guest 的排程器離開。
可用的組合是「不綁定 + 依 stddev 剔除離群輪次」。Windows 兩者都不需要 —— 同樣的量測在那邊
全距 4～6%，不必剔除任何一輪。
