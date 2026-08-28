*[English](README.md)*

# llama-bench-lab

在權重放不進 VRAM 的模型上，`llama.app` 的 CUDA binary 在 **Windows 上慢 2.1 倍**、
**Linux 上慢 1.7 倍**。它附帶的 CPU backend 完全沒有向量指令。

## 哪個 binary 是哪個

兩個發行管道都提供現成的執行檔，所以單說「prebuilt」會有歧義：

| 本文的稱法 | 來源 | 存放於 |
|---|---|---|
| **`llama.app`** | [`ggml-org/llama-install.sh`](https://github.com/ggml-org/llama-install.sh) —— 一行指令的 `install.ps1` / `install.sh` | Hugging Face |
| **llama.cpp release** | [`ggml-org/llama.cpp/releases`](https://github.com/ggml-org/llama.cpp/releases) | GitHub Releases |

兩條獨立的建置管線，所以 CPU 旗標才會不同。**慢的那個，正是 `llama-install.sh` 安裝器給你的。**

## 該用哪個

| 平台 | 用 | 理由 |
|---|---|---|
| **Windows** | [llama.cpp release，一個 zip](docs/windows/RESULTS.zh-TW.md) —— 137 MB | 82.7 t/s，與自編相當，零建置工具 |
| **Linux** | [自編](docs/linux/RESULTS.zh-TW.md) —— `-march=x86-64-v3` 可編一次散布 | Linux 沒有 CUDA release binary；約 82～83.5 對 `llama.app` 的 49.4 |

| 文件 | |
|---|---|
| [`docs/windows/RESULTS.zh-TW.md`](docs/windows/RESULTS.zh-TW.md) | Windows 的結論、數字、隔離出的因素 |
| [`docs/linux/RESULTS.zh-TW.md`](docs/linux/RESULTS.zh-TW.md) | Linux 的結論與安裝成本權衡 |
| [`docs/ISSUE.zh-TW.md`](docs/ISSUE.zh-TW.md) | 給 `ggml-org/llama-install.sh` 的 issue 草稿 |

每份都有對應的英文版。

## 怎麼量才不會量錯

這個題目天真地量會得到「很有自信但錯誤」的答案。以下三個陷阱各自造成過一次錯誤結論，
而且每次都是靠重測才抓到。

1. **每顆 binary 各自暖機後再取數。** CPU 側權重是 `mmap` 進來的，所以**每顆 binary 的第一次
   執行**都要付冷讀成本。暖了這顆不代表暖了下一顆。
2. **交錯執行，不要一顆連跑完再換下一顆。** 後跑的那顆會被拖慢。
3. **絕對不要跨批次比較。** 這台機器的絕對吞吐量在不同執行之間漂移 14～18% ——
   同一組設定曾量到 33.7 和 45.8。只有同批次內的比值有意義。

`-r N` 偵測不到前兩者：第一次迭代就把後面暖起來了。

**如果只需要比較 CPU backend，用小模型加 `-ngl 0`。** 全部計算都落在 CPU，於是 GPU 與
offload 切分退出，下面講的漂移也大半隨之消失 —— Windows 上全距 4～6%，而 offload 工作負載是
12～24%，而且完全不需要綁定。它回答的問題比部署數字窄，但回答得乾淨，而且正是它最後分離出
offload 工作負載分不出來的那些建置旗標。

WSL2 上同樣的量測更吵，而且吵法不同：干擾是**單向的** —— 表現為偶爾某一輪的行程內 stddev
跳到 ±44 而平均值塌掉。請依 stddev 剔除那些輪次，不要把它們平均進去；而且在那裡**不要**綁
E-core —— Windows 把自己的背景工作排在 E-core 上，所以 `--cpu-strict 1` 只會更糟。

### 漂移的原因

不是熱節流，也不是記憶體。十次儀器化執行（`scripts/windows/drift_probe.ps1`）橫跨
66.98 → 81.01 t/s，而 GPU 全程平在 50 °C、2857 MHz、約 34.7 W，可用記憶體 105 GB。
真正的原因是**混合架構 P-core/E-core CPU 上的執行緒配置**：

| 配置 | 四輪 | 全距 |
|---|---|---|
| 預設 `-t 10`，不綁定 | 75.66 / 80.06 / 70.89 / 73.51 | **12.2%** |
| 4 個 P-core，`-t 4 -C 0x00F --cpu-strict 1` | 60.75 / 57.91 / 59.41 / 63.50 | 9.3% |
| 6 個 E-core，`-t 6 -C 0x3F0 --cpu-strict 1` | 66.46 / 68.10 / 67.37 / 67.52 | **2.4%** |
| 全部 10 核，`-t 10 -C 0x3FF --cpu-strict 1` | 55.92 / 58.16 / 66.13 / 51.84 | 24.4% |

`ggml-cpu.c` 在**每一個 graph node** 之後都有屏障（這張圖有 2187 個），所以最慢的執行緒決定
每一道。執行緒池橫跨兩種速度的核心時，每道屏障都要付代價，而哪些執行緒落在哪一種核心，
每次啟動行程都不同。綁到 E-core 讓全距塌到 2.4%，代價是約 10% 吞吐。把**每個**核心都
strict 綁定反而比不綁更差 —— 那會阻止排程器把執行緒遷離競爭。

**要穩定的絕對值就綁 E-core；要最高吞吐就不綁，並且只在同一交錯批次內比較。**

## 測試環境

RTX 5060 Ti 16 GB (sm_120, driver 610.62) / Core Ultra 5 225 (10c/10t, AVX2 + AVX-VNNI,
無 AVX-512) / 128 GB DDR5-5600。Windows 11，以及 WSL2 的 Ubuntu 24.04，**gcc 13.3** ——
請先用 `wsl -l -v` 與 `gcc --version` 確認，因為更新的預設 distro 會帶來不同的
編譯器，而建置旗標的比較只在同一組工具鏈內成立。

模型 `ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF` 的 `Q4_0`，17.59 GiB —— 大於 VRAM，
所以 `-ncmoe 10` 會把 10 層的 MoE expert 放在 CPU 上，那正是讓 CPU backend 產生影響的關鍵。
**必須是這個 repo**：其他鏡像有同名檔案但來自不同的量化批次。

標準呼叫方式，搭配 `CUDA_VISIBLE_DEVICES=0`：

```
llama-bench -m <model> -p 0 -n 128 -ngl 99 -ncmoe 10 -t 10 -r 10 -o md
```

比較之前已驗證兩邊完全一致：`llama-bench -o json` 的 32 個欄位，以及 offload 切分
（`CPU_Mapped 3354.74 MiB` / `CUDA0 14917.64 MiB`、`graph nodes = 2187`）。

**限制**：只有一台機器。Linux 數字來自 WSL2（GPU 走 passthrough），而且是綁 E-core 量的，
Windows 那組沒綁 —— **所以請比較各平台內部的比值，絕不跨平台比絕對值。**

## 重現方式

```bash
uv run scripts/download_models.py --dest ./models --check-only   # 只查大小，不下載
uv run scripts/download_models.py --dest ./models
```

### Windows

```powershell
LLAMA_VERSION=b10107 powershell -File install.ps1        # llama.app

gh release download b10107 -R ggml-org/llama.cpp `       # llama.cpp release，一個 asset 就夠
  -p "llama-b10107-bin-win-cuda-13.3-x64.zip"

scripts/windows/build.ps1 -Config native                 # 自編
scripts/windows/build.ps1 -Config replica                # llama.app 自己的設定
```

### Linux

```bash
scripts/linux/build.sh v3                 # -march=x86-64-v3 —— 可散布，落後 native 1.4%
scripts/linux/build.sh native             # GGML_NATIVE=ON —— 最快，但只能每台自編
scripts/linux/build.sh replica            # llama.app 自己的設定
scripts/linux/bench3.sh <model.gguf> 5    # 部署工作負載

# 只量 CPU backend，噪音低得多 —— native 與 v3 就是靠這個分離出來的
BENCH_ARGS="-p 0 -n 64 -ngl 0 -t 6 -r 5" scripts/linux/bench3.sh <small-model.gguf> 8
```

## 腳本

`scripts/` 放共用工具，`scripts/windows/` 和 `scripts/linux/` 放各平台的部分。

| 腳本 | 用途 |
|---|---|
| `scripts/download_models.py` | 下載兩個 GGUF 模型（`uv run`；選擇性） |
| `scripts/windows/build.ps1` | 建置 `llama-bench` —— `native`、`replica`、`replica-shared`、`noomp` |
| `scripts/windows/bench.ps1` | 交錯的 N 方比較；用來複現 Windows 的各批次 |
| `scripts/windows/run_variants.ps1` | 逐一強制指定 llama.cpp 的 CPU 變體 DLL |
| `scripts/windows/drift_probe.ps1` | 重複同一量測並採樣 GPU 與記憶體狀態 |
| `scripts/linux/build.sh` | 建置 `llama-bench` —— `native`、`v3`、`v3vnni`、`replica` |
| `scripts/linux/bench_linux.sh` | `llama.app` 對自編版，交錯執行 |
| `scripts/linux/bench3.sh` | 交錯的三方比較；三顆 binary、標籤、參數都可覆寫 |

兩支建置腳本裡，**`replica` 指的是「`llama-install.sh` 強制的那組設定」，所以它會複製出
`llama.app`**；`native` 是 `GGML_NATIVE=ON`；`v3` 是固定的 `-march=x86-64-v3`；`v3vnni` 再加 `-mavxvnni`，用來隔離那一個旗標。

build 目錄、llama.cpp 原始碼、下載的壓縮檔、跑分原始輸出都不進 git —— 腳本可以重建，
重要的數字都在文件裡。
