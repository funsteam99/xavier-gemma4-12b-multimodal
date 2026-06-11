# Gemma 4 12B QAT + MTP Multimodal Trial on Jetson Xavier

本專案記錄在 NVIDIA Jetson Xavier 上使用 `llama.cpp` 執行 **Gemma 4 12B QAT + MTP 多模態模型** 的實測狀態、安裝流程、前端測試工具與待解問題。

這個 repo 建議對應 GitHub：

```text
funsteam99/xavier-gemma4-12b-multimodal
```

先前已完成且較穩定的 E4B baseline 另存於：

```text
https://github.com/funsteam99/xavier-gemma4-e4b-multimodal.git
```

目前定位：

- **Gemma 4 12B QAT + MTP**：本 repo 主線。已完成 `llama.cpp` CUDA build（特別在編譯期停用 VMM 以防多模態音訊/影像推導發生 OOM 崩潰），且已完成首輪效能驗證與 systemd 自動化服務部署。
- **Gemma 4 E4B**：只作 baseline 參照；正式 E4B 紀錄請看 `xavier-gemma4-e4b-multimodal`。

## Hardware

```text
Device: NVIDIA Jetson Xavier
Hostname: ubuntu.local
User: nvidia
RAM visible to system: about 14 GiB
CUDA device: Xavier, compute capability 7.2
CUDA memory reported by llama.cpp: 14886 MiB
```

模型與前端部署在 SD 卡：

```text
/media/nvidia/sd
```

`llama.cpp` 原始碼位置：

```text
/home/nvidia/src/llama.cpp
```

本 repo 在 Xavier 上的工作目錄：

```text
/home/nvidia/XAVIER
```

## Model Focus

### Gemma 4 12B

12B QAT + MTP 是本 repo 的主要測試目標。模型使用 QAT Q4_0 主模型、Q8_0 MTP drafter 與 BF16 多模態 projector。

```text
llama.cpp source: official master
verified source commit: ac4cddeb0 (2026-06-10)
CUDA architecture: 72
```

模型集中在獨立目錄：

```text
/media/nvidia/sd/models/gemma-4-12b-qat-mtp/
  gemma-4-12B-it-QAT-Q4_0.gguf
  gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf
  mmproj-gemma-4-12B-it-QAT-BF16.gguf
```

Runtime 目錄與服務：

```text
Repo:    /home/nvidia/XAVIER
Console: /media/nvidia/sd/gemma4-12b-qat-mtp-console
Backend: http://ubuntu.local:18085
Console: http://ubuntu.local:18091
```

預設啟動設定：

```text
n_ctx: 4096
batch: 512
ubatch: 512
parallel: 1
spec_type: draft-mtp
spec_draft_n_max: 4
main GPU layers: 99
draft GPU layers: 99
image_max_tokens: 768
```

舊 Q4_K_M baseline 曾測得：

```text
RAM used: about 9 GiB
RAM available: about 4 to 5 GiB
Swap used: small, about 80 to 100 MiB
```

舊 Q4_K_M baseline 速度：

```text
Prompt eval: about 5.3 tokens/second
Generation: about 3.5 tokens/second
```

Gemma 4 12B QAT + MTP 在 Xavier 上的實測速度與記憶體佔用（2026-06-11 實測，CTX=4096, q4_0 cache）：
*   **記憶體佔用**：載入後整體系統使用約 8.8 GiB（available 剩餘約 5.5 GiB）
*   **Text Baseline 速度**：約 3.85 tokens/second (耗時 3.38 秒，輸出 13 tokens)
*   **Image OCR 速度**：約 3.85 tokens/second (耗時 33.23 秒，輸出 128 tokens)
*   **Audio ASR 速度**：約 3.98 tokens/second (耗時 6.03 秒，輸出 24 tokens)

音訊/ASR 經編譯期與快取優化後已能正常執行，未發生 OOM 閃退，且辨識結果精準。

### E4B Baseline

E4B 是先前完成的較穩定版本，速度較適合 Xavier 日常互動。此 repo 只保留 E4B 作比較基準，完整 E4B 專案請看：

```text
https://github.com/funsteam99/xavier-gemma4-e4b-multimodal.git
```

已知 E4B 服務配置：

```text
Backend: http://ubuntu.local:18084
Console: http://ubuntu.local:18090
```

代表性效能：

```text
Image processing: about 2 to 6.7 seconds
Audio processing: about 1.5 to 3.6 seconds
Prompt eval: about 60 to 103 tokens/second
Text generation: about 4.5 to 5.3 tokens/second
```

## Initial Install

以下操作在 Xavier 上執行。

### 1. 準備 `llama.cpp`

```bash
mkdir -p /home/nvidia/src
cd /home/nvidia/src
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
```

如果已經有 repo，更新：

```bash
cd /home/nvidia/src/llama.cpp
git pull --ff-only
```

設定 CUDA build（特別加入 `-DGGML_CUDA_NO_VMM=ON` 以免多模態語音/影像編碼造成 VMM 崩潰）：

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=72 -DGGML_CUDA_NO_VMM=ON
```

Xavier 記憶體有限，編譯不要用無限制 `-j`。未啟動服務時，空閒記憶體有 11GiB 左右，建議使用 `-j6` 快速編譯：

```bash
cmake --build build -j6 --target llama-server llama-cli
```

如果記憶體吃緊，改用：

```bash
cmake --build build -j2 --target llama-server llama-cli
```

確認版本：

```bash
/home/nvidia/src/llama.cpp/build/bin/llama-server --version
```

### 2. 放置模型

E4B：

```bash
mkdir -p /media/nvidia/sd/models/gemma-4-e4b-it
```

12B QAT + MTP：

```bash
mkdir -p /media/nvidia/sd/models/gemma-4-12b-qat-mtp
cd /media/nvidia/sd/models/gemma-4-12b-qat-mtp

curl -fL -C - -o gemma-4-12B-it-QAT-Q4_0.gguf \
  https://huggingface.co/lmstudio-community/gemma-4-12B-it-QAT-GGUF/resolve/main/gemma-4-12B-it-QAT-Q4_0.gguf

curl -fL -C - -o mmproj-gemma-4-12B-it-QAT-BF16.gguf \
  https://huggingface.co/lmstudio-community/gemma-4-12B-it-QAT-GGUF/resolve/main/mmproj-gemma-4-12B-it-QAT-BF16.gguf

curl -fL -C - -o gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf \
  https://huggingface.co/Janvitos/gemma-4-12B-it-qat-assistant-MTP-Q8_0-GGUF/resolve/main/gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf
```

下載前後建議檢查：

```bash
free -h
df -h /media/nvidia/sd
```

### 3. 部署本 repo

將本 repo 放在 Xavier：

```text
/home/nvidia/XAVIER
```

確認腳本可執行：

```bash
cd /home/nvidia/XAVIER
chmod +x scripts/init-xavier.sh
chmod +x scripts/init-xavier-12b.sh
```

## Running Services

### E4B Baseline

```bash
cd /home/nvidia/XAVIER
./scripts/init-xavier.sh check
./scripts/init-xavier.sh start
```

常用命令：

```bash
./scripts/init-xavier.sh status
./scripts/init-xavier.sh stop
./scripts/init-xavier.sh start-backend
./scripts/init-xavier.sh start-console
```

### 12B QAT + MTP

```bash
cd /home/nvidia/XAVIER
./scripts/init-xavier-12b.sh check
./scripts/init-xavier-12b.sh start
```

常用命令：

```bash
./scripts/init-xavier-12b.sh status
./scripts/init-xavier-12b.sh stop
./scripts/init-xavier-12b.sh start-backend
./scripts/init-xavier-12b.sh start-console
```

QAT/MTP 可用環境變數調整：

```bash
CTX_SIZE=4096 \
BATCH_SIZE=512 \
UBATCH_SIZE=512 \
SPEC_DRAFT_N_MAX=4 \
IMAGE_MAX_TOKENS=768 \
./scripts/init-xavier-12b.sh start
```

腳本會在啟動前執行 `free -h` 與 `tegrastats`，預設要求至少 `10240 MiB` 的 `MemAvailable`。不要同時啟動 E4B、舊 12B 或其他大型推理模型。

等 systemd unit 更新為 QAT/MTP 後，正式服務操作為：

```bash
systemctl --user start gemma-12b-backend.service
systemctl --user start gemma-12b-console.service
systemctl --user status gemma-12b-backend.service
systemctl --user status gemma-12b-console.service
```

在 unit 更新完成前，以 repo 腳本為準，不要啟動仍指向舊 Q4_K_M 路徑的 unit。

如果遇到 context 不足：

```text
request (...) exceeds the available context size
```

提高 `CTX_SIZE`。目前 12B 預設是 `4096`。

## Health Checks

Backend：

```bash
curl http://127.0.0.1:18084/health
curl http://127.0.0.1:18085/health
```

Models：

```bash
curl http://127.0.0.1:18084/v1/models
curl http://127.0.0.1:18085/v1/models
```

Console：

```bash
curl -fsS -X POST -H "Content-Type: application/json" -d "{}" \
  http://127.0.0.1:18090/api/health

curl -fsS -X POST -H "Content-Type: application/json" -d "{}" \
  http://127.0.0.1:18091/api/health
```

Memory：

```bash
free -h
tegrastats --interval 1000
```

## Frontend Console

前端是輕量瀏覽器測試台，由 Python HTTP server 提供靜態頁面並代理到 `llama-server`，避免瀏覽器 CORS 問題。

主要功能：

- Backend health check
- Model capability display
- Text prompt baseline
- Image OCR
- Chart / UI understanding
- ASR audio transcription
- Image + audio combined prompt
- Raw JSON inspection
- Response time
- Prompt tokens / output tokens
- Estimated output tok/s

支援輸入格式：

```text
Images: PNG, JPG, JPEG, WebP
Audio: WAV, MP3
```

不支援：

```text
webm
m4a
browser default recording formats that are not WAV/MP3
```

### Request Formats

圖片：

```json
{
  "type": "image_url",
  "image_url": {
    "url": "data:image/png;base64,..."
  }
}
```

音訊：

```json
{
  "type": "input_audio",
  "input_audio": {
    "data": "...base64...",
    "format": "wav"
  }
}
```

## Important Fixes

### Image Batch Assert

曾遇到圖片上傳造成：

```text
GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) && "non-causal attention requires n_ubatch >= n_tokens") failed
```

原因是圖片 token 數超過 physical batch。修正：

```text
-ub 1024
```

### Invalid Media Decode

錯誤：

```json
{
  "error": {
    "code": 400,
    "message": "Failed to load image or audio file",
    "type": "invalid_request_error"
  }
}
```

常見原因：

- 把音訊拖到圖片區
- 使用不支援的圖片格式
- 使用不支援的音訊格式

前端已限制圖片與音訊上傳區的格式。

### Context Too Small

錯誤：

```text
request (...) exceeds the available context size
```

12B 多模態請求容易超過 `2048` context，目前 12B 預設已改為：

```text
CTX_SIZE=4096
```

## Pending Issues

### 1. 12B ASR/audio 已成功測試並解決 VMM 崩潰

透過編譯期停用虛擬記憶體管理選項（-DGGML_CUDA_NO_VMM=ON），且搭配優化快取參數（q4_0 KV cache），ASR 語音辨識已可正常運作，不再發生 OOM 崩潰。不過 `llama.cpp` 音訊輸入仍有實驗性性質，後續可持續追蹤 ASR 尾端重複片段的優化。

### 2. 12B 記憶體餘裕有限

舊 12B Q4_K_M 載入後約使用 9 GiB RAM。QAT + MTP 會另外載入 drafter，實際餘裕必須重新測量。啟動前、長任務前、調高 context 前都要先查：

```bash
free -h
tegrastats --interval 1000
```

### 3. 12B 速度慢

12B 在 Xavier 上可跑，但生成速度約 3 到 4 tok/s。長回答會很慢，建議：

- `max_tokens` 控制在 64 到 256
- ASR 控制在 96 到 160
- 圖片先用小圖
- `--parallel 1`

### 4. Frontend 可再加強

待做：

- ASR 重複片段偵測與截斷
- 顯示目前 context limit
- 顯示目前 request token 估算
- 測試紀錄匯出
- 針對 E4B / 12B 的比較頁

### 5. QAT + MTP 效能對比評估 (Xavier vs Mac)

為了評估 Jetson Xavier 的移植效能與瓶頸，以下整理了 **Gemma 4 12B QAT (Q4_0) + MTP (Q8_0)** 在 Jetson Xavier 與 Apple Silicon Mac 上的對比數據：

| 評估維度 / 平台 | NVIDIA Jetson Xavier | Apple Silicon Mac (參照基準) |
| :--- | :--- | :--- |
| **硬體核心** | 8核 Carmel ARM CPU, 512核 Volta GPU | Apple Silicon (如 M 系列 Max/Pro) |
| **統一記憶體與頻寬** | ~14.8 GiB 統一記憶體 @ **137 GB/s** | 統一記憶體 @ **150 ~ 400+ GB/s** |
| **KV Cache 設定** | 優化量化 `q4_0` (大幅節省 VRAM 佔用) | 預設 `q8_0` 或 `f16` |
| **記憶體管理方式** | 停用 VMM (`-DGGML_CUDA_NO_VMM=ON`) | 啟用預設 VMM 虛擬記憶體池 |
| **Text Generation 速度**| **3.85 t/s** (2026-06-11 實測) | **12.68 t/s** (投機解碼實驗數據) |
| **ASR 語音辨識速度** | **3.98 t/s** (297 tokens WAV) | N/A (一般不限於邊緣 ASR 推理) |
| **部署狀態與限制** | 透過 systemd 用戶服務常駐後端與控制台 | 本機開發端/命令列測試 |

> [!NOTE]
> *   **效能差距主因**：Xavier 的生成速度約為 Mac 的 **30.4%**。這主要受限於 Xavier 的記憶體頻寬（137 GB/s）與硬體代次（Volta 架構，無專用 Tensor Cores 優化多模態投影）。
> *   **記憶體取捨**：Xavier 必須使用 `q4_0` KV Cache 並編譯關閉 VMM 才能避免 OOM，這對速度有微幅影響，但確保了在 14.8 GiB 記憶體下的穩定運行。
> *   詳細評估報告與建置歷史請參閱 [gemma-4-12b-xavier-eval.md](file:///C:/Users/pondahai/gemma-4-12b-xavier-eval.md)。


## Current Recommendation

本 repo 的主線是 `Gemma 4 12B QAT + MTP` 在 Xavier 上的可行性與多模態 trial。

日常互動與 ASR 目前仍先用 E4B baseline。

12B 保留為獨立 trial service，用於：

- 較難 OCR
- 圖片/介面理解
- 多步驟推理
- 文字品質比較

12B audio/ASR 暫時等待 `llama.cpp` 正式穩定後再評估。
