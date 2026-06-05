# Gemma 4 12B Multimodal Trial on Jetson Xavier

本專案記錄在 NVIDIA Jetson Xavier 上使用 `llama.cpp` 執行 **Gemma 4 12B 多模態模型** 的實測狀態、安裝流程、前端測試工具與待解問題。

這個 repo 建議對應 GitHub：

```text
funsteam99/xavier-gemma4-12b-multimodal
```

先前已完成且較穩定的 E4B baseline 另存於：

```text
https://github.com/funsteam99/xavier-gemma4-e4b-multimodal.git
```

目前定位：

- **Gemma 4 12B**：本 repo 主角，已成功載入並可測文字/多模態，但 audio/ASR 仍視為實驗性，不作正式採用。
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

12B 是本 repo 的主要測試目標。它已在 Xavier 上成功載入，使用新版 `llama.cpp`：

```text
llama.cpp version: 9522 (3ecfb150a)
```

模型檔：

```text
/media/nvidia/sd/models/gemma-4-12b-it/
  gemma-4-12B-it-Q4_K_M.gguf
  mmproj-gemma-4-12B-it-Q8_0.gguf
```

服務：

```text
Backend: http://ubuntu.local:18085
Console: http://ubuntu.local:18091
```

目前 12B 啟動設定：

```text
n_ctx: 4096
batch: 1024
ubatch: 1024
parallel: 1
image_max_tokens: 768
```

載入後記憶體大約：

```text
RAM used: about 9 GiB
RAM available: about 4 to 5 GiB
Swap used: small, about 80 to 100 MiB
```

12B 文字 smoke test 成功，但速度較慢：

```text
Prompt eval: about 5.3 tokens/second
Generation: about 3.5 tokens/second
```

重要限制：12B 的 audio/ASR 暫不作正式採用。`llama.cpp` runtime log 明確顯示：

```text
init_audio: audio input is in experimental stage and may have reduced quality
```

實測 ASR 會出現尾端重複片段。前端已加入保守 ASR prompt、`temperature: 0`、`repeat_penalty: 1.18`、ASR 預設 `max_tokens: 160`，但這只是緩解，不代表穩定。

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

設定 CUDA build：

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=72
```

Xavier 記憶體有限，編譯不要用無限制 `-j`。建議：

```bash
cmake --build build -j2 --target llama-server llama-cli
```

如果記憶體吃緊，改用：

```bash
cmake --build build -j1 --target llama-server llama-cli
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

12B：

```bash
mkdir -p /media/nvidia/sd/models/gemma-4-12b-it
cd /media/nvidia/sd/models/gemma-4-12b-it

wget -c -O gemma-4-12B-it-Q4_K_M.gguf \
  https://huggingface.co/ggml-org/gemma-4-12B-it-GGUF/resolve/main/gemma-4-12B-it-Q4_K_M.gguf

wget -c -O mmproj-gemma-4-12B-it-Q8_0.gguf \
  https://huggingface.co/ggml-org/gemma-4-12B-it-GGUF/resolve/main/mmproj-gemma-4-12B-it-Q8_0.gguf
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

### 12B

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

12B 可用環境變數調整：

```bash
CTX_SIZE=4096 \
BATCH_SIZE=1024 \
UBATCH_SIZE=1024 \
IMAGE_MAX_TOKENS=768 \
./scripts/init-xavier-12b.sh start
```

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

### 1. 12B ASR/audio 等待正式支援

目前 `llama.cpp` audio input 仍標示 experimental。12B ASR 會出現尾端重複片段，因此 12B audio/ASR 只保留為測試功能。

目前決策：

- E4B 繼續作為可用 baseline
- 12B 可測文字、圖片理解、OCR、較難推理
- 12B ASR/audio 等待 `llama.cpp` 後續正式支援或更穩定 runtime

### 2. 12B 記憶體餘裕有限

12B Q4_K_M 載入後約使用 9 GiB RAM，Xavier 剩約 4 到 5 GiB 可用。啟動前、長任務前、調高 context 前都要先查：

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

## Current Recommendation

本 repo 的主線是 `Gemma 4 12B` 在 Xavier 上的可行性與多模態 trial。

日常互動與 ASR 目前仍先用 E4B baseline。

12B 保留為獨立 trial service，用於：

- 較難 OCR
- 圖片/介面理解
- 多步驟推理
- 文字品質比較

12B audio/ASR 暫時等待 `llama.cpp` 正式穩定後再評估。
