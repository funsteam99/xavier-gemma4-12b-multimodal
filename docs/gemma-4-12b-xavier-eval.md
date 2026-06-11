# Gemma 4 12B on Jetson Xavier Evaluation

Date: 2026-06-11 (Updated)

## Summary
Gemma 4 12B QAT + MTP multimodal setup has been successfully brought up and tested on Jetson Xavier. By optimizing compilation options and runtime parameters, we successfully resolved initial CUDA Out of Memory (OOM) crashes during audio ASR, enabling full Text, Image OCR, and Audio ASR capabilities.

This setup is now fully integrated with systemd user services for persistent backend and web console hosting.

---

## Benchmark Results (2026-06-11)

All tests were performed on the Jetson Xavier using the compiled `NO_VMM` binary, `-c 4096`, and `q4_0` cache.

| Test Case | Input | Output / Transcription | Latency | 客戶端總均速 | 後端純生成速率 (MTP) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Text Baseline** | 29 tokens | `I am a Large Language Model based on a transformer architecture.` | **3.38s** | 3.85 t/s | **8.25 ~ 15.99 t/s** |
| **Image OCR** | 744 tokens (PNG) | Detailed layout description of the Linux desktop top bar, terminal windows, and `ngrok` output. | **33.23s** | 3.85 t/s | **7.18 ~ 8.25 t/s** |
| **Audio ASR** | 297 tokens (WAV) | `and so my fellow americans ask not what your country can do for you ask what you can do for your country` | **6.03s** | 3.98 t/s | **7.18 ~ 8.25 t/s** |

> [!NOTE]
> *   **客戶端總均速**：以瀏覽器端計量之總回應時間（Latency）計算，包含了 Prompt 評估時間以及網路/CORS 資料傳輸延遲。
> *   **後端純生成速率**：由 `llama-server` 的後端 `timings` 直接回傳之純 GPU 解碼輸出速度。在啟用 MTP 投機解碼（Speculative Decoding，草稿模型接受率約 36% ~ 43%）後，文字生成純速度可達 **8.25 ~ 15.99 t/s**。

---

## Critical Issues & Solutions

### 1. Audio ASR CUDA Out of Memory (OOM) Crash
*   **Symptom:** During audio input processing, `llama-server` crashed with `CUDA error: out of memory` during virtual memory allocation (`cuMemAddressReserve(&pool_addr, CUDA_POOL_VMM_MAX_SIZE, 0, 0, 0)`).
*   **Root Cause:**
    1.  The default CUDA allocator in `llama.cpp` uses Virtual Memory Management (VMM) pools. On unified memory systems like Jetson Tegra, reserving large virtual memory pools is highly prone to failing.
    2.  Loading a 12B QAT model (6.5GB) + MTP assistant draft model (444MB) + multimodal projector (168MB) leaves very little headroom. When the audio projector ran, it triggered a VMM pool extension that failed.
*   **Solution:**
    *   **Compile-time Fix:** Recompiled `llama.cpp` on Xavier with `-DGGML_CUDA_NO_VMM=ON` to disable the VMM memory pool. This forces the system to use standard `cudaMalloc` allocations.
    *   **Runtime Cache Optimization:** Configured KV Cache to use `q4_0` instead of `q8_0` for both target and draft models (`--cache-type-k q4_0 --cache-type-v q4_0 --cache-type-k-draft q4_0 --cache-type-v-draft q4_0`).
    *   **Context Scaling:** Scaled down context size to `-c 2048` and batch sizes to `-b 512 -ub 512` to reduce peak memory activations.

---

## Compile & Build History
For reference, here is the command used to compile `llama.cpp` on the Xavier with the VMM allocator disabled:

```bash
cd /home/nvidia/src/llama.cpp
# 1. Configure CMake with NO_VMM and Xavier architecture (7.2)
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=72 -DGGML_CUDA_NO_VMM=ON

# 2. Build llama-server and llama-cli using 6 cores
cmake --build build -j6 --target llama-server llama-cli
```

---

## Model Layout
*   **Main Model:** `/media/nvidia/sd/models/gemma-4-12b-qat-mtp/gemma-4-12B-it-QAT-Q4_0.gguf` (6.5 GB)
*   **Draft Model (MTP):** `/media/nvidia/sd/models/gemma-4-12b-qat-mtp/gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf` (444 MB)
*   **Multimodal Projector:** `/media/nvidia/sd/models/gemma-4-12b-qat-mtp/mmproj-gemma-4-12B-it-QAT-BF16.gguf` (168 MB)

---

## Systemd User Services Configuration

Both services have been updated and reloaded under systemd user config (`~/.config/systemd/user/`).

### 1. Backend Service (`gemma-12b-backend.service`)
```ini
[Unit]
Description=Gemma-4-12b llama-server Backend on Port 18085
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/nvidia/src/llama.cpp
Environment=GGML_CUDA_NO_VMM=1 PYTHONUNBUFFERED=1
ExecStart=/home/nvidia/src/llama.cpp/build/bin/llama-server \
  -m /media/nvidia/sd/models/gemma-4-12b-qat-mtp/gemma-4-12B-it-QAT-Q4_0.gguf \
  --model-draft /media/nvidia/sd/models/gemma-4-12b-qat-mtp/gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 4 \
  --mmproj /media/nvidia/sd/models/gemma-4-12b-qat-mtp/mmproj-gemma-4-12B-it-QAT-BF16.gguf \
  --media-path /tmp \
  --host 0.0.0.0 --port 18085 \
  -c 4096 -b 512 -ub 512 \
  -ngl 99 -ngld 99 \
  --threads 3 --parallel 1 \
  --image-max-tokens 768 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --cache-type-k-draft q4_0 --cache-type-v-draft q4_0 \
  --cache-ram 0 --reasoning off --no-warmup
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
```

### 2. Console Service (`gemma-12b-console.service`)
```ini
[Unit]
Description=Gemma-4-12b Web Console on Port 18091
After=network-online.target gemma-12b-backend.service
Wants=network-online.target gemma-12b-backend.service

[Service]
Type=simple
WorkingDirectory=/media/nvidia/sd/gemma4-12b-qat-mtp-console
Environment=PORT=18091 GEMMA_BACKEND=http://127.0.0.1:18085 APP_TITLE="Gemma 4 12B QAT + MTP Console" MODEL_HINT="gemma-4-12B-it-QAT-Q4_0.gguf + MTP Q8_0" PYTHONUNBUFFERED=1
ExecStart=/usr/bin/python3 /media/nvidia/sd/gemma4-12b-qat-mtp-console/server.py
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
```

### 3. Service Commands
*   **Start services:**
    ```bash
    systemctl --user start gemma-12b-backend.service gemma-12b-console.service
    ```
*   **Check status:**
    ```bash
    systemctl --user status gemma-12b-backend.service gemma-12b-console.service
    ```
*   **Stop services:**
    ```bash
    systemctl --user stop gemma-12b-backend.service gemma-12b-console.service
    ```
