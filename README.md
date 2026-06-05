# Gemma 4 E4B Multimodal on Jetson Xavier

This repo documents a working Gemma 4 E4B multimodal setup on an NVIDIA Jetson Xavier using `llama.cpp`.

The setup runs:

- Gemma 4 E4B instruction model in GGUF format
- Gemma 4 multimodal projector with image and audio encoder
- `llama-server` with CUDA enabled
- A lightweight browser test console for text, image, audio, ASR, and image+audio tests

## Hardware

- Device: NVIDIA Jetson Xavier
- Hostname: `ubuntu.local`
- User: `nvidia`
- RAM visible to system: about 14 GiB
- CUDA device detected by llama.cpp:

```text
Device 0: Xavier, compute capability 7.2, VRAM: 14886 MiB
```

## Storage

The model files are stored on the SD card:

```text
/media/nvidia/sd/models/gemma-4-e4b-it/
```

Known working files:

```text
gemma-4-E4B-it-Q4_K_M.gguf
mmproj-gemma-4-E4B-it-Q8_0.gguf
```

The SD card was mounted at:

```text
/media/nvidia/sd
```

## llama.cpp Version

The working build is:

```text
repo: /home/nvidia/src/llama.cpp
commit: 0dedb9ef7
version: 8881
commit message: hexagon: add support for FILL op (#22198)
build: GNU 10.5.0, Linux aarch64
```

CUDA is enabled:

```text
GGML_CUDA: ON
CMAKE_CUDA_ARCHITECTURES: 72
CUDA toolkit: 11.4.315
```

Runtime logs confirmed CUDA usage:

```text
ggml_cuda_init: found 1 CUDA devices
using device CUDA0 (Xavier)
offloading 32 repeating layers to GPU
offloaded 33/43 layers to GPU
clip_ctx: CLIP using CUDA0 backend
Flash Attention was auto, set to enabled
```

## Backend

The backend runs on port `18084`:

```bash
/home/nvidia/src/llama.cpp/build/bin/llama-server \
  -m /media/nvidia/sd/models/gemma-4-e4b-it/gemma-4-E4B-it-Q4_K_M.gguf \
  --mmproj /media/nvidia/sd/models/gemma-4-e4b-it/mmproj-gemma-4-E4B-it-Q8_0.gguf \
  --media-path /tmp \
  --host 0.0.0.0 \
  --port 18084 \
  -c 4096 \
  -b 2048 \
  -ub 1024 \
  -ngl 33 \
  --threads 3 \
  --parallel 1 \
  --image-max-tokens 768 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --cache-ram 0 \
  --reasoning off \
  --no-warmup
```

Health check:

```bash
curl http://127.0.0.1:18084/health
```

Expected response:

```json
{"status":"ok"}
```

Model capability check:

```bash
curl http://127.0.0.1:18084/v1/models
```

The model reports both completion and multimodal capability.

## Initialization Script

This repo includes an Xavier-side initialization script:

```text
scripts/init-xavier.sh
```

After cloning the repo on Xavier, run:

```bash
chmod +x scripts/init-xavier.sh
./scripts/init-xavier.sh check
./scripts/init-xavier.sh start
```

The `start` command:

- validates `llama.cpp`, model, and mmproj paths
- copies the console files to `/media/nvidia/sd/gemma4-e4b-console`
- starts or restarts `llama-server` on port `18084`
- starts or restarts the console on port `18090`
- prints backend and console health checks

Useful commands:

```bash
./scripts/init-xavier.sh status
./scripts/init-xavier.sh stop
./scripts/init-xavier.sh start-backend
./scripts/init-xavier.sh start-console
```

The script defaults match the working Xavier setup. Paths can be overridden:

```bash
MODEL=/path/to/model.gguf \
MMPROJ=/path/to/mmproj.gguf \
LLAMA_CPP_DIR=/home/nvidia/src/llama.cpp \
./scripts/init-xavier.sh start
```

## Test Console

A small test console runs on Xavier at:

```text
http://192.168.0.124:18090
```

This repo includes the console source under:

```text
console/
├── server.py
└── public/
    ├── index.html
    ├── styles.css
    └── app.js
```

Remote files:

```text
/media/nvidia/sd/gemma4-e4b-console/server.py
/media/nvidia/sd/gemma4-e4b-console/public/index.html
/media/nvidia/sd/gemma4-e4b-console/public/styles.css
/media/nvidia/sd/gemma4-e4b-console/public/app.js
```

Logs:

```text
/tmp/gemma4_e4b_console.log
/tmp/llama_gemma4_e4b.log
```

The console supports:

- Text baseline prompts
- Image OCR
- Chart and UI understanding
- ASR audio transcription
- Image + audio multimodal prompts
- Raw JSON inspection
- Timing and token usage display

The console proxies browser requests to `llama-server`, avoiding browser-side CORS issues.

To run the console on Xavier:

```bash
cd /media/nvidia/sd/gemma4-e4b-console
python3 server.py
```

By default, it listens on:

```text
0.0.0.0:18090
```

and proxies to:

```text
http://127.0.0.1:18084
```

## Multimodal Request Formats

Image input uses OpenAI-compatible `image_url` content parts:

```json
{
  "type": "image_url",
  "image_url": {
    "url": "data:image/png;base64,..."
  }
}
```

Audio input uses `input_audio` content parts:

```json
{
  "type": "input_audio",
  "input_audio": {
    "data": "...base64...",
    "format": "wav"
  }
}
```

The tested llama.cpp build accepts audio formats:

- `wav`
- `mp3`

It does not accept `webm`, `m4a`, or arbitrary browser recording formats through this path.

## Important Fixes

### Image Batch Assert

An image upload originally crashed `llama-server` with:

```text
GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) && "non-causal attention requires n_ubatch >= n_tokens") failed
```

The image produced about `748` image tokens while the default physical batch size was `512`.

Fix:

```bash
-ub 1024
```

### Invalid Media Decode

Some failed uploads returned:

```json
{
  "error": {
    "code": 400,
    "message": "Failed to load image or audio file",
    "type": "invalid_request_error"
  }
}
```

The backend log showed:

```text
mtmd_helper_bitmap_init_from_buf: failed to decode image bytes
```

This happened when non-image bytes were sent as an `image_url` content part, such as dragging audio into the image upload area or using an unsupported image format.

Fixes added to the test console:

- Image upload accepts only PNG, JPG, JPEG, or WebP
- Audio upload accepts only WAV or MP3
- Dragging files into the wrong zone is blocked before sending to `llama-server`

### ASR Support

The multimodal projector reports:

```text
has vision encoder
has audio encoder
```

Audio input is experimental in llama.cpp:

```text
init_audio: audio input is in experimental stage and may have reduced quality
```

ASR works through the same chat completions endpoint by sending an `input_audio` content part and prompting the model to transcribe.

## Observed Performance

This setup is usable but slow on Xavier.

Representative timings:

```text
Image processing: about 2 to 6.7 seconds
Audio processing: about 1.5 to 3.6 seconds
Prompt eval: about 60 to 103 tokens/second, depending on multimodal token count
Text generation: about 4.5 to 5.3 tokens/second
```

Long answers are the slowest part. Generating 256 tokens can take close to one minute.

Practical tuning options:

- Reduce `max_tokens`
- Use smaller images
- Lower `--image-max-tokens`
- Keep `--parallel 1`
- Keep `-ub 1024` for larger image inputs
- Consider smaller or more aggressively quantized models for faster interaction

## Quick Test Payloads

Text:

```bash
curl http://127.0.0.1:18084/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-E4B-it-Q4_K_M.gguf",
    "messages": [
      {
        "role": "user",
        "content": "請用繁體中文簡短回答：你正在 Jetson Xavier 上運作嗎？"
      }
    ],
    "max_tokens": 64,
    "temperature": 0.2
  }'
```

Image and audio requests are easiest to test through the browser console because it converts local files to base64.

## Current Status

Working:

- CUDA-enabled llama.cpp build
- Gemma 4 E4B text generation
- Image input
- Audio input
- ASR-style transcription prompts
- Image + audio combined prompts
- Browser test console on Xavier

Known limitations:

- Slow token generation on Xavier
- Audio support is experimental in llama.cpp
- Audio upload is limited to WAV and MP3
- Image upload is limited to PNG, JPG, JPEG, and WebP
- Very large images increase image token count and latency
