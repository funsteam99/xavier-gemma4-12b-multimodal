# Gemma 4 12B on Jetson Xavier Evaluation

Date: 2026-06-05

## Summary

Gemma 4 12B should be testable on the current Jetson Xavier, but it is a tight fit and should be treated as an experimental bring-up, not a guaranteed daily-driver replacement for the working Gemma 4 E4B setup.

Current decision: keep Gemma 4 12B multimodal as a trial setup only. Text and general multimodal loading work, but audio/ASR should wait for more mature `llama.cpp` support.

Recommended first target:

- `ggml-org/gemma-4-12B-it-GGUF`
- `gemma-4-12B-it-Q4_K_M.gguf`
- `mmproj-gemma-4-12B-it-Q8_0.gguf`
- short context first: `-c 2048`
- single request only: `--parallel 1`
- low output cap during tests: `max_tokens` 64 to 128

## Why It Might Fit

Current Xavier facts from the working E4B setup:

- Visible shared RAM / CUDA memory: about 14 GiB
- Existing E4B Q4_K_M multimodal setup works with CUDA through `llama.cpp`
- E4B text generation is about 4.5 to 5.3 tokens/second
- E4B prompt eval is about 60 to 103 tokens/second

Published 12B GGUF file sizes:

- Q4_K_M model: about 7.38 GB
- Q8_0 model: about 12.7 GB
- bf16 model: about 23.8 GB
- Q8_0 mmproj: about 159 MB
- bf16 mmproj: about 175 MB

The Q4_K_M model plus mmproj leaves room for runtime buffers and KV cache if context is kept modest. Q8_0 and bf16 are not realistic targets on Xavier.

## Main Risk

The current Xavier `llama.cpp` build is:

```text
version: 8881
commit: 0dedb9ef7
```

Gemma 4 12B was released on 2026-06-03, and the official GGUF repository was reconverted within the last day. Community reports mention loader/projector issues around the new 12B unified multimodal projector format. Before testing 12B, update and rebuild `llama.cpp` on Xavier.

## Audio / ASR Hold

The 12B model loads with the multimodal projector, and `llama-server` reports multimodal capability. However, the runtime log explicitly marks audio input as experimental:

```text
init_audio: audio input is in experimental stage and may have reduced quality
```

Observed 12B ASR behavior on Xavier includes repeated trailing phrases after the useful transcription, especially when the audio ends and the model keeps generating. Console-side mitigations were added:

- stricter ASR prompt
- `temperature: 0` for audio tasks
- `repeat_penalty: 1.18` for audio tasks
- ASR default `max_tokens: 160`

These reduce the risk but do not make ASR production-ready. Keep Gemma 4 E4B as the current practical ASR/multimodal baseline, and revisit Gemma 4 12B ASR when `llama.cpp` audio support is no longer experimental or when a newer projector/runtime clearly fixes repetition.

## Expected Performance

Gemma 4 12B has roughly 3x the parameter count of E4B. On Xavier, expect a clear slowdown:

- Text generation: likely around 1.5 to 2.5 tokens/second
- Image/audio prompt processing: likely usable but noticeably slower than E4B
- Long answers: likely frustrating; keep generation short

The win is quality, reasoning, and native image/audio/video input support. The cost is latency.

## Suggested Model Layout

```text
/media/nvidia/sd/models/gemma-4-12b-it/
  gemma-4-12B-it-Q4_K_M.gguf
  mmproj-gemma-4-12B-it-Q8_0.gguf
```

## Dedicated Script

This repo now includes a dedicated Xavier-side script:

```text
scripts/init-xavier-12b.sh
```

It keeps 12B separate from the working E4B service:

- backend: `18085`
- console: `18091`
- console directory: `/media/nvidia/sd/gemma4-12b-console`
- backend log: `/tmp/llama_gemma4_12b.log`
- console log: `/tmp/gemma4_12b_console.log`

Run on Xavier:

```bash
chmod +x scripts/init-xavier-12b.sh
./scripts/init-xavier-12b.sh check
./scripts/init-xavier-12b.sh start
```

## Suggested First Launch

Start conservatively:

```bash
/home/nvidia/src/llama.cpp/build/bin/llama-server \
  -m /media/nvidia/sd/models/gemma-4-12b-it/gemma-4-12B-it-Q4_K_M.gguf \
  --mmproj /media/nvidia/sd/models/gemma-4-12b-it/mmproj-gemma-4-12B-it-Q8_0.gguf \
  --media-path /tmp \
  --host 0.0.0.0 \
  --port 18085 \
  -c 2048 \
  -b 1024 \
  -ub 512 \
  -ngl 99 \
  --threads 3 \
  --parallel 1 \
  --image-max-tokens 512 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --cache-ram 0 \
  --reasoning off \
  --no-warmup
```

If it runs out of memory:

1. Reduce context to `-c 1024`
2. Reduce `-b 512`
3. Reduce `--image-max-tokens 256`
4. Try fewer GPU layers, for example `-ngl 40`, then `-ngl 24`

If image input hits the same non-causal attention batch assertion seen with E4B, raise `-ub` back to `1024`.

## Test Order

1. Text health: `/health` and `/v1/models`
2. Short text prompt with `max_tokens: 32`
3. Longer text prompt with `max_tokens: 128`
4. Small image OCR with `--image-max-tokens 256`
5. WAV or MP3 audio ASR
6. Combined image + audio

Record:

- load success or error
- peak memory from `tegrastats`
- prompt eval tokens/second
- generation tokens/second
- image/audio preprocessing time
- output quality compared with E4B

## Recommendation

Proceed with a controlled trial. Keep the existing E4B service intact on port `18084`, run 12B separately on port `18085`, and only promote it if it loads reliably and produces materially better answers for Xavier's target tasks.

For daily interactive use on Xavier, E4B will probably remain the faster default. Gemma 4 12B is worth testing for harder OCR, reasoning, code, and multi-step image understanding where slower output is acceptable. Do not treat 12B ASR/audio as the default path yet.
