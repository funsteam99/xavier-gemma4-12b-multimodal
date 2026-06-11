import urllib.request
import json
import base64
import time
import os

URL = "http://127.0.0.1:18085/v1/chat/completions"

def run_completion(payload, desc=""):
    print(f"\n=== Running {desc} ===")
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(URL, data=json.dumps(payload).encode('utf-8'), headers=headers)
    
    t0 = time.time()
    try:
        with urllib.request.urlopen(req) as response:
            res_data = response.read().decode('utf-8')
            t1 = time.time()
            res = json.loads(res_data)
            
            # Print response
            print(f"Status Code: {response.status}")
            if "choices" in res and len(res["choices"]) > 0:
                msg = res["choices"][0]["message"]
                print(f"Response Content:\n{msg.get('content', '')}")
                if "reasoning_content" in msg:
                    print(f"Reasoning:\n{msg['reasoning_content']}")
            
            # Print stats
            dt = t1 - t0
            print(f"Time taken: {dt:.2f} seconds")
            if "usage" in res:
                usage = res["usage"]
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                print(f"Prompt tokens: {prompt_tokens}")
                print(f"Completion tokens: {completion_tokens}")
                if completion_tokens > 0:
                    print(f"Gen speed: {completion_tokens / dt:.2f} t/s (rough estimation)")
            return res
    except Exception as e:
        print(f"Error running {desc}: {e}")
        if hasattr(e, 'read'):
            try:
                print(e.read().decode('utf-8'))
            except Exception:
                pass
        return None

# Test 1: Simple Text
text_payload = {
    "model": "gemma-4-12b-qat-mtp",
    "messages": [
        {"role": "user", "content": "Hi! Please tell me what architecture you are based on in one short sentence."}
    ],
    "temperature": 0.0,
    "max_tokens": 64,
    "stream": False
}
run_completion(text_payload, "Text Baseline")

# Test 2: Image OCR (if exists)
img_path = "/home/nvidia/Pictures/Screenshot from 2024-12-31 08-39-24.png"
if os.path.exists(img_path):
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode('utf-8')
    image_payload = {
        "model": "gemma-4-12b-qat-mtp",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe what text is visible in this image."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.0,
        "max_tokens": 128,
        "stream": False
    }
    run_completion(image_payload, "Image OCR")
else:
    print(f"Image not found at {img_path}")

# Test 3: Audio ASR (if exists)
audio_path = "/media/nvidia/sd/ggml-breeze-asr-26-webui/third_party/whisper.cpp/samples/jfk.wav"
if os.path.exists(audio_path):
    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode('utf-8')
    audio_payload = {
        "model": "gemma-4-12b-qat-mtp",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe this audio in English:"},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_b64,
                            "format": "wav"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.0,
        "max_tokens": 128,
        "stream": False
    }
    run_completion(audio_payload, "Audio ASR")
else:
    print(f"Audio not found at {audio_path}")
