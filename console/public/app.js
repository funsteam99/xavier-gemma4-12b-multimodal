const el = (id) => document.getElementById(id);

const presets = {
  asr: "請逐字轉寫音訊為繁體中文。只輸出實際聽到的內容，不要摘要、不要改寫、不要補充背景知識、不要自行延伸。音訊結束就停止；如果後面沒有聲音或聽不清楚，請不要重複前面的詞句。若有不確定的詞，請以 [不確定] 標記。",
  text: "請用繁體中文簡短回答：你目前在 Jetson Xavier 上透過 llama.cpp 執行。請列出你能做的三種測試。",
  ocr: "請讀取圖片中的所有可見文字，逐行輸出；如果不確定，請標記為 unsure。",
  chart: "請描述這張圖片或介面的主要內容，指出關鍵數字、狀態、異常或可操作的觀察。",
  av: "請同時參考圖片和音訊，整理使用者想表達的重點。若音訊是在描述圖片，請指出兩者是否一致。",
  free: "",
};

const presetDefaults = {
  asr: { maxTokens: 160 },
  text: { maxTokens: 128 },
  ocr: { maxTokens: 256 },
  chart: { maxTokens: 256 },
  av: { maxTokens: 256 },
  free: { maxTokens: 256 },
};

const state = {
  imageData: null,
  imageName: null,
  audioData: null,
  audioName: null,
  audioFormat: null,
  audioUrl: null,
  modelId: "gemma-4-12B-it-QAT-Q4_0.gguf",
  contextLimit: 4096,
};

const imageTypes = new Set(["image/png", "image/jpeg", "image/jpg", "image/webp"]);

// Navigation Tabs Setup
function initTabs() {
  const tabs = [
    { btn: el("btnConsole"), pane: el("tabConsole") },
    { btn: el("btnComparison"), pane: el("tabComparison") },
    { btn: el("btnLogs"), pane: el("tabLogs") }
  ];

  tabs.forEach(tab => {
    tab.btn.addEventListener("click", () => {
      tabs.forEach(t => {
        t.btn.classList.remove("active");
        t.pane.classList.remove("active");
      });
      tab.btn.classList.add("active");
      tab.pane.classList.add("active");
    });
  });
}

// Token Estimation
function estimateTokens(text) {
  if (!text) return 0;
  // Estimate Chinese characters (1 token each)
  const chineseChars = text.match(/[\u4e00-\u9fa5\u3000-\u303f\uff00-\uffef]/g) || [];
  const cleanText = text.replace(/[\u4e00-\u9fa5\u3000-\u303f\uff00-\uffef]/g, " ");
  // Estimate non-Chinese words (approx words * 1.3)
  const words = cleanText.trim().split(/\s+/).filter(Boolean);
  return chineseChars.length + Math.round(words.length * 1.3);
}

function updateTokenEstimate() {
  const promptText = el("prompt").value;
  let tokens = estimateTokens(promptText);
  
  if (state.imageData) {
    tokens += 768; // llama.cpp image max tokens
  }
  
  if (state.audioData) {
    tokens += 200; // estimated audio tokens
  }
  
  el("tokenEstimate").textContent = tokens;
  
  const limit = state.contextLimit;
  if (tokens >= limit * 0.9) {
    el("tokenEstimate").style.color = "var(--danger)";
    el("tokenWarning").style.display = "block";
  } else if (tokens >= limit * 0.7) {
    el("tokenEstimate").style.color = "var(--accent-orange)";
    el("tokenWarning").style.display = "none";
  } else {
    el("tokenEstimate").style.color = "var(--accent)";
    el("tokenWarning").style.display = "none";
  }
}

// ASR Repetition Filtering
function filterASRRepetitions(text, isASR) {
  if (!isASR) return { truncated: false, text };
  
  const len = text.length;
  // Look for consecutive repeats of size 10 to 150 characters
  for (let size = 10; size <= Math.min(150, len / 2); size++) {
    const suffix = text.slice(-size);
    const prev = text.slice(-2 * size, -size);
    if (suffix === prev) {
      let cleanText = text.slice(0, -size);
      // Clean recursively if it repeated even more times
      while (cleanText.slice(-size) === suffix) {
        cleanText = cleanText.slice(0, -size);
      }
      return {
        truncated: true,
        text: cleanText.trim() + " ... [已自動截斷重複語句]"
      };
    }
  }
  
  // Check for short repeats repeating 3+ times (e.g. 4-9 chars)
  for (let size = 4; size < 10; size++) {
    if (len >= size * 3) {
      const suffix = text.slice(-size);
      const prev1 = text.slice(-2 * size, -size);
      const prev2 = text.slice(-3 * size, -2 * size);
      if (suffix === prev1 && suffix === prev2) {
        let cleanText = text.slice(0, -2 * size);
        while (cleanText.slice(-size) === suffix) {
          cleanText = cleanText.slice(0, -size);
        }
        return {
          truncated: true,
          text: cleanText.trim() + " ... [已自動截斷重複語句]"
        };
      }
    }
  }
  
  return { truncated: false, text };
}

// Test History Log Management
const LOG_KEY = "xavier_bench_logs";
let runHistory = JSON.parse(localStorage.getItem(LOG_KEY) || "[]");

function saveHistory() {
  localStorage.setItem(LOG_KEY, JSON.stringify(runHistory));
  renderHistoryTable();
}

function addHistoryItem(preset, prompt, image, audio, latency, promptTok, outputTok, speed, output) {
  const item = {
    id: Date.now(),
    time: new Date().toLocaleTimeString(),
    date: new Date().toLocaleDateString(),
    preset: preset,
    prompt: prompt || "(None)",
    media: [image, audio].filter(Boolean).join(", ") || "None",
    latency: latency,
    tokens: `${promptTok} / ${outputTok}`,
    speed: speed,
    output: output
  };
  runHistory.unshift(item);
  saveHistory();
}

function renderHistoryTable() {
  const tbody = el("logTableBody");
  if (runHistory.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" class="empty-state">尚無測試歷史紀錄。</td></tr>`;
    return;
  }
  
  tbody.innerHTML = runHistory.map(item => `
    <tr>
      <td>${item.date}<br><small>${item.time}</small></td>
      <td><code>${item.preset}</code></td>
      <td title="${item.prompt}">${item.prompt.length > 40 ? item.prompt.substring(0, 38) + '...' : item.prompt}</td>
      <td>${item.media}</td>
      <td>${item.latency}</td>
      <td>${item.tokens}</td>
      <td><strong>${item.speed}</strong></td>
      <td>
        <button class="btn btn-delete" onclick="deleteHistoryItem(${item.id})">刪除</button>
      </td>
    </tr>
  `).join("");
}

window.deleteHistoryItem = function(id) {
  runHistory = runHistory.filter(item => item.id !== id);
  saveHistory();
};

function clearLogs() {
  if (confirm("確定要清除所有測試歷史紀錄嗎？")) {
    runHistory = [];
    saveHistory();
  }
}

function exportLogs() {
  if (runHistory.length === 0) {
    alert("沒有可導出的測試記錄！");
    return;
  }
  const blob = new Blob([JSON.stringify(runHistory, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `xavier_bench_logs_${Date.now()}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// System Status and Config Loads
function setHealth(ok, text, latency) {
  el("healthText").textContent = text;
  el("healthText").className = ok ? "ok" : "bad";
  el("latencyText").textContent = latency == null ? "-" : `${latency} ms`;
}

function setBusy(isBusy) {
  document.querySelectorAll("button, select, input, textarea").forEach((node) => {
    node.disabled = isBusy;
  });
}

function renderRaw(payload) {
  el("rawJson").textContent = JSON.stringify(payload, null, 2);
}

function showMessage(text, kind = "warn") {
  el("answer").textContent = text;
  el("answer").className = `answer ${kind}`;
}

async function postJson(path, body = {}) {
  const started = performance.now();
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  data.client_latency_ms = Math.round(performance.now() - started);
  return data;
}

async function loadConfig() {
  try {
    const config = await postJson("/api/config");
    if (config.app_title) {
      document.title = config.app_title;
      el("titleText").textContent = config.app_title;
    }
    if (config.model_hint) {
      state.modelId = config.model_hint;
      el("modelText").textContent = config.model_hint;
    }
  } catch (error) {
    renderRaw({ ok: false, error: String(error) });
  }
}

async function checkBackend() {
  try {
    const health = await postJson("/api/health");
    setHealth(Boolean(health.ok), health.ok ? "ok" : "error", health.latency_ms);
    renderRaw(health);

    if (health.slots && health.slots.length > 0) {
      const activeSlot = health.slots[0];
      if (activeSlot.n_ctx) {
        state.contextLimit = activeSlot.n_ctx;
        el("ctxLimit").textContent = activeSlot.n_ctx;
        updateTokenEstimate();
      }
    }

    const models = await postJson("/api/models");
    const model = models?.data?.data?.[0]?.id;
    if (model) {
      state.modelId = model.split("/").pop();
      el("modelText").textContent = state.modelId;
    }
  } catch (error) {
    setHealth(false, "offline", null);
    renderRaw({ ok: false, error: String(error) });
  }
}

// Media Preprocessing and loading
function readDataUrl(file, callback) {
  const reader = new FileReader();
  reader.onload = () => callback(String(reader.result));
  reader.readAsDataURL(file);
}

function imageOk(file) {
  const name = file.name.toLowerCase();
  return imageTypes.has(file.type) || name.endsWith(".png") || name.endsWith(".jpg") || name.endsWith(".jpeg") || name.endsWith(".webp");
}

function audioFormatFor(file) {
  const name = file.name.toLowerCase();
  if (file.type.includes("wav") || name.endsWith(".wav")) return "wav";
  if (file.type.includes("mpeg") || file.type.includes("mp3") || name.endsWith(".mp3")) return "mp3";
  return null;
}

function clearImage() {
  state.imageData = null;
  state.imageName = null;
  el("imageInput").value = "";
  el("preview").hidden = true;
  el("preview").removeAttribute("src");
  el("imageTitle").textContent = "選擇圖片";
  updateTokenEstimate();
}

function clearAudio() {
  state.audioData = null;
  state.audioName = null;
  state.audioFormat = null;
  el("audioInput").value = "";
  if (state.audioUrl) URL.revokeObjectURL(state.audioUrl);
  state.audioUrl = null;
  el("audioPreview").hidden = true;
  el("audioPreview").removeAttribute("src");
  el("audioTitle").textContent = "選擇音訊";
  updateTokenEstimate();
}

function convertWebpToPng(file, callback) {
  const img = new Image();
  const reader = new FileReader();
  reader.onload = (e) => {
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0);
      const pngDataUrl = canvas.toDataURL("image/png");
      callback(pngDataUrl);
    };
    img.onerror = () => {
      callback(null);
    };
    img.src = String(e.target.result);
  };
  reader.readAsDataURL(file);
}

function loadImage(file) {
  if (!file) return;
  if (!imageOk(file)) {
    showMessage(`不支援的圖片格式：${file.name}。請使用 PNG / JPG / WebP。`);
    return;
  }

  const isWebp = file.type === "image/webp" || file.name.toLowerCase().endsWith(".webp");

  if (isWebp) {
    showMessage("偵測到 WebP 格式，正在瀏覽器端自動轉換為 PNG...", "warn");
    convertWebpToPng(file, (pngDataUrl) => {
      if (!pngDataUrl) {
        showMessage(`WebP 轉換失敗：${file.name}，請改用 PNG 或 JPG。`, "bad");
        return;
      }
      state.imageData = pngDataUrl;
      state.imageName = file.name.replace(/\.webp$/i, ".png");
      el("preview").src = pngDataUrl;
      el("preview").hidden = false;
      el("imageTitle").textContent = file.name + " (已自動轉 PNG)";
      showMessage(`WebP 圖片已成功轉碼為 PNG：${file.name}`, "ok");
      updateTokenEstimate();
    });
  } else {
    readDataUrl(file, (dataUrl) => {
      state.imageData = dataUrl;
      state.imageName = file.name;
      el("preview").src = dataUrl;
      el("preview").hidden = false;
      el("imageTitle").textContent = file.name;
      showMessage(`圖片已載入：${file.name}`, "ok");
      updateTokenEstimate();
    });
  }
}

async function preprocessAudio(file) {
  const AudioContext = window.AudioContext || window.webkitAudioContext;
  if (!AudioContext) {
    throw new Error("Web Audio API is not supported in this browser");
  }
  const audioCtx = new AudioContext();
  const arrayBuffer = await file.arrayBuffer();
  
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
  
  const targetSampleRate = 16000;
  const offlineCtx = new OfflineAudioContext(1, Math.round(audioBuffer.duration * targetSampleRate), targetSampleRate);
  
  const source = offlineCtx.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(offlineCtx.destination);
  source.start();
  
  const resampledBuffer = await offlineCtx.startRendering();
  const pcmData = resampledBuffer.getChannelData(0);
  
  let maxVal = 0;
  for (let i = 0; i < pcmData.length; i++) {
    const val = Math.abs(pcmData[i]);
    if (val > maxVal) maxVal = val;
  }
  if (maxVal > 0) {
    const scale = 0.89 / maxVal;
    for (let i = 0; i < pcmData.length; i++) {
      pcmData[i] *= scale;
    }
  }
  
  return encodeWav(pcmData, targetSampleRate);
}

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  
  const writeString = (view, offset, string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };
  
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); 
  view.setUint16(22, 1, true); 
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); 
  view.setUint16(32, 2, true); 
  view.setUint16(34, 16, true); 
  writeString(view, 36, 'data');
  view.setUint32(40, samples.length * 2, true);
  
  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }
  
  return new Blob([view], { type: 'audio/wav' });
}

async function loadAudio(file) {
  if (!file) return;
  
  showMessage(`正在預處理音訊（解碼、重採樣至 16kHz 單聲道、音量標準化）：${file.name}...`, "warn");
  
  try {
    const wavBlob = await preprocessAudio(file);
    
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = String(reader.result);
      state.audioData = dataUrl.split(",", 2)[1] || "";
      state.audioName = file.name;
      state.audioFormat = "wav"; 
      
      if (state.audioUrl) URL.revokeObjectURL(state.audioUrl);
      state.audioUrl = URL.createObjectURL(wavBlob);
      el("audioPreview").src = state.audioUrl;
      el("audioPreview").hidden = false;
      el("audioTitle").textContent = file.name + " (已優化轉檔)";
      showMessage(`音訊預處理完成並已載入：${file.name}`, "ok");
      updateTokenEstimate();
    };
    reader.readAsDataURL(wavBlob);
  } catch (err) {
    console.error(err);
    showMessage(`音訊解碼/預處理失敗 (${err.message})，改用直接載入模式...`, "warn");
    
    const format = audioFormatFor(file);
    if (!format) {
      showMessage(`不支援的音訊格式：${file.name}。llama.cpp input_audio 目前請使用 WAV 或 MP3。`);
      return;
    }
    readDataUrl(file, (dataUrl) => {
      state.audioData = dataUrl.split(",", 2)[1] || "";
      state.audioName = file.name;
      state.audioFormat = format;
      if (state.audioUrl) URL.revokeObjectURL(state.audioUrl);
      state.audioUrl = URL.createObjectURL(file);
      el("audioPreview").src = state.audioUrl;
      el("audioPreview").hidden = false;
      el("audioTitle").textContent = file.name;
      showMessage(`音訊已直接載入（未預處理）：${file.name}`, "ok");
      updateTokenEstimate();
    });
  }
}

// Payload Construction
function buildPayload() {
  const prompt = el("prompt").value.trim();
  const preset = el("preset").value;
  const isAudioTask = Boolean(state.audioData) || preset === "asr";
  const content = [{ type: "text", text: prompt || presets.asr }];
  if (state.imageData) content.push({ type: "image_url", image_url: { url: state.imageData } });
  if (state.audioData) content.push({ type: "input_audio", input_audio: { data: state.audioData, format: state.audioFormat } });
  return {
    model: state.modelId,
    messages: [{ role: "user", content }],
    temperature: isAudioTask ? 0 : 0.2,
    repeat_penalty: isAudioTask ? 1.18 : 1.08,
    max_tokens: Number(el("maxTokens").value || 256),
    stream: true,
    stream_options: { include_usage: true },
  };
}

// Run Test
async function runTest(event) {
  event.preventDefault();
  const preset = el("preset").value;
  if (!state.imageData && !state.audioData && preset === "asr") {
    showMessage("請先載入 WAV 或 MP3，再執行 ASR 測試。");
    return;
  }

  setBusy(true);
  el("answer").className = "answer";
  el("answer").textContent = "";
  el("responseTime").textContent = "-";
  el("promptTokens").textContent = "-";
  el("outputTokens").textContent = "-";
  el("tokRate").textContent = "-";
  el("repeatStatus").style.display = "none";

  const started = performance.now();
  let choiceText = "";
  let latencyText = "-";
  let promptTokens = "-";
  let completionTokens = "-";
  let rateText = "-";
  
  try {
    const payload = buildPayload();
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ payload }),
    });

    if (!res.ok) {
      const text = await res.text();
      let errPayload;
      try {
        errPayload = JSON.parse(text);
      } catch (e) {}
      throw new Error(errPayload?.error || `HTTP error ${res.status}: ${text}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    let lastData = null;
    let finalUsage = null;
    let finalTimings = null;
    let isASRRepetitionDetected = false;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop(); 

      for (const line of lines) {
        const cleaned = line.trim();
        if (!cleaned) continue;
        if (cleaned.startsWith("data: ")) {
          const dataStr = cleaned.slice(6).trim();
          if (dataStr === "[DONE]") continue;
          try {
            const parsed = JSON.parse(dataStr);
            lastData = parsed;
            if (parsed.usage) finalUsage = parsed.usage;
            if (parsed.timings) finalTimings = parsed.timings;
            const delta = parsed?.choices?.[0]?.delta;
            if (delta && delta.content) {
              choiceText += delta.content;
              
              // Run real-time repetition detection
              const filterResult = filterASRRepetitions(choiceText, preset === "asr" || Boolean(state.audioData));
              if (filterResult.truncated) {
                choiceText = filterResult.text;
                el("answer").textContent = choiceText;
                el("repeatStatus").style.display = "inline-block";
                isASRRepetitionDetected = true;
                // Cancel reader and break
                reader.cancel();
                break;
              }
              el("answer").textContent = choiceText;
            }
          } catch (e) {
            // Ignore incomplete JSON chunks
          }
        }
      }
      if (isASRRepetitionDetected) break;
    }

    const latency = Math.round(performance.now() - started);
    latencyText = `${latency} ms`;
    el("responseTime").textContent = latencyText;

    const usage = finalUsage || lastData?.usage || {};
    const timings = finalTimings || lastData?.timings || {};

    promptTokens = usage.prompt_tokens ?? timings.prompt_n ?? "-";
    completionTokens = usage.completion_tokens ?? timings.predicted_n ?? "-";
    
    if (timings.predicted_per_second) {
      rateText = timings.predicted_per_second.toFixed(2);
    } else if (completionTokens !== "-" && completionTokens !== 0 && latency) {
      rateText = (completionTokens / (latency / 1000)).toFixed(2);
    }

    el("promptTokens").textContent = promptTokens;
    el("outputTokens").textContent = completionTokens;
    el("tokRate").textContent = rateText;

    renderRaw(lastData || { ok: true });
    setHealth(true, "ok", latency);
    
    // Add to history
    addHistoryItem(
      preset,
      el("prompt").value.trim(),
      state.imageName,
      state.audioName,
      latencyText,
      promptTokens,
      completionTokens,
      rateText,
      choiceText
    );

  } catch (error) {
    showMessage(String(error), "bad");
    renderRaw({ ok: false, error: String(error) });
    setHealth(false, "error", null);
  } finally {
    setBusy(false);
  }
}

function wireDrop(zoneId, loader) {
  const zone = el(zoneId);
  zone.addEventListener("dragover", (event) => {
    event.preventDefault();
    zone.style.borderColor = "var(--accent)";
  });
  zone.addEventListener("dragleave", () => {
    zone.style.borderColor = "var(--border)";
  });
  zone.addEventListener("drop", (event) => {
    event.preventDefault();
    zone.style.borderColor = "var(--border)";
    loader(event.dataTransfer.files[0]);
  });
}

// Event Listeners Wire-up
function initEvents() {
  el("preset").addEventListener("change", () => {
    const preset = el("preset").value;
    el("prompt").value = presets[preset];
    if (presetDefaults[preset]) {
      el("maxTokens").value = presetDefaults[preset].maxTokens;
    }
    updateTokenEstimate();
  });
  
  el("prompt").addEventListener("input", updateTokenEstimate);
  el("imageInput").addEventListener("change", (event) => loadImage(event.target.files[0]));
  el("audioInput").addEventListener("change", (event) => loadAudio(event.target.files[0]));
  el("clearImage").addEventListener("click", clearImage);
  el("clearAudio").addEventListener("click", clearAudio);
  
  el("clearAll").addEventListener("click", () => {
    el("preset").value = "asr";
    el("prompt").value = presets.asr;
    el("maxTokens").value = presetDefaults.asr.maxTokens;
    clearImage();
    clearAudio();
    renderRaw({});
    el("answer").className = "answer";
    el("answer").textContent = "準備測試。";
    el("repeatStatus").style.display = "none";
    updateTokenEstimate();
  });
  
  el("checkBtn").addEventListener("click", checkBackend);
  el("testForm").addEventListener("submit", runTest);
  
  el("btnExportLogs").addEventListener("click", exportLogs);
  el("btnClearLogs").addEventListener("click", clearLogs);
  
  wireDrop("imageDropzone", loadImage);
  wireDrop("audioDropzone", loadAudio);
}

// Init execution
initTabs();
initEvents();
renderHistoryTable();

el("prompt").value = presets.asr;
el("maxTokens").value = presetDefaults.asr.maxTokens;
updateTokenEstimate();

loadConfig().then(checkBackend);
