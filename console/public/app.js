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
  modelId: "gemma-4-E4B-it-Q4_K_M.gguf",
};

const imageTypes = new Set(["image/png", "image/jpeg", "image/jpg", "image/webp"]);

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
}

function loadImage(file) {
  if (!file) return;
  if (!imageOk(file)) {
    showMessage(`不支援的圖片格式：${file.name}。請使用 PNG / JPG / WebP。`);
    return;
  }
  readDataUrl(file, (dataUrl) => {
    state.imageData = dataUrl;
    state.imageName = file.name;
    el("preview").src = dataUrl;
    el("preview").hidden = false;
    el("imageTitle").textContent = file.name;
    showMessage(`圖片已載入：${file.name}`, "ok");
  });
}

function loadAudio(file) {
  if (!file) return;
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
    showMessage(`音訊已載入：${file.name}`, "ok");
  });
}

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
    stream: false,
  };
}

async function runTest(event) {
  event.preventDefault();
  if (!state.imageData && !state.audioData && el("preset").value === "asr") {
    showMessage("請先載入 WAV 或 MP3，再執行 ASR 測試。");
    return;
  }

  setBusy(true);
  el("answer").className = "answer";
  el("answer").textContent = "Running...";
  el("responseTime").textContent = "-";
  el("promptTokens").textContent = "-";
  el("outputTokens").textContent = "-";
  el("tokRate").textContent = "-";

  try {
    const data = await postJson("/api/chat", { payload: buildPayload() });
    renderRaw(data);
    const choice = data?.data?.choices?.[0]?.message?.content;
    const usage = data?.data?.usage || {};
    const latency = data.latency_ms || data.client_latency_ms;
    el("answer").textContent = choice || JSON.stringify(data?.data || data, null, 2);
    el("responseTime").textContent = `${latency} ms`;
    el("promptTokens").textContent = usage.prompt_tokens ?? "-";
    el("outputTokens").textContent = usage.completion_tokens ?? "-";
    el("tokRate").textContent = usage.completion_tokens && latency ? (usage.completion_tokens / (latency / 1000)).toFixed(2) : "-";
    setHealth(Boolean(data.ok), data.ok ? "ok" : "error", data.latency_ms);
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
  zone.addEventListener("dragover", (event) => event.preventDefault());
  zone.addEventListener("drop", (event) => {
    event.preventDefault();
    loader(event.dataTransfer.files[0]);
  });
}

el("preset").addEventListener("change", () => {
  const preset = el("preset").value;
  el("prompt").value = presets[preset];
  if (presetDefaults[preset]) {
    el("maxTokens").value = presetDefaults[preset].maxTokens;
  }
});
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
});
el("checkBtn").addEventListener("click", checkBackend);
el("testForm").addEventListener("submit", runTest);
wireDrop("imageDropzone", loadImage);
wireDrop("audioDropzone", loadAudio);

el("prompt").value = presets.asr;
el("maxTokens").value = presetDefaults.asr.maxTokens;
loadConfig().then(checkBackend);
