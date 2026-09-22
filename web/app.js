const SAMPLE_RATE = 16000;
const WINDOW_SECONDS = 2;
const STEP_SECONDS = 0.5;
const WINDOW_SIZE = SAMPLE_RATE * WINDOW_SECONDS;
const VEHICLE_ENERGY_RATIO = 0.018;
const FREQUENCY_RISE_HZ = 40;

const state = {
  audioContext: null,
  stream: null,
  source: null,
  processor: null,
  buffer: new Float32Array(0),
  previousFrequency: null,
  previousEnergy: null,
  vehicleStreak: 0,
  windows: 0,
  history: [],
};

const $ = (id) => document.getElementById(id);

function setStatus(level, label, reason) {
  const card = $("statusCard");
  card.className = `card status ${level}`;
  $("statusLabel").textContent = label;
  $("statusReason").textContent = reason;
}

function dominantFrequency(samples, sampleRate) {
  let bestFrequency = 0;
  let bestMagnitude = 0;
  for (let k = 1; k < samples.length / 2; k += 2) {
    let real = 0;
    let imag = 0;
    for (let n = 0; n < samples.length; n += 8) {
      const angle = (2 * Math.PI * k * n) / samples.length;
      real += samples[n] * Math.cos(angle);
      imag -= samples[n] * Math.sin(angle);
    }
    const magnitude = real * real + imag * imag;
    if (magnitude > bestMagnitude) {
      bestMagnitude = magnitude;
      bestFrequency = (k * sampleRate) / samples.length;
    }
  }
  return bestFrequency;
}

function analyzeWindow(samples, sourceLabel) {
  const energy = Math.sqrt(samples.reduce((sum, value) => sum + value * value, 0) / samples.length);
  const frequency = dominantFrequency(samples, SAMPLE_RATE);
  const frequencyRise = state.previousFrequency == null ? 0 : frequency - state.previousFrequency;
  const energyRise = state.previousEnergy == null ? 0 : (energy - state.previousEnergy) / Math.max(state.previousEnergy, 1e-6);
  const vehicleLike = energy > VEHICLE_ENERGY_RATIO;
  state.vehicleStreak = vehicleLike ? state.vehicleStreak + 1 : 0;
  const persistent = state.vehicleStreak >= 2;
  const rising = frequencyRise >= FREQUENCY_RISE_HZ || energyRise >= 0.2;
  let level = "safe";
  let label = "SAFE";
  let reason = "No persistent vehicle-like acoustic evidence.";
  if (persistent && rising) {
    level = "risk";
    label = "HIGH ACOUSTIC RISK";
    reason = "Persistent vehicle-like audio with rising energy or dominant frequency.";
  } else if (vehicleLike) {
    level = persistent ? "detected" : "caution";
    label = persistent ? "VEHICLE SOUND DETECTED" : "CAUTION";
    reason = "Vehicle-like acoustic energy detected; waiting for persistence.";
  }

  state.previousFrequency = frequency;
  state.previousEnergy = energy;
  state.windows += 1;
  $("frequencyValue").textContent = `${Math.round(frequency)} Hz`;
  $("energyValue").textContent = energy.toFixed(4);
  $("vehicleValue").textContent = `${Math.round(Math.min(100, energy / VEHICLE_ENERGY_RATIO * 100))}%`;
  $("windowCount").textContent = `${state.windows} windows`;
  setStatus(level, label, reason);
  state.history.unshift({ source: sourceLabel, label, frequency, energy });
  state.history = state.history.slice(0, 8);
  $("history").innerHTML = state.history.map((item) =>
    `<div class="history-row"><span>${item.source} · ${item.label}</span><span>${Math.round(item.frequency)} Hz · ${item.energy.toFixed(3)}</span></div>`
  ).join("");
  drawFeature(samples, frequency, energy, level);
}

function drawFeature(samples, frequency, energy, level) {
  const canvas = $("spectrogram");
  const context = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  context.clearRect(0, 0, width, height);
  const gradient = context.createLinearGradient(0, 0, 0, height);
  gradient.addColorStop(0, "#132c48");
  gradient.addColorStop(1, "#06101c");
  context.fillStyle = gradient;
  context.fillRect(0, 0, width, height);
  const step = Math.max(1, Math.floor(samples.length / width));
  for (let x = 0; x < width; x++) {
    let sum = 0;
    for (let i = x * step; i < Math.min(samples.length, (x + 1) * step); i++) sum += Math.abs(samples[i]);
    const amplitude = Math.min(1, sum / step * 5);
    context.fillStyle = `hsl(${205 - amplitude * 150}, 85%, ${42 + amplitude * 30}%)`;
    context.fillRect(x, height - amplitude * height, 1, amplitude * height);
  }
  context.fillStyle = level === "risk" ? "#ff5964" : "#80dcff";
  context.font = "16px system-ui";
  context.fillText(`Dominant frequency: ${Math.round(frequency)} Hz`, 18, 28);
  context.fillText(`Energy: ${energy.toFixed(4)}`, 18, 52);
}

function appendSamples(samples) {
  const combined = new Float32Array(state.buffer.length + samples.length);
  combined.set(state.buffer);
  combined.set(samples, state.buffer.length);
  state.buffer = combined;
  while (state.buffer.length >= WINDOW_SIZE) {
    analyzeWindow(state.buffer.slice(0, WINDOW_SIZE), "LIVE");
    state.buffer = state.buffer.slice(Math.floor(SAMPLE_RATE * STEP_SECONDS));
  }
}

async function startMicrophone() {
  if (!navigator.mediaDevices?.getUserMedia) {
    $("connectionStatus").textContent = "This browser does not expose microphone access.";
    return;
  }
  try {
    state.audioContext = new AudioContext();
    state.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    state.source = state.audioContext.createMediaStreamSource(state.stream);
    state.processor = state.audioContext.createScriptProcessor(4096, 1, 1);
    state.processor.onaudioprocess = (event) => appendSamples(event.inputBuffer.getChannelData(0));
    state.source.connect(state.processor);
    state.processor.connect(state.audioContext.destination);
    $("startButton").disabled = true;
    $("stopButton").disabled = false;
    $("connectionStatus").textContent = "Microphone connected. Listening in 2-second windows.";
  } catch (error) {
    $("connectionStatus").textContent = `Microphone error: ${error.message}`;
  }
}

function stopMicrophone() {
  if (state.processor) state.processor.disconnect();
  if (state.source) state.source.disconnect();
  if (state.stream) state.stream.getTracks().forEach((track) => track.stop());
  if (state.audioContext) state.audioContext.close();
  state.processor = null;
  state.source = null;
  state.stream = null;
  state.buffer = new Float32Array(0);
  $("startButton").disabled = false;
  $("stopButton").disabled = true;
  $("connectionStatus").textContent = "Microphone stopped.";
}

async function analyzeFile(file) {
  try {
    const context = new AudioContext();
    const decoded = await context.decodeAudioData(await file.arrayBuffer());
    const channel = decoded.getChannelData(0);
    const ratio = decoded.sampleRate / SAMPLE_RATE;
    const resampled = new Float32Array(Math.floor(channel.length / ratio));
    for (let i = 0; i < resampled.length; i++) resampled[i] = channel[Math.floor(i * ratio)];
    state.buffer = new Float32Array(0);
    for (let offset = 0; offset + WINDOW_SIZE <= resampled.length; offset += SAMPLE_RATE * STEP_SECONDS) {
      analyzeWindow(resampled.slice(offset, offset + WINDOW_SIZE), "FILE");
    }
    await context.close();
    $("connectionStatus").textContent = `Analyzed ${file.name}.`;
  } catch (error) {
    $("connectionStatus").textContent = `File analysis error: ${error.message}`;
  }
}

$("startButton").addEventListener("click", startMicrophone);
$("stopButton").addEventListener("click", stopMicrophone);
$("fileInput").addEventListener("change", (event) => {
  if (event.target.files[0]) analyzeFile(event.target.files[0]);
});
