import { AudioEngine } from './audioEngine.js';
import { AlertSystem } from './alertSystem.js';

document.addEventListener('DOMContentLoaded', () => {
  const toggleBtn = document.getElementById('toggleBtn');
  const statusIndicator = document.getElementById('statusIndicator');
  const statusText = document.getElementById('statusText');
  const volumeBar = document.getElementById('volumeBar');
  const volumeValue = document.getElementById('volumeValue');
  const detectionBox = document.getElementById('detectionBox');
  const sensitivitySlider = document.getElementById('sensitivitySlider');
  const sensitivityValue = document.getElementById('sensitivityValue');
  
  // We no longer need demo mode if we have real DL running
  const demoToggle = document.getElementById('demoToggle');
  demoToggle.parentElement.style.opacity = '0.5';
  demoToggle.disabled = true;

  const canvas = document.getElementById('waveformCanvas');
  const canvasCtx = canvas.getContext('2d');
  
  const resizeCanvas = () => {
    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = canvas.parentElement.clientHeight;
  };
  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();

  const engine = new AudioEngine();
  const alertSystem = new AlertSystem();

  let isMonitoring = false;
  let animationFrameId;

  sensitivitySlider.addEventListener('input', (e) => {
    sensitivityValue.textContent = `${e.target.value}%`;
  });

  toggleBtn.addEventListener('click', async () => {
    if (isMonitoring) {
      stopMonitoring();
    } else {
      await startMonitoring();
    }
  });

  async function startMonitoring() {
    try {
      // Connect WebSocket callback
      engine.onPredictionResult = handleDeepLearningPrediction;
      
      await engine.start();
      isMonitoring = true;
      toggleBtn.textContent = 'Stop Protection';
      toggleBtn.classList.add('stop');
      
      updateStatusUI('Safe', 'Streaming to AI Server...');
      drawLoop();
    } catch (err) {
      alert("Error: " + err.message);
      console.error(err);
    }
  }

  function stopMonitoring() {
    isMonitoring = false;
    cancelAnimationFrame(animationFrameId);
    
    engine.stop();
    
    toggleBtn.textContent = 'Start Protection';
    toggleBtn.classList.remove('stop');
    
    updateStatusUI('Safe', 'System Idle');
    
    volumeBar.style.width = '0%';
    volumeValue.textContent = '0%';
    
    detectionBox.innerHTML = '<span class="placeholder">Awaiting Audio...</span>';
    canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
  }

  function drawLoop() {
    if (!isMonitoring) return;

    // Visuals only (logic handled by DL Server)
    const visualData = engine.getVisualData();
    if (visualData) {
        drawWaveform(visualData);
        // Compute rough UI volume
        let rms = 0;
        for (let i=0; i<visualData.length; i++) {
            let v = (visualData[i] - 128) / 128.0;
            rms += v*v;
        }
        rms = Math.sqrt(rms / visualData.length);
        let volPct = Math.min(100, rms * 400);
        updateVolumeUI(volPct);
    }

    animationFrameId = requestAnimationFrame(drawLoop);
  }

  function handleDeepLearningPrediction(prediction) {
      // prediction: { top_class, top_score, hazard_detected, hazard_class, hazard_score }
      let sensitivity = parseInt(sensitivitySlider.value, 10);
      let threshold = 0.8 - (sensitivity / 100 * 0.5); // 0.3 to 0.8 threshold
      
      let isDanger = prediction.hazard_detected && prediction.hazard_score > threshold;
      
      if (isDanger) {
          updateStatusUI('Danger', 'DANGER DETECTED!');
          updateDetectionUI('Danger', `🚨 ${prediction.hazard_class.toUpperCase()} 🚨<br><small>Confidence: ${Math.round(prediction.hazard_score * 100)}%</small>`);
          alertSystem.triggerAlert(prediction.hazard_class);
      } else if (prediction.hazard_detected && prediction.hazard_score > threshold * 0.5) {
          // Warning state
          updateStatusUI('Warning', 'Hazardous Event Approaching');
          updateDetectionUI('Warning', `⚠️ Warning: ${prediction.hazard_class}<br><small>Confidence: ${Math.round(prediction.hazard_score * 100)}%</small>`);
      } else {
          // Safe state
          if (!alertSystem.isAlerting) {
             updateStatusUI('Safe', 'Safe Environment');
             updateDetectionUI('Safe', `Ambient: ${prediction.top_class} (${Math.round(prediction.top_score * 100)}%)`);
          }
      }
  }

  function updateStatusUI(state, text) {
    statusIndicator.className = `status-badge ${state.toLowerCase()}`;
    statusText.textContent = text;
  }

  function updateVolumeUI(volPct) {
    const p = Math.min(100, Math.max(0, volPct));
    volumeBar.style.width = `${p}%`;
    volumeValue.textContent = `${Math.round(p)}%`;
  }

  function updateDetectionUI(state, textHtml) {
    if (state === 'Safe') {
        detectionBox.innerHTML = `<span style="color: var(--text-muted)">${textHtml}</span>`;
    } else if (state === 'Warning') {
        detectionBox.innerHTML = `<span style="color: var(--warning-color)">${textHtml}</span>`;
    } else {
        detectionBox.innerHTML = `<span style="color: var(--danger-color); font-weight: 800">${textHtml}</span>`;
    }
  }

  function drawWaveform(dataArray) {
    canvasCtx.fillStyle = 'rgba(0, 0, 0, 0.2)';
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

    if (!dataArray) return;

    canvasCtx.lineWidth = 2;
    canvasCtx.strokeStyle = 'rgba(59, 130, 246, 0.8)';
    canvasCtx.beginPath();

    const sliceWidth = canvas.width * 1.0 / dataArray.length;
    let x = 0;

    for (let i = 0; i < dataArray.length; i++) {
        const v = dataArray[i] / 128.0;
        const y = v * canvas.height / 2;

        if (i === 0) {
            canvasCtx.moveTo(x, y);
        } else {
            canvasCtx.lineTo(x, y);
        }

        x += sliceWidth;
    }

    canvasCtx.lineTo(canvas.width, canvas.height / 2);
    canvasCtx.stroke();
  }
});
