export class AudioEngine {
  constructor() {
    this.audioContext = null;
    this.mediaStream = null;
    this.sourceNode = null;
    this.scriptNode = null;
    this.analyserNode = null;
    this.visualArray = null;
    this.socket = null;
    this.isRunning = false;
    this.onPredictionResult = null; // Callback for DL results
  }

  async start() {
    if (this.isRunning) return;

    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      // Force 16kHz for YAMNet
      this.audioContext = new AudioContext({ sampleRate: 16000 });
      
      // Setup WebSocket connection to our Python FastAPI DL Server
      this.socket = new WebSocket('ws://localhost:8000/ws/audio');
      
      this.socket.onopen = () => {
        console.log("WebSocket connected. Deep Learning stream active.");
      };

      this.socket.onmessage = (event) => {
        const prediction = JSON.parse(event.data);
        if (this.onPredictionResult) {
            this.onPredictionResult(prediction);
        }
      };

      this.socket.onerror = (error) => {
        console.error("Deep Learning WebSocket Error:", error);
      };
      
      // Setup Audio Nodes
      this.sourceNode = this.audioContext.createMediaStreamSource(this.mediaStream);
      
      // AnalyserNode purely for visual UI
      this.analyserNode = this.audioContext.createAnalyser();
      this.analyserNode.fftSize = 256;
      this.visualArray = new Uint8Array(this.analyserNode.frequencyBinCount);
      
      this.sourceNode.connect(this.analyserNode);
      
      // Buffer size 4096 (approx 250ms chunks at 16kHz)
      this.scriptNode = this.audioContext.createScriptProcessor(4096, 1, 1);
      
      this.scriptNode.onaudioprocess = (audioProcessingEvent) => {
        if (!this.isRunning || !this.socket || this.socket.readyState !== WebSocket.OPEN) return;
        
        // Retrieve the Float32Array PCM data (mono)
        const inputBuffer = audioProcessingEvent.inputBuffer;
        const inputData = inputBuffer.getChannelData(0);
        
        // Send directly as binary via WebSocket
        this.socket.send(inputData.buffer);
      };
      
      // We must connect scriptNode to destination for it to run, but we can gain it to 0
      const gainNode = this.audioContext.createGain();
      gainNode.gain.value = 0;
      this.analyserNode.connect(this.scriptNode);
      this.scriptNode.connect(gainNode);
      gainNode.connect(this.audioContext.destination);
      
      this.isRunning = true;
    } catch (err) {
      console.error('Error starting audio/websocket:', err);
      throw new Error('Microphone access or Server connection failed. Ensure Python server is running.');
    }
  }

  stop() {
    if (!this.isRunning) return;
    
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
    }
    
    if (this.audioContext && this.audioContext.state !== 'closed') {
      this.audioContext.close();
    }
    
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
        this.socket.close();
    }
    
    this.mediaStream = null;
    this.sourceNode = null;
    this.scriptNode = null;
    this.socket = null;
    this.isRunning = false;
  }

  getVisualData() {
      if (!this.isRunning || !this.analyserNode) return null;
      this.analyserNode.getByteTimeDomainData(this.visualArray);
      return this.visualArray;
  }
}
