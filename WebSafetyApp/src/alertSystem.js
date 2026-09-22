export class AlertSystem {
  constructor() {
    this.overlay = document.getElementById('alertOverlay');
    this.isAlerting = false;
    this.audioContext = null;
    this.oscillator = null;
    this.gainNode = null;
    this.alertTimeout = null;
  }

  triggerAlert(message) {
    if (this.isAlerting) return;
    this.isAlerting = true;

    // Visual Alert
    this.overlay.querySelector('.alert-message').textContent = message || "Danger detected!";
    this.overlay.classList.remove('hidden');
    document.body.classList.add('danger-mode');

    // Vibration Alert (if supported)
    if (navigator.vibrate) {
      navigator.vibrate([300, 100, 300, 100, 500]);
    }

    // Audio Alert
    this.playAlertSound();

    // Auto dismiss after 3 seconds
    if (this.alertTimeout) clearTimeout(this.alertTimeout);
    this.alertTimeout = setTimeout(() => {
      this.dismissAlert();
    }, 3000);
  }

  playAlertSound() {
    try {
      if (!this.audioContext) {
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        this.audioContext = new AudioContext();
      }

      if (this.audioContext.state === 'suspended') {
        this.audioContext.resume();
      }

      this.oscillator = this.audioContext.createOscillator();
      this.gainNode = this.audioContext.createGain();

      this.oscillator.type = 'square';
      this.oscillator.frequency.setValueAtTime(800, this.audioContext.currentTime); // 800Hz
      this.oscillator.frequency.exponentialRampToValueAtTime(1200, this.audioContext.currentTime + 0.1);
      
      this.gainNode.gain.setValueAtTime(0, this.audioContext.currentTime);
      this.gainNode.gain.linearRampToValueAtTime(0.5, this.audioContext.currentTime + 0.05);
      this.gainNode.gain.exponentialRampToValueAtTime(0.01, this.audioContext.currentTime + 0.5);

      this.oscillator.connect(this.gainNode);
      this.gainNode.connect(this.audioContext.destination);

      this.oscillator.start();
      this.oscillator.stop(this.audioContext.currentTime + 0.5);
    } catch (e) {
      console.log("Audio alert failed", e);
    }
  }

  dismissAlert() {
    this.isAlerting = false;
    this.overlay.classList.add('hidden');
    document.body.classList.remove('danger-mode');
    
    if (this.oscillator) {
      try {
        this.oscillator.stop();
      } catch (e) {}
    }
  }
}
