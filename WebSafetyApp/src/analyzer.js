export class AudioAnalyzer {
  constructor() {
    this.historyLength = 30; 
    this.volumes = [];
    this.midPeakHistory = [];
    this.dangerScore = 0; // Cumulative score for alerts
  }

  analyze(audioData, sensitivity) {
    if (!audioData || !audioData.freqData) {
      return { status: 'Safe', volume: 0, reason: null, confidence: 0 };
    }

    const { freqData, timeData, bufferLength, sampleRate } = audioData;
    
    // RMS volume calculation
    let sumSquares = 0;
    for (let i = 0; i < timeData.length; i++) {
        let val = (timeData[i] - 128) / 128;
        sumSquares += val * val;
    }
    let rms = Math.sqrt(sumSquares / timeData.length);
    let volumePercent = Math.min(100, rms * 400); 
    
    this.volumes.push(volumePercent);
    if (this.volumes.length > this.historyLength) this.volumes.shift();
    
    const sensitivityFactor = 1 - (sensitivity / 100);
    const attackVolThreshold = 10 + (sensitivityFactor * 20); // Spike threshold
    
    // Decay score to require sustained evidence
    this.dangerScore = Math.max(0, this.dangerScore - 2); 

    // --- FREQUENCY BAND ANALYSIS ---
    // Instead of using error-prone flat maths, we extract peaks and averages for strict band matching.
    const binSize = sampleRate / 2 / bufferLength; // ~21.5 Hz per bin
    
    const lowEndBin = Math.floor(350 / binSize); // 0 - 350Hz (Voice focus, music kicks, bass)
    const midEndBin = Math.floor(3000 / binSize); // 350 - 3000Hz (Horns, Sirens, screeching brakes)
    
    let maxLow = 0, avgLow = 0;
    let maxMid = 0, avgMid = 0;
    let midPeakBin = 0;

    for (let i = 0; i < bufferLength; i++) {
        let v = freqData[i];
        if (i <= lowEndBin) {
            if (v > maxLow) maxLow = v;
            avgLow += v;
        } else if (i <= midEndBin) {
            if (v > maxMid) {
                maxMid = v;
                midPeakBin = i;
            }
            avgMid += v;
        }
    }

    avgLow /= Math.max(1, lowEndBin + 1);
    avgMid /= Math.max(1, midEndBin - lowEndBin);

    // Track sweeping frequencies in Mid band to detect Sirens
    this.midPeakHistory.push(midPeakBin);
    if (this.midPeakHistory.length > 25) this.midPeakHistory.shift(); // ~400ms memory

    let sweepAmount = 0;
    if (this.midPeakHistory.length > 5) {
        let maxP = 0, minP = 999;
        for(let p of this.midPeakHistory) {
            if (p > maxP) maxP = p;
            if (p < minP) minP = p;
        }
        sweepAmount = maxP - minP;
    }

    // --- IDENTIFICATION LOGIC ---
    
    // 1. Attack Check: A sudden burst
    const pastVol = this.volumes.length > 5 ? this.volumes[this.volumes.length - 5] : volumePercent;
    const hasFastAttack = (volumePercent - pastVol) > attackVolThreshold;

    // 2. Tonal Prominence: True horns have massive singular energy peaks in the midband.
    // Random noise (snare drums, street wash) has a relatively flat plateau (`avgMid` will be high alongside `maxMid`).
    const isTonal = maxMid > avgMid + 35; 
    
    // 3. Voice/Music Suppression: 
    // Human voice and music possess massive energy in the low frequencies (0-350Hz).
    // A car horn has almost zero fundamental energy below 350Hz, so if the low band is 
    // competing with or overpowering the mid band, it's virtually guaranteed to be a voice or music.
    const isVoiceOrMusic = (maxLow > maxMid + 10) || (avgLow > avgMid * 1.25);

    // Classification Rules:
    const isHornSignature = maxMid > 130 && isTonal && !isVoiceOrMusic;
    const isSirenSignature = maxMid > 100 && isTonal && !isVoiceOrMusic && (sweepAmount > 3 && sweepAmount < 40);

    let status = 'Safe';
    let reason = null;

    // --- DECISION STATE MACHINE ---
    if (volumePercent > 20) {
        if (isVoiceOrMusic) {
            this.dangerScore = 0; // Wipe score entirely for speech/music chunks
            reason = "Safe (Voice/Music signature)";
        } else if (isHornSignature || isSirenSignature) {
            // Rapidly build confidence if pattern matches
            if (hasFastAttack) {
                this.dangerScore = Math.min(100, this.dangerScore + 40); // Sharp honk burst
            } else {
                this.dangerScore = Math.min(100, this.dangerScore + 15); // Sustained blare
            }
        } else {
            // Not voice, but not a clear horn either (ambient noise)
            this.dangerScore *= 0.5;
            reason = "Analyzing ambient noise...";
        }
    }

    if (this.dangerScore >= 75) {
        status = 'Danger';
        reason = isSirenSignature ? '🚨 Siren Wailing Detected!' : '🚨 Vehicle Horn Detected!';
    } else if (this.dangerScore >= 40) {
        status = 'Warning';
        reason = 'Hazardous sound pattern building...';
    } else if (!reason) {
        reason = 'Monitoring environment...';
    }

    return {
        status,
        volume: volumePercent,
        reason,
        confidence: this.dangerScore
    };
  }
}
