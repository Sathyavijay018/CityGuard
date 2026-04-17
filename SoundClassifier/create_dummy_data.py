import numpy as np
import soundfile as sf
import os
import scipy.signal

def generate_engine_rumble(duration, sr):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Base engine rumble frequency
    f0 = np.random.uniform(50, 100) 
    sig = 0.5 * scipy.signal.sawtooth(2 * np.pi * f0 * t)
    # Add harmonics
    sig += 0.3 * np.sin(2 * np.pi * (f0 * 2) * t)
    sig += 0.2 * np.sin(2 * np.pi * (f0 * 3) * t)
    # Low-pass filter to sound muffled/bassy like an engine
    b, a = scipy.signal.butter(4, 400 / (sr / 2), 'low')
    sig = scipy.signal.filtfilt(b, a, sig)
    # Add varying amplitude to simulate engine sputtering
    envelope = 1.0 + 0.2 * np.sin(2 * np.pi * 5 * t)
    return sig * envelope

def generate_passing_car(duration, sr):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Doppler effect: frequency drops as it passes
    f_start = np.random.uniform(250, 400)
    f_end = f_start * 0.7  # pitch drops
    frequencies = np.linspace(f_start, f_end, len(t))
    phase = 2 * np.pi * np.cumsum(frequencies) / sr
    sig = np.sin(phase)
    # Add noise resembling tires on road
    noise = np.random.randn(len(t))
    b, a = scipy.signal.butter(2, [1000 / (sr / 2), 4000 / (sr / 2)], 'bandpass')
    tire_noise = scipy.signal.filtfilt(b, a, noise)
    sig += 0.5 * tire_noise
    # Volume envelope: gets louder then quieter (movement!)
    envelope = scipy.signal.windows.hann(len(t))
    return sig * envelope

def generate_room_silence(duration, sr):
    # Absolute silence with tiny room tone/microphone hiss to train it that room noise = NO VEHICLE
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    noise = np.random.randn(len(t))
    b, a = scipy.signal.butter(2, 500 / (sr / 2), 'high')
    room_hiss = scipy.signal.filtfilt(b, a, noise)
    return 0.001 * room_hiss  # Very quiet room hiss

def generate_dataset(base_dir="data", samples_per_class=200):
    vehicle_dir = os.path.join(base_dir, "vehicle")
    non_vehicle_dir = os.path.join(base_dir, "non_vehicle")
    os.makedirs(vehicle_dir, exist_ok=True)
    os.makedirs(non_vehicle_dir, exist_ok=True)

    sr = 22050
    duration = 2.0

    print("Generating HIGH REALISM vehicle samples (engines, passing movement)...")
    for i in range(samples_per_class):
        if i % 2 == 0:
            sig = generate_engine_rumble(duration, sr)
        else:
            sig = generate_passing_car(duration, sr)
        
        # Add slight ambient noise to everything
        sig += generate_room_silence(duration, sr) * 5
        sig = sig / (np.max(np.abs(sig)) + 1e-6) # normalize
        sf.write(os.path.join(vehicle_dir, f"vehicle_{i:03d}.wav"), sig, sr)

    print("Generating NON-VEHICLE samples (room silence, ambient noise, non-vehicle sounds)...")
    for i in range(samples_per_class):
        if i < samples_per_class // 2:
            # PURE SILENCE / ROOM AMBIENCE (Critical to fix the "silent room" bug)
            sig = generate_room_silence(duration, sr)
        else:
            # Other random bright sounds
            f1 = np.random.uniform(2000, 3000)
            t = np.linspace(0, duration, int(sr * duration), endpoint=False)
            sig = np.sin(2*np.pi*f1*t) * np.exp(-3*t)  # Ping sound
        
        sig = sig / (np.max(np.abs(sig)) + 1e-6)
        # Random lower amplitude for non-vehicle so model learns quiet = safe
        vol = np.random.uniform(0.1, 0.8)
        if i < samples_per_class // 2: vol = 0.05 # Very quiet for room noise
        sf.write(os.path.join(non_vehicle_dir, f"non_vehicle_{i:03d}.wav"), sig * vol, sr)

if __name__ == "__main__":
    generate_dataset()
    print("Advanced High-Realism Dataset generation complete.")
