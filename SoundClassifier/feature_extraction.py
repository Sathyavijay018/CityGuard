import librosa
import numpy as np

def extract_features(audio, sr):
    features = []
    
    # 1. MFCC (13) + Delta + Delta-Delta
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    
    # Mean of MFCCs
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.mean(mfcc_delta, axis=1))
    features.extend(np.mean(mfcc_delta2, axis=1))
    
    # 2. Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    features.append(np.mean(spectral_centroid))
    
    # 3. Spectral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    features.append(np.mean(spectral_rolloff))
    
    # 4. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(audio)
    features.append(np.mean(zcr))
    
    return np.array(features)

def extract_from_file(file_path):
    # Load audio file (resample to 22050 for consistency)
    audio, sr = librosa.load(file_path, sr=22050)
    return extract_features(audio, sr)

if __name__ == "__main__":
    # Test feature extraction on random noise
    dummy_audio = np.random.randn(22050)
    feats = extract_features(dummy_audio, 22050)
    print(f"Extracted {len(feats)} features. Expected: 13 + 13 + 13 + 1 + 1 + 1 = 42")
