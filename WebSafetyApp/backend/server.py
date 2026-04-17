import asyncio
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import csv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YAMNet Model
print("Loading YAMNet model...")
model = hub.load('https://tfhub.dev/google/yamnet/1')

# Load Class Names
class_map_path = model.class_map_path().numpy().decode('utf-8')
class_names = []
with open(class_map_path) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        class_names.append(row['display_name'])

print(f"YAMNet loaded. Loaded {len(class_names)} classes.")

# Define Target Hazard Classes
# We consider anything related to vehicles, horns, or sirens as a potential hazard/vehicle classification.
HAZARD_CLASSES = {
    "Vehicle", "Car", "Motorcycle", "Truck", "Bus", "Train", 
    "Bicycle", "Bicycle bell", 
    "Vehicle horn, car horn, honking", "Siren", 
    "Police car (siren)", "Ambulance (siren)", "Fire engine, fire truck (siren)",
    "Train horn", "Car alarm", "Skidding", "Tire squeal"
}

@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected to audio stream.")
    
    # We maintain a small buffer to run 0.975s chunks expected by YAMNet (15600 samples)
    # But YAMNet natively accepts arbitrary length arrays and chunks internally.
    # It returns scores of shape (N_chunks, 521).
    
    # We will accumulate short bursts and run prediction
    audio_buffer = np.array([], dtype=np.float32)
    
    try:
        while True:
            # Receive binary float32 array from JS
            data = await websocket.receive_bytes()
            chunk = np.frombuffer(data, dtype=np.float32)
            
            # Append to buffer
            audio_buffer = np.concatenate((audio_buffer, chunk))
            
            # Feed to model if we have at least ~0.5 seconds of audio (8000 samples)
            if len(audio_buffer) >= 8000:
                # Cap the buffer to the most recent 1 second to stay real-time (16000 samples)
                if len(audio_buffer) > 16000:
                    audio_buffer = audio_buffer[-16000:]
                    
                waveform = audio_buffer
                
                # Run YAMNet model
                scores, embeddings, spectrogram = model(waveform)
                
                # scores shape: (num_frames, 521)
                # Take the mean score across frames for this chunk
                mean_scores = np.mean(scores.numpy(), axis=0)
                
                # Top 1 prediction
                top_class_index = np.argmax(mean_scores)
                top_class_name = class_names[top_class_index]
                top_score = float(mean_scores[top_class_index])
                
                # Check for hazards anywhere in the top 3 (to catch simultaneous sounds like horn + street noise)
                top_3_indices = np.argsort(mean_scores)[::-1][:3]
                
                detected_hazard = None
                hazard_score = 0.0
                
                for idx in top_3_indices:
                    cls_name = class_names[idx]
                    if cls_name in HAZARD_CLASSES:
                        detected_hazard = cls_name
                        hazard_score = float(mean_scores[idx])
                        break
                
                # Prepare payload
                response = {
                    "top_class": top_class_name,
                    "top_score": top_score,
                    "hazard_detected": detected_hazard is not None,
                    "hazard_class": detected_hazard,
                    "hazard_score": hazard_score,
                    "timestamp": time.time()
                }
                
                await websocket.send_json(response)
                
                # Clear buffer after processing
                audio_buffer = np.array([], dtype=np.float32)
                
    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"WebSocket error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
