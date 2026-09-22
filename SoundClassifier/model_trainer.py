import os
import glob
import numpy as np
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from feature_extraction import extract_from_file

def load_data(base_dir="data"):
    X = []
    y = []
    
    # Class 1: vehicle
    vehicle_files = glob.glob(os.path.join(base_dir, "vehicle", "*.wav"))
    print(f"Found {len(vehicle_files)} vehicle audio files.")
    for f in vehicle_files:
        try:
            feats = extract_from_file(f)
            X.append(feats)
            y.append("vehicle")
        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    # Class 0: non_vehicle
    non_vehicle_files = glob.glob(os.path.join(base_dir, "non_vehicle", "*.wav"))
    print(f"Found {len(non_vehicle_files)} non-vehicle audio files.")
    for f in non_vehicle_files:
        try:
            feats = extract_from_file(f)
            X.append(feats)
            y.append("non_vehicle")
        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    return np.array(X), np.array(y)

def train_model():
    X, y = load_data()
    if len(X) == 0:
        print("No data found! Cannot train model.")
        return
        
    # Optional: ensure we have at least 2 classes
    if len(np.unique(y)) < 2:
        print(f"Found only {len(np.unique(y))} classes. Need 2 to train.")
        return

    print(f"Total dataset size: {X.shape[0]} samples with {X.shape[1]} features.")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    print("Training RandomForestClassifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    # Evaluate
    print("Evaluating model...")
    y_pred = clf.predict(X_test_scaled)
    target_names = clf.classes_
    
    report = classification_report(y_test, y_pred, target_names=target_names)
    print("Classification Report:")
    print(report)
    
    conf_mat = confusion_matrix(y_test, y_pred, labels=target_names)
    print("Confusion Matrix:")
    print(conf_mat)
    
    # Feature Importances (useful for UI)
    feature_names = [f"MFCC_{i+1}" for i in range(13)] + \
                    [f"Delta_{i+1}" for i in range(13)] + \
                    [f"Delta2_{i+1}" for i in range(13)] + \
                    ["Spectral Centroid", "Spectral Rolloff", "Zero Crossing Rate"]
    
    importances = clf.feature_importances_
    df_importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Save model, scaler and metadata
    os.makedirs("models", exist_ok=True)
    with open("models/sound_classifier.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
        
    # Save test stats for the Streamlit app to show
    metadata = {
        'report': classification_report(y_test, y_pred, target_names=target_names, output_dict=True),
        'confusion_matrix': conf_mat.tolist(),
        'classes': target_names.tolist(),
        'feature_importances': df_importances.head(10).to_dict(orient='records')
    }
    with open("models/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
        
    print("Model and scaler saved to models/")

if __name__ == "__main__":
    train_model()
