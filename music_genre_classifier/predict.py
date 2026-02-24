import numpy as np
import librosa
import tensorflow as tf
import pickle

# Constants - MUST MATCH PREPROCESS.PY
SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

def process_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        
        if len(y) < SAMPLES_PER_TRACK:
            y = np.pad(y, (0, SAMPLES_PER_TRACK - len(y)))
        else:
            y = y[:SAMPLES_PER_TRACK]
            
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db.T # (Time, n_mels)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def predict_genre(file_path, model_path='best_model.h5', classes_path='classes.npy'):
    if not isinstance(file_path, str): # Handle uploaded file object from streamlit
         # Save temp file if needed or process directly (librosa load accepts path, not file-like mostly, need workaround)
         pass

    # For now assume file_path is a string path
    features = process_audio(file_path)
    if features is None:
        return None
        
    features = np.expand_dims(features, axis=0) # Batch dim
    
    try:
        from model import SelfAttention
        model = tf.keras.models.load_model(model_path, custom_objects={'SelfAttention': SelfAttention})
    except ImportError:
        # Fallback if model.py is not easily importable or if using standard layers
        model = tf.keras.models.load_model(model_path)
    
    classes = np.load(classes_path)
    
    prediction = model.predict(features)
    predicted_index = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_index]
    
    return classes[predicted_index], confidence, prediction[0]

if __name__ == "__main__":
    # Test with a dummy file if available
    pass
