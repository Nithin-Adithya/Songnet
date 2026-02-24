import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'fma_small')
METADATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'fma_metadata', 'tracks.csv')
SAMPLE_RATE = 22050
DURATION = 30  # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

def load_metadata(path):
    # Load metadata with correct header
    tracks = pd.read_csv(path, index_col=0, header=[0, 1])
    
    # relevant columns: split, subset, genre_top, artist_name, track_date_created
    # Note: 'artist' and 'track' are top-level headers.
    
    keep_cols = [
        ('set', 'split'), 
        ('set', 'subset'), 
        ('track', 'genre_top'),
        ('artist', 'name'),
        ('track', 'date_created')
    ]
    
    # Select only relevant columns
    df_all = tracks[keep_cols]
    
    # Flatten columns
    df_all.columns = ['split', 'subset', 'genre_top', 'artist_name', 'track_date']
    
    # Filter for small subset and top-level genres
    df = df_all[df_all['subset'] == 'small']
    df = df[df['genre_top'].notnull()]
    
    return df

def get_audio_path(track_id):
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(DATA_DIR, tid_str[:3], tid_str + '.mp3')

def extract_mel_spectrogram(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Padding or Truncating
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

def process_data():
    print("Loading metadata...")
    df = load_metadata(METADATA_PATH)
    print(f"Found {len(df)} tracks in FMA-small.")
    
    # Save Metadata Mapping (Track ID -> {Artist, Year, Genre})
    # Filter only tracks that exist
    # verify existence first
    
    track_ids = df.index.tolist()
    genres = df['genre_top'].tolist()
    artists = df['artist_name'].tolist()
    dates = df['track_date'].tolist()
    
    X = []
    y = []
    valid_ids = []
    metadata_map = {}
    
    print("Extracting features (this may take a while)...")
    for tid, genre, artist, date in tqdm(zip(track_ids, genres, artists, dates), total=len(track_ids)):
        file_path = get_audio_path(tid)
        if not os.path.exists(file_path):
            continue
            
        mel_spec = extract_mel_spectrogram(file_path)
        if mel_spec is not None:
            X.append(mel_spec)
            y.append(genre)
            valid_ids.append(tid)
            
            # extract year from date string
            year = str(date)[:4] if pd.notnull(date) else "Unknown"
            
            metadata_map[str(tid)] = {
                'artist': str(artist),
                'year': year,
                'genre': str(genre)
            }
            
    X = np.array(X)
    y = np.array(y)
    
    print(f"Processed {len(X)} tracks.")
    
    # Label Encoding
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = le.classes_
    
    # Save processed data
    np.save('X.npy', X)
    np.save('y.npy', y_enc)
    np.save('classes.npy', classes)
    
    # Save metadata
    import json
    with open('metadata.json', 'w') as f:
        json.dump(metadata_map, f)
        
    print("Data saved to X.npy, y.npy, classes.npy, metadata.json")
    
    return X, y_enc, classes

if __name__ == "__main__":
    process_data()
