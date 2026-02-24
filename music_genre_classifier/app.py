import streamlit as st
import numpy as np
import os
import tempfile
from predict import predict_genre
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import json
import re

# Page Config
st.set_page_config(
    page_title="SongNet Music Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .main {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h1 {
        color: #1DB954; /* Spotify-like Green */
    }
</style>
""", unsafe_allow_html=True)

# Load Metadata if available
METADATA_FILE = 'metadata.json'
metadata_map = {}
if os.path.exists(METADATA_FILE):
    try:
        with open(METADATA_FILE, 'r') as f:
            metadata_map = json.load(f)
    except:
        pass

def get_track_id_from_filename(filename):
    match = re.search(r'(\d+)', filename)
    if match:
        return str(int(match.group(1))) 
    return None

# Sidebar
with st.sidebar:
    st.title("🎵 SongNet")
    st.info("""
    **Model Architecture:**
    - C-RNN (CNN + GRU)
    - Self-Attention Mechanism
    
    **Dataset:**
    - FMA-Small (8 Genres)
    """)
    st.markdown("---")
    st.caption("Built with TensorFlow & Streamlit")

# Main Layout
st.title("🎵 SongNet: Music Genre Classifier")
st.markdown("##### AI-Powered Music Analysis")

col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.subheader("1. Upload Audio")
    uploaded_file = st.file_uploader("Upload MP3/WAV file", type=['mp3', 'wav'])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/mp3')
        
        # Display Metadata
        track_id = get_track_id_from_filename(uploaded_file.name)
        if track_id and track_id in metadata_map:
            meta = metadata_map[track_id]
            st.success(f"**Artist:** {meta.get('artist', 'Unknown')}\n\n**Year:** {meta.get('year', 'Unknown')}\n\n**True Genre:** {meta.get('genre', 'Unknown')}")
        else:
            st.info("No metadata found for this track.")

        if st.button("Analyze Track", type="primary", use_container_width=True):
            with st.spinner("Analyzing audio..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    # Predict
                    result = predict_genre(tmp_path)
                    
                    if result:
                        st.session_state['result'] = result
                        st.session_state['tmp_path'] = tmp_path
                    else:
                        st.error("Prediction failed.")
                except Exception as e:
                    st.error(f"Error: {e}")

with col2:
    st.subheader("2. Analysis Results")
    
    if uploaded_file is None:
        st.info("Please upload a file to start analysis.")
    elif 'result' in st.session_state and uploaded_file:
        genre, confidence, probabilities = st.session_state['result']
        tmp_path = st.session_state['tmp_path']
        
        # Top Prediction
        st.metric(label="Predicted Genre", value=genre, delta=f"{confidence*100:.1f}% Confidence")
        st.progress(float(confidence))
        
        # Tabs for details
        tab1, tab2 = st.tabs(["📊 Probabilities", "🌊 Waveform"])
        
        with tab1:
            try:
                classes = np.load('classes.npy')
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(x=classes, y=probabilities, ax=ax, palette='viridis')
                ax.set_ylabel("Probability")
                ax.set_ylim(0, 1)
                plt.xticks(rotation=45)
                st.pyplot(fig)
            except:
                st.warning("Could not plot probabilities.")
                
        with tab2:
            try:
                y, sr = librosa.load(tmp_path, sr=22050, duration=30)
                fig_wave, ax_wave = plt.subplots(figsize=(8, 3))
                librosa.display.waveshow(y, sr=sr, ax=ax_wave, color='#1DB954')
                ax_wave.set_yticks([]) # Hide Y-axis (amplitude)
                ax_wave.set_xlabel("Time (seconds)")
                sns.despine(left=True, bottom=False) # Clean borders
                st.pyplot(fig_wave)
            except Exception as e:
                st.warning(f"Waveform error: {e}")
                

