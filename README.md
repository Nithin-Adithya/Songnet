<div align="center">
  
# 🎵 SongNet: Music Genre Classifier

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue.style=for-the-badge)

*An AI-powered music analysis tool built with a custom C-RNN architecture and Self-Attention mechanism.*

</div>

---

## 🌟 Overview

**SongNet** is an advanced machine learning project designed to automatically classify the genre of musical tracks. By leveraging a sophisticated Deep Learning pipeline combining **Convolutional Neural Networks (CNN)**, **Gated Recurrent Units (GRU)**, and **Bahdanau-style Self-Attention**, SongNet extracts deep spatial and temporal features from audio signals to provide accurate genre predictions.

The project features a sleek, interactive **Streamlit** web application for real-time inference, allowing users to upload MP3/WAV files, analyze waveforms, and view detailed probability distributions of predicted genres.

---

## ✨ Key Features

- **End-to-End Classification:** Instantly classify track genres among 8 popular categories (based on the FMA-Small dataset).
- **Interactive UI:** A beautiful, responsive Streamlit web interface with drag-and-drop file uploading.
- **Waveform Analysis:** Visually inspect the waveform of uploaded audio samples in real-time using `librosa`.
- **Intelligent Architecture:** Custom C-RNN model utilizing self-attention to focus on the most important temporal segments of a song.
- **Edge Deployment Ready:** Includes tools to convert the trained `.h5` model to **TensorFlow Lite (`.tflite`)** for mobile/edge use cases.
- **Rich Metadata Extraction:** Fetches and displays track metadata (Artist, Year) if available in the database.

---

## 🧠 Model Architecture

The core of SongNet is built on a hybrid neural network architecture:

1. **Feature Extraction:** 3 blocks of 1D Convolutions combined with Batch Normalization, ReLU activation, and MaxPooling to capture local spatial features.
2. **Sequential Modeling:** A `GRU` (or optionally `LSTM`) layer processes the sequence of extracted features to capture temporal dependencies.
3. **Self-Attention Mechanism:** A custom Bahdanau-style attention layer weights the most critical time steps, heavily improving the model's contextual understanding.
4. **Classification Head:** Dense layers output a probability distribution across 8 genres using a Softmax activation.

---

## 🛠️ Tech Stack

- **Machine Learning / AI:** TensorFlow, Keras, TensorFlow Lite
- **Audio Processing:** Librosa
- **Web Interface:** Streamlit
- **Data Visualization:** Matplotlib, Seaborn
- **Language:** Python 3.x

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed on your system.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Nithin-Adithya/Songnet.git
   cd Songnet
   ```

2. **Create a virtual environment (Optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r music_genre_classifier/requirements.txt
   ```

### Running the Application

To launch the Streamlit interface locally:

```bash
cd music_genre_classifier
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser to interact with SongNet.

### Model Conversion (TFLite)

If you wish to deploy the model on mobile or edge devices, you can convert the saved Keras model using the provided utility:

```bash
python music_genre_classifier/convert_tflite.py
```
*Outputs `songnet.tflite`.*

---

## 📅 Dataset

This project utilizes the **FMA-Small** dataset (Free Music Archive).
- Tracks: 8,000
- Genres: 8 (Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, Rock)
- Format: 30-second clips

---

## 🤝 Contributing

Contributions are welcome! If you have ideas for improvements, new features, or bug fixes, feel free to open an issue or submit a pull request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

<div align="center">
  <p>Made with ❤️ by <a href="https://github.com/Nithin-Adithya">Nithin Adithya</a></p>
</div>
