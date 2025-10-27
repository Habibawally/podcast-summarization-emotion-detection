# 🎧 Podcast Summarization & Emotion Detection System
<img width="830" height="496" alt="Screenshot 2025-10-27 185723" src="https://github.com/user-attachments/assets/18930d66-aa19-471c-b7f0-ae9786a9ae0c" />

> **AI that Listens, Understands & Analyzes Emotion**

This project combines **Automatic Speech Recognition (ASR)**, **Text Summarization**, and **Emotion Detection** to analyze podcast audio files.  
It transcribes spoken content using **Whisper**, summarizes it using **DistilBART**, and detects the emotional tone using **audio feature analysis** with `Librosa`.

---

## 🚀 Project Overview

The **Podcast Summarization & Emotion Detection System** aims to help users quickly understand long audio content by providing:
- 🎙️ **Speech-to-text transcription** (via Whisper)
- ✂️ **Smart summarization** (via DistilBART)
- ❤️‍🔥 **Emotion analysis** from voice tone and acoustic features
- 📊 **Visualizations** of waveform, MFCC, RMS, and energy for deeper insight

The system provides a **simple Tkinter-based GUI**, allowing users to upload audio files and view real-time analysis results.

---

## 🧠 System Architecture
🎧 Audio Input (.mp3 / .wav / .m4a)
│
▼
🗣️ Whisper ASR → Transcription
│
▼
📝 DistilBART Summarizer → Smart Summary
│
▼
🎭 EmotionDetector (Librosa + Rule-based Model)
│
▼
📊 Visualization (Waveform, MFCC, RMS, Spectrogram)
│
▼
🖥️ Output in Tkinter GUI

--

## 🧩 Key Components

### 1. **Speech Recognition (Whisper)**
- **Model:** `openai/whisper-tiny`
- **Purpose:** Converts podcast audio to text.
- **Libraries Used:**
  - `transformers`
  - `torch`
  - `librosa`

### 2. **Summarization (DistilBART)**
- **Model:** `sshleifer/distilbart-cnn-6-6`
- **Purpose:** Summarizes long transcripts into concise, meaningful summaries.
- **Library Used:**
  - `transformers.pipeline("summarization")`

### 3. **Emotion Detection (Rule-based via Audio Features)**
- **Library:** `Librosa`
- **Extracted Features:**
  - Zero Crossing Rate (ZCR)
  - Root Mean Square Energy (RMS)
  - Spectral Centroid
  - Spectral Bandwidth
  - Spectral Rolloff
  - MFCCs (13 coefficients)
  - 
- **Classification Logic:** Rule-based model mapping energy, pitch, and ZCR to emotion categories:
  - Neutral
  - Happy
  - Sad
  - Angry
  - Fearful
  - Surprise

### 4. **Visualization**
Using `matplotlib` and `librosa.display`, the system generates:
- Waveform (Amplitude over Time)
- RMS Energy Plot
- MFCC Spectrogram

Visual outputs are saved in the `/visualizations` directory.

### 5. **GUI (Tkinter)**
A clean desktop interface to:
- Load models
- Select audio file
- Display summary and emotion results interactively

---

## 🛠️ Libraries and Dependencies

| Library | Purpose |
|----------|----------|
| `torch` | Running Whisper and Transformer models |
| `transformers` | Pretrained models (Whisper, DistilBART) |
| `librosa` | Audio feature extraction |
| `matplotlib` | Data visualization (waveform, MFCCs, etc.) |
| `numpy` | Numerical operations |
| `pydub` | Audio format conversion |
| `tkinter` | GUI creation |
| `PIL` | Image handling in GUI |
| `re` | Text splitting for summarization |
| `os`, `tempfile`, `threading` | File management and parallel processing |

---

## ⚙️ Installation

```bash
# 1️⃣ Clone the repository
git clone https://github.com/YOUR_USERNAME/podcast-summarization-emotion-detection.git
cd podcast-summarization-emotion-detection

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Run the application
python main.py
bash
Copy code
📦 podcast-summarization-emotion-detection
├── emotion_detector.py        # Handles feature extraction & emotion detection
├── main.py                    # Main system + GUI
├── requirements.txt           # Project dependencies
├── visualizations/            # Auto-generated plots (Waveform, MFCC, RMS)
└── README.md     
```bash
pip install -r requirements.txt
python podcast_summary_system.py


