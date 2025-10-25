import os
import re
import threading
import tempfile
import torch
import librosa
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
from pydub import AudioSegment
from emotion_detector import EmotionDetector
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk


class PodcastSummarySystem:
    def __init__(self):
        print("Initializing Podcast Summary System...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.whisper_processor = None
        self.whisper_model = None
        self.summarizer = None
        self.emotion_detector = EmotionDetector()
        print("System initialized (models not loaded yet).")

    def load_models(self):
        """Load Whisper and summarization models."""
        print("Loading models... please wait, this may take a while.")
        self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").to(self.device)
        self.summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-6-6",
            device=0 if torch.cuda.is_available() else -1
        )
        print("Models loaded successfully!")

    def process_audio_file(self, audio_file_path):
        audio = AudioSegment.from_file(audio_file_path)
        wav_path = os.path.join(tempfile.gettempdir(), "uploaded_audio.wav")
        audio.export(wav_path, format="wav")
        return wav_path, os.path.splitext(os.path.basename(audio_file_path))[0]

    def transcribe_audio(self, audio_path, chunk_length=30):
        if self.whisper_model is None or self.whisper_processor is None:
            raise RuntimeError("Models not loaded yet! Please load models first.")
        print(f"Transcribing long audio...")
        y, sr = librosa.load(audio_path, sr=16000)
        total_duration = librosa.get_duration(y=y, sr=sr)
        full_transcript = ""
        for i in range(0, int(total_duration), chunk_length):
            chunk = y[int(i * sr):int(min(i + chunk_length, total_duration) * sr)]
            if len(chunk) == 0:
                continue
            input_features = self.whisper_processor(chunk, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
            with torch.no_grad():
                predicted_ids = self.whisper_model.generate(input_features)
            text = self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            full_transcript += " " + text
        return full_transcript.strip()

    def summarize_text(self, text):
        if self.summarizer is None:
            raise RuntimeError("Summarization model not loaded yet!")
        if len(text) < 100:
            return "Text too short to summarize."
        chunks = re.split(r'(?<=[.!?])\s+', text)
        summaries = []
        for chunk in chunks:
            if len(chunk.split()) > 50:
                input_length = len(chunk.split())
                summary = self.summarizer(
                    chunk,
                    max_length=int(input_length * 0.6),
                    min_length=int(input_length * 0.3),
                    do_sample=False
                )[0]['summary_text']
                summaries.append(summary)
        return " ".join(summaries) if summaries else "Summary could not be generated."

    def process_podcast(self, audio_file_path):
        audio_path, title = self.process_audio_file(audio_file_path)
        transcript = self.transcribe_audio(audio_path)
        summary = self.summarize_text(transcript)
        emotions = self.emotion_detector.detect_emotions(audio_path)
        return {
            "title": title,
            "transcript": transcript,
            "summary": summary,
            "emotions": emotions
        }


class PodcastApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎧 Podcast Analyzer with Emotion Detection")
        self.system = PodcastSummarySystem()
        self.setup_ui()

    def setup_ui(self):
        self.frame = ttk.Frame(self.root, padding=10)
        self.frame.pack(fill=tk.BOTH, expand=True)

        ttk.Button(self.frame, text="Load Models", command=self.load_models_thread).pack(pady=5)
        ttk.Button(self.frame, text="Select Audio File", command=self.start_thread).pack(pady=10)

        self.text = tk.Text(self.frame, wrap=tk.WORD, height=15)
        self.text.pack(fill=tk.BOTH, expand=True)

    def load_models_thread(self):
        thread = threading.Thread(target=self.load_models)
        thread.start()

    def load_models(self):
        try:
            self.system.load_models()
            self.root.after(0, lambda: self.text.insert(tk.END, "✅ Models loaded successfully!\n"))
        except Exception as e:
            self.root.after(0, lambda: self.text.insert(tk.END, f"❌ Error loading models: {e}\n"))

    def start_thread(self):
        thread = threading.Thread(target=self.select_file)
        thread.start()

    def display_results(self, result):
        self.text.delete(1.0, tk.END)
        self.text.insert(tk.END, f"🎧 Title: {result['title']}\n\n")
        self.text.insert(tk.END, f"📝 Summary:\n{result['summary']}\n\n")
        self.text.insert(tk.END, f"💬 Primary Emotion: {result['emotions']['primary_emotion']}\n\n")

    def select_file(self):
        path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.mp3 *.wav *.m4a")])
        if not path:
            return
        self.root.after(0, lambda: self.text.delete(1.0, tk.END))
        self.root.after(0, lambda: self.text.insert(tk.END, "Processing...\n"))

        try:
            result = self.system.process_podcast(path)
            self.root.after(0, lambda: self.display_results(result))
        except Exception as e:
            self.root.after(0, lambda: self.text.insert(tk.END, f"❌ Error: {e}\n"))


if __name__ == "__main__":
    root = tk.Tk()
    app = PodcastApp(root)
    root.mainloop()
