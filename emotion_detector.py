import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import warnings
import traceback

warnings.filterwarnings('ignore')

class EmotionDetector:
    def __init__(self):
        print("Initializing Emotion Detection System...")
        self.emotions = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'surprise']
        print("Emotion detection initialized successfully")

    def extract_features(self, audio_path, sr=None):
        print("Extracting audio features for emotion detection...")
        if sr is None:
            try:
                y, sr = librosa.load(audio_path, sr=22050)
            except Exception as e:
                print(f"❌ Error loading audio: {e}")
                traceback.print_exc()
                return {}, None, None
        else:
            y = audio_path

        features = {}
        try:
            # Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y)
            features['zcr_mean'] = float(np.mean(zcr))

            # Root Mean Square Energy
            rmse = librosa.feature.rms(y=y)
            features['rmse_mean'] = float(np.mean(rmse))

            # MFCC
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc{i+1}_mean'] = float(np.mean(mfccs[i]))

            # Spectral Features
            features['spectral_centroid_mean'] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
            features['spectral_bandwidth_mean'] = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
            features['spectral_rolloff_mean'] = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))

            print("✅ Feature extraction completed successfully")
            return features, y, sr
        except Exception as e:
            print(f"❌ Error during feature extraction: {e}")
            traceback.print_exc()
            return features, y, sr

    def detect_emotions(self, audio_path):
        """Detect the emotion based on extracted features."""
        try:
            features, y, sr = self.extract_features(audio_path)
            if not features or y is None:
                return {'primary_emotion': 'unknown', 'emotion_scores': {e: 0 for e in self.emotions}}

            energy = features.get('rmse_mean', 0)
            pitch = features.get('spectral_centroid_mean', 0)
            zcr = features.get('zcr_mean', 0)

            # Simple rule-based emotion classification
            if energy > 0.1 and pitch > 2000:
                primary = 'happy'
            elif energy > 0.08 and zcr > 0.1:
                primary = 'angry'
            elif energy < 0.05:
                primary = 'sad'
            elif pitch > 2200:
                primary = 'surprise'
            elif energy < 0.07 and pitch < 1800:
                primary = 'fearful'
            else:
                primary = 'neutral'

            scores = {e: 0.1 for e in self.emotions}
            scores[primary] = 0.5

            # Generate visualization for debugging/insight
            self.visualize_audio((y, sr), features, save_dir="visualizations")

            return {'primary_emotion': primary, 'emotion_scores': scores, 'audio_data': (y, sr)}
        except Exception as e:
            print(f"❌ Error detecting emotions: {e}")
            traceback.print_exc()
            return {'primary_emotion': 'unknown', 'emotion_scores': {e: 0 for e in self.emotions}}

    def visualize_audio(self, audio_data, features, save_dir="visualizations"):
        """Visualize how we detect emotions using audio features."""
        try:
            y, sr = audio_data
            os.makedirs(save_dir, exist_ok=True)

            # 🔹 1. Waveform with ZCR
            plt.figure(figsize=(10, 4))
            librosa.display.waveshow(y, sr=sr, alpha=0.7)
            plt.title(f"Waveform (ZCR Mean: {features['zcr_mean']:.4f})")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "waveform_zcr.png"))
            plt.close()

            # 🔹 2. RMS Energy over time
            rms = librosa.feature.rms(y=y)[0]
            plt.figure(figsize=(10, 4))
            plt.plot(rms, color='r')
            plt.title(f"Energy (RMS Mean: {features['rmse_mean']:.4f})")
            plt.xlabel("Frame")
            plt.ylabel("RMS Energy")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "rms_energy.png"))
            plt.close()

            # 🔹 3. MFCC Visualization
            plt.figure(figsize=(10, 4))
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            librosa.display.specshow(mfccs, sr=sr, x_axis='time')
            plt.colorbar(format='%+2.0f dB')
            plt.title("MFCC Coefficients")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "mfcc_visualization.png"))
            plt.close()

            print("✅ Visualizations saved in 'visualizations' folder.")

        except Exception as e:
            print(f"❌ Error in visualization: {e}")
            traceback.print_exc()
