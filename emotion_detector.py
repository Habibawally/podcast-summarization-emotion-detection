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
                print(f"Error loading audio: {e}")
                traceback.print_exc()
                return {}, None, None
        else:
            y = audio_path

        features = {}
        try:
            zcr = librosa.feature.zero_crossing_rate(y)
            features['zcr_mean'] = float(np.mean(zcr))

            rmse = librosa.feature.rms(y=y)
            features['rmse_mean'] = float(np.mean(rmse))

            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc{i+1}_mean'] = float(np.mean(mfccs[i]))

            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))

            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))

            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))

            print("Feature extraction completed successfully")
            return features, y, sr
        except Exception as e:
            print(f"Error during feature extraction: {e}")
            traceback.print_exc()
            return features, y, sr

    def detect_emotions(self, audio_path):
        try:
            features, y, sr = self.extract_features(audio_path)
            if not features or y is None:
                return {'primary_emotion': 'unknown', 'emotion_scores': {e: 0 for e in self.emotions}}

            energy = features.get('rmse_mean', 0)
            pitch = features.get('spectral_centroid_mean', 0)
            zcr = features.get('zcr_mean', 0)

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

            return {'primary_emotion': primary, 'emotion_scores': scores, 'audio_data': (y, sr)}
        except Exception as e:
            print(f"Error detecting emotions: {e}")
            traceback.print_exc()
            return {'primary_emotion': 'unknown', 'emotion_scores': {e: 0 for e in self.emotions}}

    def visualize_audio(self, audio_data, save_dir="visualizations"):
        try:
            y, sr = audio_data
            os.makedirs(save_dir, exist_ok=True)
            paths = {}

            # Waveform
            plt.figure(figsize=(10, 4))
            librosa.display.waveshow(y, sr=sr)
            path_wave = os.path.join(save_dir, "waveform.png")
            plt.title("Waveform")
            plt.savefig(path_wave); plt.close()
            paths["waveform"] = path_wave

            # Spectrogram
            plt.figure(figsize=(10, 4))
            spec = librosa.feature.melspectrogram(y=y, sr=sr)
            spec_db = librosa.power_to_db(spec, ref=np.max)
            librosa.display.specshow(spec_db, sr=sr, x_axis='time', y_axis='mel')
            path_spec = os.path.join(save_dir, "spectrogram.png")
            plt.title("Spectrogram")
            plt.savefig(path_spec); plt.close()
            paths["spectrogram"] = path_spec

            # MFCC
            plt.figure(figsize=(10, 4))
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            librosa.display.specshow(mfccs, sr=sr, x_axis='time')
            path_mfcc = os.path.join(save_dir, "mfcc.png")
            plt.title("MFCC")
            plt.savefig(path_mfcc); plt.close()
            paths["mfcc"] = path_mfcc

            return paths
        except Exception as e:
            print(f"Error in visualization: {e}")
            return {}
