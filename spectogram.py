import argparse
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def plot_mel_spectrogram(audio_file):
    # Load audio and retrieve sample rate using librosa
    librosa_audio, sr = librosa.load(audio_file, sr=None)  # Load audio with sample rate information

    # Ensure audio is padded or trimmed to fit 30 seconds
    target_length = int(30 * sr)
    current_length = len(librosa_audio)
    if current_length < target_length:
        # Pad the audio
        librosa_audio = np.pad(librosa_audio, (0, target_length - current_length), mode='constant')
    elif current_length > target_length:
        # Trim the audio
        librosa_audio = librosa_audio[:target_length]

    # Make log-Mel spectrogram
    mel = librosa.feature.melspectrogram(y=librosa_audio, sr=sr, n_mels=128, hop_length=512)

    # Plot the spectrogram of the spoken sequence
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(mel, ref=np.max), sr=sr, hop_length=512, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.show()

if __name__ == "__main__":
    # Load file that is parsed
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to the audio file")
    args = parser.parse_args()
    # Plot spectogram
    plot_mel_spectrogram(args.file)