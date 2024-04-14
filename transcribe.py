import argparse
from googletrans import Translator
from transformers import pipeline
import whisper

def transcribe_audio(audio_file):
    # Load the Whisper model
    model = whisper.load_model("base")

    # Load audio and ensure it fits the set duration
    audio = whisper.load_audio(audio_file)
    audio = whisper.pad_or_trim(audio)

    # Make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Detect the spoken language
    _, probs = model.detect_language(mel)
    spoken_language = max(probs, key=probs.get)
    print(f"Detected language: {spoken_language}")

    # Decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    # Retrieve the transcription text
    transcribed_text = result.text if result else None

    if transcribed_text:
        print("\nTranscription:")
        print(transcribed_text)

        if spoken_language != 'en':
            # Translate the text to English by providing the detected language using Google Translate
            translator = Translator()
            translated_text = translator.translate(transcribed_text, src=spoken_language, dest="en").text
            print("\nTranslated Transcription (English):")
            print(translated_text)
        else:
            translated_text = transcribed_text
            print("\nDetected language is English. Using original transcription.")

        # Sentiment Analysis
        sentiment_analysis = pipeline("sentiment-analysis", framework="pt", model="SamLowe/roberta-base-go_emotions")
        sentiment_results = sentiment_analysis(translated_text)
        print("\nDetected Sentiment:")
        for result in sentiment_results:
            print(result['label'])

    else:
        print("\nTranscription failed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to the audio file")
    args = parser.parse_args()
    transcribe_audio(args.file)