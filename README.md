# HerShield Proof of Concept
Audio Transcription and Sentiment Analysis

## Description
 <!-- Briefly describe what the script does -->
This Python codebase transcribes audio files, translates the transcription (if needed),
and performs sentiment analysis on the transcribed text using machine learning models.

Another file allows generation of the visual spectogram. 

## Prerequisites
<!-- List any prerequisites needed to run the script -->
1. Python installed on your system.
2. Required libraries:
    - `argparse`
    - `googletrans`
    - `transformers`
    - `whisper`
    - `librosa`
    - `numpy`
    You can install these libraries using pip:
```bash
pip install argparse googletrans transformers whisper
```

## Usage
<!-- Explain how to use the script -->
1. **Download the Script**: Download the Python script provided.
2. **Run the Script**: Open a terminal or command prompt and navigate to the directory where the script is located.
    ```bash
    python script_name.py audio_file_path
    ```
    Replace `script_name.py` with the name of the script file and `audio_file_path` with the path to the audio file you want to transcribe. We provide 2 sample files, one in English and one in Spanish. But feel free to test more. 
3. **View Transcription**: The script will display the transcription of the audio file. If the detected language is not English, it will also provide a translation of the transcription into English.
4. **Sentiment Analysis**: After transcribing and translating (if necessary), the script will perform sentiment analysis on the transcribed text and display the detected sentiment labels.

## Example
<!-- Provide an example command to run the script -->
```bash
python transcribe.py sample.m4a
```

Similarly, you can run the `spectogram.py` file to visualize the spectogram. 

## Spectogram
<!-- Provide an example command to run the script -->
```bash
python spectogram.py sample.m4a
```
