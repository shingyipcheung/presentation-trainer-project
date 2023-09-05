# Presentation Trainer Project

## Overview

This project aims to combine Speech-to-Text (STT), Speech Emotion Recognition (SER), and Facial Emotion Recognition (FER) technologies to evaluate your presentation skills. We employ a hybrid architecture that combines Convolutional Neural Networks (CNN) for the SER model. For pitch detection, an autocorrelation method is used.

## Features

- Speech-to-Text (STT): Transcribes your speech.
- Speech Emotion Recognition (SER): Identifies the emotional tone in your speech.
- Facial Emotion Recognition (FER): Recognizes facial expressions.
- Pitch Detection: Monitors the pitch of your voice.
- Words per Minute (WPM) Monitoring: Calculates your speaking speed.
- Loudness Monitoring: Monitors the loudness of your voice.


## Installation
Install the required packages:

```bash
pip install -r requirements.txt
```

## How to Run

To run the demo, use the following command:

```bash
python.exe -m streamlit run app.py
```

## Project Structure

- `speech_emotion_recognition.py`: Contains the code for the SER model and inference functions.
- `pitch_detection.py`: The code for pitch detection.
- `utils.py`: Utility functions for audio processing.
- `app.py`: Streamlit app for the demo.
- `models/ser/`: Folder containing pre-trained models.
- `views.py`: Streamlit UI structures.

## Blog Post

For a more detailed understanding of the project, you can refer to the [Blog Post](#https://jamesy.dev/blog/how-to-build-a-presentation-trainer-using-pretrained-models).



## Acknowledgments

This project is inspired by various research papers and open-source contributions in the area of audio signal processing and machine learning.

