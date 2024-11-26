Real-Time Gesture-to-Speech Translation System
Project Overview
This project enables real-time recognition of Indian Sign Language (ISL) gestures, translation into six regional Indian languages, and text-to-speech (TTS) conversion. Designed to bridge the communication gap, it supports individuals with speech and hearing impairments, enabling seamless interaction in diverse linguistic contexts.

Key Features
Real-Time Gesture Recognition: An RNN-based model processes input gestures to identify corresponding English phrases.
Language Translation: Recognized text is dynamically translated into one of six regional Indian languages: Hindi, Marathi, Telugu, Kannada, Tamil, and Malayalam.
Text-to-Speech Conversion: Converts translated text into spoken language with regional accents for natural communication.
Web Interface:
Live Video Feed: Hand gestures captured via webcam.
Language Selection: Users can select their preferred language.
Audio Output: Real-time speech feedback.
Tech Stack
Backend: Python, Flask, TensorFlow, Hugging Face Transformers.
Frontend: HTML, CSS, JavaScript.
Libraries:
Gesture Recognition: MediaPipe, OpenCV.
Language Translation: Pre-trained transformer models.
Speech Synthesis: VitsModel for TTS conversion.
Data Handling: pandas, numpy.
