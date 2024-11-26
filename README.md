This project enables real-time recognition of Indian Sign Language (ISL) gestures, translating them into six regional Indian languages, and performing text-to-speech (TTS) conversion. Designed to bridge the communication gap, it supports individuals with speech and hearing impairments, enabling seamless interaction in diverse linguistic contexts.

Key Features: 
The system incorporates multiple features to enhance accessibility and usability. It supports real-time gesture recognition through an RNN-based model, which processes input gestures and identifies corresponding English phrases. Recognized text is then dynamically translated into one of six regional Indian languages: Hindi, Marathi, Telugu, Kannada, Tamil, and Malayalam. The final translated text is converted into spoken language with regional accents to ensure natural communication.

The user interface provides an interactive experience, featuring live video feed to capture hand gestures via a webcam, options for selecting the preferred language, and real-time audio feedback based on gesture recognition.

Tech Stack: 
This project uses a robust technology stack, integrating both backend and frontend technologies. The backend is powered by Python, Flask, TensorFlow, and Hugging Face Transformers, while the frontend leverages HTML, CSS, and JavaScript. Key libraries include MediaPipe and OpenCV for gesture recognition, pre-trained transformer models for language translation, and VitsModel for TTS conversion. Data handling is performed using pandas and numpy.

To run this application, we first need to create a virtual environment, followed by activation of virtual environment. Then navigate to directory containing your project and run the app.py file. The commands for executing this is as follows:
1. python -m venv myenv
2. myenv\Scripts\activate
3. cd <path_to_directory>
4. python app.py

Upon initialization of TensorFlow, Hugging Face Transformers and the Flask Application, Flask will start a server at http://127.0.0.1:5000, which will direct you to our application.

Note: TensorFlow version used is 2.15.0
