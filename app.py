import os
from flask import Flask, request, jsonify, render_template, Response
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TFAutoModelForSeq2SeqLM
from gtts import gTTS
import time

import cv2
import numpy as np
import tensorflow as tf
import keras
import pandas as pd
from hand_gesture_label import process_frame

app = Flask(__name__)

model = keras.models.load_model("SignToTextModel.keras") # Sign-to-text model
label_df = pd.read_csv("Phrase_Labels.csv")  # Labels file
camera = cv2.VideoCapture(0)  # Webcam

language_models = {
    'telugu': {
        'tokenizer': MBart50TokenizerFast.from_pretrained("aryaumesh/english-to-telugu"),
        'model': MBartForConditionalGeneration.from_pretrained("aryaumesh/english-to-telugu"),
    },
    'marathi': {
        'tokenizer': MBart50TokenizerFast.from_pretrained("aryaumesh/english-to-marathi"),
        'model': MBartForConditionalGeneration.from_pretrained("aryaumesh/english-to-marathi"),
    },
    'tamil': {
        'tokenizer': AutoTokenizer.from_pretrained("aryaumesh/english-to-tamil"),
        'model': AutoModelForSeq2SeqLM.from_pretrained("aryaumesh/english-to-tamil"),
    },
    'hindi': {
        'tokenizer': AutoTokenizer.from_pretrained("aryaumesh/tf_model"),
        'model': TFAutoModelForSeq2SeqLM.from_pretrained("aryaumesh/tf_model"),
    },
    'malayalam': {
        'tokenizer': AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ml"),
        'model': TFAutoModelForSeq2SeqLM.from_pretrained("ananya03/tf_model_malyalam"),
    },
    'kannada': {
        'tokenizer': AutoTokenizer.from_pretrained("ananya03/Kannada_model"),
        'model': TFAutoModelForSeq2SeqLM.from_pretrained("ananya03/Kannada_model"),
    }
}


# Variables to manage recording
is_recording = False
recorded_frames = []

# Extract keypoints from a frame (Replace with your actual keypoint extraction method)
def extract_keypoints(frame):
    # Process frame for hand landmarks
    processed_frame, keypoints = process_frame(frame)

    return keypoints

# Route for live video feed
def generate_frames():
    global is_recording, recorded_frames
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # If recording, store the frame for prediction
            if is_recording:
                recorded_frames.append(frame)

            # Encode the frame for live streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


################################### Integrating front-end with back-end ###################################

@app.route('/')
def index():
    return render_template('index.html')

# Route to start recording
@app.route('/start_recording', methods=['POST'])
def start_recording():
    global is_recording, recorded_frames
    is_recording = True
    print(is_recording)
    recorded_frames = []  # Clear any previous frames
    return jsonify({"status": "Recording started"})

# Route to stop recording and predict gesture
@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global is_recording
    is_recording = False
    print(is_recording)
    
    # Process frames and predict gesture
    if recorded_frames:
        keypoints_sequences = [extract_keypoints(frame) for frame in recorded_frames]
        keypoints_padded = keras.preprocessing.sequence.pad_sequences(
            [keypoints_sequences], padding="post", value=-4, maxlen=178
        )

        # Predict gesture
        y_pred = model.predict(keypoints_padded, verbose=1)
        predicted_label_index = np.argmax(y_pred)
        predicted_label = label_df["Unique Labels"][predicted_label_index]

        # Clear recorded frames after prediction
        recorded_frames.clear()
        return jsonify({"gesture": predicted_label})
    
    return jsonify({"gesture": "No frames recorded"})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.get_json()
        detected_text = data['detected_text']
        selected_language = data['language']
        
        tokenizer = language_models[selected_language]['tokenizer']
        model = language_models[selected_language]['model']
        
        translated_text = None

        if(selected_language in ["hindi", "malayalam", "kannada"]):
            tokenized = tokenizer(detected_text, return_tensors='np')
            out = model.generate(**tokenized, max_length=128)
            with tokenizer.as_target_tokenizer():
                translated_text = tokenizer.decode(out[0], skip_special_tokens=True)
        else:
            inputs = tokenizer(detected_text, return_tensors="pt")
            outputs = model.generate(**inputs)
            translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        unique_id = int(time.time())  
        audio_path = f'static/audio/translated_audio_{unique_id}.mp3'
        
        gtts_lang_map = {
            'telugu': 'te',
            'marathi': 'mr',
            'tamil': 'ta',
            'hindi': 'hi', 
            'malayalam': 'ml',
            'kannada': 'kn',
        }
        
        gtts_lang = gtts_lang_map[selected_language]
        
        tts = gTTS(translated_text, lang=gtts_lang)  
        tts.save(audio_path) 

        return jsonify({
            'translated_text': translated_text,
            'audio_url': '/' + audio_path  
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500 

if __name__ == "__main__":
    os.makedirs('static/audio', exist_ok=True)
    app.run(debug=True)
