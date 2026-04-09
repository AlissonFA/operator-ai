from fasthtml.common import *
import cv2
import numpy as np
import os
import json
from core.models import ObjectDetector, GestureRecognizer, CustomGestureClassifier
from core.processor import FrameProcessor
from core.dataset import GestureDataset
from core.visualizer import Visualizer

# --- INITIALIZATION ---
detector = ObjectDetector(model_path='models/efficientdet_lite0.tflite') if os.path.exists('models/efficientdet_lite0.tflite') else None
recognizer = GestureRecognizer(model_path='models/gesture_recognizer.task') if os.path.exists('models/gesture_recognizer.task') else None
classifier = CustomGestureClassifier(model_path='models/gesture_model.pkl') if os.path.exists('models/gesture_model.pkl') else None
dataset = GestureDataset(file_path='gesture_dataset.csv')
processor = FrameProcessor(detector, recognizer, classifier)
visualizer = Visualizer()

app, rt = fast_app(static_path='assets')

@rt("/")
def get():
    return Title("NLW Operator AI - FastHTML"), Body(
        H1("NLW Operator AI"),
        Div(
            Select(
                Option("Object Detection", value="Object Detection", selected=True),
                Option("Gesture Recognition", value="Gesture Recognition"),
                Option("Data Collection", value="Data Collection"),
                id="mode-select", onchange="updateState()"
            ),
            Input(type="text", id="label-input", value="Gesto_X", oninput="updateState()"),
            Button("Start Recording", id="record-btn", onclick="toggleRecording()"),
            P(id="status-text")
        ),
        Div(
            Canvas(id="output-canvas", width="640", height="480"),
            Video(id="video", width="640", height="480", style="display:none", autoplay=True),
        ),
        Script(src="/js/app.js")
    )

@app.ws('/ws')
async def ws(ws):
    await ws.accept()
    
    conn_mode = "Object Detection"
    conn_label = "Gesto_X"
    conn_is_recording = False

    while True:
        try:
            msg = await ws.receive()
            
            if 'text' in msg:
                config = json.loads(msg['text'])
                if config.get('type') == 'config':
                    conn_mode = config.get('mode', conn_mode)
                    conn_label = config.get('label', conn_label)
                    conn_is_recording = config.get('isRecording', conn_is_recording)
                continue

            if 'bytes' in msg:
                frame_bytes = msg['bytes']
                nparr = np.frombuffer(frame_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None: continue

                # Process according to current mode
                if conn_mode == "Object Detection":
                    if detector: img = processor.process_object_detection(img)
                elif conn_mode == "Gesture Recognition":
                    img = cv2.flip(img, 1)
                    if recognizer: img = processor.process_gesture_recognition(img)
                elif conn_mode == "Data Collection":
                    img = cv2.flip(img, 1)
                    if recognizer:
                        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        results = recognizer.recognize(rgb_img)
                        if conn_is_recording and results.hand_landmarks:
                            for landmarks in results.hand_landmarks:
                                dataset.save_landmarks(landmarks, conn_label)
                        if results.hand_landmarks:
                            visualizer.draw_hand_landmarks(img, results.hand_landmarks[0])
                        status_text = f"REC: {conn_label}" if conn_is_recording else "PAUSED"
                        color = (0, 0, 255) if conn_is_recording else (255, 0, 0)
                        visualizer.draw_status(img, status_text, color=color)

                _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
                await ws.send_bytes(buffer.tobytes())

        except Exception as e:
            break

if __name__ == "__main__":
    serve()


