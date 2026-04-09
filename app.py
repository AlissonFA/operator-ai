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
        Style("""
            body { 
                background-color: #0b0e14; 
                color: #f0f6fc; 
                font-family: 'Inter', sans-serif;
                margin: 0;
                padding: 20px;
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            h1 {
                background: linear-gradient(90deg, #58a6ff, #bc85ff);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 30px;
            }
            select, input, button {
                background: #21262d;
                color: white;
                border: 1px solid #30363d;
                padding: 8px 12px;
                border-radius: 6px;
                margin: 5px;
            }
            button#record-btn {
                background: #238636;
                border: none;
                cursor: pointer;
            }
            button#record-btn:hover { background: #2ea043; }
        """),
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
            Div(
                Canvas(id="output-canvas", width="640", height="480"),
                Video(id="video", width="640", height="480", style="display:none", autoplay=True),
                Img(id="gesture-image", src="", style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); display: none; width: 150px; opacity: 0.8; transition: all 0.3s ease;"),
                style="position: relative; border-radius: 12px; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.5); border: 1px solid #30363d;"
            ),
            Div(id="labels-container", 
                style="margin-left: 20px; min-width: 200px; display: flex; flex-direction: column; gap: 10px;"),
            style="display: flex; flex-direction: row; justify-content: center; align-items: flex-start; margin-top: 20px;"
        ),
        Script(src="/js/app.js")
    )

@app.ws('/ws')
async def ws(ws):
    # Com o FastHTML, o accept() é automático ou gerenciado pelo decorator.
    # Removendo o accept() manual para evitar erro de protocolo ASGI.
    
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
                labels = []
                image_to_show = None
                if conn_mode == "Object Detection":
                    if detector: 
                        img, labels = processor.process_object_detection(img)
                    else:
                        labels = ["Detector não carregado"]
                elif conn_mode == "Gesture Recognition":
                    img = cv2.flip(img, 1)
                    if recognizer: 
                        img, labels, image_to_show = processor.process_gesture_recognition(img)
                    else:
                        labels = ["Reconhecedor não carregado"]
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
                        labels = [status_text]
                
                # Envia JSON com os resultados das detecções
                payload = {"type": "labels", "data": labels}
                if image_to_show:
                    payload["image"] = image_to_show
                    
                await ws.send_json(payload)
                
                # Envia os bytes da imagem processada
                _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
                await ws.send_bytes(buffer.tobytes())

        except Exception as e:
            # print(f"WS Error: {e}")
            break

if __name__ == "__main__":
    serve()


