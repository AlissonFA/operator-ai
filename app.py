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

app, rt = fast_app()

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
        Script("""
            let ws;
            let video = document.getElementById('video');
            let canvas = document.getElementById('output-canvas');
            let ctx = canvas.getContext('2d');
            let modeSelect = document.getElementById('mode-select');
            let labelInput = document.getElementById('label-input');
            let recordBtn = document.getElementById('record-btn');
            let isRecording = false;
            let lastFrameTime = 0;

            async function setupWebcam() {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
                    video.srcObject = stream;
                    return new Promise((resolve) => video.onloadedmetadata = () => { video.play(); resolve(); });
                } catch (e) { console.error("Webcam error:", e); }
            }

            function updateState() {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({
                        type: 'config',
                        mode: modeSelect.value,
                        label: labelInput.value,
                        isRecording: isRecording
                    }));
                }
            }

            function toggleRecording() {
                isRecording = !isRecording;
                recordBtn.innerText = isRecording ? "Stop Recording" : "Start Recording";
                updateState();
            }

            function connectWS() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
                ws.binaryType = 'arraybuffer';

                ws.onopen = () => {
                    console.log('Connected to WS');
                    updateState();
                    requestAnimationFrame(sendFrame);
                };

                ws.onmessage = async (event) => {
                    if (event.data instanceof ArrayBuffer) {
                        const blob = new Blob([event.data], { type: 'image/jpeg' });
                        const url = URL.createObjectURL(blob);
                        const img = new Image();
                        img.onload = () => {
                            ctx.drawImage(img, 0, 0);
                            URL.revokeObjectURL(url);
                            requestAnimationFrame(sendFrame);
                        };
                        img.src = url;
                    }
                };
                
                ws.onclose = () => setTimeout(connectWS, 1000);
            }

            const tempCanvas = document.createElement('canvas');
            const tempCtx = tempCanvas.getContext('2d');
            tempCanvas.width = 640;
            tempCanvas.height = 480;

            function sendFrame(time) {
                if (time - lastFrameTime < 33) { // Cap at ~30fps
                    requestAnimationFrame(sendFrame);
                    return;
                }
                lastFrameTime = time;

                if (ws && ws.readyState === WebSocket.OPEN && video.readyState >= 2) {
                    tempCtx.drawImage(video, 0, 0, 640, 480);
                    tempCanvas.toBlob((blob) => {
                        if (blob) ws.send(blob);
                    }, 'image/jpeg', 0.6);
                } else {
                    requestAnimationFrame(sendFrame);
                }
            }

            setupWebcam().then(connectWS);
        """)
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


