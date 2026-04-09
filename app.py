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
            :root {
                --primary: #58a6ff;
                --secondary: #bc85ff;
                --bg-gradient: radial-gradient(circle at center, #1e293b 0%, #0f172a 100%);
                --glass: rgba(255, 255, 255, 0.03);
                --glass-border: rgba(255, 255, 255, 0.1);
            }
            body { 
                background: var(--bg-gradient);
                background-attachment: fixed;
                color: #f0f6fc; 
                font-family: 'Outfit', 'Inter', sans-serif;
                margin: 0;
                padding: 15px;
                display: flex;
                flex-direction: column;
                align-items: center;
                min-height: 100vh;
                overflow-x: hidden;
            }
            .glass-panel {
                background: var(--glass);
                backdrop-filter: blur(12px);
                border: 1px solid var(--glass-border);
                border-radius: 20px;
                box-shadow: 0 15px 35px -12px rgba(0, 0, 0, 0.5);
            }
            h1 {
                font-size: 1.8rem;
                font-weight: 800;
                background: linear-gradient(135deg, var(--primary), var(--secondary));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin: 10px 0 20px 0;
                letter-spacing: -0.025em;
            }
            .dashboard {
                display: flex;
                flex-direction: row;
                align-items: flex-start;
                gap: 20px;
                width: 100%;
                max-width: 1100px;
                justify-content: center;
            }
            .canvas-container {
                position: relative;
                border-radius: 24px;
                overflow: hidden;
                border: 2px solid var(--glass-border);
                line-height: 0;
                flex-shrink: 0;
            }
            .sidebar {
                display: flex;
                flex-direction: column;
                gap: 20px;
                flex-grow: 1;
                max-width: 350px;
            }
            .controls {
                display: flex;
                flex-direction: column;
                gap: 12px;
                padding: 20px;
            }
            select, input, button {
                background: #1e293b;
                color: white;
                border: 1px solid #334155;
                padding: 10px 15px;
                border-radius: 10px;
                font-size: 0.9rem;
                transition: all 0.2s;
                width: 100%;
                box-sizing: border-box;
            }
            button#record-btn {
                background: linear-gradient(135deg, #10b981, #059669);
                border: none;
                cursor: pointer;
                font-weight: 600;
            }
            button#record-btn:hover { 
                transform: translateY(-2px);
                box-shadow: 0 8px 12px -3px rgba(16, 185, 129, 0.4);
            }
            #labels-container {
                display: flex;
                flex-direction: column;
                gap: 15px;
            }
            #status-text { margin: 0; font-size: 0.8rem; opacity: 0.7; text-align: center; }
        """),
        H1("NLW Operator AI"),
        Div(
            Div(
                Canvas(id="output-canvas", width="640", height="480"),
                Video(id="video", width="640", height="480", style="display:none", autoplay=True),
                cls="canvas-container"
            ),
            Div(
                Div(
                    Select(
                        Option("Object Detection", value="Object Detection", selected=True),
                        Option("Gesture Recognition", value="Gesture Recognition"),
                        id="mode-select", onchange="updateState()"
                    ),
                    P(id="status-text"),
                    cls="controls glass-panel"
                ),
                Div(id="labels-container"),
                cls="sidebar"
            ),
            cls="dashboard"
        ),
        Script(src="/js/app.js")
    )

@app.ws('/ws')
async def ws(ws):
    # Com o FastHTML, o accept() é automático ou gerenciado pelo decorator.
    # Removendo o accept() manual para evitar erro de protocolo ASGI.
    
    conn_mode = "Object Detection"

    while True:
        try:
            msg = await ws.receive()
            
            if 'text' in msg:
                config = json.loads(msg['text'])
                if config.get('type') == 'config':
                    conn_mode = config.get('mode', conn_mode)
                continue

            if 'bytes' in msg:
                frame_bytes = msg['bytes']
                nparr = np.frombuffer(frame_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None: continue

                # Process according to current mode
                labels = []
                images_to_show = []
                if conn_mode == "Object Detection":
                    if detector: 
                        img, labels = processor.process_object_detection(img)
                    else:
                        labels = ["Detector não carregado"]
                elif conn_mode == "Gesture Recognition":
                    img = cv2.flip(img, 1)
                    if recognizer: 
                        img, labels, images_to_show = processor.process_gesture_recognition(img)
                    else:
                        labels = ["Reconhecedor não carregado"]
                
                # Envia JSON com os resultados das detecções
                payload = {"type": "labels", "data": labels}
                if images_to_show:
                    payload["images"] = images_to_show
                    
                await ws.send_json(payload)
                
                # Envia os bytes da imagem processada
                _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
                await ws.send_bytes(buffer.tobytes())

        except Exception as e:
            # print(f"WS Error: {e}")
            break

if __name__ == "__main__":
    serve()


