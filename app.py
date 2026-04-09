from fasthtml.common import *
import cv2
import numpy as np
import time
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
    return Title("NLW Operator AI - FastHTML"), Link(rel="icon", href="/favicon.png", type="image/png"), Body(
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
                padding: 20px;
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
                gap: 30px;
                width: 100%;
                max-width: 1600px;
                justify-content: center;
                padding: 0 10px;
                box-sizing: border-box;
            }
            .canvas-container {
                position: relative;
                border-radius: 24px;
                overflow: hidden;
                border: 2px solid var(--glass-border);
                line-height: 0;
                flex: 2;
                max-width: 850px;
                max-height: 70vh;
                background: #000;
                box-shadow: 0 20px 50px rgba(0,0,0,0.5);
                display: flex;
                align-items: center;
                justify-content: center;
            }
            #output-canvas {
                width: 100%;
                height: 100%;
                object-fit: contain;
                display: block;
            }
            .sidebar {
                display: flex;
                flex-direction: column;
                gap: 20px;
                flex: 1;
                min-width: 320px;
                max-width: 450px;
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
                gap: 10px;
            }
            .control-group {
                display: flex;
                flex-direction: column;
                gap: 8px;
            }
            .control-label {
                font-size: 0.8rem;
                font-weight: 600;
                color: var(--primary);
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            .checkbox-group {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 15px;
                cursor: pointer;
                user-select: none;
                padding: 12px 18px;
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid var(--glass-border) !important;
                border-radius: 16px;
                transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
                width: 100%;
                box-sizing: border-box;
            }
            .checkbox-group:hover {
                background: rgba(255, 255, 255, 0.08);
                border-color: rgba(255, 255, 255, 0.2) !important;
            }
            /* Switch Container */
            .switch {
                position: relative;
                display: block !important;
                width: 50px !important;
                height: 26px !important;
                min-width: 50px !important;
                margin: 0 !important;
                padding: 0 !important;
                border: none !important;
                background: none !important;
                cursor: pointer;
            }
            .switch input { 
                opacity: 0 !important;
                width: 0 !important;
                height: 0 !important;
                position: absolute !important;
            }
            /* The track */
            .slider-toggle {
                position: absolute;
                cursor: pointer;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background-color: #334155;
                transition: .3s;
                border-radius: 34px;
                border: none !important;
            }
            /* The thumb */
            .slider-toggle:before {
                position: absolute;
                content: "";
                height: 20px !important;
                width: 20px !important;
                left: 3px !important;
                top: 3px !important;
                background-color: white !important;
                transition: .3s cubic-bezier(0.68, -0.55, 0.265, 1.55);
                border-radius: 50%;
                box-shadow: 0 2px 5px rgba(0,0,0,0.3);
                z-index: 2;
            }
            input:checked + .slider-toggle {
                background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
                box-shadow: 0 0 15px rgba(88, 166, 255, 0.3);
            }
            input:checked + .slider-toggle:before {
                transform: translateX(24px) !important;
            }
            #status-text { margin: 0; font-size: 0.8rem; opacity: 0.7; text-align: center; }
            
            /* Custom Range Slider */
            input[type="range"] {
                -webkit-appearance: none;
                background: transparent;
                padding: 0;
                border: none;
                height: 20px;
                width: 100%;
            }
            input[type="range"]:focus { outline: none; }
            
            /* Webkit (Chrome/Safari/Edge) */
            input[type="range"]::-webkit-slider-runnable-track {
                background: #334155;
                height: 6px;
                border-radius: 10px;
            }
            input[type="range"]::-webkit-slider-thumb {
                -webkit-appearance: none;
                height: 20px;
                width: 20px;
                background: #ffffff;
                border: 3px solid #0f172a;
                border-radius: 50%;
                margin-top: -7px;
                box-shadow: 0 0 10px rgba(255, 255, 255, 0.2);
                cursor: pointer;
                transition: all 0.2s;
            }
            input[type="range"]::-webkit-slider-thumb:hover {
                transform: scale(1.1);
                border-color: var(--primary);
            }
            
            /* Firefox */
            input[type="range"]::-moz-range-track {
                background: #334155;
                height: 6px;
                border-radius: 10px;
            }
            input[type="range"]::-moz-range-thumb {
                height: 16px;
                width: 16px;
                background: #ffffff;
                border: 3px solid #0f172a;
                border-radius: 50%;
                box-shadow: 0 0 10px rgba(255, 255, 255, 0.2);
                cursor: pointer;
            }
            #fps-counter {
                position: absolute;
                top: 15px;
                right: 15px;
                background: rgba(15, 23, 42, 0.7);
                padding: 6px 12px;
                border-radius: 12px;
                font-size: 0.75rem;
                font-weight: 700;
                color: var(--primary);
                border: 1px solid var(--glass-border);
                backdrop-filter: blur(8px);
                z-index: 10;
                display: flex;
                align-items: center;
                gap: 6px;
            }
            .fps-dot {
                width: 6px;
                height: 6px;
                background: #10b981;
                border-radius: 50%;
                box-shadow: 0 0 8px #10b981;
                animation: pulse 2s infinite;
            }
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.4; }
                100% { opacity: 1; }
            }
            
            @media (max-width: 1100px) {
                .dashboard {
                    flex-direction: column;
                    align-items: center;
                }
                .canvas-container {
                    width: 100%;
                    max-width: 800px;
                }
                .sidebar {
                    width: 100%;
                    max-width: 800px;
                }
            }
        """),
        H1("NLW Operator AI"),
        Div(
            Div(
                Div(Span(cls="fps-dot"), Span("FPS: --", id="fps-value"), id="fps-counter"),
                Canvas(id="output-canvas", width="640", height="480"),
                Video(id="video", width="640", height="480", style="display:none", autoplay=True),
                cls="canvas-container"
            ),
            Div(
                Div(
                    Div(
                        Label("Modo de Operação", _for="mode-select", cls="control-label"),
                        Select(
                            Option("Object Detection", value="Object Detection", selected=True),
                            Option("Gesture Recognition", value="Gesture Recognition"),
                            id="mode-select", onchange="updateState()"
                        ),
                        cls="control-group"
                    ),
                    Div(
                        Label("Qualidade da Imagem", _for="quality-slider", cls="control-label"),
                        Input(type="range", id="quality-slider", min="10", max="100", value="70"),
                        cls="control-group"
                    ),
                    Div(
                        Div(
                            Span("Mostrar Landmarks", cls="control-label", style="margin:0"),
                            Label(
                                Input(type="checkbox", id="landmarks-checkbox", checked="checked"),
                                Span(cls="slider-toggle"),
                                cls="switch"
                            ),
                            cls="checkbox-group"
                        ),
                        cls="control-group"
                    ),
                    Div(
                        Div(
                            Span("Mostrar FPS", cls="control-label", style="margin:0"),
                            Label(
                                Input(type="checkbox", id="fps-checkbox", checked="checked"),
                                Span(cls="slider-toggle"),
                                cls="switch"
                            ),
                            cls="checkbox-group"
                        ),
                        cls="control-group"
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
    quality = 70
    show_landmarks = True
    last_time = time.time()

    while True:
        try:
            msg = await ws.receive()
            
            if 'text' in msg:
                config = json.loads(msg['text'])
                if config.get('type') == 'config':
                    conn_mode = config.get('mode', conn_mode)
                    quality = config.get('quality', quality)
                    show_landmarks = config.get('show_landmarks', show_landmarks)
                continue

            if 'bytes' in msg:
                # Calcula FPS
                current_time = time.time()
                fps = 1.0 / (current_time - last_time) if current_time > last_time else 0
                last_time = current_time

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
                        img, labels, images_to_show = processor.process_gesture_recognition(img, show_landmarks=show_landmarks)
                    else:
                        labels = ["Reconhecedor não carregado"]
                
                # Envia JSON com os resultados das detecções e o FPS
                payload = {"type": "labels", "data": labels, "fps": round(fps, 1)}
                if images_to_show:
                    payload["images"] = images_to_show
                    
                await ws.send_json(payload)
                
                # Envia os bytes da imagem processada
                _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
                await ws.send_bytes(buffer.tobytes())

        except Exception as e:
            # print(f"WS Error: {e}")
            break

if __name__ == "__main__":
    serve()


