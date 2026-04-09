import streamlit as st
import cv2
import av
import numpy as np
import threading
import os
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

from core.models import ObjectDetector, GestureRecognizer, CustomGestureClassifier
from core.processor import FrameProcessor
from core.dataset import GestureDataset
from core.visualizer import Visualizer

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="CV Operator - Real-Time AI",
    page_icon="🦾",
    layout="wide",
)

# --- CONFIGURAÇÕES WEB-RTC ---
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- ESTILIZAÇÃO ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .main {
        background-color: #0b0e14;
    }
    
    .stApp {
        background: radial-gradient(circle at top right, #1a2332 0%, #0b0e14 100%);
    }

    h1 {
        background: linear-gradient(90deg, #58a6ff, #bc85ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        letter-spacing: -1px;
    }

    .stSidebar {
        background-color: rgba(13, 17, 23, 0.8) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    .status-card {
        padding: 1.5rem;
        border-radius: 16px;
        background: rgba(33, 38, 45, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 1rem;
    }

    .metric-label {
        color: #8b949e;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-value {
        color: #f0f6fc;
        font-size: 1.2rem;
        font-weight: 600;
    }

    /* Estilização dos widgets do Streamlit */
    .stSelectbox label, .stTextInput label {
        color: #c9d1d9 !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- CACHE DE MODELOS ---
@st.cache_resource
def load_object_detector():
    if not os.path.exists('models/efficientdet_lite0.tflite'):
        return None
    return ObjectDetector(model_path='models/efficientdet_lite0.tflite')

@st.cache_resource
def load_gesture_recognizer():
    if not os.path.exists('models/gesture_recognizer.task'):
        return None
    return GestureRecognizer(model_path='models/gesture_recognizer.task')

@st.cache_resource
def load_custom_classifier():
    if not os.path.exists('models/gesture_model.pkl'):
        return None
    try:
        return CustomGestureClassifier(model_path='models/gesture_model.pkl')
    except:
        return None

@st.cache_resource
def load_dataset():
    return GestureDataset(file_path='gesture_dataset.csv')

# --- LOGICA DO PROCESSADOR DE VÍDEO ---
class VideoProcessor:
    def __init__(self):
        self.mode = "Object Detection"
        self.detector = load_object_detector()
        self.recognizer = load_gesture_recognizer()
        self.classifier = load_custom_classifier()
        self.dataset = load_dataset()
        self.processor = FrameProcessor(self.detector, self.recognizer, self.classifier)
        self.visualizer = Visualizer()
        
        self.label = ""
        self.is_recording = False
        self.lock = threading.Lock()

    def update_params(self, mode, label, is_recording):
        with self.lock:
            self.mode = mode
            self.label = label
            self.is_recording = is_recording

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        with self.lock:
            mode = self.mode
            label = self.label
            is_recording = self.is_recording

        if mode == "Object Detection":
            if self.detector:
                img = self.processor.process_object_detection(img)
            else:
                cv2.putText(img, "Modelo Detection nao encontrado", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        elif mode == "Gesture Recognition":
            img = cv2.flip(img, 1) # Espelhamento
            if self.recognizer:
                img = self.processor.process_gesture_recognition(img)
            else:
                cv2.putText(img, "Modelo Gesture nao encontrado", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        elif mode == "Data Collection":
            img = cv2.flip(img, 1)
            if self.recognizer:
                # MediaPipe exige RGB
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.recognizer.recognize(rgb_img)
                
                if is_recording and results.hand_landmarks:
                    for landmarks in results.hand_landmarks:
                        self.dataset.save_landmarks(landmarks, label)
                
                if results.hand_landmarks:
                    self.visualizer.draw_hand_landmarks(img, results.hand_landmarks[0])
                
                status_text = f"RECORDING: {label}" if is_recording else "PAUSED"
                color = (0, 0, 255) if is_recording else (255, 0, 0)
                self.visualizer.draw_status(img, status_text, color=color)
            else:
                cv2.putText(img, "Modelo Recognizer nao encontrado", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI PRINCIPAL ---
def main():
    if "processor" not in st.session_state:
        st.session_state.processor = VideoProcessor()

    st.title("🦾 CV Operator AI")
    st.markdown("### Próxima Geração de Reconhecimento em Tempo Real")
    
    # Sidebar
    st.sidebar.title("⚙️ Painel de Controle")
    mode = st.sidebar.selectbox(
        "Selecione o Fluxo de Trabalho",
        ["Object Detection", "Gesture Recognition", "Data Collection"]
    )
    
    label = ""
    is_recording = False
    
    st.sidebar.markdown("---")
    
    if mode == "Data Collection":
        st.sidebar.subheader("💎 Coleta de Dados")
        label = st.sidebar.text_input("Rótulo do Gesto:", value="Gesto_X")
        
        if "is_recording" not in st.session_state:
            st.session_state.is_recording = False
            
        if st.sidebar.button("🔴 Alternar Gravação"):
            st.session_state.is_recording = not st.session_state.is_recording
        
        is_recording = st.session_state.is_recording
        
        record_status = "GRAVANDO" if is_recording else "PARADO"
        color = "#ff4b4b" if is_recording else "#8b949e"
        st.sidebar.markdown(f"Status: <span style='color:{color}; font-weight:bold;'>{record_status}</span>", unsafe_allow_html=True)
        st.sidebar.info("As marcas das mãos (landmarks) são salvas em `gesture_dataset.csv`.")

    # Atualiza parâmetros no processador (thread-safe)
    st.session_state.processor.update_params(mode, label, is_recording)

    # Layout com Colunas
    col_video, col_info = st.columns([3, 1])
    
    with col_video:
        webrtc_streamer(
            key="cv-operator-stream",
            video_frame_callback=st.session_state.processor.recv,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with col_info:
        st.markdown("#### Meta-Dados")
        
        # Card 1: Modo
        st.markdown(f"""
        <div class="status-card">
            <div class="metric-label">Modo Operacional</div>
            <div class="metric-value">{mode}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Card 2: Modelos
        models_status = "Online" if st.session_state.processor.detector or st.session_state.processor.recognizer else "Offline"
        st.markdown(f"""
        <div class="status-card">
            <div class="metric-label">Motores de IA</div>
            <div class="metric-value">{models_status}</div>
        </div>
        """, unsafe_allow_html=True)
        
        if mode == "Gesture Recognition":
            if st.session_state.processor.classifier:
                st.success("🎯 Classificador Customizado Ativo")
            else:
                st.warning("⚠️ Modelo 'gesture_model.pkl' ausente")

        if mode == "Data Collection" and is_recording:
            st.error(f"📡 Capturando [{label}]")

    st.markdown("---")
    st.caption("Desenvolvido para NLW 22 - Operator (Python)")

if __name__ == "__main__":
    main()

