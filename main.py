import sys
# Importa as classes necessárias do pacote core
from core.webcam import WebcamHandler
from core.models import ObjectDetector, GestureRecognizer, CustomGestureClassifier
from core.processor import FrameProcessor

def run_object_detection():
    """
    Executa o modo de detecção de objetos em tempo real via webcam.
    """
    print("Starting Object Detection...")
    # Inicializa o detector com o modelo pré-treinado
    detector = ObjectDetector(model_path='models/efficientdet_lite0.tflite')
    # Configura o processador de frames com o detector
    processor = FrameProcessor(detector=detector)
    # Inicializa o manipulador da webcam
    webcam = WebcamHandler()
    
    try:
        while True:
            # Captura um novo frame da webcam
            frame = webcam.read_frame()
            if frame is None: break
            
            # Processa o frame para detectar objetos
            frame = processor.process_object_detection(frame)
            # Exibe o frame processado em uma janela
            webcam.show_frame("Object Detection", frame)
            
            # Verifica se o usuário pressionou a tecla de saída (ESC ou 'q')
            if webcam.check_exit():
                break
    finally:
        # Garante que a webcam seja liberada corretamente
        webcam.release()

def run_gesture_recognition():
    """
    Executa o modo de reconhecimento de gestos em tempo real.
    """
    print("Starting Gesture Recognition...")
    # Inicializa o reconhecedor de gestos do MediaPipe
    recognizer = GestureRecognizer(model_path='models/gesture_recognizer.task')
    try:
        # Tenta carregar o classificador de gestos customizado (modelo treinado)
        classifier = CustomGestureClassifier(model_path='models/gesture_model.pkl')
    except Exception as e:
        print(f"Warning: Custom gesture model not found or failed to load: {e}")
        classifier = None
        
    # Configura o processador com o reconhecedor e o classificador (se disponível)
    processor = FrameProcessor(recognizer=recognizer, classifier=classifier)
    webcam = WebcamHandler()
    
    try:
        while True:
            # Captura o frame (com espelhamento para facilitar a interação)
            frame = webcam.read_frame(flip=True)
            if frame is None: break
            
            # Processa o frame para identificar mãos e gestos
            frame = processor.process_gesture_recognition(frame)
            webcam.show_frame("Gesture Recognition", frame)
            
            if webcam.check_exit():
                break
    finally:
        webcam.release()

if __name__ == "__main__":
    # Menu principal da aplicação via CLI
    print("\n--- CV Operator CLI ---")
    print("1: Object Detection")
    print("2: Gesture Recognition")
    print("q: Quit")
    
    choice = input("\nSelect mode: ").strip().lower()
    
    if choice == '1':
        run_object_detection()
    elif choice == '2':
        run_gesture_recognition()
    elif choice == 'q':
        sys.exit()
    else:
        print("Invalid choice.")
