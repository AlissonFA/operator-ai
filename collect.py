from core.webcam import WebcamHandler
from core.models import GestureRecognizer
from core.dataset import GestureDataset
from core.visualizer import Visualizer

def main():
    # Solicita ao usuário o nome do gesto que será coletado
    LABEL = input("Enter gesture label to collect: ").strip()
    if not LABEL:
        print("Empty label. Exiting.")
        return

    # Inicializa os componentes necessários
    webcam = WebcamHandler()
    recognizer = GestureRecognizer(model_path='models/gesture_recognizer.task')
    dataset = GestureDataset(file_path='gesture_dataset.csv')
    visualizer = Visualizer()

    is_recording = False
    print(f"\nCollecting for: {LABEL}")
    print("Commands: 's' to Toggle Recording, 'q' to Quit\n")

    try:
        while True:
            # Lê o frame da webcam com espelhamento
            frame = webcam.read_frame(flip=True)
            if frame is None: break

            # Reconhece as mãos no frame atual
            results = recognizer.recognize(frame)
            
            # Se estiver gravando e houver mãos visíveis, salva os pontos no dataset
            if is_recording and results.hand_landmarks:
                for landmarks in results.hand_landmarks:
                    dataset.save_landmarks(landmarks, LABEL)

            # --- Interface do Usuário (UI) ---
            # Desenha os pontos da mão se detectada
            visualizer.draw_hand_landmarks(frame, results.hand_landmarks[0]) if results.hand_landmarks else None
            
            # Define o status e cor do texto na tela
            status = f"RECORDING: {LABEL}" if is_recording else "PAUSED"
            color = (0, 0, 255) if is_recording else (255, 0, 0)
            visualizer.draw_status(frame, status, color=color)
            
            # Exibe o frame na janela
            webcam.show_frame("Data Collection", frame)

            # Verifica comandos do teclado
            if webcam.check_exit(ord('q')): # Sair
                break
            if webcam.check_exit(ord('s')): # Ligar/Desligar gravação
                is_recording = not is_recording
                print(f"Status: {'Recording' if is_recording else 'Paused'}")

    finally:
        # Libera a câmera ao finalizar
        webcam.release()
        print("Capture stopped.")

if __name__ == "__main__":
    main()
