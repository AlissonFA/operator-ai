import cv2
from .visualizer import Visualizer

class FrameProcessor:
    """Orquestra a inferência dos modelos e a visualização dos resultados."""
    
    def __init__(self, detector=None, recognizer=None, classifier=None):
        self.detector = detector
        self.recognizer = recognizer
        self.classifier = classifier
        self.visualizer = Visualizer()

    def process_object_detection(self, frame):
        """Executa a detecção de objetos e anota o frame com caixas e rótulos."""
        # MediaPipe exige imagem em RGB, OpenCV usa BGR
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.detect(rgb_frame)
        # Chama o visualizador para desenhar os objetos detectados no frame original
        return self.visualizer.draw_objects(frame, results)

    def process_gesture_recognition(self, frame):
        """Executa o rastreamento de mãos e a classificação de gestos customizados."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.recognizer.recognize(rgb_frame)
        
        # Se houver mãos detectadas
        if results.hand_landmarks:
            legends = []
            for i, landmarks in enumerate(results.hand_landmarks):
                # Desenha os pontos e esqueletos das mãos
                self.visualizer.draw_hand_landmarks(frame, landmarks)
                
                # Se houver um classificador treinado, identifica o gesto
                if self.classifier:
                    gesture_name = self.classifier.predict(landmarks)
                    
                    # Identifica se a mão é Esquerda ou Direita
                    # Obs: O MediaPipe inverte a lateralidade em imagens espelhadas, filtramos aqui
                    hand_info = results.handedness[i][0]
                    side = "Right" if hand_info.category_name == "Left" else "Left"
                    legends.append(f"{side}: {gesture_name.upper()}")
            
            # Desenha as legendas dos gestos no canto superior da tela
            for idx, text in enumerate(legends):
                self.visualizer.draw_status(frame, text, position=(20, 50 + idx*40))
                
        return frame
