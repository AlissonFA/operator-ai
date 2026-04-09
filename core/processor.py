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
        """Executa a detecção de objetos e anota o frame com caixas."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.detect(rgb_frame)
        
        detections = []
        for detection in results.detections:
            category = detection.categories[0]
            detections.append(f"{category.category_name} ({round(category.score, 2)})")
            
        frame = self.visualizer.draw_objects(frame, results)
        return frame, detections

    def process_gesture_recognition(self, frame):
        """Executa o rastreamento de mãos e a classificação de gestos customizados."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.recognizer.recognize(rgb_frame)
        
        legends = []
        image_to_show = None
        gestures = []
        
        # Se houver mãos detectadas
        if results.hand_landmarks:
            for i, landmarks in enumerate(results.hand_landmarks):
                # Desenha os pontos e esqueletos das mãos
                self.visualizer.draw_hand_landmarks(frame, landmarks)
                
                # Se houver um classificador treinado, identifica o gesto
                if self.classifier:
                    gesture_name = self.classifier.predict(landmarks)
                    gestures.append(gesture_name)
                    
                    # Identifica se a mão é Esquerda ou Direita
                    hand_info = results.handedness[i][0]
                    side = "Right" if hand_info.category_name == "Left" else "Left"
                    legends.append(f"{side}: {gesture_name.upper()}")
            
            if len(gestures) == 2 and gestures[0] == gestures[1]:
                # O gesto deve bater com o nome da imagem (ex: "coracao-coreano" -> coracao-coreano.png)
                image_to_show = f"{gestures[0]}.png"
                
        return frame, legends, image_to_show
