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

    def process_gesture_recognition(self, frame, show_landmarks=True):
        """Executa o rastreamento de mãos e a classificação de gestos customizados."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.recognizer.recognize(rgb_frame)
        
        legends = []
        images_to_show = []
        
        # Se houver mãos detectadas
        if results.hand_landmarks:
            for i, landmarks in enumerate(results.hand_landmarks):
                # Desenha os pontos e esqueletos das mãos se habilitado
                if show_landmarks:
                    self.visualizer.draw_hand_landmarks(frame, landmarks)
                
                # Se houver um classificador treinado, identifica o gesto
                if self.classifier:
                    gesture_name = self.classifier.predict(landmarks)
                    
                    # Padroniza para formato de arquivo (ex: "mao aberta" -> "mao-aberta")
                    gesture_name_formatted = gesture_name.replace(" ", "-").replace("_", "-").lower()
                    
                    images_to_show.append(f"{gesture_name_formatted}.png")
                    
                    # Identifica se a mão é Esquerda ou Direita
                    hand_info = results.handedness[i][0]
                    side = "Right" if hand_info.category_name == "Left" else "Left"
                    legends.append(f"{side}: {gesture_name.upper()}")
        else:
            legends.append("Nenhum gesto detectado")
                
        return frame, legends, images_to_show
