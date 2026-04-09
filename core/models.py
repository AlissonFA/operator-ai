import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib
import numpy as np

class ObjectDetector:
    """Invólucro para a Detecção de Objetos do MediaPipe."""
    def __init__(self, model_path='models/efficientdet_lite0.tflite', score_threshold=0.5):
        # Configurações básicas para carregar o modelo TFLite
        base_options = python.BaseOptions(model_asset_path=model_path)
        # Configura as opções do detector, incluindo o limiar de confiança (score_threshold)
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            score_threshold=score_threshold,
            running_mode=vision.RunningMode.IMAGE
        )
        # Cria a instância do detector
        self.detector = vision.ObjectDetector.create_from_options(options)

    def detect(self, numpy_image):
        """Converte a imagem do OpenCV para o formato MediaPipe e realiza a detecção."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)
        return self.detector.detect(mp_image)


class GestureRecognizer:
    """Invólucro para o Reconhecimento de Gestos (Landmarks) do MediaPipe."""
    def __init__(self, model_path='models/gesture_recognizer.task', num_hands=2):
        # Carrega o arquivo de tarefa (.task) do MediaPipe
        base_options = python.BaseOptions(model_asset_path=model_path)
        # Configura o reconhecedor para detectar até num_hands mãos
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)

    def recognize(self, numpy_image):
        """Realiza o reconhecimento de pontos e gestos básicos na imagem."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)
        return self.recognizer.recognize(mp_image)


class CustomGestureClassifier:
    """Invólucro para o modelo de classificação de gestos treinado com scikit-learn."""
    def __init__(self, model_path='models/gesture_model.pkl'):
        # Carrega o modelo salvo em formato .pkl usando joblib
        self.model = joblib.load(model_path)

    def predict(self, hand_landmarks):
        """Recebe os pontos brutos da mão e prevê o gesto usando o modelo scikit-learn."""
        features = []
        # Converte os objetos de landmark para uma lista plana de coordenadas [x, y, z, x, y, z, ...]
        for lm in hand_landmarks:
            features.extend([lm.x, lm.y, lm.z])
        
        # Realiza a predição (retorna uma lista, pegamos o primeiro elemento)
        prediction = self.model.predict([features])
        return prediction[0]
