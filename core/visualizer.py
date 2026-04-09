import cv2

class Visualizer:
    """Responsável por desenhar detecções, marcos (landmarks) das mãos e sobreposições nos frames."""
    
    @staticmethod
    def draw_objects(image, detection_result):
        """Desenha caixas delimitadoras (bounding boxes) nos frames."""
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            
            # Desenha o retângulo da caixa delimitadora
            cv2.rectangle(image, start_point, end_point, (0, 255, 0), 3)

        return image

    @staticmethod
    def draw_hand_landmarks(image, hand_landmarks, connections=None):
        """Desenha os pontos (landmarks) e as conexões da mão."""
        h, w, _ = image.shape
        
        # Conexões padrão do MediaPipe para as mãos se não forem fornecidas
        HAND_CONNECTIONS = connections or [
            (0, 1), (1, 2), (2, 3), (3, 4), # polegar
            (0, 5), (5, 6), (6, 7), (7, 8), # indicador
            (5, 9), (9, 10), (10, 11), (11, 12), # médio
            (9, 13), (13, 14), (14, 15), (15, 16), # anelar
            (13, 17), (17, 18), (18, 19), (19, 20), # mínimo
            (0, 17) # palma
        ]

        # Desenha as linhas de conexão entre os pontos
        for connection in HAND_CONNECTIONS:
            p1 = hand_landmarks[connection[0]]
            p2 = hand_landmarks[connection[1]]
            cv2.line(image, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), (200, 200, 200), 1)

        # Desenha cada ponto (círculo) com cores específicas para cada dedo
        for idx, landmark in enumerate(hand_landmarks):
            color = (255, 255, 255)
            if 1 <= idx <= 4: color = (0, 255, 255)    # Polegar
            elif 5 <= idx <= 8: color = (0, 255, 0)     # Indicador
            elif 9 <= idx <= 12: color = (255, 255, 0)  # Médio
            elif 13 <= idx <= 16: color = (255, 0, 0)   # Anelar
            elif 17 <= idx <= 20: color = (255, 0, 255) # Mínimo
            cv2.circle(image, (int(landmark.x*w), int(landmark.y*h)), 5, color, -1)
            
        return image

    @staticmethod
    def draw_status(image, text, position=(20, 50), color=(0, 255, 255)):
        """Desenha um texto de status (como gesto atual ou status de gravação)."""
        # Desenha uma borda preta para melhorar a legibilidade (contorno)
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        # Desenha o texto principal por cima
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        return image
