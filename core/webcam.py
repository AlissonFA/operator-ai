import cv2

def flip_frame(frame):
    """Inverte o frame horizontalmente (efeito espelho)."""
    return cv2.flip(frame, 1)

class WebcamHandler:
    """Gerencia a captura da webcam via OpenCV e o gerenciamento básico de janelas."""
    
    def __init__(self, camera_index=0, width=None, height=None):
        # Abre a conexão com a câmera (índice 0 é geralmente a padrão)
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Não foi possível abrir a webcam.")
            
        # Configura a largura e altura da imagem, se especificado
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
    def read_frame(self, flip=False):
        """Lê um frame da webcam e opcionalmente o inverte."""
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Inverte o frame se solicitado (útil para interação tipo espelho)
        if flip:
            frame = flip_frame(frame)
        return frame
    
    def show_frame(self, window_name, frame):
        """Exibe o frame em uma janela do OpenCV."""
        cv2.imshow(window_name, frame)
        
    def check_exit(self, key_code=ord('q')):
        """Retorna True se a tecla de saída ('q' por padrão) for pressionada."""
        return cv2.waitKey(1) & 0xFF == key_code
        
    def release(self):
        """Libera os recursos da webcam e fecha todas as janelas abertas."""
        self.cap.release()
        cv2.destroyAllWindows()
