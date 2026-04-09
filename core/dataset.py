import csv
import os

class GestureDataset:
    """Gerencia os dados brutos de pontos (landmarks) das mãos armazenados em CSV."""
    
    def __init__(self, file_path="gesture_dataset.csv"):
        self.file_path = file_path
        
    def save_landmarks(self, hand_landmarks, label):
        """Achata (flattens) os pontos e os salva no CSV junto com o rótulo."""
        data = []
        # Converte cada ponto (x,y,z) em valores individuais na lista
        for landmark in hand_landmarks:
            data.extend([landmark.x, landmark.y, landmark.z])
        
        # Adiciona o nome do gesto (label) como a última coluna
        data.append(label)
        
        # Garante que o diretório exista antes de salvar
        os.makedirs(os.path.dirname(os.path.abspath(self.file_path)), exist_ok=True)
        
        # Abre o arquivo em modo 'append' (adicionar ao final)
        with open(self.file_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data)
            
    def get_distribution(self):
        """Retorna a contagem de amostras coletadas por classe (gesto)."""
        if not os.path.exists(self.file_path):
            return {}
            
        import pandas as pd
        # Lê o CSV e conta as ocorrências da última coluna
        df = pd.read_csv(self.file_path, header=None)
        return df.iloc[:, -1].value_counts().to_dict()
