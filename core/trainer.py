import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

class GestureTrainer:
    """Gerencia o treinamento do modelo de reconhecimento de gestos."""
    
    def __init__(self, dataset_path="gesture_dataset.csv"):
        self.dataset_path = dataset_path
        
    def train(self, model_save_path="models/gesture_model.pkl"):
        """Carrega os dados, treina um Random Forest e salva o modelo."""
        if not os.path.exists(self.dataset_path):
            print(f"Error: Dataset {self.dataset_path} not found.")
            return False
            
        print(f"Loading dataset from {self.dataset_path}...")
        df = pd.read_csv(self.dataset_path, header=None)
        
        # Separa os recursos (X) dos rótulos (y) - Convertendo explicitamente para NumPy
        X = df.iloc[:, :-1].to_numpy(dtype='float32')
        y = df.iloc[:, -1].to_numpy()
        
        # Verifica se há pelo menos 2 classes para permitir a divisão estratificada
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            print(f"Error: You need at least 2 different gesture labels to train. Found: {unique_classes}")
            return False

        # Divide os dados em treino (80%) e teste (20%)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("Training Random Forest model...")
        # Inicializa o classificador com 100 árvores de decisão
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
        # Avalia a precisão do modelo no conjunto de teste
        y_pred = clf.predict(X_test)
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Garante que o diretório de destino exista e salva o modelo .pkl
        os.makedirs(os.path.dirname(os.path.abspath(model_save_path)), exist_ok=True)
        joblib.dump(clf, model_save_path)
        print(f"Model saved to {model_save_path}")
        return True
