from core.trainer import GestureTrainer

def main():
    # Inicializa o treinador com o caminho do dataset CSV
    trainer = GestureTrainer(dataset_path='gesture_dataset.csv')
    
    # Executa o treinamento e salva o modelo resultante
    success = trainer.train(model_save_path='models/gesture_model.pkl')
    
    if success:
        print("\nTraining completed successfully!")
    else:
        print("\nTraining failed.")

if __name__ == "__main__":
    main()
