import time
import logging
import numpy as np
import pandas as pd
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from dataset_preprocessing import preprocess_dataset

# Configurar logging
logging.basicConfig(
    filename='svm_performance.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('svm_monitor')


def log_model_performance(model, X_train, y_train, X_test, y_test, model_name="SVM"):

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    

    start_time = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_time
    

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    

    logger.info(f"Model: {model_name}")
    logger.info(f"Training time: {train_time:.4f} seconds")
    logger.info(f"Prediction time: {predict_time:.4f} seconds")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Weighted precision: {precision:.4f}")
    logger.info(f"Weighted recall: {recall:.4f}")
    logger.info(f"Weighted F1-score: {f1:.4f}")
    
    # Registrar informe de clasificación detallado
    class_report = classification_report(y_test, y_pred)
    logger.info(f"Informe de clasificación:\n{class_report}")
    
    # Registrar matriz de confusión
    conf_matrix = confusion_matrix(y_test, y_pred)
    logger.info(f"Matriz de confusión:\n{conf_matrix}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'train_time': train_time,
        'predict_time': predict_time,
        'confusion_matrix': conf_matrix,
        'y_pred': y_pred
    }

def visualize_results(results, save_path='svm_results.png'):
    """Genera visualizaciones de los resultados y los guarda en un archivo"""
    # Crear figura con subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfico de barras para métricas
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    values = [results['accuracy'], results['precision'], results['recall'], results['f1']]
    axs[0].bar(metrics, values, color='skyblue')
    axs[0].set_ylim(0, 1)
    axs[0].set_title('Model Performance')
    
    # Visualizar matriz de confusión
    im = axs[1].imshow(results['confusion_matrix'], interpolation='nearest', cmap=plt.cm.Blues)
    axs[1].set_title('Confusion matrix')
    
    # Añadir valores a la matriz de confusión
    for i in range(results['confusion_matrix'].shape[0]):
        for j in range(results['confusion_matrix'].shape[1]):
            axs[1].text(j, i, str(results['confusion_matrix'][i, j]),
                     ha="center", va="center", color="white" if results['confusion_matrix'][i, j] > results['confusion_matrix'].max()/2 else "black")
    
    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f"Visualization saved in {save_path}")

def save_features_to_csv(features, labels, feature_names, filename):
    """
    Saves extracted features and labels to a CSV file.
    
    Args:
        features (np.ndarray): Feature matrix.
        labels (np.ndarray): Corresponding labels.
        feature_names (list): List of feature names.
        filename (str): Name of the CSV file.
    """
    # Crear un DataFrame con las características y los nombres de las columnas
    df = pd.DataFrame(features, columns=feature_names)  # Usar feature_names para las columnas
    df["Label"] = labels  # Agregar las etiquetas como última columna
    
    # Guardar el DataFrame en el archivo CSV
    df.to_csv(filename, index=False)
    logger.info(f"Características guardadas en {filename}")
    
# Argument parser for selecting preprocessing mode
parser = argparse.ArgumentParser(description="Choose preprocessing mode: simple or complex.")
parser.add_argument("--mode", choices=["simple", "complex"], default="simple", help="Preprocessing type")
args = parser.parse_args()
print(f"selected: {args.mode}")
# Define filenames based on the chosen mode
train_csv = f"train_dataset_{args.mode}.csv"
test_csv = f"test_dataset_{args.mode}.csv"

if __name__ == "__main__":
    if os.path.exists(train_csv) and os.path.exists(test_csv):
        print("CSV files found. Loading data...")
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)

        # Separate features and labels
        X_train = train_df.drop(columns=['label'])
        y_train = train_df['label']
        X_test = test_df.drop(columns=['label'])
        y_test = test_df['label']
    else:
        logger.info("=== CSV not found, initiating preprocessing... ===")
        start_total = time.time()
        logger.info(f"Loading and preprocessing data using {args.mode} mode...")

        X_train, y_train, X_test, y_test, feature_names = preprocess_dataset(mode=args.mode)

        # Log dataset info
        logger.info(f"Shape of X_train: {X_train.shape}")
        logger.info(f"Shape of X_test: {X_test.shape}")
        logger.info(f"Class Distribution in train: {np.unique(y_train, return_counts=True)}")
        logger.info(f"Class Distribution in test: {np.unique(y_test, return_counts=True)}")

        save_features_to_csv(X_train, y_train, feature_names, train_csv)
        save_features_to_csv(X_test, y_test, feature_names, test_csv)

    try:
        # Train and evaluate SVM models
        logger.info("Training SVM base model...")
        svm_model = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
        results = log_model_performance(svm_model, X_train, y_train, X_test, y_test)

        logger.info("Trying SVM with linear Kernel...")
        svm_linear = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))
        linear_results = log_model_performance(svm_linear, X_train, y_train, X_test, y_test, "SVM-linear")

        logger.info("Trying SVM with C=10...")
        svm_c10 = make_pipeline(StandardScaler(), SVC(C=10, gamma='auto', probability=True))
        c10_results = log_model_performance(svm_c10, X_train, y_train, X_test, y_test, "SVM-C10")

        visualize_results(results)

        print(f"SVM Base - Accuracy: {results['accuracy']:.4f}")
        print(f"SVM Linear - Accuracy: {linear_results['accuracy']:.4f}")
        print(f"SVM C=10 - Accuracy: {c10_results['accuracy']:.4f}")
        print("See complete details in svm_performance.log")

        total_time = time.time() - start_total
        logger.info(f"=== PROCESS COMPLETED IN {total_time:.2f} SECONDS ===")

    except Exception as e:
        logger.error(f"ERROR: {str(e)}", exc_info=True)
        logger.error("=== PROCESS FINISHED WITH ERRORS ===")

