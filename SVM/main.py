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
from tqdm import tqdm  # Importamos tqdm para mostrar barras de progreso

# Configurar logging
logging.basicConfig(
    filename='svm_performance.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('svm_monitor')

# Configurar un handler adicional para mostrar logs en la consola
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)


def log_model_performance(model, X_train, y_train, X_test, y_test, model_name="SVM"):
    print(f"\n===== Entrenando modelo: {model_name} =====")
    
    # Entrenamiento con barra de progreso
    start_time = time.time()
    print("Entrenando modelo...")
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"Entrenamiento completado en {train_time:.4f} segundos")
    
    # Predicción con barra de progreso
    start_time = time.time()
    print("Realizando predicciones...")
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_time
    print(f"Predicción completada en {predict_time:.4f} segundos")
    
    # Cálculo de métricas
    print("Calculando métricas de rendimiento...")
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    # Mostrar métricas en la consola
    print(f"\n----- Resultados de {model_name} -----")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted precision: {precision:.4f}")
    print(f"Weighted recall: {recall:.4f}")
    print(f"Weighted F1-score: {f1:.4f}")
    
    # Guardar en el log
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
    print(f"\nInforme de clasificación:\n{class_report}")
    
    # Registrar matriz de confusión
    conf_matrix = confusion_matrix(y_test, y_pred)
    logger.info(f"Matriz de confusión:\n{conf_matrix}")
    print(f"\nMatriz de confusión:\n{conf_matrix}")
    
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
    print("\nGenerando visualizaciones...")
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
    print(f"Visualización guardada en {save_path}")

def save_features_to_csv(features, labels, feature_names, filename):
    """
    Saves extracted features and labels to a CSV file.
    
    Args:
        features (np.ndarray): Feature matrix.
        labels (np.ndarray): Corresponding labels.
        feature_names (list): List of feature names.
        filename (str): Name of the CSV file.
    """
    print(f"\nGuardando características en {filename}...")
    # Crear un DataFrame con las características y los nombres de las columnas
    df = pd.DataFrame(features, columns=feature_names)  # Usar feature_names para las columnas
    df["Label"] = labels  # Agregar las etiquetas como última columna
    
    # Guardar el DataFrame en el archivo CSV
    df.to_csv(filename, index=False)
    logger.info(f"Características guardadas en {filename}")
    print(f"Características guardadas correctamente en {filename}")
    
# Argument parser for selecting preprocessing mode
parser = argparse.ArgumentParser(description="Choose preprocessing mode: simple or complex.")
parser.add_argument("--mode", choices=["simple", "complex"], default="simple", help="Preprocessing type")
args = parser.parse_args()
print(f"Modo seleccionado: {args.mode}")

# Define filenames based on the chosen mode
train_csv = f"train_dataset_{args.mode}.csv"
test_csv = f"test_dataset_{args.mode}.csv"

if __name__ == "__main__":
    print("\n====== INICIANDO PROCESO SVM ======")
    start_total = time.time()
    
    if os.path.exists(train_csv) and os.path.exists(test_csv):
        print(f"Archivos CSV encontrados. Cargando datos de {train_csv} y {test_csv}...")
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        print(f"Datos cargados - Train: {train_df.shape}, Test: {test_df.shape}")

        # Separate features and labels
        X_train = train_df.drop(columns=['Label'])
        y_train = train_df['Label']
        X_test = test_df.drop(columns=['Label'])
        y_test = test_df['Label']
        print(f"Características y etiquetas separadas correctamente")
        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        # Mostrar distribución de clases
        train_dist = pd.Series(y_train).value_counts().to_dict()
        test_dist = pd.Series(y_test).value_counts().to_dict()
        print(f"Distribución de clases en train: {train_dist}")
        print(f"Distribución de clases en test: {test_dist}")
    else:
        print("=== CSV no encontrado, iniciando preprocesamiento... ===")
        logger.info("=== CSV not found, initiating preprocessing... ===")
        logger.info(f"Loading and preprocessing data using {args.mode} mode...")
        print(f"Cargando y preprocesando datos usando modo {args.mode}...")
        
        # Usar tqdm para mostrar el progreso del preprocesamiento
        print("Este proceso puede tomar tiempo, espere por favor...")
        X_train, y_train, X_test, y_test, feature_names = preprocess_dataset(mode=args.mode)

        # Log dataset info
        logger.info(f"Shape of X_train: {X_train.shape}")
        logger.info(f"Shape of X_test: {X_test.shape}")
        logger.info(f"Class Distribution in train: {np.unique(y_train, return_counts=True)}")
        logger.info(f"Class Distribution in test: {np.unique(y_test, return_counts=True)}")
        
        print(f"Preprocesamiento completado:")
        print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
        print(f"Distribución de clases en train: {np.unique(y_train, return_counts=True)}")
        print(f"Distribución de clases en test: {np.unique(y_test, return_counts=True)}")

        save_features_to_csv(X_train, y_train, feature_names, train_csv)
        save_features_to_csv(X_test, y_test, feature_names, test_csv)

    try:
        # Train and evaluate SVM models
        print("\n====== ENTRENAMIENTO DE MODELOS ======")
        
        print("\n1. Entrenando SVM básico (kernel RBF por defecto)...")
        svm_model = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True), verbose=True)
        results = log_model_performance(svm_model, X_train, y_train, X_test, y_test)

        print("\n2. Entrenando SVM con kernel lineal...")
        svm_linear = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True), verbose=True)
        linear_results = log_model_performance(svm_linear, X_train, y_train, X_test, y_test, "SVM-linear")

        print("\n3. Entrenando SVM con C=10...")
        svm_c10 = make_pipeline(StandardScaler(), SVC(C=10, gamma='auto', probability=True), verbose=True)
        c10_results = log_model_performance(svm_c10, X_train, y_train, X_test, y_test, "SVM-C10")

        # Generar visualizaciones
        visualize_results(c10_results, save_path='svm_results.png')

        # Resumen final
        print("\n====== RESUMEN DE RESULTADOS ======")
        print(f"SVM Base (RBF) - Accuracy: {results['accuracy']:.4f}")
        print(f"SVM Linear - Accuracy: {linear_results['accuracy']:.4f}")
        print(f"SVM C=10 (RBF) - Accuracy: {c10_results['accuracy']:.4f}")
        print("Ver detalles completos en svm_performance.log")

        total_time = time.time() - start_total
        print(f"\n====== PROCESO COMPLETADO EN {total_time:.2f} SEGUNDOS ======")
        logger.info(f"=== PROCESS COMPLETED IN {total_time:.2f} SECONDS ===")

    except Exception as e:
        logger.error(f"ERROR: {str(e)}", exc_info=True)
        logger.error("=== PROCESS FINISHED WITH ERRORS ===")
        print(f"\n¡ERROR!: {str(e)}")
        print("El proceso finalizó con errores. Consulte svm_performance.log para más detalles.")