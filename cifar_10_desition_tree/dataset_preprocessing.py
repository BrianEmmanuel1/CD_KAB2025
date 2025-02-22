from imports import Cifar10 as cif
import numpy as np
import albumentations as A
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def split_cifar_data():
    data = cif()

    x_train = np.array(data.train_data.data).reshape(len(data.train_data), -1) / 255
    y_train = np.array(data.train_data.targets)
    
    x_test = np.array(data.test_data.data).reshape(len(data.test_data), -1) / 255
    y_test = np.array(data.test_data.targets)
    
    return x_train, y_train, x_test, y_test

def split_cifar_data_no_reshape():
    data = cif()

    # NO aplanar las imágenes, mantener el formato original (32,32,3)
    x_train = np.array(data.train_data.data) / 255.0  # Normalizar
    y_train = np.array(data.train_data.targets)
    
    x_test = np.array(data.test_data.data) / 255.0  # Normalizar
    y_test = np.array(data.test_data.targets)
    
    return x_train, y_train, x_test, y_test


def patadas_de_ahogado():
    
    print("Cargando datos...")
    X_train, y_train, X_test, y_test = split_cifar_data_no_reshape()
    print(f"Tamaño inicial del conjunto de entrenamiento: {X_train.shape}, {y_train.shape}")

    # Data Augmentation con Albumentations
    augmenters = A.Compose([
        A.HorizontalFlip(p=0.5),  # Voltea horizontalmente el 50% de las imágenes
        A.Rotate(limit=10, p=1.0),  # Rotación de -10° a 10°
    ])

    num_augmentations = len(X_train)
    print(f"Generando {num_augmentations} imágenes aumentadas...")
    
    X_train_aug = np.array([augmenters(image=img)["image"] for img in X_train[:num_augmentations]])
    y_train_aug = y_train[:num_augmentations]

    # Aplanar las imágenes después de la augmentación
    X_train_aug = X_train_aug.reshape(len(X_train_aug), -1)
    X_train = X_train.reshape(len(X_train), -1)

    print(f"Dimensión después de aplanar: {X_train.shape}")

    # Combinamos datos originales con los aumentados
    X_train_combined = np.concatenate([X_train, X_train_aug])
    y_train_combined = np.concatenate([y_train, y_train_aug])

    print(f"TAamaño después de la augmentación: {X_train_combined.shape}, {y_train_combined.shape}")

    # Balanceo de Clases con SMOTE
    print("Aplicando SMOTE para balanceo de clases...")
    smote = SMOTE()
    X_train_bal, y_train_bal = smote.fit_resample(X_train_combined, y_train_combined)
    
    print(f"Tamaño después de SMOTE: {X_train_bal.shape}, {y_train_bal.shape}")

    # Optimización del Conjunto de Entrenamiento - Eliminación de Ejemplos Difíciles
    print("Entrenando Árbol de Decisión para encontrar ejemplos difíciles...")
    clf_temp = DecisionTreeClassifier(max_depth=10, min_samples_split=10, min_samples_leaf=5, criterion='gini')
    clf_temp.fit(X_train_bal, y_train_bal)
    
    y_pred_train = clf_temp.predict(X_train_bal)
    error_indices = np.where(y_pred_train != y_train_bal)[0]

    num_remove = int(len(error_indices) * 0.05)
    worst_examples = error_indices[:num_remove]

    print(f"Eliminando {num_remove} muestras difíciles de clasificar...")

    X_train_filtered = np.delete(X_train_bal, worst_examples, axis=0)
    y_train_filtered = np.delete(y_train_bal, worst_examples, axis=0)

    print(f"Tamaño final del conjunto de entrenamiento: {X_train_filtered.shape}, {y_train_filtered.shape}")

    return X_train_filtered, y_train_filtered, X_test, y_test