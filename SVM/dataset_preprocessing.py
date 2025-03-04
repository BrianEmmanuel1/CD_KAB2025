import cv2
import numpy as np
from scipy.stats import kurtosis, skew, entropy
from import_dataset import load_dataset
from sklearn.preprocessing import StandardScaler
import pywt  # Para transformadas wavelet
import logging
import time
import os

# Configure logging system
def setup_logging():
    """
    Configures the logging system for stroing it in svm_preprocessing.log
    """
    # Creates log directory if not exists
    log_dir = os.path.dirname("svm_preprocessing.log")
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configure the logger
    logger = logging.getLogger("svm_preprocessing")
    logger.setLevel(logging.INFO)
    
    # Creates a file handler
    file_handler = logging.FileHandler("svm_preprocessing.log")
    file_handler.setLevel(logging.INFO)
    
    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Defines log format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Adds logger handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Logger global
logger = setup_logging()

def preprocess_dataset(mode='simple'):
    """
    Loads the CamelyonPatch dataset, preprocesses images, and extracts features.
    
    Returns:
        train_x_features (np.ndarray): Extracted features from training images
        train_y (np.ndarray): Training labels
        test_x_features (np.ndarray): Extracted features from test images
        test_y (np.ndarray): Test labels
        feature_names (list): List of feature names for the dataset
    """
    logger.info("Iniciando preprocesamiento del dataset")
    start_time = time.time()
    
    # Load dataset
    logger.info("Cargando dataset...")
    train_x, train_y, test_x, test_y = load_dataset()
    logger.info(f"Dataset cargado. Tamaño de entrenamiento: {train_x.shape}, Tamaño de prueba: {test_x.shape}")
    
    # Check and convert data types if necessary
    if train_x.dtype == bool:
        logger.info("Convirtiendo imágenes de tipo booleano a uint8")
        train_x = train_x.astype(np.uint8) * 255
        test_x = test_x.astype(np.uint8) * 255
    
    # Convert to grayscale
    logger.info("Convirtiendo imágenes a escala de grises...")
    gray_start_time = time.time()
    train_x_gray = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in train_x])
    test_x_gray = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in test_x])
    logger.info(f"Conversión a escala de grises completada en {time.time() - gray_start_time:.2f} segundos")
    
    # Extract features and feature names
    logger.info("Extrayendo características del conjunto de entrenamiento...")
    train_start_time = time.time()
    train_x_features = []
    feature_names = []
    
    for image in train_x_gray:
        features, names = extract_features(image, mode)
        train_x_features.append(features)
        if not feature_names:
            feature_names = names  # Set feature names from the first image
    
    train_x_features = np.array(train_x_features)
    logger.info(f"Extracción de características de entrenamiento completada en {time.time() - train_start_time:.2f} segundos")
    
    logger.info("Extrayendo características del conjunto de prueba...")
    test_start_time = time.time()
    test_x_features = []
    
    for image in test_x_gray:
        features, _ = extract_features(image, mode)
        test_x_features.append(features)
    
    test_x_features = np.array(test_x_features)
    logger.info(f"Extracción de características de prueba completada en {time.time() - test_start_time:.2f} segundos")
    
    # Normalize features
    logger.info("Normalizando características...")
    scaler = StandardScaler()
    train_x_features = scaler.fit_transform(train_x_features)
    test_x_features = scaler.transform(test_x_features)
    
    # Check for NaN or infinite values
    nan_count_train = np.isnan(train_x_features).sum()
    inf_count_train = np.isinf(train_x_features).sum()
    
    if nan_count_train > 0 or inf_count_train > 0:
        logger.warning(f"Detectados valores inválidos en el conjunto de entrenamiento: {nan_count_train} NaN, {inf_count_train} infinitos")
        logger.info("Reemplazando valores inválidos...")
    
    train_x_features = np.nan_to_num(train_x_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    nan_count_test = np.isnan(test_x_features).sum()
    inf_count_test = np.isinf(test_x_features).sum()
    
    if nan_count_test > 0 or inf_count_test > 0:
        logger.warning(f"Detectados valores inválidos en el conjunto de prueba: {nan_count_test} NaN, {inf_count_test} infinitos")
        logger.info("Reemplazando valores inválidos...")
    
    test_x_features = np.nan_to_num(test_x_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    logger.info(f"Extracción de características completada. Forma de las características de train: {train_x_features.shape}")
    logger.info(f"Extracción de características completada. Forma de las características de test: {test_x_features.shape}")
    logger.info(f"Procesamiento completo en {time.time() - start_time:.2f} segundos")
    
    return train_x_features, train_y, test_x_features, test_y, feature_names


def extract_features(image, mode='simple'):
    """
    Extract statistical and textural features from a grayscale image.
    
    Args:
        image (np.ndarray): Grayscale image
        
    Returns:
        tuple: Feature vector (list of feature values), corresponding column names (list of feature names)
    """
    features = []
    feature_names = []  # Lista para almacenar los nombres de las características
    
    try:
        # Extraer características básicas
        features.append(calculate_mean(image))
        feature_names.append("mean")
        
        features.append(calculate_variance(image))
        feature_names.append("variance")
        
        features.append(calculate_kurtosis(image))
        feature_names.append("kurtosis")
        
        features.append(calculate_skew(image))
        feature_names.append("skew")
        
        features.append(calculate_fractal_dimension(image))
        feature_names.append("fractal_dimension")
        
        features.append(calculate_entropy(image))
        feature_names.append("entropy")
        
    except Exception as e:
        logger.error(f"Error al calcular características básicas: {e}")
        # Rellenar con ceros si hay error
        features.extend([0.0] * 6)
        feature_names.extend(["mean", "variance", "kurtosis", "skew", "fractal_dimension", "entropy"])

    if mode == 'complex':
        try:
            # Extraer características GLCM
            glcm_features = calculate_glcm_features(image)
            features.extend(glcm_features)
            feature_names.extend([f"glcm_feature_{i}" for i in range(len(glcm_features))])
        except Exception as e:
            logger.error(f"Error al calcular características GLCM: {e}")
            features.extend([0.0] * 4)
            feature_names.extend([f"glcm_feature_{i}" for i in range(4)])
    
    try:
        # Gradientes X e Y
        grad_x, grad_y = calculate_gradients(image)
        features.append(np.mean(np.abs(grad_x)))  # Media del gradiente X
        feature_names.append("mean_grad_x")
        
        features.append(np.std(np.abs(grad_x)))   # Desviación estándar del gradiente X
        feature_names.append("std_grad_x")
        
        features.append(np.mean(np.abs(grad_y)))  # Media del gradiente Y
        feature_names.append("mean_grad_y")
        
        features.append(np.std(np.abs(grad_y)))   # Desviación estándar del gradiente Y
        feature_names.append("std_grad_y")
        
    except Exception as e:
        logger.error(f"Error al calcular gradientes: {e}")
        features.extend([0.0] * 4)
        feature_names.extend(["mean_grad_x", "std_grad_x", "mean_grad_y", "std_grad_y"])
    
    try:
        # Magnitud y ángulo del gradiente
        mag, angle = calculate_gradient_magnitude_angle(grad_x, grad_y)
        features.append(np.mean(mag))     # Media de la magnitud del gradiente
        feature_names.append("mean_grad_mag")
        
        features.append(np.std(mag))      # Desviación estándar de la magnitud
        feature_names.append("std_grad_mag")
        
        features.append(np.mean(angle))   # Media del ángulo del gradiente
        feature_names.append("mean_grad_angle")
        
        features.append(np.std(angle))    # Desviación estándar del ángulo
        feature_names.append("std_grad_angle")
        
    except Exception as e:
        logger.error(f"Error al calcular magnitud y ángulo del gradiente: {e}")
        features.extend([0.0] * 4)
        feature_names.extend(["mean_grad_mag", "std_grad_mag", "mean_grad_angle", "std_grad_angle"])
    
    try:
        # Energía de la imagen
        features.append(calculate_energy(image))
        feature_names.append("energy")
        
    except Exception as e:
        logger.error(f"Error al calcular energía de la imagen: {e}")
        features.append(0.0)
        feature_names.append("energy")
    
    if mode == 'complex':
        try:
            # Características Wavelet db9
            wavelet_features = calculate_wavelet_features(image, wavelet='db9', level=2)
            features.extend(wavelet_features)
            feature_names.extend([f"wavelet_feature_{i}" for i in range(len(wavelet_features))])
        except Exception as e:
            logger.error(f"Error al calcular características wavelet: {e}")
            features.extend([0.0] * 28)
            feature_names.extend([f"wavelet_feature_{i}" for i in range(28)])
    
    return features, feature_names


def calculate_mean(image):
    """Calculate the mean intensity of the image"""
    try:
        return np.mean(image)
    except Exception as e:
        print(f"Error al calcular la media: {e}")
        return 0  # Retorna 0 en caso de error

def calculate_variance(image):
    """Calculate the variance of the image"""
    try:
        return np.var(image)
    except Exception as e:
        print(f"Error al calcular la varianza: {e}")
        return 0  # Retorna 0 en caso de error

def calculate_kurtosis(image):
    """Calculate the kurtosis of the image intensity distribution"""
    # Ensure the image has sufficient variance
    if np.var(image) < 1e-6:
        logger.warning(f"Varianza insuficiente para calcular curtosis, retornando 0.0")
        return 0.0
    
    try:
        return kurtosis(image.flatten(), nan_policy='omit')
    except Exception as e:
        logger.error(f"Error al calcular curtosis: {e}")
        return 0.0

def calculate_skew(image):
    """Calculate the skewness of the image intensity distribution"""
    # Ensure the image has sufficient variance
    if np.var(image) < 1e-6:
        logger.warning(f"Varianza insuficiente para calcular asimetría en imagen , retornando 0.0")
        return 0.0
    
    try:
        return skew(image.flatten(), nan_policy='omit')
    except Exception as e:
        logger.error(f"Error al calcular asimetría: {e}")
        return 0.0

def calculate_fractal_dimension(image, threshold=None):
    """
    Calculate the fractal dimension of the image using box counting method.
    
    Args:
        image (np.ndarray): Input grayscale image
        threshold (float, optional): Threshold for binarization. If None, uses Otsu's method
        
    Returns:
        float: Fractal dimension
    """
    try:
        # Normalize image to [0, 255] for consistent thresholding
        img_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Use Otsu's method if threshold not provided
        if threshold is None:
            _, binary = cv2.threshold(img_norm, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(img_norm, int(threshold * 255), 1, cv2.THRESH_BINARY)
        
        # Ensure image is 2D
        assert(len(binary.shape) == 2)
        
        def box_count(img, k):
            # Count boxes of size k×k
            S = np.add.reduceat(
                np.add.reduceat(img, np.arange(0, img.shape[0], k), axis=0),
                                np.arange(0, img.shape[1], k), axis=1)
            # Count non-empty (0) and non-full boxes (k*k)
            return len(np.where((S > 0) & (S < k*k))[0])
        
        # Determine range of box sizes
        p = min(binary.shape)
        n = 2**np.floor(np.log(p)/np.log(2))
        n = int(np.log(n)/np.log(2))
        sizes = 2**np.arange(n, 1, -1)
        
        # Count boxes of different sizes
        counts = []
        for size in sizes:
            count = box_count(binary, size)
            if count > 0:
                counts.append(count)
        
        # If we have at least 2 valid counts
        if len(counts) >= 2:
            sizes = sizes[:len(counts)]
            coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
            return -coeffs[0]
        else:
            logger.warning(f"Número insuficiente de conteos válidos para calcular dimensión fractal en imagen ")
            return 0.0
    except Exception as e:
        logger.error(f"Error al calcular dimensión fractal: {e}")
        return 0.0

def calculate_entropy(image):
    """
    Calculate the Shannon entropy of the image intensity distribution.
    
    Args:
        image (np.ndarray): Input grayscale image
        
    Returns:
        float: Entropy value
    """
    try:
        # Normalize to 0-255 and convert to uint8 for consistent histogram calculation
        image_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Calculate histogram
        hist = cv2.calcHist([image_norm], [0], None, [256], [0, 256])
        hist = hist / hist.sum()  # Normalize to get probability distribution
        
        # Filter out zeros to avoid log(0)
        hist = hist[hist > 0]
        
        if len(hist) > 0:
            entropy_value = -np.sum(hist * np.log2(hist))
            return float(entropy_value)
        else:
            logger.warning("Histograma vacío para calcular entropía")
            return 0.0
    except Exception as e:
        logger.error(f"Error al calcular entropía: {e}")
        return 0.0

def calculate_glcm_features(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """
    Calculate texture features using Gray Level Co-occurrence Matrix
    
    Args:
        image (np.ndarray): Input grayscale image
        distances (list): List of distances for GLCM calculation
        angles (list): List of angles for GLCM calculation
        
    Returns:
        list: GLCM features (contrast, correlation, energy, homogeneity)
    """
    try:
        from skimage.feature import graycomatrix, graycoprops
        # Normalize and quantize to 8 levels to reduce computation
        image_norm = cv2.normalize(image, None, 0, 7, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Calculate GLCM
        glcm = graycomatrix(image_norm, distances=distances, angles=angles, 
                           levels=8, symmetric=True, normed=True)
        
        # Extract properties
        contrast = graycoprops(glcm, 'contrast').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        energy = graycoprops(glcm, 'energy').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        
        return [contrast, correlation, energy, homogeneity]
    except ImportError:
        logger.error("No se pudo importar skimage para calcular características GLCM")
        return [0.0, 0.0, 0.0, 0.0]
    except Exception as e:
        logger.error(f"Error al calcular características GLCM: {e}")
        return [0.0, 0.0, 0.0, 0.0]

# Nuevas funciones para las características solicitadas

def calculate_gradients(image):
    """
    Calcula los gradientes Sobel en las direcciones X e Y.
    
    Args:
        image (np.ndarray): Imagen en escala de grises
        
    Returns:
        tuple: (gradiente_x, gradiente_y)
    """
    try:
        # Asegurar que la imagen es de tipo float para cálculos precisos
        if image.dtype != np.float32 and image.dtype != np.float64:
            image_float = image.astype(np.float32)
        else:
            image_float = image.copy()
        
        # Calcular gradientes usando Sobel
        grad_x = cv2.Sobel(image_float, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image_float, cv2.CV_32F, 0, 1, ksize=3)
        
        return grad_x, grad_y
    except Exception as e:
        logger.error(f"Error al calcular gradientes: {e}")
        # Devolver arrays de ceros si hay error
        return np.zeros_like(image, dtype=np.float32), np.zeros_like(image, dtype=np.float32)

def calculate_gradient_magnitude_angle(grad_x, grad_y):
    """
    Calcula la magnitud y el ángulo del gradiente.
    
    Args:
        grad_x (np.ndarray): Gradiente en dirección X
        grad_y (np.ndarray): Gradiente en dirección Y
        
    Returns:
        tuple: (magnitud, ángulo en radianes)
    """
    try:
        # Calcular magnitud y dirección
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        angle = np.arctan2(grad_y, grad_x)
        
        return magnitude, angle
    except Exception as e:
        logger.error(f"Error al calcular magnitud y ángulo del gradiente: {e}")
        # Devolver arrays de ceros si hay error
        return np.zeros_like(grad_x, dtype=np.float32), np.zeros_like(grad_x, dtype=np.float32)

def calculate_energy(image):
    """
    Calcula la energía de la imagen (suma de los cuadrados de los valores de píxeles).
    
    Args:
        image (np.ndarray): Imagen en escala de grises
        
    Returns:
        float: Energía de la imagen
    """
    try:
        # Normalizar imagen a [0,1] para valores consistentes
        if image.dtype != np.float32 and image.dtype != np.float64:
            image_norm = image.astype(np.float32) / 255.0
        else:
            max_val = np.max(image)
            if max_val > 0:
                image_norm = image / max_val
            else:
                image_norm = image
        
        return np.sum(image_norm**2)
    except Exception as e:
        logger.error(f"Error al calcular energía de la imagen: {e}")
        return 0.0

def calculate_wavelet_features(image, wavelet='db9', level=2):
    """
    Extrae características basadas en la transformada wavelet.
    
    Args:
        image (np.ndarray): Imagen en escala de grises
        wavelet (str): Familia wavelet a utilizar
        level (int): Nivel de descomposición
        
    Returns:
        list: Características wavelet (estadísticas de coeficientes)
    """
    try:
        # Verificar que la imagen tenga dimensiones que permitan la descomposición wavelet
        # La transformada wavelet requiere dimensiones divisibles por 2^level
        h, w = image.shape
        new_h, new_w = h, w
        
        # Redimensionar si es necesario
        if h % (2**level) != 0 or w % (2**level) != 0:
            new_h = ((h // (2**level)) + 1) * (2**level)
            new_w = ((w // (2**level)) + 1) * (2**level)
            # Rellenar con ceros para alcanzar las dimensiones necesarias
            padded_img = np.zeros((new_h, new_w), dtype=np.float32)
            padded_img[:h, :w] = image
            image = padded_img
            logger.info(f"Imagen redimensionada para wavelet: {h}x{w} -> {new_h}x{new_w}")
        
        # Aplicar transformada wavelet 2D
        coeffs = pywt.wavedec2(image, wavelet, level=level)
        
        # Extraer características de cada nivel de coeficientes
        features = []
        
        # Añadir características del coeficiente de aproximación (LL)
        approximation = coeffs[0]
        features.extend([
            np.mean(approximation),
            np.std(approximation),
            np.max(approximation),
            np.min(approximation)
        ])
        
        # Añadir características de los coeficientes de detalle (LH, HL, HH)
        for detail_level in range(1, len(coeffs)):
            for detail_type in range(3):  # LH, HL, HH
                detail = coeffs[detail_level][detail_type]
                features.extend([
                    np.mean(np.abs(detail)),
                    np.std(detail),
                    np.max(np.abs(detail)),
                    np.sum(detail**2) / (detail.shape[0] * detail.shape[1])  # Energía
                ])
        
        return features
    except Exception as e:
        logger.error(f"Error al calcular características wavelet: {e}")
        # Retornar valores default si ocurre algún error (4 para aprox + 4*3*level para detalles)
        return [0.0] * (4 + 4 * 3 * level)