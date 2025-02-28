import cv2
from import_dataset import load_dataset
import numpy as np
from scipy.stats import kurtosis, skew, entropy

def preprocess_dataset():
    train_x, train_y,test_x, test_y = load_dataset()
    # # Convertir bool -> uint8
    # train_x = train_x.astype(np.uint8) * 255
    # test_x = test_x.astype(np.uint8) * 255
    
    train_x_gray = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in train_x])
    test_x_gray = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in test_x])
    
    train_x_features = np.array([extract_features(image) for image in train_x_gray])
    test_x_features = np.array([extract_features(image) for image in test_x_gray])
    
    return train_x_features, train_y, test_x_features, test_y

def extract_features(image):
    # Extract features from image
    features = [
        calculate_mean(image),
        calculate_variance(image),
        calculate_kurtosis(image),
        calculate_skew(image),
        calculate_fractal_dimension(image),
        calculate_determinant(image),
        calculate_entropy(image)
    ]
    return features

def calculate_mean(image):
    # Calculate the mean of the image
    return np.mean(image)

def calculate_variance(image):
    # Calculate the variance of the image
    return np.var(image)

def calculate_kurtosis(image):
    # Calculate the kurtosis of the image
    return kurtosis(image)

def calculate_skew(image):
    # Calculate the skew of the image
    return skew(image, axis=None)

def calculate_fractal_dimension(image, threshold=0.9):
    # Calculate the fractal dimension of the image
    assert(len(image.shape) == 2)
    
    def box_count(image, k):
        S = np.add.reduceat(
            np.add.reduceat(image, np.arange(0, image.shape[0], k), axis=0),
                               np.arange(0, image.shape[1], k), axis=1)

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k*k))[0])
    
    image = (image < threshold)
    p = min(image.shape)
    n = 2**np.floor(np.log(p)/np.log(2))
    n = int(np.log(n)/np.log(2))
    sizes = 2**np.arange(n, 1, -1)
    counts = []
    for size in sizes:
        counts.append(box_count(image, size))
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

def calculate_determinant(image):
    # Calculate the determinant of the image
     
    height, width = image.shape

    # Create a matrix where every row represents a pixel (x, y, intensity) 
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    pixels = np.column_stack((x.ravel(), y.ravel(), image.ravel()))

    
    cov_matrix = np.cov(pixels.T)

    det_value = np.linalg.det(cov_matrix)

    return det_value

def calculate_entropy(image):
    # Calculate the entropy of the image
    hist, _ = np.histogram(image, bins=256, range=(0, 255), density=True)
    
    entropy_value = entropy(hist, base=2)
    return entropy_value