import cv2
import numpy as np
from scipy.stats import kurtosis, skew, entropy
from import_dataset import load_dataset
from sklearn.preprocessing import StandardScaler

def preprocess_dataset():
    """
    Loads the CamelyonPatch dataset, preprocesses images, and extracts features.
    
    Returns:
        train_x_features (np.ndarray): Extracted features from training images
        train_y (np.ndarray): Training labels
        test_x_features (np.ndarray): Extracted features from test images
        test_y (np.ndarray): Test labels
    """
    # Load dataset
    train_x, train_y, test_x, test_y = load_dataset()
    
    # Check and convert data types if necessary
    if train_x.dtype == bool:
        train_x = train_x.astype(np.uint8) * 255
        test_x = test_x.astype(np.uint8) * 255
    
    # Convert to grayscale
    print("Converting images to grayscale...")
    train_x_gray = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in train_x])
    test_x_gray = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in test_x])
    
    # Extract features
    print("Extracting features from training set...")
    train_x_features = np.array([extract_features(image) for image in train_x_gray])
    
    print("Extracting features from test set...")
    test_x_features = np.array([extract_features(image) for image in test_x_gray])
    
    # Normalize features
    scaler = StandardScaler()
    train_x_features = scaler.fit_transform(train_x_features)
    test_x_features = scaler.transform(test_x_features)
    
    # Check for NaN or infinite values
    train_x_features = np.nan_to_num(train_x_features, nan=0.0, posinf=0.0, neginf=0.0)
    test_x_features = np.nan_to_num(test_x_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"Feature extraction complete. Feature shape: {train_x_features.shape}")
    
    return train_x_features, train_y, test_x_features, test_y

def extract_features(image):
    """
    Extract statistical and textural features from a grayscale image.
    
    Args:
        image (np.ndarray): Grayscale image
        
    Returns:
        list: Feature vector
    """
    # Extract features from image
    features = [
        calculate_mean(image),
        calculate_variance(image),
        calculate_kurtosis(image),
        calculate_skew(image),
        calculate_fractal_dimension(image),
        calculate_entropy(image),
        *calculate_glcm_features(image)  # Unpack GLCM features
    ]
    return features

def calculate_mean(image):
    """Calculate the mean intensity of the image"""
    return np.mean(image)

def calculate_variance(image):
    """Calculate the variance of the image"""
    return np.var(image)

def calculate_kurtosis(image):
    """Calculate the kurtosis of the image intensity distribution"""
    # Ensure the image has sufficient variance
    if np.var(image) < 1e-6:
        return 0.0
    
    try:
        return kurtosis(image.flatten(), nan_policy='omit')
    except:
        return 0.0

def calculate_skew(image):
    """Calculate the skewness of the image intensity distribution"""
    # Ensure the image has sufficient variance
    if np.var(image) < 1e-6:
        return 0.0
    
    try:
        return skew(image.flatten(), nan_policy='omit')
    except:
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
        return 0.0

def calculate_entropy(image):
    """
    Calculate the Shannon entropy of the image intensity distribution.
    
    Args:
        image (np.ndarray): Input grayscale image
        
    Returns:
        float: Entropy value
    """
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
        from skimage.feature import greycomatrix, greycoprops
        
        # Normalize and quantize to 8 levels to reduce computation
        image_norm = cv2.normalize(image, None, 0, 7, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Calculate GLCM
        glcm = greycomatrix(image_norm, distances=distances, angles=angles, 
                           levels=8, symmetric=True, normed=True)
        
        # Extract properties
        contrast = greycoprops(glcm, 'contrast').mean()
        correlation = greycoprops(glcm, 'correlation').mean()
        energy = greycoprops(glcm, 'energy').mean()
        homogeneity = greycoprops(glcm, 'homogeneity').mean()
        
        return [contrast, correlation, energy, homogeneity]
    except:
        # If scikit-image is not available or other error
        return [0.0, 0.0, 0.0, 0.0]

# Removed calculate_determinant function as it was computationally expensive 
# and its interpretation was unclear

if __name__ == "__main__":
    # Test the preprocessing pipeline
    train_features, train_labels, test_features, test_labels = preprocess_dataset()
    print(f"Training data shape: {train_features.shape}")
    print(f"Test data shape: {test_features.shape}")