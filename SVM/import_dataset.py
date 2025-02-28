import h5py
import pandas as pd
import numpy as np
import gzip
import shutil
import os

def decompress_gz(gz_path):
    """Descomprime un archivo .gz si aún no ha sido descomprimido."""
    output_path = gz_path.replace('.gz', '')
    
    if not os.path.exists(output_path):  # Evita descomprimir si ya existe
        with gzip.open(gz_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    
    return output_path

def load_h5(file_path):
    """Carga un archivo HDF5, descomprimiéndolo si es necesario."""
    if file_path.endswith('.gz'):
        file_path = decompress_gz(file_path)
    
    with h5py.File(file_path, 'r') as f:
        print("Claves disponibles:", list(f.keys()))
        key = list(f.keys())[0]
        data = np.array(f[key][:])
    return data

def load_csv_file(file_path):
    return pd.read_csv(file_path)

def load_dataset():
    train_x = load_h5('./data/camelyonpatch_level_2_split_train_x.h5-002.gz')
    train_y = load_h5('./data/camelyonpatch_level_2_split_train_y.h5.gz')
    test_x = load_h5('./data/camelyonpatch_level_2_split_test_x.h5.gz')
    test_y = load_h5('./data/camelyonpatch_level_2_split_test_y.h5.gz')
    
    return train_x, train_y, test_x, test_y
