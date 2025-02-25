import numpy as np
from keras import datasets
from utils.preprocess_data import normalize_img, to_categorical_labels, getFlowImg


class ImageDataset():
    def __init__(self, dataset: datasets, label_size):        
        #load train and test images and labels from the dataset
        (self.train_img, self.train_lbl), (self.test_img, self.test_lbl) = dataset.load_data()
        
        #normalize images rgb values [0, 255] -> [0, 1]
        self.train_img, self.test_img = normalize_img(self.train_img, self.test_img)
        
        #hotencoding labels
        self.train_lbl, self.test_lbl = to_categorical_labels(self.train_lbl, self.test_lbl, size=label_size)
        
        self.input_shape = np.shape(self.train_img)[1:]
    
    def get_flowimg_datagen(self, validation_split, batch_size, horizontal_flip = False):
        #get splitted trian and test data with modifications 
        return getFlowImg(
            self.train_img, self.train_lbl, 
            self.test_img, self.test_lbl, 
            batch_size_=batch_size, 
            val_split=validation_split,
            horizontal_flip=horizontal_flip)
        
    def get_data (self):
        #get simple splited - data
        return self.train_img, self.train_lbl, self.test_img, self.test_lbl

class MNIST(ImageDataset):
    def __init__(self):
        super().__init__(datasets.mnist, label_size=10)
        #since mnis are just black and white images, an additional channel is nedded
        self.train_img = np.expand_dims(self.train_img, axis=-1)
        self.test_img = np.expand_dims(self.test_img, axis=-1)
        #update the shape value of images
        self.input_shape = np.shape(self.train_img)[1:]
        #set the output size
        self.output_size = 10

class CIFAR100 (ImageDataset):    
    def __init__(self):
        super().__init__(datasets.cifar100, label_size=100)
        self.output_size = 100 
        
class CIFAR10(ImageDataset):
    def __init__(self):
        super().__init__(datasets.cifar10, label_size=10)
        self.output_size = 10        

