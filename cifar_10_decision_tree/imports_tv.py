import numpy as np
import torchvision

class ImportData():
    def __init__(self):
        pass
    
class Cifar10():
    def __init__(self):
        self.train_data = torchvision.datasets.CIFAR10(root='../data', train=True, download=True)
        self.test_data = torchvision.datasets.CIFAR10(root='../data', train=False, download=True)
        
    def split_data(self):
        x_train = self.train_data.data
        y_train = self.train_data.targets 
        
        x_test = self.test_data.data
        y_test = self.test_data.targets
        
        return x_train, y_train, x_test, y_test