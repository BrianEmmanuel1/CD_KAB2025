import torchvision

class ImportData():
    def __init__(self):
        pass
    
class Cifar10():
    def __init__(self):
        self.train_data = torchvision.datasets.CIFAR10(root='../data', train=True, download=True)
        self.test_data = torchvision.datasets.CIFAR10(root='../data', train=False, download=True)