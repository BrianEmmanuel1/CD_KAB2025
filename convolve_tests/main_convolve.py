from keras import layers, models
import utils.imports_tf as datasets 
from utils.model import Model

#hyperparameters
batch_size = 64
epochs = 100
validation_split = 0.2

#load dataset
dataset = datasets.CIFAR10()

#shapes of input and output data
input_shape = dataset.input_shape
output_size = dataset.output_size

#train test images and labels
train_img, train_labels, test_img, test_labels = dataset.get_data()

#train and test data with dinamic tranformations
train_data, test_data = dataset.get_flowimg_datagen(
    validation_split=validation_split, 
    batch_size=batch_size, 
    horizontal_flip=False)

#setup model architecture

model_arch = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu", input_shape=input_shape),
    layers.BatchNormalization(),
        
    layers.Conv2D(filters = 32, kernel_size=(3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.2),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    
    layers.Conv2D(filters = 64, kernel_size=(3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.3),
    
    layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    
    layers.Conv2D(filters = 128, kernel_size=(3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.5),    
    
    layers.Flatten(),
    
    layers.Dense(256, activation = "relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    
    layers.Dense(output_size, activation="softmax")
])

model = Model(model_arch)
model.compile()
model.set_patience(5)
model.fit(train_data, test_data, epochs=epochs)
model.evaluate(test_img, test_labels)
model.test_examples(test_img[:15], test_labels[:15])