from keras import models, layers, applications
from utils.imports_tf import CIFAR100
from utils.model import Model

#hyperparameters
batch_size = 64
epochs = 100
validation_split = 0.2

cifar100 = CIFAR100()

#load dataset train and test images and labels
train_img, train_lbl, test_img, test_lbl = cifar100.get_data()
train_dataflow, test_dataflow = cifar100.get_flowimg_datagen(validation_split=validation_split, batch_size=batch_size)

#?
img_train_preproc = applications.vgg16.preprocess_input(cifar100.train_img)
img_test_preppoc = applications.vgg16.preprocess_input(cifar100.test_img)


#Load the base model and remove the fully connected layers
base_model = applications.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(32, 32, 3)
)

#desactivate base model training
base_model.trainable = False 

inputs = layers.Input(shape=(32, 32, 3))
x = base_model(inputs, training=False) #set the input layer to the base model
x = layers.GlobalAveragePooling2D()(x) 
x = layers.Dense(512, activation="relu")(x) #add a fully connected hidden layer 
outputs = layers.Dense(100, activation="softmax")(x) #add the output layer 

#build the whole model
model_arch = models.Model(inputs, outputs)

model = Model(model_arch)
model.set_learning_rate(0.001)
model.compile()
model.fit(train_dataflow, test_dataflow, epochs=epochs)
model.evaluate(test_img, test_lbl)
model.test_examples(test_img [:5], test_lbl[:5])
