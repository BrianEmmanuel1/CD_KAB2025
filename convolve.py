from keras.utils import to_categorical  # type: ignore
from keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator   # type: ignore
from keras import datasets  # type: ignore
from keras import optimizers
from keras import callbacks
from keras import models

(train_img, train_labels), (test_img, test_labels) = datasets.cifar10.load_data()

#normalizacion de imagenes
train_img = train_img.astype("float32") / 255.
test_img = test_img.astype("float32") / 255.

lbl_train = to_categorical(train_labels, 10)
lbl_test = to_categorical(test_labels, 10)

model = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu", input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(filters=32, kernel_size=(4, 4), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.2) ,
        
    layers.Conv2D(filters = 32, kernel_size=(3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.2),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(filters=64, kernel_size=(4, 4), padding="same", activation="relu"),
    layers.BatchNormalization(),
    
    layers.Conv2D(filters = 64, kernel_size=(3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.3),
    
    layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Conv2D(filters=128, kernel_size=(4, 4), padding="same", activation="relu"),
    layers.BatchNormalization(),
    
    layers.Conv2D(filters = 128, kernel_size=(3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.4),    
    
    layers.Flatten(),
    
    layers.Dense(256, activation = "relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    
    layers.Dense(10, activation="softmax")
])

callback = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
optimizer = optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])



train_datagen = ImageDataGenerator(
    rotation_range = 15,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    horizontal_flip = True,
    validation_split = 0.20
)

val_datagen = ImageDataGenerator(
    validation_split=0.20
)

train_generator = train_datagen.flow(
    train_img, lbl_train,
    batch_size = 64,
    subset = "training",
    shuffle = True
)

val_generator = val_datagen.flow(
    train_img, lbl_train,
    batch_size = 64,
    subset = "validation",
    shuffle = False
)

train_datagen.fit(train_img)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=100, 
    callbacks=[callback])

print("\nEvaluation stage")
eval = model.evaluate(test_img, lbl_test)

print(f"\nTest loss: {eval[0]:.4f}")
print(f"Test accuracy: {eval[1]*100:.2f}%\n")


print("Few testing examples")
predictions = model.predict(test_img[:20, :])
predictions_idx = predictions.argmax(1) 
print(predictions_idx)
print(test_labels[:20].flatten())