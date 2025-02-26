from tensorflow.keras.preprocessing.image import ImageDataGenerator   # type: ignore
from keras import utils

def normalize_img (train_img, test_img, float_precission = 16):
    #normalize images rgb values [0, 255] -> [0, 1]
    #low aproximation is set improve training process velocity
    train_img = train_img.astype(f"float{float_precission}") / 255    
    test_img = test_img.astype(f"float{float_precission}") / 255
    
    return train_img, test_img

def to_categorical_labels(train_lbl, test_lbl, size):
    #hot encoding for labels: [0, 0, ..., 1(at n index), ... , 0, 0] -> [n]
    train_lbl = utils.to_categorical(train_lbl, size)
    test_lbl = utils.to_categorical(test_lbl, size)
    return train_lbl, test_lbl


#generate random transformations to test and training images to improve model generalization
def getFlowImg(img_train, lbl_train, img_test, lbl_test, val_split, batch_size_, horizontal_flip = False):
    
    #tranformation for train data
    train_datagen = ImageDataGenerator(
        rotation_range = 0.15,
        height_shift_range = 0.1,
        width_shift_range = 0.1,
        horizontal_flip = horizontal_flip,
        validation_split = val_split
    )
    
    #no transformation is nedeed for test data
    test_datagen = ImageDataGenerator(
        validation_split = val_split
    )
    
    #generate a flow of random images based on tranformations for training data
    traing_gen = train_datagen.flow(
        img_train, lbl_train,
        batch_size = batch_size_,
        shuffle = True
    )
    
    test_gen = test_datagen.flow(
        img_test, lbl_test,
        batch_size = batch_size_,
        shuffle = False
    )
    
    return traing_gen, test_gen