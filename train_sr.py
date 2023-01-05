import os
import json
import numpy as np
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split

LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 40
DATA_PATH = "Data\Speech_Commands_Data_Set.json"
SAVE_MODEL_PATH = "Data\model.h5"
NUM_OF_KEYWORDS = 10

def load_dataset(data_path):

    with open(data_path, "r") as fp:
        data = json.load(fp)
    X = np.array(data["MFCCs"])
    y = np.array(data["labels"])
    
    return X, y

def get_data_splits(data_path, test_size=0.1, val_size=.2):
    # load dataset
    X, y = load_dataset(data_path)
    
    # create train test splits
    X_train, X_test, y_train, y_test = train_test_split(X, 
        y, test_size=test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size)
    
    X_train = X_train[..., 1]
    X_val = X_val[..., 1]
    X_test = X_test[..., 1]
    
    # return
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model(input_shape, my_learning_rate):
    # build CNN network
    model = keras.Sequential()

    # 3 conv layers
    model.add(keras.layers.Conv3D(filters=64, kernel_size=(3,3), 
        activation="relu", input_shape=input_shape, kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    
    
    model.add(keras.layers.Conv3D(filters=32, kernel_size=(2, 2),
        activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    
    model.add(keras.layers.Conv3D(filters=32, kernel_size=(2, 2),
        activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    
    # flatten convs and feed into Dense
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation_function="relu"))
    model.add(keras.layers.Dropeout(0.3))

    # softmax
    model.add(keras.layers.Dense(NUM_OF_KEYWORDS, activation="softmax"))
    
    my_optimizer = keras.optimizers.Adam(learning_rate=my_learning_rate)
    model.compile(optimizer=my_optimizer,
                  loss="categorical_cross_entropy",
                  metrics=["accuracy"])

    print(model.summary())
    return model
    

def main ():
    # train test split
    X_train, X_val, X_test, y_train, y_val, y_test = get_data_splits(DATA_PATH)
    
    # build CNN
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]) 
    model = build_model(input_shape, LEARNING_RATE)

    # train model
    model.fit(X_train,y_train, 
                NUM_EPOCHS=40, 
                batch_size=BATCH_SIZE,
                validation_data=(X_val,y_val))

    # eval model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test error: {test_loss}\n Test Accuracy: {test_acc}")

    # save model
    model.save(SAVE_MODEL_PATH)