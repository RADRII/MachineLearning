import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.dummy import DummyClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle
plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True
import sys

# i a
##Function: Convert a square of numbers in a 2d array to a 1d list of numbers
def squareTo1D(arr2d, rows, cols, initR, initC):
    output = []
    i = initR
    while( i < initR + rows):
        j = initC
        while( j < initC + cols):
            output.append(arr2d[i][j])
            j = j + 1
        i = i + 1
    return output


##Function: Convolve kernel to input array and return result
def convolve(nn, kk):
    ##height = heightnn - heightkk + 1
    ##Same for width
    rows = len(nn) - len(kk) + 1
    cols = len(nn[0]) - len(kk[0]) + 1

    arrkk = squareTo1D(kk, len(kk), len(kk[0]), 0, 0)

    output = [[-1000 for i in range(cols)] for j in range(rows)]

    for i in range(rows):
        for j in range(cols):
            arrnn = squareTo1D(nn, len(kk), len(kk[0]), i, j)
            num = 0
            for k in range(len(arrnn)):
                num = num + (arrnn[k] * arrkk[k])
            output[i][j] = num
    return output

# i b
##Read in image of blue triangle and convert to 2d array of pixels
im = Image.open('triangle.png')
rgb = np.array(im.convert('RGB'))
r=rgb[:,:,0] # array of r pixels
Image.fromarray(np.uint8(r)).show()

##Convolve on two seperate kernels
kernel1 = [[-1,-1,-1], [-1,8,-1], [-1,-1,-1]]
kernel2 = [[0,-1,0], [-1,8,-1], [0,-1,0]]

output = convolve(r, kernel1)
Image.fromarray(np.uint8(output)).show()
output = convolve(r, kernel2)
Image.fromarray(np.uint8(output)).show()

# ii a
def week8(num, l):
    # Model / data parameters
    num_classes = 10
    input_shape = (32, 32, 3)

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    n=num
    x_train = x_train[1:n]; y_train=y_train[1:n]
    #x_test=x_test[1:500]; y_test=y_test[1:500]

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    print("orig x_train shape:", x_train.shape)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    use_saved_model = False
    if use_saved_model:
        model = keras.models.load_model("cifar.model")
    else:
        model = keras.Sequential()
        model.add(Conv2D(16, (3,3), padding='same', input_shape=x_train.shape[1:],activation='relu'))
        model.add(Conv2D(16, (3,3), strides=(2,2), padding='same', activation='relu'))
        model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
        model.add(Conv2D(32, (3,3), strides=(2,2), padding='same', activation='relu'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax',kernel_regularizer=regularizers.l1(l)))
        model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
        model.summary()

        batch_size = 128
        epochs = 20
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
        model.save("cifar.model")
        plt.subplot(211)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.subplot(212)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss'); plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

    print("Conv Model: Train")
    preds = model.predict(x_train)
    y_pred = np.argmax(preds, axis=1)
    y_train1 = np.argmax(y_train, axis=1)
    print(classification_report(y_train1, y_pred))
    print(confusion_matrix(y_train1,y_pred))

    print("Conv Model: Test")
    preds = model.predict(x_test)
    y_pred = np.argmax(preds, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    print(classification_report(y_test1, y_pred))
    print(confusion_matrix(y_test1,y_pred))

    dclf = DummyClassifier(strategy = 'uniform')
    dclf.fit(x_train, y_train)

    print("Dummy Model: Train")
    preds = dclf.predict(x_train)
    y_pred = np.argmax(preds, axis=1)
    y_train1 = np.argmax(y_train, axis=1)
    print(classification_report(y_train1, y_pred))
    print(confusion_matrix(y_train1,y_pred))

    print("Dummy Model: Test")
    preds = dclf.predict(x_test)
    y_pred = np.argmax(preds, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    print(classification_report(y_test1, y_pred))
    print(confusion_matrix(y_test1,y_pred))

week8(5000, 0.001)

# ii b iii
print("")
print("Using 10k training points: ")
print("")
week8(1000, 0.001)

print("")
print("Using 20k training points: ")
print("")
week8(20000, 0.001)

print("")
print("Using 40k training points: ")
print("")
week8(40000, 0.001)


# ii b iv

print("")
print("Using L1 of 0: ")
print("")
week8(5000, 0)

print("")
print("Using L1 of 0.0000001: ")
print("")
week8(5000, 0.0000001)

print("")
print("Using L1 of 0.1: ")
print("")
week8(5000, 0.005)

print("")
print("Using L1 of 0.1: ")
print("")
week8(5000, 0.1)

def week8Pool():
    # Model / data parameters
    num_classes = 10
    input_shape = (32, 32, 3)

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    n=5000
    x_train = x_train[1:n]; y_train=y_train[1:n]
    #x_test=x_test[1:500]; y_test=y_test[1:500]

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    print("orig x_train shape:", x_train.shape)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    use_saved_model = False
    if use_saved_model:
        model = keras.models.load_model("cifar.model")
    else:
        model = keras.Sequential()
        model.add(Conv2D(16, (3,3), padding='same', input_shape=x_train.shape[1:],activation='relu'))
        model.add(Conv2D(16, (3,3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
        model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax',kernel_regularizer=regularizers.l1(0.001)))
        model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
        model.summary()

        batch_size = 128
        epochs = 20
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
        model.save("cifar.model")
        plt.subplot(211)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.subplot(212)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss'); plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

    print("Conv Model: Train")
    preds = model.predict(x_train)
    y_pred = np.argmax(preds, axis=1)
    y_train1 = np.argmax(y_train, axis=1)
    print(classification_report(y_train1, y_pred))
    print(confusion_matrix(y_train1,y_pred))

    print("Conv Model: Test")
    preds = model.predict(x_test)
    y_pred = np.argmax(preds, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    print(classification_report(y_test1, y_pred))
    print(confusion_matrix(y_test1,y_pred))

    dclf = DummyClassifier(strategy = 'uniform')
    dclf.fit(x_train, y_train)

    print("Dummy Model: Train")
    preds = dclf.predict(x_train)
    y_pred = np.argmax(preds, axis=1)
    y_train1 = np.argmax(y_train, axis=1)
    print(classification_report(y_train1, y_pred))
    print(confusion_matrix(y_train1,y_pred))

    print("Dummy Model: Test")
    preds = dclf.predict(x_test)
    y_pred = np.argmax(preds, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    print(classification_report(y_test1, y_pred))
    print(confusion_matrix(y_test1,y_pred))

print("")
print("Using Max pool: ")
print("")
week8Pool()