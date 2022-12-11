'''
This a network that predicts a handwritten digits digit
Date Made: 9/17/22
'''
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from test_network import test

def train():
    mnist = keras.datasets.mnist # dataset

    (x_train, y_train), (x_test, y_test) = mnist.load_data() # x is the image and y is the answer.

   
    x_train = keras.utils.normalize(x_train, axis = 1)
    
    # apply the neural network

    model = keras.models.Sequential()

    model.add(keras.layers.Flatten(input_shape=(28, 28))) # turns it from 28x28 2D grid to a 784 1D line
    model.add(keras.layers.Dense(128, activation = 'relu'))
    model.add(keras.layers.Dense(128, activation = 'relu'))
    model.add(keras.layers.Dense(10, activation = 'softmax'))

    model.compile(optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy', 
    metrics = ["accuracy"])


    best = 0
    for i in range(5): # train model 5 times for best accuracy
        model.fit(x_train, y_train, epochs=3) # train model

        loss, accuracy = model.evaluate(x_test, y_test)

        if accuracy > best:
            best = accuracy
            model.save('handwritten_digit.model')

        print(f"Model saved with {loss} loss and {accuracy} accuracy") # 97% accuracy
    
    print(f"Final Model saved with {loss} loss and {best} accuracy")
    run_test = input("Would you like to test the network? (y/n) ")

    if run_test.lower() == "y":
        test()
    elif run_test.lower() == "n":
        exit()
    else:
        exit()

if __name__ == "__main__":
    train()