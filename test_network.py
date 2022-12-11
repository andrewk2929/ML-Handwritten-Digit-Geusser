import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def test():
    mnist = keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_test = keras.utils.normalize(x_test, axis = 1) # normalize np array

    model = tf.keras.models.load_model("handwritten_digit.model")

    x = int(input("How many images would you like for me to predict? "))

    prediction = model.predict(x_test)

    score = 0
    for i in range(x):
        guess = (np.argmax(prediction[i]))
        answer = y_test[i]

        # display image
        plt.grid(False)
        plt.imshow(x_test[i], cmap = plt.cm.binary)
        plt.xlabel(f"Answer: {answer}")
        plt.title(f"Prediction: {guess}")
        plt.show()

        print(f"I predict this number is a {guess} ")
        print(f"This number is actually a {answer} ")

        if guess == answer:
            score += 1

    print(f"I guessed {score} out of {x}")
    print(f"{int(round(100 * (score/x), 1))} % accuracy") # round to nearest tenth

if __name__ == "__main__":
    test()