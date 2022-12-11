'''
After playing around with the network, I found that if I were to give it a random image it still guesses a number.
In my case 3 (digit_16.png)
'''
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


model = tf.keras.models.load_model('handwritten_digit.model')

image_number = 1
while os.path.isfile(f"digits_img/digit_{image_number}.png"):
    try:
        img = cv2.imread(f"digits_img/digit_{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"I predict this number is a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap = plt.cm.binary)
        plt.show()
    except:
        print("Error! Wrong resolution!")
    finally:
        image_number += 1