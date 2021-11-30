# importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import os
from pathlib import Path
import argparse
import cv2

# defining the constants
IMG_ROWS = 28
IMG_COLS = 28

# paths
MODEL_DIR = os.path.join(os.getcwd(),'model')

# Create a dictionary for each type of label 
labels = {0 : "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
          5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}

# data preprocessing
def data_preprocessing(filename):
	# load the image
	img = load_img(filename, color_mode = "grayscale", target_size=(28, 28))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 1 channel
	img = img.reshape(1, IMG_ROWS, IMG_COLS, 1)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img

def main():
    parser = argparse.ArgumentParser(description='Predict the image')
    parser.add_argument('--image', type=str, help='Image to predict')
    args = parser.parse_args()
    # image_path = os.path.join(os.getcwd(),args.image)
    image_path = args.image
    try:
        x = data_preprocessing(image_path)
        try:
            model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'model_fashion.h5'))
            prediction = np.array_str(np.argmax(model.predict(x), axis=-1))
            prediction = str(prediction)[1:-1]
            print("Model Prediction : ", labels[int(prediction)])
        except:
            print('Model not found')
    except:
        print('Invalid image path')
        exit(1)

if __name__ == "__main__":
  main()