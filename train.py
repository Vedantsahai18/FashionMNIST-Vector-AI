
# importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D
import matplotlib.pyplot as plt
import os
from tensorflow.keras.callbacks import EarlyStopping
from pathlib import Path
import argparse

# Create a dictionary for each type of label 
labels = {0 : "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
          5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}

# defining the constants
IMG_ROWS = 28
IMG_COLS = 28
NUM_CLASSES = 10
TEST_SIZE = 0.2
RANDOM_STATE = 2018

# paths
PATH = os.path.join(os.getcwd(),'input')
MODEL_DIR = os.path.join(os.getcwd(),'model')
IMAGE_DIR = os.path.join(os.getcwd(),'images')

#creating the paths
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
Path(IMAGE_DIR).mkdir(parents=True, exist_ok=True)

#Model constants
NO_EPOCHS = 50
BATCH_SIZE = 128

"""## <a id="51">Prepare the model</a>

## Data preprocessing

First we will do a data preprocessing to prepare for the model.

We reshape the columns  from (784) to (28,28,1). We also save label (target) feature as a separate vector.
"""

# data preprocessing
def data_preprocessing(raw):
    out_y = to_categorical(raw.label, NUM_CLASSES)
    num_images = raw.shape[0]
    x_as_array = raw.values[:,1:]
    x_shaped_array = x_as_array.reshape(num_images, IMG_ROWS, IMG_COLS, 1)
    out_x = x_shaped_array / 255
    return out_x, out_y

# define the model
def build_model():

  # Model
  model = Sequential()
  # Add convolution 2D
  model.add(Conv2D(32, kernel_size=(3, 3),
                  activation='relu',
                  kernel_initializer='he_normal',
                  input_shape=(IMG_ROWS, IMG_COLS, 1)))
  model.add(MaxPooling2D((2, 2)))
  # Add dropouts to the model
  model.add(Dropout(0.25))
  model.add(Conv2D(64, 
                  kernel_size=(3, 3), 
                  activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  # Add dropouts to the model
  model.add(Dropout(0.25))
  model.add(Conv2D(128, (3, 3), activation='relu'))
  # Add dropouts to the model
  model.add(Dropout(0.4))
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  # Add dropouts to the model
  model.add(Dropout(0.3))
  model.add(Dense(NUM_CLASSES, activation='softmax'))


  model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                optimizer='adam',
                metrics=['accuracy'])

  # inspecting the model
  print(model.summary())

  return model


def plot_training_history(history):

   # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(IMAGE_DIR + '/Model_Acc.png')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(IMAGE_DIR + '/Model_loss.png')
    plt.show()


"""## <a id="57">Prediction accuracy with the new model</a>

Let's evaluate the prediction accuracy with the new model.
"""

def model_evaluation(model, X_test, y_test,test_data):

  # test prediction accuracy with the new model.
  score = model.evaluate(X_test, y_test, verbose=0)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--train", type=str,required=True,
    help="enter the CSV train file path")
  ap.add_argument("--test", type=str, required=True,
    help="enter the CSV test file path")
  args = vars(ap.parse_args())

  train_data = pd.read_csv(args["train"])
  test_data = pd.read_csv(args["test"])

  print("Fashion MNIST train -  rows:",train_data.shape[0]," columns:", train_data.shape[1])
  print("Fashion MNIST test -  rows:",test_data.shape[0]," columns:", test_data.shape[1])

  # prepare the data
  X, y = data_preprocessing(train_data)
  X_test, y_test = data_preprocessing(test_data)

  """## Split train in train and validation set

  We further split the train set in train and validation set. The validation set will be 20% from the original train set, therefore the split will be train/validation of 0.8/0.2.
  """

  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

  # The dimmension of the processed train, validation and test set are as following:
  print("Fashion MNIST train -  rows:",X_train.shape[0]," columns:", X_train.shape[1:4])
  print("Fashion MNIST valid -  rows:",X_val.shape[0]," columns:", X_val.shape[1:4])
  print("Fashion MNIST test -  rows:",X_test.shape[0]," columns:", X_test.shape[1:4])

  # building the model
  model = build_model()

  # callback
  early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)

  #  training the model
  train_model = model.fit(X_train, y_train,
                    batch_size=BATCH_SIZE,
                    epochs=NO_EPOCHS,
                    verbose=1,
                    validation_data=(X_val, y_val),callbacks=[early_stop])

  # saving the model archi + weights
  model.save(os.path.join(MODEL_DIR + "/model_fashion.h5"))

  #plotting the training history
  plot_training_history(train_model)

  # evaluate the model
  model_evaluation(model, X_test, y_test,test_data)

if __name__ == "__main__":
  main()