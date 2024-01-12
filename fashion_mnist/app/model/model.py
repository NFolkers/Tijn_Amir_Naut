from pathlib import Path
from tkinter.tix import IMAGE
from keras.models import load_model
from PIL import Image
import numpy as np

from keras import backend as K

# custom metrics used during training, required to load models 
# https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# model_dir = Path("./model/trained_cnn_200epochs")  Old model

model_dir = Path("./model/new_model_save_200_epochs_valLoss")

__version__ = "0.1.0"

def downscale(img):
    resized_img = img.resize((28, 28), Image.LANCZOS )
    return resized_img

def preprocess(pictures):
    done = []
    # pil_images = Image.fromarray(pictures)
    for picture in pictures:
        downscaled = downscale(picture)
        grayscale = downscaled.convert("L") 
        back_to_numpy = np.array(grayscale)
        normalized = abs((back_to_numpy.astype("float32") / 255.0) - 1 )
        reshaped = normalized.reshape((28, 28))
        done.append(reshaped)
    return np.asarray(done)

def loadModel():
    custom_objects = {'f1_m' : f1_m, 'precision_m' : precision_m, 'recall_m' : recall_m}
    model = load_model(model_dir, custom_objects)
    return model
