from fastapi import FastAPI, File, UploadFile
from matplotlib.pylab import f
from pydantic import BaseModel
from model.model import preprocess, loadModel
from model.model import __version__ as model_version
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import mnist_reader
import uuid
import os


app = FastAPI()


class network:
    def __init__(self) -> None:
        self.model = loadModel()
        self.mappings = [ "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag" , "Ankle boot"]
    
    def makePredictions(self, img):
        probs = (self.model.predict(img))
        chosen_class = np.argmax(probs)
        return self.mappings[chosen_class]

    def preprocess_img(self, image_array):
        processed_images = preprocess(image_array)
        return processed_images

    def pipeline(self, filename):
        img = Image.open(f'./images/{filename}')
        precessed_img = self.preprocess_img([img])
        prediction = self.makePredictions(precessed_img)
        return prediction

    text: str

ai = network()

@app.get('/')
def home():
    return {"health_check": "OK", "model_version": model_version}

@app.post("/upload/")
async def create_upload_files(file: UploadFile = File(...)):

    # check for bad file uploads
    extension = file.filename[-4:]
    if extension != '.png' and extension != '.jpg' and extension != 'jpeg':
        return {"Error code:422": "Please upload a valid image with extension jpg or png."}

    # open file from POST operation
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()

    # store image to analyse it
    with open(f"./images/{file.filename}", "wb") as f:
        f.write(contents)
    
    # get prediction
    prediction = ai.pipeline(file.filename)

    # remove data from the server after making a prediction
    os.remove(f"./images/{file.filename}")

    return {"prediction": prediction}



# data = Image.open("./shirt.png")
# processed_image = ai.preprocess_img([data])

# ai.makePredictions(processed_image)


# 0	T-shirt/top
# 1	Trouser
# 2	Pullover
# 3	Dress
# 4	Coat
# 5	Sandal
# 6	Shirt
# 7	Sneaker
# 8	Bag
# 9	Ankle boot