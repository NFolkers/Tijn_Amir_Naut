from fastapi import FastAPI, File, UploadFile
from model.model import preprocess, loadModel
from model.model import __version__ as model_version
from PIL import Image
import numpy as np
import uuid
import os
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


app = FastAPI()


origins = [
    "http://localhost:3000",
    "localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

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


# retrieve a file from 
@app.post("/upload")
async def create_upload_files(file: UploadFile = File(...)):

    # check for bad file uploads
    # print(file)
    # print(file.filename)
    extension = file.filename[-4:]
    if extension != '.png' and extension != '.jpg' and extension != 'jpeg':
        return {"prediction": "null", "error": "Please upload a valid image with extension jpg or png."}

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