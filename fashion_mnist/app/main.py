from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from model.model import preprocess, loadModel
from model.model import __version__ as model_version
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import mnist_reader


app = FastAPI()


class network:
    def __init__(self) -> None:
        self.model = loadModel()
        self.mappings = [ "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag" , "Ankle boot"]
    
    def makePredictions(self, img):
        probs = (self.model.predict(img))
        chosen_class = np.argmax(probs)
        print(self.mappings[chosen_class])

    def process(self, image_array):
        processed_images = preprocess(image_array)
        return processed_images



    text: str

class ImageIn(BaseModel):
    images: UploadFile

@app.get('/')
def home():
    return {"health_check": "OK", "model_version": model_version}

@app.post("/files/")
async def create_files(images: bytes = File()):
    return {"file_sizes": [len(file) for file in images]}


@app.post("/uploadfiles/")
async def create_upload_files(images: UploadFile = File()):
    return {"filenames": [file.filename for file in images]}



data = Image.open("./shirt.png")
ai = network()
processed_image = ai.process(np.array[data])


# print(X_train[0])
# print(processed_image[0])

plt.imshow(processed_image[0], cmap=plt.cm.binary)
plt.show()
ai.makePredictions(processed_image)


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