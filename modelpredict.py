from keras.models import load_model
import cv2
import numpy as np
from PIL import Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
# load an image from file
image = load_img('D:/savs/IMG_CLF/data/TB_Diseases/Atelectasis/00000047_003.png', target_size=(256, 256))

# convert the image pixels to a numpy array
image = img_to_array(image)

# prepare the image for the VGG model
image = preprocess_input(image)

image = np.expand_dims(image, axis=0)

model = load_model('D:/savs/IMG_CLF/models/ResNet50_best.h5')
preds = model.predict(image)
print(/*////preds)
