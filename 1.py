from keras.layers import Dense,Flatten,Dropout,Conv2D,MaxPool2D,Input
from keras.models import Model,save_model,load_model,Sequential
from keras.callbacks import ModelCheckpoint,TensorBoard,EarlyStopping
from keras.optimizers import Adam,SGD,RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50,Xception,InceptionResNetV2,DenseNet169
from keras import backend as K
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import os
