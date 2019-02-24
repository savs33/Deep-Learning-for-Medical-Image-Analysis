seed = 42

import numpy as np
np.random.seed(seed)

import scipy as sp
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from keras.models import load_model,Model
from keras import backend as K
from PIL import Image

import cv2
import helpers

class Segmenter():
	def __init__(self,model_path):
		self.img_rows = 512
		self.img_cols = 512
		self.img_channels = 1

		self.threshold = 50
		self.model_path = model_path
		self.model = self.get_model()

	def get_dice_coeff(self, y_pred, y_true):
		y_pred = K.flatten(y_pred)
		y_true = K.flatten(y_true)

		intersect = K.sum(y_pred * y_true)
		denominator = K.sum(y_pred) + K.sum(y_true)

		dice_coeff = (2.0 * intersect) / (denominator + 1e-6)
		return dice_coeff

	def custom_loss(self, y_pred, y_true):
		dice_coeff = self.get_dice_coeff(y_pred, y_true)
		return 1.0 - dice_coeff

	def load_model(self, path):
		model = load_model(path, custom_objects={
						   'custom_loss': self.custom_loss})
		return model

	def get_model(self):
		model = self.load_model(self.model_path)
		# model.summary()
		return model

	def preprocess_img(self,img):
		img = cv2.resize(img,(self.img_rows,self.img_cols))
		img = img/255.0
		return img

	def read_img(self,path):
		img = cv2.imread(path,0)
		img = self.preprocess_img(img)
		img = img.reshape(self.img_rows,self.img_cols,self.img_channels)
		return img

	def get_seg_mask(self,imgs):
		y_pred_score = self.model.predict(imgs)
		y_pred = y_pred_score * 255
		y_pred = y_pred > self.threshold
		y_pred = y_pred * 255.0
		return y_pred

	def save_seg_mask(self,in_path,out_path):
		paths = [in_path]
		imgs = np.array([self.read_img(x) for x in paths])
		results = self.get_seg_mask(imgs)
		seg_mask = [x for x in results]
		seg_mask = seg_mask[0]

		heatmap = cv2.applyColorMap(np.uint8(seg_mask), cv2.COLORMAP_AUTUMN)
		# heatmap = cv2.applyColorMap(np.uint8(seg_mask), cv2.COLORMAP_JET)
		status = cv2.imwrite(out_path,heatmap)

		img = cv2.imread(in_path)
		status = cv2.imwrite('static/tmp_imgs/3.png',img)

		print(out_path,status)


if __name__ == '__main__':

	seg = Segmenter(model_path='models/Unet_5.h5')

	img_paths = [
		'data/mtgm/normal/106.png',
		'data/mtgm/normal/14.png',
		'data/mtgm/normal/227.png',
		
		'data/mtgm/diseased/106.png',
		'data/mtgm/diseased/14.png',
		'data/mtgm/diseased/227.png',
		]

	# seg.process_batch(img_paths)

	for i,path in enumerate(img_paths):
		out_path = 'tmp/'+str(i)+'.jpg'
		seg.save_seg_mask(path,out_path=out_path)
