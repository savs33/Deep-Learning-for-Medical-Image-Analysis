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

class Visualizer():
	def __init__(self,model_path):
		self.img_rows = 224
		self.img_cols = 224
		self.img_channels = 3

		self.model_path = model_path
		self.last_conv_layer_name = 'conv5_block16_concat'
		self.pred_layer_name = 'dense_1'
		self.model = self.get_model()

	def get_model(self):
		model = load_model(self.model_path)
		last_conv_layer = model.get_layer(self.last_conv_layer_name)
		pred_layer = model.get_layer(self.pred_layer_name)
		model = Model(model.input,[last_conv_layer.output,pred_layer.output])
		# model.summary()
		return model

	def preprocess_img(self,img):
		# img = img.resize((self.img_rows,self.img_cols),Image.NEAREST)
		# img = np.asarray(img)
		img = cv2.resize(img,(self.img_rows,self.img_cols))
		img = img/255.0
		return img

	def read_img(self,path):
		# img = Image.open(path)
		img = cv2.imread(path)
		img = self.preprocess_img(img)
		return img

	def get_pred_cam(self,imgs):
		features,results = self.model.predict(imgs)
		gap_weights = self.model.layers[-1].get_weights()[0]

		cam_results = []

		for idx in range(len(imgs)):
			results = 1 - results
			results = results.flatten()
			result = results[0]

			original_img = imgs[idx]*255
			feature = features[idx]

			cam_output = np.dot(feature, gap_weights)
			original_img = cv2.resize(original_img,(self.img_rows,self.img_cols))
			cam_output = cv2.resize(cam_output,(self.img_rows,self.img_cols))
			cam_output /= np.max(cam_output)
			heatmap = cv2.applyColorMap(np.uint8(255*cam_output), cv2.COLORMAP_JET)
			heatmap[np.where(cam_output < 0.2)] = 0
			overlay = heatmap*0.5 + original_img
			cam_results.append([original_img,heatmap,overlay,result])

		return cam_results

	def process_batch(self,paths):

		helpers.clear_folder('tmp/cam/img/')
		helpers.clear_folder('tmp/cam/overlay/')

		imgs = np.array([self.read_img(x) for x in paths])
		helpers.write_imgs_serially(imgs*255,base_path='tmp/cam/img/')
		
		results = self.get_pred_cam(imgs)
		overlays = np.array([heatmap for original_img,heatmap,overlay,result in results])
		helpers.write_imgs_serially(overlays,base_path='tmp/cam/overlay/')

	def save_tb_overlay(self,in_path,out_path):
		
		paths = [in_path]
		imgs = np.array([self.read_img(x) for x in paths])
		results = self.get_pred_cam(imgs)
		overlays = np.array([heatmap for original_img,heatmap,overlay,result in results])
		scores = np.array([result for original_img,heatmap,overlay,result in results])
		
		overlay = overlays[0]
		status = cv2.imwrite(out_path,overlay)
		print(out_path,status)
		score = scores[0]
		
		img = cv2.imread(in_path)
		status = cv2.imwrite('static/tmp_imgs/1.png',img)
		print(status)

		return score


if __name__ == '__main__':

	vis = Visualizer(model_path='models/TB_detection.h5')

	img_paths = [
		'data/mtgm/normal/106.png',
		'data/mtgm/normal/14.png',
		'data/mtgm/normal/227.png',
		
		'data/mtgm/diseased/106.png',
		'data/mtgm/diseased/14.png',
		'data/mtgm/diseased/227.png',
		]

	# vis.process_batch(img_paths)

	for i,path in enumerate(img_paths):
		out_path = 'tmp/'+str(i)+'.jpg'
		score = vis.save_tb_overlay(path,out_path=out_path)
		print(score)
