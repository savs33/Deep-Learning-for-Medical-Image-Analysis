from keras.layers import Dense,Flatten,Dropout,Conv2D,MaxPool2D,Input
from keras.models import Model,save_model,load_model,Sequential
from keras.callbacks import ModelCheckpoint,TensorBoard,EarlyStopping
from keras.optimizers import Adam,SGD,RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50,Xception,InceptionResNetV2,DenseNet201
from keras import backend as K
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import os
import pandas as pd


class ImagePred(object):
	
	def __init__(self):

		K.clear_session()
		self.img_rows,self.img_cols = 512,512
		self.channels = 3
		self.input_shape = (self.img_rows,self.img_cols,self.channels)

		self.num_classes = 2
		self.seed = 42
		
		self.batch_size = 4

		# self.img_path = 'data/TB_Diseases/'
		self.img_path = 'D:/savs/IMG_CLF/data/validation_equal/'

		self.name = 'ResNet50_Pnuemonia'
		self.save_path = ''.join(['models/',self.name,'_best','.h5'])
		self.metrics_path = 'logs/'+self.name+'_metrics.txt'

	def get_data_generator(self,path):
		img_gen = ImageDataGenerator(
			rescale = 1/255.0)

		img_gen = img_gen.flow_from_directory(
			path,
			target_size = (self.img_rows,self.img_cols),
			batch_size = self.batch_size,
			seed = self.seed,
			class_mode = 'categorical',
			color_mode = 'rgb')

		return img_gen

	def get_predictions(self,save=True):
		
		self.model = load_model(self.save_path)
		data_generator = self.get_data_generator(self.img_path)

		y_actual = data_generator.classes
		y_actual = y_actual.reshape(-1,1)

		y_pred = self.model.predict_generator(
			generator = data_generator,
			verbose = 1,
			)

		if save:
			np.save('tmp/'+self.name+'_pred.npy',y_pred)
			np.save('tmp/'+self.name+'_true.npy',y_actual)
			
			y_pred = np.load('tmp/'+self.name+'_pred.npy')
			y_actual = np.load('tmp/'+self.name+'_true.npy')
			
			class1_prob,class2_prob = np.hsplit(y_pred,2)

			df = pd.DataFrame()
			df['class1'] = pd.Series(class1_prob.flatten())
			df['class2'] = pd.Series(class2_prob.flatten())
			df['pred'] = pd.Series(np.argmax(y_pred,axis=1))
			df['true'] = pd.Series(y_actual.flatten())

			df.to_csv('logs/predn_'+self.name+'.csv',index=False)

		return y_actual,y_pred

if __name__ == '__main__':

	clf = ImagePred()
	clf.get_predictions(save=True)
	del clf
