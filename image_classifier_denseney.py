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


# from matplotlib import pyplot as plt

class ImageClassifier(object):
	"""docstring for ImageClassifier"""
	def __init__(self):

		K.clear_session()
		# self.img_rows,self.img_cols = 197,197//ResNet50,Xception,
		self.img_rows,self.img_cols = 256,256
		self.channels = 3
		self.input_shape = (self.img_rows,self.img_cols,self.channels)

		self.num_classes = 15
		self.seed = 42

		self.batch_size = 2

		self.train_img_path = 'data/train_equal/'
		self.validation_img_path = 'data/validation_equal/'


		self.steps_per_epoch = self.get_count_in_folder(self.train_img_path)//self.batch_size,
		self.validation_steps = self.get_count_in_folder(self.validation_img_path)//self.batch_size,

		# self.name = 'ResNet50'
		self.name = 'DenseNet201'
		self.save_path = ''.join(['models/',self.name,'_best','.h5'])
		self.metrics_path = 'logs/'+self.name+'_metrics.txt'

		self.model = self.get_model()
		# print (self.model.summary())

	def get_count_in_folder(self,path):
		total = np.sum([len(files) for root,dirs,files in os.walk(path)])
		return total

	def cnn_model(self):
		model = Sequential()
		model.add(Conv2D(64,3,activation='relu',input_shape=self.input_shape))
		model.add(Conv2D(64,3))
		model.add(MaxPool2D(pool_size=(2,2)))
		model.add(Conv2D(128,3))
		model.add(Conv2D(128,3))
		model.add(MaxPool2D(pool_size=(2,2)))
		model.add(Flatten())
		model.add(Dense(512,activation='relu'))
		model.add(Dense(512,activation='relu'))
		model.add(Dense(self.num_classes,activation='relu'))
		return model

	def get_model(self):
		return self.pretrained_model()

	def pretrained_model(self):
		# base_model = ResNet50(include_top=False,input_shape=self.input_shape,weights='imagenet')
		base_model = DenseNet201(include_top=False,input_shape=self.input_shape,weights='imagenet')
		# base_model = Xception(include_top=False,input_shape=self.input_shape,weights='imagenet')
		# base_model = InceptionResNetV2(include_top=False,input_shape=self.input_shape,weights='imagenet')

		x = base_model.output
		x = Flatten()(x)
		x = Dense(512,activation='relu')(x)
		x = Dropout(0.5)(x)
		x = Dense(512,activation='relu')(x)
		x = Dropout(0.5)(x)
		predictions = Dense(self.num_classes,activation='softmax')(x)

		model = Model(base_model.inputs,predictions)
		return model

	def build_model(self,lr=1e-3):

		opt = Adam(lr=lr)
		self.model.compile(
			optimizer = opt,
			loss = 'binary_crossentropy',
			metrics = ['accuracy']
			)

	def get_train_generator(self,path):
		img_gen = ImageDataGenerator(
			zoom_range = 0.0,
			width_shift_range = 0.0,
			height_shift_range = 0.0,
			horizontal_flip = False,
			rotation_range = 0.0,
			rescale = 1/255.0)

		img_gen = img_gen.flow_from_directory(
			path,
			target_size = (self.img_rows,self.img_cols),
			batch_size = self.batch_size,
			seed = self.seed,
			class_mode = 'categorical',
			color_mode = 'rgb'
			)

		return img_gen

	def get_validation_generator(self,path):
		img_gen = ImageDataGenerator(
			rescale = 1/255.0)

		img_gen = img_gen.flow_from_directory(
			path,
			target_size = (self.img_rows,self.img_cols),
			batch_size = self.batch_size,
			seed =self.seed,
			class_mode = 'categorical',
			color_mode = 'rgb',
			shuffle = False
			)

		return img_gen

	def get_test_generator(self,path):
		img_gen = ImageDataGenerator(
			rescale = 1/255.0)

		img_gen = img_gen.flow_from_directory(
			path,
			target_size = (self.img_rows,self.img_cols),
			batch_size = self.batch_size,
			seed =self.seed,
			class_mode = None,
			color_mode = 'rgb',
			shuffle = False
			)

		return img_gen

	def get_callbacks(self):
		early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=5, verbose=1, mode='auto')
		checkpointer = ModelCheckpoint(filepath=self.save_path, verbose=1, save_best_only=True)
		# tensorboard = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=self.batch_size, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
		# return [early_stopping,checkpointer,tensorboard]
		# return [early_stopping,checkpointer]
		return [early_stopping,checkpointer]

	def train(self,lr=1e-4,num_epochs=2):

		self.build_model(lr)

		train_generator = self.get_train_generator(self.train_img_path)
		validation_generator = self.get_validation_generator(self.validation_img_path)

		hist = self.model.fit_generator(
			generator = train_generator,
			epochs = num_epochs,
			validation_data = validation_generator,
			callbacks = self.get_callbacks(),
			)

	def continue_training(self,lr=1e-4,num_epochs=10):

		self.model = load_model(self.save_path)
		# self.build_model(lr)

		train_generator = self.get_train_generator(self.train_img_path)
		validation_generator = self.get_validation_generator(self.validation_img_path)

		hist = self.model.fit_generator(
			generator = train_generator,
			epochs  = num_epochs,
			validation_data = validation_generator,
			callbacks = self.get_callbacks(),
			)

	def get_predictions(self,save=True):

		self.model = load_model(self.save_path)
		validation_generator = self.get_validation_generator(self.validation_img_path)

		validation_steps = self.validation_steps[0]

		y_actual = np.empty((0,self.num_classes))
		y_pred = np.empty((0,self.num_classes))

		y_actual = validation_generator.classes
		y_pred = self.model.predict_generator(
			generator = validation_generator,
			verbose = 1,
			)

		if save:
			np.save('tmp/'+self.name+'_pred.npy',y_pred)
			np.save('tmp/'+self.name+'_true.npy',y_actual)

		return y_actual,y_pred

	def get_metrics(self,save_logs=True):
		y_pred = np.load('tmp/'+self.name+'_pred.npy')
		y_actual = np.load('tmp/'+self.name+'_true.npy')

		y_pred = np.argmax(y_pred,axis=1)

		print (y_actual.shape)
		print (y_pred.shape)

		cm = confusion_matrix(y_actual,y_pred)
		report = classification_report(y_actual,y_pred)
		accuracy = accuracy_score(y_actual,y_pred)

		print (cm)
		print (report)
		print ('Accuracy :',accuracy)

		if save_logs:
			file = open(self.metrics_path,'w')
			print (cm,end="",depend=file)
			print (report,end="",depend=file)
			file.close()

	def get_all_pred(self):

		self.model = load_model(self.save_path)
		validation_generator = self.get_validation_generator(self.test_img_path)

		y_actual = np.empty((0,self.num_classes))
		y_pred = np.empty((0,self.num_classes))

		y_pred = self.model.predict_generator(
			generator = validation_generator,
			verbose = 1,
			)

		return y_pred



if __name__ == '__main__':

	clf = ImageClassifier()
	clf.train(lr=1e-3,num_epochs=5)
	del clf

	clf = ImageClassifier()
	clf.get_predictions(save=True)
	del clf

	clf = ImageClassifier()
	clf.get_metrics(save_logs=True)
	del clf
