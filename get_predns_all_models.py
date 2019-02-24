from get_predictions import ImagePred
from glob import glob



def get_model_names(path):
	model_names = glob(path)
	model_names = [x.split('/')[-1] for x in model_names]
	model_names = [x.split('.')[0] for x in model_names]
	model_names = [x.replace('_best','') for x in model_names]
	return model_names

def run_for_model(model_name):
	print model_name
	
	clf = ImagePred()
	clf.name = model_name
	
	clf.get_predictions(save=True)
	del clf

if __name__ == '__main__':

	model_names = get_model_names(path='models/*.h5')
	for model_name in model_names:
		run_for_model(model_name)

