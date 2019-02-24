import shutil
import os
import numpy as np
import cv2
import errno


def clear_dir_contents(path):
	if os.path.exists(path):
		shutil.rmtree(path)
	os.makedirs(path)

def normalize(img,lower=0,upper=255):

	diff = np.max(img)-np.min(img)
	img = img.astype(np.float32)

	if diff != 0:
		img = ((upper-lower)*(img-np.min(img)))/(np.max(img)-np.min(img))
	
	return img

def create_folder(path):
	if os.path.exists(path):
		return
	os.makedirs(path)

def move_folder(src,dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src,dst)
    shutil.rmtree(src)

def get_count(path):
	for root,dirs,files in os.walk(path):
		print(root)
		count = sum([len(f) for r,d,f in os.walk(root)])
		print(root,':',count)


def know_shape(path):
	arr = np.load(path)
	print(arr.shape)

if __name__ == '__main__':
	get_count('data/train/')
	get_count('data/valid/')
	# get_count('data/vol/')

