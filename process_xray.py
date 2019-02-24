import cv2
import time
import numpy as np

def convt_img(raw_img):
	img = cv2.imdecode(np.frombuffer(raw_img, np.uint8), -1)
	return img

def get_tb_results(img):
	heatmap = cv2.bitwise_not(img)
	time.sleep(2)
	return img,heatmap,'100'

def get_seg_results(img):
	heatmap = cv2.bitwise_not(img)
	heatmap = heatmap + 100
	# time.sleep(2)
	return heatmap

def process_tb(raw_img,filename):
	img = convt_img(raw_img)
	img,heatmap,percentage = get_tb_results(img)
	return heatmap,percentage

def process_seg(raw_img,filename):
	img = convt_img(raw_img)
	seg_mask = get_seg_results(img)
	return seg_mask

def write_img(img,path):
	status = cv2.imwrite(path,img)
	print(status)

if __name__ == '__main__':
	img = cv2.imread('static/imgs/xray.png')
	img,heatmap,percentage = process_img(img)
	print(img.shape)
	print(heatmap.shape)
	print(percentage)

	write_img(heatmap,'out.jpg')