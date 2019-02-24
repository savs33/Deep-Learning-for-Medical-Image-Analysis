import Augmentor
import numpy as np
import os


def generate_samples(src,dst,num_samples,seed):
	p = Augmentor.Pipeline(
		source_directory = src,
		output_directory = dst)

	p.set_seed(seed=seed)
	p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)
	# p.zoom(probability=0.5, min_factor=1.0, max_factor=1.2)
	# p.crop_random(probability=0.1, percentage_area=0.9)
	p.random_distortion(probability=0.5,grid_height=10,grid_width=10,magnitude=2)
	p.skew_left_right(probability=0.5,magnitude=0.1)
	p.skew_top_bottom(probability=0.5,magnitude=0.1)
	p.resize(probability=1.0, width=512, height=512)
	p.sample(num_samples)

	del p

def equal_sample(src,dst,samples_per_class,seed):
	folders = [r for r,d,f in os.walk(src)][1:]
	for folder in folders:
		class_name  = folder.split('/')[-1]
		print (class_name)
		in_folder = src + class_name +'/' 
		out_folder = dst + class_name +'/' 
		generate_samples(src=in_folder,dst=out_folder,num_samples=samples_per_class,seed=seed)

if __name__ == '__main__':
	
	equal_sample(src='data/TB_Diseases/',dst='../../train_equal/',samples_per_class=1000,seed=42)
	equal_sample(src='data/TB_Diseases/',dst='../../validation_equal/',samples_per_class=50,seed=1)