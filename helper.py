import os
import shutil

def delete_folder(path):
	folder_paths = [root for root,dirs,files in os.walk(path)]

	for folder_path in folder_paths:
		if 'equal' in folder_path:
			print folder_path
			try:
				shutil.rmtree(folder_path)
			except:
				print '*'+folder_path

def add_gitignore(path):
	folder_paths = [root for root,dirs,files in os.walk(path)]

	for folder_path in folder_paths:

		file_path = folder_path+'/.gitignore'
		
		if os.path.exists(file_path):
			print '*'+file_path
			return

		file = open(file_path,'w')

		content = '*.jpg\n*.png\n'
		file.write(content)
		
		print file_path
		file.close()

def file_counts(path):
	for root,dirs,files in os.walk(path):
		num_files = len(files)
		if num_files==0:
			num_files = sum([len(f) for r,d,f in os.walk(root)])
		print root,':',num_files

if __name__ == '__main__':

	# path = '.'
	# add_gitignore(path)
	
	path = 'data/'
	file_counts(path)

	# delete_folder(path)
