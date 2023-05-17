"""
Split the VOC dataset into train and validation splitting according to the probability
"""
import os
import argparse
import shutil
import numpy as np 

from makedir import smart_makedir

def train_test_split(main_dataset_path, p_train, p_val):
	"""
	Split the dataset in train and test accordind to the probability

	Parameters
	----------
	main_dataset_path: string
		path of original VOC dataset

	p_train : float
		propability to split in train - val_test

	p_val : float
		probability to split in val - test		
	"""
	# total names list
	n_tot = int(len(os.listdir(main_dataset_path + '/images')))
	name_list = [i.split('.')[0] for i in os.listdir(main_dataset_path + '/images')]

	# train - val_test splitting
	size_train = int(n_tot*p_train)
	train_name = np.random.choice(name_list, size=size_train, replace=False)
	
	# val - test splitting
	size_val = int(n_tot*p_val)
	val_test_name = [i for i in name_list if i not in train_name] 
	val_name = np.random.choice(val_test_name, size=size_val, replace=False)
	test_name = [i for i in val_test_name if i not in val_name] 

	# Create the sub folder
	dict_splitting = {'train':train_name, 'validation':val_name, 'test':test_name}
	for k in dict_splitting.keys():
		print(k)
		k_images = main_dataset_path + '/' + k + '/images/'
		k_annotations = main_dataset_path + '/' + k + '/Annotations/'
		smart_makedir(k_images)
		smart_makedir(k_annotations)
		for name in dict_splitting[k]:
			shutil.copyfile(main_dataset_path + '/images/' + name + '.png', 	
							k_images + name + '.png')

			shutil.copyfile(main_dataset_path + '/Annotations/' + name + '.xml', 	
							k_annotations + name +'.xml')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Splitting precedure train/val')
	parser.add_argument("data_directory", type=str, help="VOC dataset directory")
	parser.add_argument("-p_train", default=0.7, type=float, help="percentage of train dataset")
	parser.add_argument("-p_val", default=0.2, type=float, help="percentage of val dataset")
	args = parser.parse_args()

	train_test_split(args.data_directory, args.p_train, args.p_val)

