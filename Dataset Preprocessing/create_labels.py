import os
# import cv2
import h5py
import torch
import shutil
import torchvision
import numpy as np
# from imutils import paths
from torchvision import transforms, datasets, models
from matplotlib import pyplot as plt

#################### List Species Name in a text file #############################################
data_directory = '/home/ashimag/wii_data_species_2022/images/wii_data_aite_species_data-2022'
labels_directory = '/home/ashimag/wii_data_species_2022/labels/wii_all_labels'
# data_directory_BB = '/home/ashimag/wii_data_species_2022/images/wii_data_aite_species_data-2022_BB'
labels = os.listdir(data_directory)
# with open('2022_aite_species.txt', 'w') as f:
#     for label in labels:
#         f.write(f"{label}\n")

# data_directory = '/home/ashimag/wii_data_tiger_project_2014'
# labels = os.listdir(data_directory)
# print(labels)
# print(len(labels))

# data_directory = '/home/ashimag/wii_data_27-09-2018/Block_1/Photographs'
# labels = os.listdir(data_directory)
# print(labels)
# print(len(labels))
# with open('2018_block1_photos.txt', 'w') as f:
#     for label in labels:
#         f.write(f"{label}\n")

# data_directory = '/home/ashimag/wii_data_27-09-2018/Block_2/Photographs'
# labels = os.listdir(data_directory)
# print(labels)
# print(len(labels))
# with open('2018_block2_photos.txt', 'w') as f:
#     for label in labels:
#         f.write(f"{label}\n")
"""
def visualize_statistics():
	labels = os.listdir(data_directory)
	num_images_list = []
	print(labels)
	count = 0 
	for each_species in labels:
		# print("current species: ", each_species, end="  ")
		species_path = os.path.join(data_directory, each_species)
		images_path = os.listdir(species_path)
		# images_path = list(paths.list_images(species_path))
		num = len(images_path)
		num_images_list.append(num)
		if num > 0: 
			print(each_species, end=" ")
			print(num)
			count += 1
			# print(each_species)
	print(count)
	print(np.sum(num_images_list))
	# with open('2014_tiger_count.txt', 'w') as f:
	# 	for (i, each_species) in enumerate(labels):
	# 		f.write(f"{num_images_list[i]}\n")

visualize_statistics()
"""
# """
# visualize statistics of dataset
num_images_list = []
classes_less_than10 = 0
classes_less_than100 = 0
classes_less_than500 = 0
classes_more_than_5000 = []
classes_less_than100_list = []
for each_species in labels:
	species_path = os.path.join(data_directory, each_species)
	images_path = os.listdir(species_path)
	# images_path = list(paths.list_images(species_path))
	num = len(images_path)
	num_images_list.append(num)
	if num == 1:
		print("1 species only: ", each_species)
	if num < 10:
		classes_less_than10 += 1
	if num < 100: 
		classes_less_than100_list.append(each_species)
		classes_less_than100 += 1
	if num < 500: 
		classes_less_than500 += 1
	if num > 5000:
		classes_more_than_5000.append(each_species)


print(classes_less_than100_list)
# print("Num of classes less than 10: ", classes_less_than10)
# print("Num of classes less than 100: ", classes_less_than100)
# print("Num of classes less than 500: ", classes_less_than500)
# print(num_images_list)
# x_axis = np.arange(1, len(labels)+1, 1)
# class_num_max = np.argmax(num_images_list)
# class_num_min = np.argmin(num_images_list)
# # print(class_num_max)
# # print(class_num_min)
# print(np.max(num_images_list))
# print(np.min(num_images_list))
# print(labels[class_num_max])
# print(labels[class_num_min])
# print(classes_more_than_5000)
# """
# """
########################################################## Remove labels (less than 100 images) from labels list ############################################
# """
print("Number of total categories: ", len(labels))
for category in classes_less_than100_list:
	labels.remove(category)
print("Number of remaining categories: ", len(labels))
# """
# """
# """
# create train, validation, test stratified splits.
image_names = []
category_names = []
label_file_names = []
from sklearn.model_selection import StratifiedShuffleSplit
# create list of all images and all labels
for each_species in labels:
	species_path = os.path.join(data_directory, each_species)
	images = os.listdir(species_path)

	species_label_path = os.path.join(labels_directory, each_species)
	label_files = os.listdir(species_label_path)

	images_path = [os.path.join(species_path, images[i]) for i in range(len(images))]
	label_files_path = [os.path.join(species_label_path, label_files[i]) for i in range(len(label_files))]

	# import pdb; pdb.set_trace()
	image_names.extend(images_path)
	label_file_names.extend(label_files_path)

	categories = [each_species for i in range(len(images))]
	category_names.extend(categories)

image_names = np.array(image_names)
category_names = np.array(category_names)
# import pdb; pdb.set_trace()

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
sss.get_n_splits(image_names, category_names)
for train_idx, left_idx in sss.split(image_names, category_names):
	X_train, left_images = image_names[train_idx], image_names[left_idx]
	Y_train, left_category = category_names[train_idx], category_names[left_idx]
	label_files_train, left_label_files = label_file_names[train_idx], label_file_names[left_idx]
	
sss_test = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
for val_idx, test_idx in sss_test.split(left_images, left_category):
	X_val, X_test = left_images[val_idx], left_images[test_idx]
	Y_val, Y_test = left_category[val_idx], left_category[test_idx]
	label_files_val, label_files_test = left_label_files[val_idx], left_label_files[test_idx]

# for left_idx, val_idx in sss.split(image_names, category_names):
# 	left_images, X_val = image_names[left_idx], image_names[val_idx]
# 	left_category, Y_val = category_names[left_idx], category_names[val_idx]

# print(len(X_val), "    ", len(Y_val))
# import pdb; pdb.set_trace()

# sss_test = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
# sss_test.get_n_splits(left_images, left_category)
# for train_idx, test_idx in sss_test.split(left_images, left_category):
# 	X_train, X_test = left_images[train_idx], left_images[test_idx]
# 	Y_train, Y_test = left_category[train_idx], left_category[test_idx]

print("X_train size: ", len(X_train))
print("X_val size: ", len(X_val))
print('X_test size: ', len(X_test))
print("Total size: ", len(X_train) + 2*len(X_val))

# import pdb; pdb.set_trace()

train_dest_dir = '/home/ashimag/wii_data_species_2022/images/train'
train_label_dest_dir = '/home/ashimag/wii_data_species_2022/labels/train'
for i, src_img_path in enumerate(X_train):
	img_cat = src_img_path.split('/')[-2]
	src_label_path = label_files_train[i]

	dest_img_path = os.path.join(train_dest_dir, img_cat)
	dest_label_file_path = os.path.join(train_label_dest_dir, img_cat)

	if not os.path.isdir(dest_img_path):
		os.mkdir(dest_img_path)
	shutil.copy(src_img_path, dest_img_path)

	if not os.path.isdir(dest_label_file_path)
		os.mkdir(dest_label_file_path)
	shutil.copy(src_label_path, dest_label_file_path)

val_dest_dir = '/home/ashimag/wii_data_species_2022/images/val'
val_label_dest_dir = '/home/ashimag/wii_data_species_2022/labels/val'
for i, src_img_path in enumerate(X_val):
	img_cat = src_img_path.split('/')[-2]
	src_label_path = label_files_val[i]

	dest_img_path = os.path.join(val_dest_dir, img_cat)
	dest_label_file_path = os.path.join(val_label_dest_dir, img_cat)

	if not os.path.isdir(dest_img_path):
		os.mkdir(dest_img_path)
	shutil.copy(src_img_path, dest_img_path)
	if not os.path.isdir(dest_label_file_path)
		os.mkdir(dest_label_file_path)
	shutil.copy(src_label_path, dest_label_file_path)


test_dest_dir = '/home/ashimag/wii_data_species_2022/images/test'
test_label_dest_dir = '/home/ashimag/wii_data_species_2022/labels/test'
for i, src_img_path in enumerate(X_val):
	img_cat = src_img_path.split('/')[-2]
	src_label_path = label_files_test[i]

	dest_img_path = os.path.join(test_dest_dir, img_cat)
	dest_label_file_path = os.path.join(test_label_dest_dir, img_cat)

	if not os.path.isdir(dest_img_path):
		os.mkdir(dest_img_path)
	shutil.copy(src_img_path, dest_img_path)

	if not os.path.isdir(dest_label_file_path)
		os.mkdir(dest_label_file_path)
	shutil.copy(src_label_path, dest_label_file_path)

# if os.path.isdir(path) and confirmation != "y":
# 	confirmation = input("Path '%s" %path + "' already exists. Overwrite? (y/n)")

# elif not os.path.isdir(path):
# 	os.mkdir(path)
# """

# plt.bar(x_axis, num_images_list)
# plt.savefig('species_dataset_statistics.png')

	# os.listdir(os.path.join())
# TRAIN_IMAGES = "/home/ashimag/Datasets/tiered_imagenet/train_images"
# TRAIN_IMAGES = "/home/ashimag/Datasets/mini_imagenet/train_images"
# train_image_paths = list(paths.list_images(TRAIN_IMAGES))
