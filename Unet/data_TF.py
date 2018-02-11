# -*- coding:utf-8 -*-
'''  
#====#====#====#====
# Project Name:     U-net 
# File Name:        data_TF 
# Date:             2/10/18 8:38 PM 
# Using IDE:        PyCharm Community Edition  
# From HomePage:    https://github.com/DuFanXin/U-net
# Author:           DuFanXin 
# BlogPage:         http://blog.csdn.net/qq_30239975  
# E-mail:           18672969179@163.com
# Copyright (c) 2018, All Rights Reserved.
#====#====#====#==== 
'''
import Augmentor
import os
import glob
import cv2
import tensorflow as tf

TRAIN_SET_NAME = 'train_set.tfrecords'
DEVELOPMENT_SET_NAME = 'development_set.tfrecords'
INPUT_IMG_WIDE, INPUT_IMG_HEIGHT, INPUT_IMG_CHANNEL = 512, 512, 1
OUTPUT_IMG_WIDE, OUTPUT_IMG_HEIGHT, OUTPUT_IMG_CHANNEL = 512, 512, 1
TAIN_SET_SIZE = 1000
DEVELOPMENT_SET_SIZE = 200


def augment():
	p = Augmentor.Pipeline(
		source_directory='../data_set/my_set/merged_origin_data_set',
		output_directory='../data_set/my_set/merged_augment_data_set'
	)
	p.rotate(probability=0.2, max_left_rotation=2, max_right_rotation=2)
	p.zoom(probability=0.2, min_factor=1.1, max_factor=1.2)
	p.skew(probability=0.2)
	p.random_distortion(probability=0.2, grid_width=100, grid_height=100, magnitude=1)
	p.shear(probability=0.2, max_shear_left=2, max_shear_right=2)
	p.crop_random(probability=0.2, percentage_area=0.8)
	p.flip_random(probability=0.2)
	p.sample(n=TAIN_SET_SIZE + DEVELOPMENT_SET_SIZE)


def split_merged_augment_data_set():
	merged_data_set_paths = glob.glob(os.path.join('../data_set/my_set/merged_augment_data_set', '*.JPEG'))
	augment_image_path = '../data_set/my_set/augment_images'
	if not os.path.lexists(augment_image_path):
		os.mkdir(augment_image_path)
	augment_label_path = '../data_set/my_set/augment_labels'
	if not os.path.lexists(augment_label_path):
		os.mkdir(augment_label_path)
	for index, merged_data_set_path in enumerate(merged_data_set_paths):
		merged_image = cv2.imread(merged_data_set_path)
		print(merged_data_set_path)
		image = merged_image[:, :, 0]
		label = merged_image[:, :, 2]
		cv2.imwrite(filename=os.path.join(augment_image_path, '%d.jpg' % index), img=image)
		cv2.imwrite(filename=os.path.join(augment_label_path, '%d.jpg' % index), img=label)
	print('Done split merged augment data_set. \nImages at %s\nLabels at %s' % (augment_image_path, augment_label_path))


def write_img_to_tfrecords():
	augment_image_path = '../data_set/my_set/augment_images'
	augment_label_path = '../data_set/my_set/augment_labels'
	train_set_writer = tf.python_io.TFRecordWriter(os.path.join('../data_set/my_set', TRAIN_SET_NAME))  # 要生成的文件
	development_set_writer = tf.python_io.TFRecordWriter(os.path.join('../data_set/my_set', DEVELOPMENT_SET_NAME))
	# test_set_writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.data_dir, 'test_set.tfrecords'))  # 要生成的文件

	for index in range(TAIN_SET_SIZE):
		train_image = cv2.imread(os.path.join(augment_image_path, '%d.jpg' % index), flags=0)
		train_label = cv2.imread(os.path.join(augment_label_path, '%d.jpg' % index), flags=0)
		# train_image = cv2.resize(src=train_image, dsize=(INPUT_IMG_WIDE, INPUT_IMG_HEIGHT))
		# train_image = np.asarray(a=train_image, dtype=np.uint8)
		train_image = cv2.resize(src=train_image, dsize=(INPUT_IMG_WIDE, INPUT_IMG_HEIGHT))
		# train_label = np.asarray(a=train_label, dtype=np.uint8)
		train_label = cv2.resize(src=train_label, dsize=(OUTPUT_IMG_WIDE, OUTPUT_IMG_HEIGHT))
		train_label[train_label <= 100] = 0
		train_label[train_label > 100] = 1
		# train_image = io.imread(file_path)
		# train_image = transform.resize(train_image, (INPUT_IMG_WIDE, INPUT_IMG_HEIGHT, INPUT_IMG_CHANNEL))
		# sample_image = train_image[:, :, 0]
		# label_image = train_image[:, :, 2]
		# label_image[label_image < 100] = 0
		# label_image[label_image > 100] = 10
		example = tf.train.Example(features=tf.train.Features(feature={
			'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[train_label.tobytes()])),
			'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[train_image.tobytes()]))
		}))  # example对象对label和image数据进行封装
		train_set_writer.write(example.SerializeToString())  # 序列化为字符串
		if index % 100 == 0:
			print('Done train_set writing %.2f%%' % (index / TAIN_SET_SIZE * 100))
	train_set_writer.close()
	print("Done train_set writing")

	for index in range(TAIN_SET_SIZE, TAIN_SET_SIZE + DEVELOPMENT_SET_SIZE):
		development_image = cv2.imread(os.path.join(augment_image_path, '%d.jpg' % index), flags=0)
		development_label = cv2.imread(os.path.join(augment_label_path, '%d.jpg' % index), flags=0)
		# development_image = cv2.resize(src=development_image, dsize=(INPUT_IMG_WIDE, INPUT_IMG_HEIGHT))
		# development_image = np.asarray(a=development_image, dtype=np.uint8)
		development_image = cv2.resize(src=development_image, dsize=(INPUT_IMG_WIDE, INPUT_IMG_HEIGHT))
		# development_label = np.asarray(a=development_label, dtype=np.uint8)
		development_label = cv2.resize(src=development_label, dsize=(OUTPUT_IMG_WIDE, OUTPUT_IMG_HEIGHT))
		development_label[development_label <= 100] = 0
		development_label[development_label > 100] = 1
		# development_image = io.imread(file_path)
		# development_image = transform.resize(development_image, (INPUT_IMG_WIDE, INPUT_IMG_HEIGHT, INPUT_IMG_CHANNEL))
		# sample_image = development_image[:, :, 0]
		# label_image = development_image[:, :, 2]
		# label_image[label_image < 100] = 0
		# label_image[label_image > 100] = 10
		example = tf.train.Example(features=tf.train.Features(feature={
			'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[development_label.tobytes()])),
			'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[development_image.tobytes()]))
		}))  # example对象对label和image数据进行封装
		development_set_writer.write(example.SerializeToString())  # 序列化为字符串
		if index % 100 == 0:
			print('Done development_set writing %.2f%%' % ((index - TAIN_SET_SIZE) / DEVELOPMENT_SET_SIZE * 100))
	development_set_writer.close()
	print("Done development_set writing")

if __name__ == '__main__':
	# split_merged_augment_data_set()
	write_img_to_tfrecords()
