# -*- coding:utf-8 -*-
"""
#====#====#====#====
# Project Name:     U-net 
# File Name:        data_Keras
# Date:             2/9/18 4:44 PM 
# Using IDE:        PyCharm Community Edition  
# From HomePage:    https://github.com/DuFanXin/U-net
# Author:           DuFanXin 
# BlogPage:         http://blog.csdn.net/qq_30239975  
# E-mail:           18672969179@163.com
# Copyright (c) 2018, All Rights Reserved.
#====#====#====#==== 
"""
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import glob
import cv2
# import tensorflow as tf
# from data_TF import TRAIN_SET_NAME, VALIDATION_SET_NAME, TEST_SET_NAME, PREDICT_SET_NAME, \
# 	TRAIN_SET_SIZE, VALIDATION_SET_SIZE, TEST_SET_SIZE, PREDICT_SET_SIZE, \
# 	ORIGIN_IMAGE_DIRECTORY, ORIGIN_LABEL_DIRECTORY, ORIGIN_PREDICT_DIRECTORY
TRAIN_SET_NAME = 'train_set.tfrecords'
VALIDATION_SET_NAME = 'validation_set.tfrecords'
TEST_SET_NAME = 'test_set.tfrecords'
PREDICT_SET_NAME = 'predict_set.tfrecords'

# ORIGIN_MERGED_SOURCE_DIRECTORY = '../data_set/my_set/merged_origin_data_set'
ORIGIN_IMAGE_DIRECTORY = '../data_set/my_set/image'
ORIGIN_LABEL_DIRECTORY = '../data_set/my_set/label'
# ORIGIN_PREDICT_DIRECTORY = '../data_set/test'
# Augmentor.Pipeline的参数'output_directory'有毒，非要绝对路径，只能这样咯
AUGMENT_OUTPUT_DIRECTORY = \
	os.getcwd()[:os.getcwd().rindex('/')] + '/data_set/my_set/merged_augment_data_set'
AUGMENT_IMAGE_PATH = '../data_set/augment_images'
AUGMENT_LABEL_PATH = '../data_set/augment_labels'

INPUT_IMG_WIDE, INPUT_IMG_HEIGHT, INPUT_IMG_CHANNEL = 512, 512, 1
OUTPUT_IMG_WIDE, OUTPUT_IMG_HEIGHT, OUTPUT_IMG_CHANNEL = 512, 512, 1
TRAIN_SET_SIZE = 2100
VALIDATION_SET_SIZE = 27
TEST_SET_SIZE = 30
PREDICT_SET_SIZE = 30


class Augmentation(object):
	# A class used to augmentate image
	# Firstly, read train image and label seperately, and then merge them together for the next process
	# Secondly, use keras preprocessing to augmentate image
	# Finally, seperate augmentated image apart into train image and label
	def __init__(
			self, train_path="../data_set/train", label_path="../data_set/label", merge_path="../data_set/merge",
			aug_merge_path="../data_set/aug_merge", aug_train_path="../data_set/aug_train",
			aug_label_path="../data_set/aug_label", img_type="tif"
	):

		# Using glob to get all .img_type form path
		self.train_imgs = glob.glob(train_path + "/*." + img_type)  # 训练集
		self.label_imgs = glob.glob(label_path + "/*." + img_type)  # label
		self.train_path = train_path
		self.label_path = label_path
		self.merge_path = merge_path
		self.img_type = img_type
		self.aug_merge_path = aug_merge_path
		self.aug_train_path = aug_train_path
		self.aug_label_path = aug_label_path
		self.slices = len(self.train_imgs)
		self.datagen = ImageDataGenerator(
			rotation_range=0.2,
			width_shift_range=0.05,
			height_shift_range=0.05,
			shear_range=0.05,
			zoom_range=0.05,
			horizontal_flip=True,
			fill_mode='nearest')

	def augmentation(self):
		# 读入3通道的train和label, 分别转换成矩阵, 然后将label的第一个通道放在train的第2个通处, 做数据增强
		print("运行 Augmentation")

		# Start augmentation.....
		trains = self.train_imgs
		labels = self.label_imgs
		path_train = self.train_path
		path_label = self.label_path
		path_merge = self.merge_path
		imgtype = self.img_type
		path_aug_merge = self.aug_merge_path
		print('%d images \n%d labels' % (len(trains), len(labels)))
		if len(trains) != len(labels) or len(trains) == 0 or len(trains) == 0:
			print("trains can't match labels")
			return 0
		if not os.path.lexists(path_merge):
			os.mkdir(path_merge)
		if not os.path.lexists(path_aug_merge):
			os.mkdir(path_aug_merge)
		for i in range(len(trains)):
			img_t = load_img(path_train + "/" + str(i) + "." + imgtype)  # 读入train
			img_l = load_img(path_label + "/" + str(i) + "." + imgtype)  # 读入label
			x_t = img_to_array(img_t)                                    # 转换成矩阵
			x_l = img_to_array(img_l)
			x_t[:, :, 2] = x_l[:, :, 0]                                  # 把label当做train的第三个通道
			img_tmp = array_to_img(x_t)
			img_tmp.save(path_merge + "/" + str(i) + "." + imgtype)      # 保存合并后的图像
			img = x_t
			img = img.reshape((1,) + img.shape)                          # 改变shape(1, 512, 512, 3)
			savedir = path_aug_merge + "/" + str(i)                      # 存储合并增强后的图像
			if not os.path.lexists(savedir):
				os.mkdir(savedir)
			print("running %d doAugmenttaion" % i)
			self.do_augmentate(img, savedir, str(i))                      # 数据增强

	def do_augmentate(self, img, save_to_dir, save_prefix, batch_size=1, save_format='tif', imgnum=70):
		# augmentate one image
		datagen = self.datagen
		i = 0
		for _ in datagen.flow(
				img,
				batch_size=batch_size,
				save_to_dir=save_to_dir,
				save_prefix=save_prefix,
				save_format=save_format):
			i += 1
			if i > imgnum:
				break

	def split_merge(self):
		# 读入合并增强之后的数据(aug_merge), 对其进行分离, 分别保存至 aug_train, aug_label
		print("running splitMerge")

		# split merged image apart
		path_merge = self.aug_merge_path       # 合并增强之后的图像
		path_train = self.aug_train_path       # 增强之后分离出来的train
		path_label = self.aug_label_path       # 增强之后分离出来的label

		if not os.path.lexists(path_train):
			os.mkdir(path_train)
		if not os.path.lexists(path_label):
			os.mkdir(path_label)

		for i in range(self.slices):
			path = path_merge + "/" + str(i)
			print(path)
			train_imgs = glob.glob(path + "/*." + self.img_type)  # 所有训练图像
			savedir = path_train + "/" + str(i)                   # 保存训练集的路径
			if not os.path.lexists(savedir):
				os.mkdir(savedir)
			savedir = path_label + "/" + str(i)                   # 保存label的路径
			if not os.path.lexists(savedir):
				os.mkdir(savedir)
			for imgname in train_imgs:         # rindex("/") 是返回'/'在字符串中最后一次出现的索引
				midname = imgname[imgname.rindex("/") + 1:imgname.rindex("." + self.img_type)]  # 获得文件名(不包含后缀)
				img = cv2.imread(imgname)      # 读入训练图像
				img_train = img[:, :, 2]  # 训练集是第2个通道, label是第0个通道
				img_label = img[:, :, 0]
				cv2.imwrite(path_train + "/" + str(i) + "/" + midname + "_train" + "." + self.img_type, img_train)  # 保存训练图像和label
				cv2.imwrite(path_label + "/" + str(i) + "/" + midname + "_label" + "." + self.img_type, img_label)


class DataProcess(object):

	def __init__(
			self, out_rows, out_cols, aug_merge_path='../data_set/aug_merge',
			aug_train_path="../data_set/aug_train", aug_label_path="../data_set/aug_label",
			test_path='../data_set/test', npy_path="../data_set/npydata", img_type="tif"
	):
		self.out_rows = out_rows
		self.out_cols = out_cols
		self.aug_merge_path = aug_merge_path
		self.aug_train_path = aug_train_path
		self.aug_label_path = aug_label_path
		self.test_path = test_path
		self.npy_path = npy_path
		self.img_type = img_type

	def create_train_data(self):
		# 将增强之后的训练集生成npy
		i = 0
		print('-' * 30)
		print('creating train image')
		print('-' * 30)
		count = 0
		for indir in os.listdir(self.aug_merge_path):
			path = os.path.join(self.aug_merge_path, indir)
			count += len(os.listdir(path))
		imgdatas = np.ndarray((count, self.out_rows, self.out_cols, 1), dtype=np.uint8)
		imglabels = np.ndarray((count, self.out_rows, self.out_cols, 1), dtype=np.uint8)
		for indir in os.listdir(self.aug_merge_path):
			trainpath = os.path.join(self.aug_train_path, indir)
			labelpath = os.path.join(self.aug_label_path, indir)
			print(trainpath, labelpath)
			imgs = glob.glob(trainpath + '/*' + '.tif')
			for imgname in imgs:
				trainmidname = imgname[imgname.rindex('/') + 1:]
				labelimgname = imgname[imgname.rindex('/') + 1:imgname.rindex('_')] + '_label.tif'
				print(trainmidname, labelimgname)
				img = load_img(trainpath + '/' + trainmidname, grayscale=True)
				label = load_img(labelpath + '/' + labelimgname, grayscale=True)
				img = img_to_array(img)
				label = img_to_array(label)
				imgdatas[i] = img
				imglabels[i] = label
				if i % 100 == 0:
					print('Done: {0}/{1} images'.format(i, len(imgs)))
				i += 1
				print(i)
		print('loading done', imgdatas.shape)
		np.save(self.npy_path + '/imgs_train.npy', imgdatas)            # 将30张训练集和30张label生成npy数据
		np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)
		print('Saving to .npy files done.')

	def create_small_train_data(self):
		# 将增强之后的训练集生成npy
		print('-' * 30)
		print('creating samll train image')
		print('-' * 30)
		imgs = glob.glob('../data_set/aug_train/0/*' + '.tif')
		count = len(imgs)
		imgdatas = np.ndarray((count, self.out_rows, self.out_cols, 1), dtype=np.uint8)
		imglabels = np.ndarray((count, self.out_rows, self.out_cols, 1), dtype=np.uint8)
		trainpath = '../data_set/aug_train/0'
		labelpath = '../data_set/aug_label/0'
		i = 0
		for imgname in imgs:
			trainmidname = imgname[imgname.rindex('/') + 1:]
			labelimgname = imgname[imgname.rindex('/') + 1:imgname.rindex('_')] + '_label.tif'
			print(trainmidname, labelimgname)
			img = load_img(trainpath + '/' + trainmidname, grayscale=True)
			label = load_img(labelpath + '/' + labelimgname, grayscale=True)
			img = img_to_array(img)
			label = img_to_array(label)
			imgdatas[i] = img
			imglabels[i] = label
			i += 1
			print(i)
		print('loading done', imgdatas.shape)
		np.save(self.npy_path + '/imgs_small_train.npy', imgdatas)  # 将30张训练集和30张label生成npy数据
		np.save(self.npy_path + '/imgs_mask_small_train.npy', imglabels)
		print('Saving to .npy files done.')

	def create_test_data(self):
		# 测试集生成npy
		i = 0
		print('-' * 30)
		print('Creating test images...')
		print('-' * 30)
		imgs = glob.glob(self.test_path + "/*." + self.img_type)           # ../data_set/train
		print(len(imgs))
		imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)
		for imgname in imgs:
			midname = imgname[imgname.rindex("/") + 1:]   # 图像的名字
			img = load_img(self.test_path + "/" + midname, grayscale=True)   # 转换为灰度图
			img = img_to_array(img)
			imgdatas[i] = img
			if i % 100 == 0:
				print('Done: {0}/{1} images'.format(i, len(imgs)))
			i += 1
		print('loading done', imgdatas.shape)
		np.save(self.npy_path + '/imgs_test.npy', imgdatas)            # 将30张训练集和30张label生成npy数据
		# np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)
		print('Saving to .npy files done.')

	def create_my_data(self):
		paths = glob.glob('../data_set/my_set/*JPEG')
		my_set_image = np.ndarray((len(paths), self.out_rows, self.out_cols, 1), dtype=np.uint8)
		my_set_label = np.ndarray((len(paths), self.out_rows, self.out_cols, 1), dtype=np.uint8)
		for index, path in enumerate(paths):
			image = cv2.imread(path)
			image = cv2.resize(src=image, dsize=(self.out_rows, self.out_cols))
			my_set_image[index] = np.reshape(a=image[:, :, 0], newshape=(self.out_rows, self.out_cols, 1))
			my_set_label[index] = np.reshape(a=image[:, :, 2], newshape=(self.out_rows, self.out_cols, 1))
		cv2.imshow('r', my_set_label[0])
		cv2.waitKey(0)
		np.save('../data_set/npydata/my_set_image.npy', my_set_image)
		np.save('../data_set/npydata/my_set_label.npy', my_set_label)
		print('Saving to .npy files done.')

	def load_my_train_data(self):
		print('-' * 30)
		print('load train images...')
		print('-' * 30)
		imgs_train = np.load('../data_set/npydata/my_set_image.npy')
		imgs_mask_train = np.load('../data_set/npydata/my_set_label.npy')
		imgs_train = imgs_train.astype('float32')
		# imgs_mask_train = imgs_mask_train.astype('float32')
		# imgs_train /= 255
		# mean = imgs_train.mean(axis=0)
		# imgs_train -= mean
		# imgs_mask_train /= 255
		imgs_mask_train[imgs_mask_train > 128] = 1
		imgs_mask_train[imgs_mask_train <= 128] = 0
		return imgs_train, imgs_mask_train

	def load_small_train_data(self):
		# 读入训练数据包括label_mask(npy格式), 归一化(只减去了均值)
		print('-' * 30)
		print('load train images...')
		print('-' * 30)
		imgs_train = np.load(self.npy_path + "/imgs_small_train.npy")
		imgs_mask_train = np.load(self.npy_path + "/imgs_mask_small_train.npy")
		imgs_train = imgs_train.astype('float32')
		imgs_mask_train = imgs_mask_train.astype('float32')
		# imgs_train /= 255
		# mean = imgs_train.mean(axis=0)
		# imgs_train -= mean
		# imgs_mask_train /= 255
		imgs_mask_train[imgs_mask_train > 128] = 1
		imgs_mask_train[imgs_mask_train <= 128] = 0
		return imgs_train, imgs_mask_train

	def load_train_data(self):
		# 读入训练数据包括label_mask(npy格式), 归一化(只减去了均值)
		print('-' * 30)
		print('load train images...')
		print('-' * 30)
		imgs_train = np.load(self.npy_path + "/imgs_train.npy")
		imgs_mask_train = np.load(self.npy_path + "/imgs_mask_train.npy")
		imgs_train = imgs_train.astype('float32')
		imgs_mask_train = imgs_mask_train.astype('float32')
		imgs_train /= 255
		mean = imgs_train.mean(axis=0)
		imgs_train -= mean
		imgs_mask_train /= 255
		imgs_mask_train[imgs_mask_train > 0.5] = 1
		imgs_mask_train[imgs_mask_train <= 0.5] = 0
		return imgs_train, imgs_mask_train

	def load_test_data(self):
		print('-' * 30)
		print('load test images...')
		print('-' * 30)
		imgs_test = np.load(self.npy_path + "/imgs_train.npy")
		imgs_test = imgs_test.astype('float32')
		imgs_test /= 255
		mean = imgs_test.mean(axis=0)
		imgs_test -= mean
		return imgs_test

	def show_test_result(self):
		imgs_test = np.load('../data_set/imgs_mask_test.npy')
		# shape[batch_size, width, height, 2] 需要argmax, 提取最大值下标
		img1 = np.argmax(a=imgs_test[1], axis=-1).astype('uint8')
		cv2.imshow('r', img1 * 100)
		cv2.waitKey(0)

	def write_img_to_tfrecords(self):
		import tensorflow as tf
		from random import shuffle
		import cv2
		train_set_writer = tf.python_io.TFRecordWriter(os.path.join('../data_set', TRAIN_SET_NAME))  # 要生成的文件
		validation_set_writer = tf.python_io.TFRecordWriter(os.path.join('../data_set', VALIDATION_SET_NAME))
		test_set_writer = tf.python_io.TFRecordWriter(os.path.join('../data_set', TEST_SET_NAME))  # 要生成的文件

		# add all files
		merged_images_path = []
		for indir in os.listdir(self.aug_merge_path):
			# trainpath = os.path.join(self.aug_train_path, indir)
			# labelpath = os.path.join(self.aug_label_path, indir)
			# print(trainpath, labelpath)
			imgs = glob.glob(os.path.join(self.aug_merge_path, indir) + '/*' + '.tif')
			# print(imgs[0])
			merged_images_path.extend(glob.glob(os.path.join(self.aug_merge_path, indir) + '/*' + '.tif'))
		print('Total %d files' % len(merged_images_path))
		# merged_images_path = merged_images_path[:merged_images_path.rindex('.tif'):1]
		shuffle(merged_images_path)

		# train set
		for index, merged_image_path in enumerate(merged_images_path[:TRAIN_SET_SIZE]):
			train_image_path = merged_image_path.replace('merge', 'train')
			train_image_path = train_image_path[:train_image_path.rindex('.tif')] + '_train.tif'
			label_image_path = merged_image_path.replace('merge', 'label')
			label_image_path = label_image_path[:label_image_path.rindex('.tif')] + '_label.tif'
			# print(train_image_path, label_image_path)
			image = cv2.imread(train_image_path, flags=0)
			label = cv2.imread(label_image_path, flags=0)
			# print(image.shape)
			label[label <= 100] = 0
			label[label > 100] = 1
			example = tf.train.Example(features=tf.train.Features(feature={
				'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tobytes()])),
				'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()]))
			}))  # example对象对label和image数据进行封装
			train_set_writer.write(example.SerializeToString())  # 序列化为字符串
			if index % 100 == 0:
				print('Done train_set writing %.2f%%' % (index / TRAIN_SET_SIZE * 100))
		train_set_writer.close()
		print("Done whole train_set writing")

		# validation set
		for index, merged_image_path in enumerate(merged_images_path[TRAIN_SET_SIZE:TRAIN_SET_SIZE + VALIDATION_SET_SIZE]):
			validation_image_path = merged_image_path.replace('merge', 'train')
			validation_image_path = validation_image_path[:validation_image_path.rindex('.tif')] + '_train.tif'
			label_image_path = merged_image_path.replace('merge', 'label')
			label_image_path = label_image_path[:label_image_path.rindex('.tif')] + '_label.tif'
			# print(train_image_path, label_image_path)
			image = cv2.imread(validation_image_path, flags=0)
			label = cv2.imread(label_image_path, flags=0)
			# print(image.shape)
			label[label <= 100] = 0
			label[label > 100] = 1
			example = tf.train.Example(features=tf.train.Features(feature={
				'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tobytes()])),
				'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()]))
			}))  # example对象对label和image数据进行封装
			validation_set_writer.write(example.SerializeToString())  # 序列化为字符串
			if index % 10 == 0:
				print('Done validation_set writing %.2f%%' % (index / VALIDATION_SET_SIZE * 100))
		validation_set_writer.close()
		print("Done whole validation_set writing")

		# test set
		for index in range(PREDICT_SET_SIZE):
			origin_image_path = ORIGIN_IMAGE_DIRECTORY
			origin_label_path = ORIGIN_LABEL_DIRECTORY
			predict_image = cv2.imread(os.path.join(origin_image_path, '%d.tif' % index), flags=0)
			predict_label = cv2.imread(os.path.join(origin_label_path, '%d.tif' % index), flags=0)
			# predict_image = cv2.resize(src=predict_image, dsize=(INPUT_IMG_WIDE, INPUT_IMG_HEIGHT))
			# predict_image = np.asarray(a=predict_image, dtype=np.uint8)
			# predict_image = cv2.resize(src=predict_image, dsize=(INPUT_IMG_WIDE, INPUT_IMG_HEIGHT))
			# predict_label = np.asarray(a=predict_label, dtype=np.uint8)
			# predict_label = cv2.resize(src=predict_label, dsize=(OUTPUT_IMG_WIDE, OUTPUT_IMG_HEIGHT))
			predict_label[predict_label <= 100] = 0
			predict_label[predict_label > 100] = 1
			# predict_image = io.imread(file_path)
			# predict_image = transform.resize(predict_image, (INPUT_IMG_WIDE, INPUT_IMG_HEIGHT, INPUT_IMG_CHANNEL))
			# sample_image = predict_image[:, :, 0]
			# label_image = predict_image[:, :, 2]
			# label_image[label_image < 100] = 0
			# label_image[label_image > 100] = 10
			example = tf.train.Example(features=tf.train.Features(feature={
				'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[predict_label.tobytes()])),
				'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[predict_image.tobytes()]))
			}))  # example对象对label和image数据进行封装
			test_set_writer.write(example.SerializeToString())  # 序列化为字符串
			if index % 10 == 0:
				print('Done test_set writing %.2f%%' % (index / TEST_SET_SIZE * 100))
		test_set_writer.close()
		print("Done whole test_set writing")

		# predict set
		# for index in range(PREDICT_SET_SIZE):
		# 	origin_image_path = ORIGIN_IMAGE_DIRECTORY
		# 	origin_label_path = ORIGIN_LABEL_DIRECTORY
		# 	predict_image = cv2.imread(os.path.join(origin_image_path, '%d.tif' % index), flags=0)
		# 	predict_label = cv2.imread(os.path.join(origin_label_path, '%d.tif' % index), flags=0)
		# 	# predict_image = cv2.resize(src=predict_image, dsize=(INPUT_IMG_WIDE, INPUT_IMG_HEIGHT))
		# 	# predict_image = np.asarray(a=predict_image, dtype=np.uint8)
		# 	# predict_image = cv2.resize(src=predict_image, dsize=(INPUT_IMG_WIDE, INPUT_IMG_HEIGHT))
		# 	# predict_label = np.asarray(a=predict_label, dtype=np.uint8)
		# 	# predict_label = cv2.resize(src=predict_label, dsize=(OUTPUT_IMG_WIDE, OUTPUT_IMG_HEIGHT))
		# 	predict_label[predict_label <= 100] = 0
		# 	predict_label[predict_label > 100] = 1
		# 	# predict_image = io.imread(file_path)
		# 	# predict_image = transform.resize(predict_image, (INPUT_IMG_WIDE, INPUT_IMG_HEIGHT, INPUT_IMG_CHANNEL))
		# 	# sample_image = predict_image[:, :, 0]
		# 	# label_image = predict_image[:, :, 2]
		# 	# label_image[label_image < 100] = 0
		# 	# label_image[label_image > 100] = 10
		# 	example = tf.train.Example(features=tf.train.Features(feature={
		# 		'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[predict_label.tobytes()])),
		# 		'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[predict_image.tobytes()]))
		# 	}))  # example对象对label和image数据进行封装
		# 	# predict_set_writer.write(example.SerializeToString())  # 序列化为字符串
		# 	if index % 10 == 0:
		# 		print('Done predict_set writing %.2f%%' % (index / PREDICT_SET_SIZE * 100))
		# # predict_set_writer.close()
		# print("Done predict_set writing")


if __name__ == '__main__':
	# aug = Augmentation()
	# aug.augmentation()
	# aug.split_merge()
	mydata = DataProcess(512, 512)
	# print(cv2.imread('../data_set/aug_label/13/13_0_285_label.tif').shape)
	mydata.write_img_to_tfrecords()
	# mydata.create_my_data()
	# mydata.create_small_train_data()
	# mydata.create_train_data()
	# mydata.create_test_data()
	# imgs_train, imgs_mask_train = mydata.load_train_data()
	# print(imgs_train.shape, imgs_mask_train.shape)
	# mydata.show_test_result()
