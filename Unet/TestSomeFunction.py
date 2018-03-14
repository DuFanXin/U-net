# -*- coding:utf-8 -*-
'''  
#====#====#====#====
# Project Name:     U-net 
# File Name:        TestSomeFunction 
# Date:             2/9/18 4:08 PM 
# Using IDE:        PyCharm Community Edition  
# From HomePage:    https://github.com/DuFanXin/U-net
# Author:           DuFanXin 
# BlogPage:         http://blog.csdn.net/qq_30239975  
# E-mail:           18672969179@163.com
# Copyright (c) 2018, All Rights Reserved.
#====#====#====#==== 
'''
import cv2
import numpy as np
import glob
from keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf


def show_test_result():
	imgs_test = np.load('../data_set/imgs_mask_test.npy')
	# imgs_test = imgs_test.astype('uint8')
	print(imgs_test[0][0].shape)
	# img1 = None
	img1 = np.argmax(a=imgs_test[1], axis=-1).astype('uint8')
	# print(img1.shape)
	cv2.imshow('r', img1 * 100)
	cv2.waitKey(0)


def load_small_train_data():
	# 读入训练数据包括label_mask(npy格式), 归一化(只减去了均值)
	print('-' * 30)
	print('load train images...')
	print('-' * 30)
	imgs_train = np.load('../data_set/npydata/imgs_small_train.npy')
	imgs_mask_train = np.load('../data_set/npydata/imgs_mask_small_train.npy')
	imgs_train = imgs_train.astype('uint8')
	imgs_mask_train = imgs_mask_train.astype('uint8')
	# imgs_train /= 255
	# mean = imgs_train.mean(axis=0)
	# imgs_train -= mean
	# imgs_mask_train /= 255
	# imgs_mask_train[imgs_mask_train > 128] = 1
	# imgs_mask_train[imgs_mask_train <= 128] = 0
	cv2.imshow('r', imgs_mask_train[0] * 100)
	cv2.waitKey(0)


def check_npy():
	imgs_test = np.load('../data_set/npydata/my_set_image.npy')
	# imgs_test = imgs_test.astype('uint8')
	print(imgs_test[0].shape)
	# img1 = None
	img1 = np.argmax(a=imgs_test[1], axis=-1).astype('uint8')
	# print(img1.shape)
	cv2.imshow('r', img1 * 100)
	cv2.waitKey(0)


def create_small_train_data():
	out_rows = 512
	out_cols = 512
	# 将增强之后的训练集生成npy
	print('-' * 30)
	print('creating samll train image')
	print('-' * 30)
	imgs = glob.glob('../data_set/aug_train/0/*' + '.tif')
	count = len(imgs)
	imgdatas = np.ndarray((count, out_rows, out_cols, 1), dtype=np.uint8)
	imglabels = np.ndarray((count, out_rows, out_cols, 1), dtype=np.uint8)
	train_path = '../data_set/aug_train/0'
	label_path = '../data_set/aug_label/0'
	i = 0
	for imgname in imgs:
		trainmidname = imgname[imgname.rindex('/') + 1:]
		labelimgname = imgname[imgname.rindex('/') + 1:imgname.rindex('_')] + '_label.tif'
		print(imgname, trainmidname, labelimgname)
		img = load_img(train_path + '/' + trainmidname, grayscale=True)
		label = load_img(label_path + '/' + labelimgname, grayscale=True)
		img = img_to_array(img)
		label = img_to_array(label)
		imgdatas[i] = img
		imglabels[i] = label
		i += 1
		print(i)
	print('loading done', imgdatas.shape)
	np.save('../data_set/npydata/imgs_small_train.npy', imgdatas)  # 将30张训练集和30张label生成npy数据
	np.save('../data_set/npydata/imgs_mask_small_train.npy', imglabels)
	print('Saving to .npy files done.')


def batch_norm(x, is_training, eps=1e-05, decay=0.9, affine=True, name='BatchNorm2d'):
	from tensorflow.python.training.moving_averages import assign_moving_average

	with tf.variable_scope(name):
		print(get_scope())
		print(tf.get_variable_scope())
		params_shape = x.shape[-1:]
		moving_mean = tf.get_variable(name='mean', shape=params_shape, initializer=tf.zeros_initializer, trainable=False)
		moving_var = tf.get_variable(name='variance', shape=params_shape, initializer=tf.ones_initializer, trainable=False)

		def mean_var_with_update():
			mean, variance = tf.nn.moments(x, list(range(len(x.shape) - 1)), name='moments')
			with tf.control_dependencies([
				assign_moving_average(moving_mean, mean, decay),
				assign_moving_average(moving_var, variance, decay)
			]):
				return tf.identity(mean), tf.identity(variance)
		mean, variance = tf.cond(is_training, mean_var_with_update, lambda: (moving_mean, moving_var))
		if affine:
			beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer)
			gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer)
			normed = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
		else:
			normed = tf.nn.batch_normalization(x, mean, variance, None, None, eps)
		return normed


def get_scope():
	means = tf.get_variable(name='means', shape=[1], initializer=tf.zeros_initializer, trainable=False)
	return means.name


def tif_merge():
	import cv2
	import numpy as np
	img1 = cv2.imread(filename='../data_set/train/0.tif')
	img2 = cv2.imread(filename='../data_set/train/1.tif')
	img_mer = np.concatenate((img1, img2), axis=-1)
	np.save(file='../data_set/train/merged.tif', arr=img_mer)
	# cv2.imwrite(filename='../data_set/train/merged.tif', img=img_mer)
	print(img_mer.shape)

if __name__ == '__main__':
	tif_merge()
	# create_small_train_data()
	# load_small_train_data()
	# show_test_result()
	# srce = tf.Variable(tf.random_normal(shape=[1, 5, 5, 3]))
	# is_train = tf.placeholder(dtype=tf.bool, shape=[])
	# with tf.variable_scope('0'):
	# 	srce = tf.get_variable(name='mean', shape=[1, 5, 5, 3], initializer=tf.zeros_initializer, trainable=False)
	# norm = batch_norm(x=srce, is_training=is_train)
	# sr = dds(x=srce, is_training=True)
	# with tf.Session() as sess:
	# 	sess.run(tf.global_variables_initializer())
	# 	sess.run(tf.local_variables_initializer())
	# 	print(get_scope())
	# 	print(sess.run(norm, feed_dict={is_train: False}).shape)
		# print(tf.get_variable_scope())
		# print(srce.shape[:-1])
		# print(list(range(len(srce.shape) - 1)))
		# axise = list(range(len(srce.shape) - 1))
		# print(axise)
		# batch_mean, batch_var = sess.run(tf.nn.moments(x=srce, axes=-1, name='moments'))
		# print(batch_mean)
		# sess.run(norm)
