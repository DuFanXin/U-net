# -*- coding:utf-8 -*-
'''  
#====#====#====#====
# Project Name:     U-net 
# File Name:        unet-Kares
# Date:             2/9/18 3:59 PM 
# Using IDE:        PyCharm Community Edition  
# From HomePage:    https://github.com/DuFanXin/U-net
# Author:           DuFanXin 
# BlogPage:         http://blog.csdn.net/qq_30239975  
# E-mail:           18672969179@163.com
# Copyright (c) 2018, All Rights Reserved.
#====#====#====#==== 
'''
from keras.callbacks import ModelCheckpoint
from keras.layers import merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Softmax
from keras.models import *
from keras.optimizers import *
from keras.utils import to_categorical

from Unet.data_Keras import DataProcess


class myUnet(object):
    def __init__(self, img_rows=512, img_cols=512):
        self.img_rows = img_rows
        self.img_cols = img_cols

    def load_train_data(self):
        mydata = DataProcess(self.img_rows, self.img_cols)
        imgs_train, imgs_mask_train = mydata.load_my_train_data()
        imgs_mask_train = to_categorical(imgs_mask_train, num_classes=2)
        return imgs_train, imgs_mask_train

    def load_test_data(self):
        mydata = DataProcess(self.img_rows, self.img_cols)
        imgs_test = mydata.load_test_data()
        return imgs_test

    def get_unet(self):
        inputs = Input((self.img_rows, self.img_cols, 1))

        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        print(conv1.shape)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        print(conv1.shape)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        print(pool1.shape)
        print('\n')

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        print(conv2.shape)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        print(conv2.shape)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        print(pool2.shape)
        print('\n')

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        print(conv3.shape)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        print(conv3.shape)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        print(pool3.shape)
        print('\n')

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        print(conv4.shape)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        print(conv4.shape)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
        print(pool4.shape)
        print('\n')

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        print(conv5.shape)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        print(conv5.shape)
        drop5 = Dropout(0.5)(conv5)
        print('\n')

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
        print(up6.shape)
        print(drop4.shape)
        merge6 = merge([drop4, up6], mode='concat', concat_axis=3)
        print('merge: ')
        print(merge6.shape)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = merge([conv3, up7], mode='concat', concat_axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = merge([conv2, up8], mode='concat', concat_axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = merge([conv1, up9], mode='concat', concat_axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        # conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
        conv10 = Softmax()(conv9)
        print(conv10.shape)

        model = Model(input=inputs, output=conv10)

        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        print('model compile')
        return model

    def train(self):
        print("loading data")
        imgs_train, imgs_mask_train = self.load_train_data()
        print("loading data done")
        model = self.get_unet()
        print("got unet")

        # 保存的是模型和权重,
        model_checkpoint = ModelCheckpoint('../data_set/unet.hdf5', monitor='loss', verbose=1, save_best_only=True)
        print('Fitting model...')
        model.fit(x=imgs_train, y=imgs_mask_train, validation_split=0.2, batch_size=1, epochs=1, verbose=1, shuffle=True,
                  callbacks=[model_checkpoint])

    def test(self):
        print("loading data")
        imgs_test = self.load_test_data()
        print("loading data done")
        model = self.get_unet()
        print("got unet")
        model.load_weights('../data_set/unet.hdf5')
        print('predict test data')
        # imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        # np.save('../data_set/imgs_mask_test.npy', imgs_mask_test)

if __name__ == '__main__':
    unet = myUnet()
    unet.get_unet()
    # unet.train()
    # unet.test()
