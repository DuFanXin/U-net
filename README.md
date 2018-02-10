# U-net
The implement of U-net with keras(unet-Keras.py) and TensorFLow(unet-TF.psy).
# Paper
https://arxiv.org/abs/1505.04597
# Note
I change a little bit of the Architecture. I don`t change the image size each step of conv. 
Thus I just copy the result of corresponding previous layers without cropping, and then concatenate.
# Data
I use the data set in ***ISBI Challenge: Segmentation of neuronal structures in EM stacks***(http://brainiac2.mit.edu/isbi_challenge/)
THe data set have been downloaded to folder data_set/train and data_set/label
