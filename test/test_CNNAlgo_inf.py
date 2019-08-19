import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
sys.path.insert(0,"../src")
sys.path.insert(0,"../src/data_gen")
from pathlib import Path
import math
import numpy as np
import scipy.io as spio
import tensorflow as tf
import json
tf.reset_default_graph() 
from gaussianfun import gaussianfun
from gen_dft_data import gen_gaussian_data
from CNNLayer import CNNLayer

json_file = open('paras.json')
paras = json.load(json_file)

N = paras['inputSize']
Ntrain = 1000
in_siz = paras['inputSize']
out_siz = paras['outputSize']
in_range = np.float32([0,1])
out_range = np.float32([0,out_siz//2])

freqidx = range(out_siz//2)
freqmag = np.fft.ifftshift(gaussianfun(np.arange(-N//2,N//2),[0,0],[2,2]))
freqmag[N//2] = 0

#=========================================================
#----- Parameters Setup

prefixed = paras['prefixed']

channel_siz = paras['channelSize'] # Num of interp pts on each dim

#x_train,y_train = gen_gaussian_data(freqmag,freqidx,Ntrain)
x_train = np.load('tftmp/x_train.npy')
y_train = np.load('tftmp/y_train.npy')
#----- Self-adjusted Parameters of BNet
# Num of levels of the BF struct, must be a even num
nlvl = 4
# Filter size for the input and output


print("======== Parameters =========")
print("Channel Size:     %6d" % (channel_siz))
print("Num Levels:       %6d" % (nlvl))
print("Prefix Coef:      %6r" % (prefixed))
print("In Range:        (%6.2f, %6.2f)" % (in_range[0], in_range[1]))
print("Out Range:       (%6.2f, %6.2f)" % (out_range[0], out_range[1]))

#=========================================================
#----- Variable Preparation
sess = tf.Session()

trainInData = tf.placeholder(tf.float32, shape=(Ntrain,in_siz,1),
        name="trainInData")
trainOutData = tf.placeholder(tf.float32, shape=(Ntrain,out_siz,1),
        name="trainOutData")

#=========================================================
#----- Training Preparation
cnn_net = CNNLayer(in_siz, out_siz, channel_siz, nlvl, prefixed)

y_train_output = cnn_net(tf.convert_to_tensor(x_train))

loss_train = tf.reduce_mean(
        tf.squared_difference(y_train, y_train_output))

# Initialize Variables
init = tf.global_variables_initializer()

print("Total Num Paras:  %6d" % ( np.sum( [np.prod(v.get_shape().as_list())
    for v in tf.trainable_variables()]) ))

#=========================================================
#----- Step by Step Training
sess.run(init)
train_dict = {trainInData: x_train, trainOutData: y_train}
train_loss = sess.run(loss_train,feed_dict=train_dict)
print("Train Loss: %10e." % (train_loss))


sess.close()