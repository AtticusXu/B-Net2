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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
tf.reset_default_graph() 
from gaussianfun import gaussianfun
from gen_dft_data import gen_gaussian_data
from Bi_ButterflyLayer import ButterflyLayer

N = 32
in_siz = 64
out_siz = 32
in_range = np.float32([0,1])
out_range = np.float32([-1/4,1/4])
sig = -1

en_mid_siz = out_siz
freqidx = range(out_siz//2)
freqmag = np.fft.ifftshift(gaussianfun(np.arange(-N//2,N//2),
                                       [1],[0.1]))
freqmag[N//2] = 0

#=========================================================
#----- Parameters Setup

prefixed = True

channel_siz = 12 # Num of interp pts on each dim

batch_siz = 1

#----- Self-adjusted Parameters of BNet
# Num of levels of the BF struct, must be a even num
nlvl = 5

print("======== Parameters =========")
print("Batch Size:       %6d" % (batch_siz))
print("Channel Size:     %6d" % (channel_siz))
print("Num Levels:       %6d" % (nlvl))
print("Prefix Coef:      %6r" % (prefixed))
print("In Range:        (%6.2f, %6.2f)" % (in_range[0], in_range[1]))
print("Out Range:       (%6.2f, %6.2f)" % (out_range[0], out_range[1]))

#=========================================================
#----- Variable Preparation
sess = tf.Session()

trainInData = tf.placeholder(tf.float32, shape=(batch_siz,in_siz,1),
        name="trainInData")
trainOutData = tf.placeholder(tf.float32, shape=(batch_siz,out_siz,1),
        name="trainOutData")

global_steps = tf.Variable(0, trainable=False)
#=========================================================
#----- Training Preparation
butterfly_net = ButterflyLayer(N, in_siz, out_siz, False,
        channel_siz, nlvl, -1, prefixed,
        in_range, out_range)

y_train_output,nl = butterfly_net(trainInData)

MSE_loss_train_y = tf.reduce_mean(
        tf.squared_difference(trainOutData, y_train_output))

# Initialize Variables
init = tf.global_variables_initializer()

print("Total Num Paras:  %6d" % ( np.sum( [np.prod(v.get_shape().as_list())
    for v in tf.trainable_variables()]) ))

#=========================================================
#----- Step by Step Training

sess.run(init)

x_train,y_train = gen_gaussian_data(freqmag,freqidx,batch_siz)
train_dict = {trainInData: x_train, trainOutData: y_train}
mse_y = sess.run(MSE_loss_train_y, feed_dict=train_dict)
print("Loss: %10e." % (mse_y))

[y,nl] = sess.run([y_train_output[0,:,0],nl], feed_dict=train_dict)
print(nl)
fig = plt.figure(0,figsize=(10,8))
plt.plot(np.linspace(0,out_siz,out_siz),y,'r*-')
plt.plot(np.linspace(0,out_siz,out_siz),y_train[0,:,0],'b')
#plt.plot(np.linspace(0,in_siz,in_siz),x_train[0,:,0],'b')
plt.savefig("test.png")