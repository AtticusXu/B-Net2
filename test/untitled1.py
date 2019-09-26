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

import matplotlib.pyplot as plt
import json
tf.reset_default_graph() 
from gaussianfun import gaussianfun
from gen_dft_data import gen_ede_Ell_sine_data,gen_ede_Ell_data
from Bi_ButterflyLayer import ButterflyLayer
from middle_layer import MiddleLayer

json_file = open('paras.json')
paras = json.load(json_file)
input_size = paras['inputSize']
N = input_size//2
in_siz = input_size*2
en_mid_siz = 128
de_mid_siz = min(in_siz,en_mid_siz)
out_siz = input_size
in_range = np.float32([0,1])
en_mid_range = np.float32([0,en_mid_siz/in_siz])
de_mid_range = np.float32([-0.25,0.25])
out_range = np.float32([0,1])
sig = paras['sigma']
freqidx = range(en_mid_siz//2)
#freqmag = np.fft.ifftshift(gaussianfun(np.arange(-N//2,N//2),
#                                       [1],[0.1]))
freqmag = np.ones(N)
freqmag[1] = 1
freqmag[N//2] = 0
N_0 = 2**10
a = np.ones(N_0+1)
m = N_0//4
for j in range(N_0+1):
    if (j-m//2)%(2*m) < m:
        a[j] = 10
batch_siz = paras['batchSize'] # Batch size during traning
channel_siz = paras['channelSize']
adam_learning_rate = paras['ADAMparas']['learningRate']
adam_learning_rate_decay = paras['ADAMparas']['learningRatedecay']
adam_beta1 = paras['ADAMparas']['beta1']
adam_beta2 = paras['ADAMparas']['beta2']

max_iter = paras['maxIter'] # Maximum num of iterations
test_batch_siz = paras['Ntest']
report_freq = paras['reportFreq'] # Frequency of reporting
record_freq = paras['recordFreq'] # Frequency of recording
en_nlvl = 6
de_nlvl = 5

#=========================================================
#----- Variable Preparation
sess = tf.Session()
trainInData = tf.placeholder(tf.float32, shape=(batch_siz,in_siz,1),
        name="trainInData")
trainMidData = tf.placeholder(tf.float32, shape=(batch_siz,en_mid_siz,1),
        name="trainMidData")
trainMidNorm = tf.placeholder(tf.float32, shape=(batch_siz),
        name="trainMidNorm")
trainOutData = tf.placeholder(tf.float32, shape=(batch_siz,out_siz//2,1),
        name="trainOutData")
trainOutNorm = tf.placeholder(tf.float32, shape=(batch_siz),
        name="trainOutNorm")
global_steps=tf.Variable(0, trainable=False)

#=========================================================
#----- Training Preparation
en_butterfly_net = ButterflyLayer(2*N, in_siz, en_mid_siz, False,
        channel_siz, en_nlvl, -1, True,
        in_range, en_mid_range)
middle_net = MiddleLayer(in_siz, en_mid_siz, a[::(2**4)],
                                    sine = True, prefixed = 2, std = 0.5)
de_butterfly_net = ButterflyLayer(N, de_mid_siz, out_siz, False,
        channel_siz, de_nlvl, 1, True,
        en_mid_range, out_range)

y_train_en_mid = en_butterfly_net(trainInData)
y_train_de_mid = middle_net(y_train_en_mid)
y_train_output = de_butterfly_net(y_train_de_mid)/N
y_train_output_r = y_train_output[:,::2,:]

learning_rate = tf.train.exponential_decay(adam_learning_rate,
                                           global_steps,100,
                                           adam_learning_rate_decay)
optimizer_adam = tf.train.AdamOptimizer(learning_rate,
        adam_beta1, adam_beta2)

MSE_loss_train_u = tf.reduce_mean(
        tf.squared_difference(trainOutData, y_train_output_r))
MSE_loss_train_y = tf.reduce_mean(
        tf.squared_difference(trainMidData[:,0::2], -y_train_en_mid[:,1::2]))

L2_loss_train_u = tf.reduce_mean(tf.divide(tf.sqrt(tf.reduce_sum(tf.squeeze(
            tf.squared_difference(trainOutData, y_train_output_r)),1)),trainOutNorm))
L2_loss_train_y = tf.reduce_mean(tf.divide(tf.sqrt(tf.reduce_sum(tf.squeeze(
            tf.squared_difference(trainMidData[:,0::2], -y_train_en_mid[:,1::2])),1)),trainMidNorm))

train_step = optimizer_adam.minimize(MSE_loss_train_u,global_step=global_steps)

# Initialize Variables
init = tf.global_variables_initializer()
print("Total Num Paras:  %6d" % ( np.sum( [np.prod(v.get_shape().as_list())
    for v in tf.trainable_variables()]) ))
    
sess.run(init)
f_train = np.load('tftmp/fft_4000_f_train_c.npy')
y_train = np.load('tftmp/fft_4000_y_train_c.npy')
u_train = np.load('tftmp/fft_4000_u_train_c.npy')
f_norm = np.load('tftmp/fft_4000_f_norm_c.npy')
y_norm = np.load('tftmp/fft_4000_y_norm_c.npy')
u_norm = np.load('tftmp/fft_4000_u_norm_c.npy')
for it in range(max_iter):
    start = (it*batch_siz)%4000
    end = ((it+1)*batch_siz-1)%4000+1
    f_train_it = f_train[start:end]
    y_train_it = y_train[start:end]
    u_train_it = u_train[start:end]
    f_norm_it = f_norm[start:end]
    y_norm_it = y_norm[start:end]
    u_norm_it = u_norm[start:end]
    train_dict = {trainInData: f_train_it, trainMidData: y_train_it,
                  trainMidNorm: y_norm_it,
                  trainOutData:u_train_it, trainOutNorm: u_norm_it}
    if it % 1 == 0:
            [y,y_L2_loss,y_loss,u_L2_loss,u_loss] = sess.run(
                    [y_train_en_mid,L2_loss_train_y,MSE_loss_train_y,L2_loss_train_u,MSE_loss_train_u],
                    feed_dict=train_dict)
            #print(y[0,:,0])
            print(it)
            print('y_mse:%10e,y_l2:%10e'%(y_loss, y_L2_loss))
            print('u_mse:%10e,u_l2:%10e'%(u_loss, u_L2_loss))

    sess.run(train_step, feed_dict=train_dict)
#print(y_en[0])
#print(y_de[0])
#print(y_out[0])
#print(y_loss)