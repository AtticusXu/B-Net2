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

N = 64
in_siz = 256
en_mid_siz = 128
de_mid_siz = min(in_siz,en_mid_siz*2)
out_siz = 64
in_range = np.float32([0,1])
en_mid_range = np.float32([0,en_mid_siz/in_siz])
de_mid_range = np.float32([-0.25,0.25])
out_range = np.float32([0,1])
sig = paras['sigma']
freqidx = range(en_mid_siz//2)
#freqmag = np.fft.ifftshift(gaussianfun(np.arange(-N//2,N//2),
#                                       [1],[0.1]))
freqmag = np.zeros(N)
freqmag[1] = 1
freqmag[N//2] = 0
m = np.floor(np.sqrt(N))//2
a = np.ones(N+1)
for i in range(N+1):
    if i%(2*m) < m:
        #a[i] = 10
        a[i] = 1
print(a)
batch_siz = 2
f_train,y_train,u_train,f_norm,y_norm,u_norm = gen_ede_Ell_data(
            batch_siz,freqidx,freqmag,a)

sess = tf.Session()
trainInData = tf.placeholder(tf.float32, shape=(batch_siz,in_siz,1),
        name="trainInData")
trainMidData = tf.placeholder(tf.float32, shape=(batch_siz,en_mid_siz,1),
        name="trainMidData")

en_butterfly_net = ButterflyLayer(2*N, in_siz, en_mid_siz, False,
        32, 5, -1, True,
        in_range, en_mid_range)
y_train_output = en_butterfly_net(trainInData)

MSE_loss_train_y = tf.reduce_mean(
        tf.squared_difference(trainMidData, y_train_output))

# Initialize Variables
init = tf.global_variables_initializer()

sess.run(init)

train_dict = {trainInData: f_train,trainMidData: y_train}

[ytrain,y_loss] = sess.run(
           [y_train_output,MSE_loss_train_y], feed_dict=train_dict)
print(ytrain)
print(y_loss)