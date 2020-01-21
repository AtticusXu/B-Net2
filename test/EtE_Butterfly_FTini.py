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
from ODE_matrix import DirSineElliptic, Dir_2_Elliptic
import matplotlib.pyplot as plt
import json
tf.reset_default_graph() 
from gaussianfun import gaussianfun
from gen_dft_data import gen_ede_Ell_sine_data,gen_ede_Ell_data
from Bi_ButterflyLayer import ButterflyLayer
from Bi_CNNLayer import CNNLayer
from middle_layer import MiddleLayer
de = True
json_file = open('paras.json')
paras = json.load(json_file)
butterfly = paras['butterfly']
input_size = paras['inputSize']
prefixed = paras['prefixed']
N = input_size
N = input_size//2
in_siz = input_size*2
en_mid_siz = 16
de_mid_siz = 32
if de:
    out_siz = input_size * 2
else:
    out_siz = input_size

in_range = np.float32([0,1])
en_mid_range = np.float32([0,en_mid_siz/in_siz])
de_mid_range = np.float32([-en_mid_siz/in_siz,en_mid_siz/in_siz])
de_mid_range = np.float32([0,de_mid_siz/in_siz])
out_range = np.float32([0,1])
sig = paras['sigma']
freqidx = range(en_mid_siz//2)
freqmag = np.zeros(N)
freqmag = np.fft.ifftshift(gaussianfun(np.arange(-N//2,N//2),
                                       [1],[0.1]))

for i in range(1,8):
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
trainset =  paras['trainset']
en_nlvl = 4
de_nlvl = 4
beta = 10**6
dir_mat = Dir_2_Elliptic(a[::(2**10//N)],N)
dir_mat = np.tile(dir_mat,(batch_siz,1,1))
#=========================================================
#----- Variable Preparation
sess = tf.Session()
trainInData = tf.placeholder(tf.float32, shape=(batch_siz,in_siz,1),
        name="trainInData")
trainInNorm = tf.placeholder(tf.float32, shape=(batch_siz),
        name="trainInNorm")
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

if de:
    en_butterfly_net = ButterflyLayer(N, in_siz, en_mid_siz, False,
        channel_siz, en_nlvl, -1, True,
        in_range, en_mid_range)
else:
    en_butterfly_net = ButterflyLayer(2*N, in_siz, en_mid_siz, False,
        channel_siz, en_nlvl, -1, True,
        in_range, en_mid_range)


middle_net = MiddleLayer(in_siz, en_mid_siz, de_mid_siz, sine = True,
                        a = a[::(2**10//N)], prefixed = 2, std = 0.03)

de_butterfly_net = ButterflyLayer(N, de_mid_siz, out_siz, False,
        channel_siz, de_nlvl, 1, True,
        de_mid_range, out_range)

y_train_en_mid = en_butterfly_net(trainInData)
y_train_de_mid = middle_net(y_train_en_mid)
y_train_output = de_butterfly_net(y_train_de_mid)/N
y_train_output_i = -y_train_output[:,1::2,:]

MSE_loss_train_u = tf.reduce_mean(
        tf.squared_difference(trainOutData, y_train_output_i))
L2_loss_train_u = tf.reduce_mean(tf.divide(tf.sqrt(tf.reduce_sum(tf.squeeze(
            tf.squared_difference(trainOutData, y_train_output_i)),1)),trainOutNorm))
# Initialize Variables
init = tf.global_variables_initializer()

print("Total Num Paras:  %6d" % ( np.sum( [np.prod(v.get_shape().as_list())
    for v in tf.trainable_variables()]) ))


sess.run(init)


for n in tf.global_variables():
    np.save('tftmp/'+n.name.split(':')[0], n.eval(session=sess))
    print(n.name.split(':')[0] + ' saved')

sess.close()
