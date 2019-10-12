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
from gen_dft_data import gen_ede_Ell_data
from Bi_ButterflyLayer import ButterflyLayer
from middle_layer import MiddleLayer

json_file = open('paras.json')
paras = json.load(json_file)
input_size = paras['inputSize']
N = input_size//2
in_siz = input_size*2
en_mid_siz = 16
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
freqmag = np.zeros(N)
for i in range(1,8):
    freqmag[i] = 1
freqmag[N//2] = 0
N_0 = 2**10
a = np.ones(N_0+1)
m = N_0//4
for j in range(N_0+1):
    if (j-m//2)%(2*m) < m:
        a[j] = 1
batch_siz = paras['batchSize'] # Batch size during traning
channel_siz = paras['channelSize']
adam_learning_rate = paras['ADAMparas']['learningRate']
adam_learning_rate_decay = paras['ADAMparas']['learningRatedecay']
adam_beta1 = paras['ADAMparas']['beta1']
adam_beta2 = paras['ADAMparas']['beta2']

max_iter = paras['maxIter'] # Maximum num of iterations
test_batch_siz = paras['Ntest']


f_train,y_train,u_train,f_norm,y_norm,u_norm = gen_ede_Ell_data(
            400,freqidx,freqmag,a)
np.save('tftmp/fft_4000_f_train_c', f_train)
np.save('tftmp/fft_4000_y_train_c', y_train)
np.save('tftmp/fft_4000_u_train_c', u_train)
np.save('tftmp/fft_4000_f_norm_c', f_norm)
np.save('tftmp/fft_4000_y_norm_c', y_norm)
np.save('tftmp/fft_4000_u_norm_c', u_norm)