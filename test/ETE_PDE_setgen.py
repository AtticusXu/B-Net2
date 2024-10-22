import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
sys.path.insert(0,"../src")
sys.path.insert(0,"../src/data_gen")

import numpy as np
import tensorflow as tf
import json
tf.reset_default_graph() 
from gen_dft_data import gen_ete_Ell_data

json_file = open('paras.json')
paras = json.load(json_file)
input_size = paras['inputSize']
N = input_size//2
in_siz = input_size*2
en_mid_siz = 16
de_mid_siz = 32
out_siz = input_size
in_range = np.float32([0,1])
en_mid_range = np.float32([0,en_mid_siz/in_siz])
de_mid_range = np.float32([-0.25,0.25])
out_range = np.float32([0,1])
sig = paras['sigma']
freqidx = range(en_mid_siz//2)
freqmag = np.zeros(N)
for i in range(1,8):
    freqmag[i] = 1
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
testset = paras['Ntest']
trainset =  paras['trainset']

f_train,y_train,u_train,f_norm,y_norm,u_norm = gen_ete_Ell_data(
            trainset,freqidx,freqmag,a)
print(np.mean(f_norm))
print(np.mean(y_norm))
print(np.mean(u_norm))
np.save('tftmp/fft_4000_f_train_c', f_train)
np.save('tftmp/fft_4000_y_train_c', y_train)
np.save('tftmp/fft_4000_u_train_c', u_train)
np.save('tftmp/fft_4000_f_norm_c', f_norm)
np.save('tftmp/fft_4000_y_norm_c', y_norm)
np.save('tftmp/fft_4000_u_norm_c', u_norm)

f_test,y_test,u_test,f_norm_test,y_norm_test,u_norm_test = gen_ete_Ell_data(
            testset,freqidx,freqmag,a)
print(np.mean(f_norm_test))
print(np.mean(y_norm_test))
print(np.mean(u_norm_test))
np.save('tftmp/fft_5000_f_test_c', f_test)
np.save('tftmp/fft_5000_y_test_c', y_test)
np.save('tftmp/fft_5000_u_test_c', u_test)
np.save('tftmp/fft_5000_f_norm_test_c', f_norm_test)
np.save('tftmp/fft_5000_y_norm_test_c', y_norm_test)
np.save('tftmp/fft_5000_u_norm_test_c', u_norm_test)
