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

json_file = open('paras.json')
paras = json.load(json_file)
butterfly = paras['butterfly']
input_size = paras['inputSize']
prefixed = paras['prefixed']
N = input_size//2
in_siz = input_size*2
en_mid_siz = 16
de_mid_siz = 32
out_siz = input_size
in_range = np.float32([0,1])
en_mid_range = np.float32([0,en_mid_siz/in_siz])
de_mid_range = np.float32([0,de_mid_siz/in_siz])
out_range = np.float32([0,1])
sig = paras['sigma']
freqidx = range(en_mid_siz//2)
#freqmag = np.fft.ifftshift(gaussianfun(np.arange(-N//2,N//2),
#                                       [1],[0.1]))
freqmag = np.zeros(N)
for i in range(1,8):
    freqmag[1] = 1
freqmag[N//2] = 0
N_0 = 2**10
a = np.ones(N_0+1)
m = N_0//4
for j in range(N_0+1):
    if (j-m//2)%(2*m) < m:
        a[j] = 1
b = 0.1
batch_siz = paras['batchSize'] # Batch size during traning
channel_siz = paras['channelSize']
adam_learning_rate = paras['ADAMparas']['learningRate']
adam_learning_rate_decay = paras['ADAMparas']['learningRatedecay']
adam_beta1 = paras['ADAMparas']['beta1']
adam_beta2 = paras['ADAMparas']['beta2']
linear = paras['linear']
max_iter = paras['maxIter'] # Maximum num of iterations
test_batch_siz = paras['Ntest']
report_freq = paras['reportFreq'] # Frequency of reporting
record_freq = paras['recordFreq'] # Frequency of recording
trainset =  paras['trainset']
test_siz = paras['Ntest']
en_nlvl = 4
de_nlvl = 4
beta = 10**4
dir_mat = DirSineElliptic(a[::(2**10//N)],N)
dir_mat_train = np.tile(dir_mat,(batch_siz,1,1))
dir_mat_test = np.tile(dir_mat,(test_siz,1,1))
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

testInData = tf.placeholder(tf.float32, shape=(test_siz,in_siz,1),
        name="testInData")
testInNorm = tf.placeholder(tf.float32, shape=(test_siz),
        name="testInNorm")
testMidData = tf.placeholder(tf.float32, shape=(test_siz,en_mid_siz,1),
        name="testMidData")
testMidNorm = tf.placeholder(tf.float32, shape=(test_siz),
        name="testMidNorm")
testOutData = tf.placeholder(tf.float32, shape=(test_siz,out_siz//2,1),
        name="testOutData")
testOutNorm = tf.placeholder(tf.float32, shape=(test_siz),
        name="testOutNorm")
global_steps=tf.Variable(0, trainable=False)

#=========================================================
#----- Training Preparation

middle_net = MiddleLayer(in_siz, en_mid_siz, de_mid_siz, a[::(2**10//N)],
                                True, prefixed, std = 0.1)

if butterfly:
    en_butterfly_net = ButterflyLayer(2*N, in_siz, en_mid_siz, False,
        channel_siz, en_nlvl, -1, prefixed,
        in_range, en_mid_range,0.5)
    de_butterfly_net = ButterflyLayer(N, de_mid_siz, out_siz, False,
        channel_siz, de_nlvl, 1, prefixed,
        de_mid_range, out_range,0.5)
    
    y_train_en_mid = en_butterfly_net(trainInData)
    y_train_de_mid = middle_net(y_train_en_mid)
    y_train_output = de_butterfly_net(y_train_de_mid)/N
    y_train_output_i = -y_train_output[:,1::2,:]
    
    y_test_en_mid = en_butterfly_net(testInData)
    y_test_de_mid = middle_net(y_test_en_mid)
    y_test_output = de_butterfly_net(y_test_de_mid)/N
    y_test_output_i = -y_test_output[:,1::2,:]
else:
    en_cnn_net = CNNLayer(2*N, in_siz, en_mid_siz, 
        channel_siz, en_nlvl, -1, prefixed,0.4)
    de_cnn_net = CNNLayer(N, de_mid_siz, out_siz,
        channel_siz, de_nlvl, 1, prefixed,0.4)
    
    y_train_en_mid = en_cnn_net(trainInData)
    y_train_de_mid = middle_net(y_train_en_mid)
    y_train_output = de_cnn_net(y_train_de_mid)/N
    y_train_output_i = -y_train_output[:,1::2,:]
    
    y_test_en_mid = en_cnn_net(testInData)
    y_test_de_mid = middle_net(y_test_en_mid)
    y_test_output = de_cnn_net(y_test_de_mid)/N
    y_test_output_i = -y_test_output[:,1::2,:]
    
if linear:
    st_train_dense = tf.Variable(dir_mat_train.astype(np.float32),
                       trainable=False,name = "ST_Train_Dense")
    st_test_dense = tf.Variable(dir_mat_test.astype(np.float32),
                       trainable=False,name = "ST_Test_Dense")
    
    f_train_output = tf.matmul(st_train_dense,y_train_output_i)
    f_test_output = tf.matmul(st_test_dense,y_test_output_i)
else:
    st_train_dense = tf.Variable(dir_mat_train.astype(np.float32),
                       trainable=False,name = "ST_Train_Dense")
    st_test_dense = tf.Variable(dir_mat_test.astype(np.float32),
                       trainable=False,name = "ST_Test_Dense")
    st_train_b = tf.Variable(tf.constant(b, dtype=np.float32,
                                   shape=[batch_siz,out_siz//2,1]),
                       trainable=False,name = "ST_Train_b")
    st_test_b = tf.Variable(tf.constant(b, dtype=np.float32,
                                   shape=[test_siz,out_siz//2,1]),
                       trainable=False,name = "ST_Train_b")
    
    f_train_linear = tf.matmul(st_train_dense,y_train_output_i)
    f_train_nonlinear = y_train_output_i*y_train_output_i*y_train_output_i
    f_train_nonlinear = st_train_b*f_train_nonlinear
    f_train_output = f_train_linear + f_train_nonlinear
    
    f_test_linear = tf.matmul(st_test_dense,y_test_output_i)
    f_test_nonlinear = y_test_output_i*y_test_output_i*y_test_output_i
    f_test_nonlinear = st_test_b*f_test_nonlinear
    f_test_output = f_test_linear + f_test_nonlinear

learning_rate = tf.train.exponential_decay(adam_learning_rate,
                                           global_steps,100,
                                           adam_learning_rate_decay)
optimizer_adam = tf.train.AdamOptimizer(learning_rate,
        adam_beta1, adam_beta2)

MSE_loss_train_u = tf.reduce_mean(
        tf.squared_difference(trainOutData, y_train_output_i))
MSE_loss_train_y = tf.reduce_mean(
        tf.squared_difference(trainMidData[:,0::2], -y_train_en_mid[:,1::2]))
MSE_loss_train_f = tf.reduce_mean(
        tf.squared_difference(trainInData[:,0:input_size:2], f_train_output))

L2_loss_train_u = tf.reduce_mean(tf.divide(tf.sqrt(tf.reduce_sum(tf.squeeze(
            tf.squared_difference(trainOutData, y_train_output_i)),1)),trainOutNorm))
L2_loss_train_y = tf.reduce_mean(tf.divide(tf.sqrt(tf.reduce_sum(tf.squeeze(
            tf.squared_difference(trainMidData[:,0::2],
                                  -y_train_en_mid[:,1::2])),1)),trainMidNorm))
L2_loss_train_f = tf.reduce_mean(tf.divide(tf.sqrt(tf.reduce_sum(tf.squeeze(
            tf.squared_difference(trainInData[:,0:input_size:2],
                                  f_train_output)),1)),trainInNorm))

boundary_loss_train = tf.reduce_mean(tf.square(y_train_output_i[:,0]))
solvetrain_loss_train = MSE_loss_train_f + beta*boundary_loss_train

MSE_loss_test_u = tf.reduce_mean(
        tf.squared_difference(testOutData, y_test_output_i))
MSE_loss_test_y = tf.reduce_mean(
        tf.squared_difference(testMidData[:,0::2], -y_test_en_mid[:,1::2]))
MSE_loss_test_f = tf.reduce_mean(
        tf.squared_difference(testInData[:,0:input_size:2], f_test_output))

L2_loss_test_u = tf.reduce_mean(tf.divide(tf.sqrt(tf.reduce_sum(tf.squeeze(
            tf.squared_difference(testOutData, y_test_output_i)),1)),testOutNorm))
L2_loss_test_y = tf.reduce_mean(tf.divide(tf.sqrt(tf.reduce_sum(tf.squeeze(
            tf.squared_difference(testMidData[:,0::2],
                                  -y_test_en_mid[:,1::2])),1)),testMidNorm))
L2_loss_test_f = tf.reduce_mean(tf.divide(tf.sqrt(tf.reduce_sum(tf.squeeze(
            tf.squared_difference(testInData[:,0:input_size:2],
                                  f_test_output)),1)),testInNorm))

boundary_loss_test = tf.reduce_mean(tf.square(y_test_output_i[:,0]))
solvetrain_loss_test = MSE_loss_test_f + beta*boundary_loss_test

#mix_loss = MSE_loss_train_u + 10**(-9)*solvetrain_loss
train_step = optimizer_adam.minimize(solvetrain_loss_train,global_step=global_steps)

# Initialize Variables
init = tf.global_variables_initializer()
print("Total Num Paras:  %6d" % ( np.sum( [np.prod(v.get_shape().as_list())
    for v in tf.trainable_variables()]) ))
    
sess.run(init)
err_list = np.zeros((3,max_iter//record_freq))
f_train = np.load('tftmp/fft_4000_f_train_c.npy')
y_train = np.load('tftmp/fft_4000_y_train_c.npy')
u_train = np.load('tftmp/fft_4000_u_train_c.npy')
f_norm = np.load('tftmp/fft_4000_f_norm_c.npy')
y_norm = np.load('tftmp/fft_4000_y_norm_c.npy')
u_norm = np.load('tftmp/fft_4000_u_norm_c.npy')
for it in range(max_iter):
    start = (it*batch_siz)%trainset
    end = ((it+1)*batch_siz-1)%trainset+1
    f_train_it = f_train[start:end]
    y_train_it = y_train[start:end]
    u_train_it = u_train[start:end]
    f_norm_it = f_norm[start:end]
    y_norm_it = y_norm[start:end]
    u_norm_it = u_norm[start:end]

    train_dict = {trainInData: f_train_it, trainInNorm: f_norm_it,
                  trainMidData: y_train_it, trainMidNorm: y_norm_it,
                  trainOutData: u_train_it, trainOutNorm: u_norm_it}
    if it % report_freq == 0:
        [f,y_L2_loss,y_loss,u_L2_loss,u_loss,f_L2_loss,f_loss] = sess.run(
                    [solvetrain_loss_train,L2_loss_train_y, MSE_loss_train_y,
                     L2_loss_train_u, MSE_loss_train_u,
                     L2_loss_train_f, MSE_loss_train_f],
                    feed_dict=train_dict)
        print('step(s): %d'%it)
        print(f)
        #print(f_train_it[0])
        print('f_mse:%10e,f_l2:%10e'%(f_loss, f_L2_loss))
        print('y_mse:%10e,y_l2:%10e'%(y_loss, y_L2_loss))
        print('u_mse:%10e,u_l2:%10e'%(u_loss, u_L2_loss))
        if it % record_freq == 0:
            err_list[0,it//record_freq] = f_L2_loss
            err_list[1,it//record_freq] = y_L2_loss
            err_list[2,it//record_freq] = u_L2_loss

    sess.run(train_step, feed_dict=train_dict)
    
f_test = np.load('tftmp/fft_5000_f_test_c.npy')
y_test = np.load('tftmp/fft_5000_y_test_c.npy')
u_test = np.load('tftmp/fft_5000_u_test_c.npy')
f_norm_test = np.load('tftmp/fft_5000_f_norm_test_c.npy')
y_norm_test = np.load('tftmp/fft_5000_y_norm_test_c.npy')
u_norm_test = np.load('tftmp/fft_5000_u_norm_test_c.npy')
test_dict = {testInData: f_test, testInNorm: f_norm_test,
                  testMidData: y_test, testMidNorm: y_norm_test,
                  testOutData: u_test, testOutNorm: u_norm_test}
[f,y_L2_loss,y_loss,u_L2_loss,u_loss,f_L2_loss,f_loss] = sess.run(
                    [solvetrain_loss_test,L2_loss_test_y, MSE_loss_test_y,
                     L2_loss_test_u, MSE_loss_test_u,
                     L2_loss_test_f, MSE_loss_test_f],
                    feed_dict=test_dict)
print("test error:")
print(f)
#print(f_train_it[0])
print('f_mse:%10e,f_l2:%10e'%(f_loss, f_L2_loss))
print('y_mse:%10e,y_l2:%10e'%(y_loss, y_L2_loss))
print('u_mse:%10e,u_l2:%10e'%(u_loss, u_L2_loss))
#print(y_en[0])
#print(y_de[0])
#print(y_out[0])
#print(y_loss)