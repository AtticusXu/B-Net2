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
from gen_dft_data import gen_ete_denoise_data
from Bi_ButterflyLayer import ButterflyLayer
from Bi_CNNLayer import CNNLayer
from middle_layer import MiddleLayer

json_file = open('paras.json')
paras = json.load(json_file)
#=========================================================
#----- Parameters Setup

input_size = paras['inputSize']
output_size = paras['outputSize']
prefixed = paras['prefixed']
N = input_size
noise = 0.01
in_siz = input_size*2
en_mid_siz = output_size
de_mid_siz = output_size*2
out_siz = input_size*2
in_range = np.float32([0,1])
en_mid_range = np.float32([0,en_mid_siz/in_siz])
de_mid_range = np.float32([-en_mid_siz/in_siz,en_mid_siz/in_siz])
out_range = np.float32([0,1])
sig = paras['sigma']
freqidx = range(en_mid_siz//2)
freqmag = np.fft.ifftshift(gaussianfun(np.arange(-N//2,N//2),
                                       [0],[2]))

#----- Self-adjusted Parameters of Net
butterfly = paras['butterfly']
batch_siz = paras['batchSize'] # Batch size during traning
channel_siz = paras['channelSize']
adam_learning_rate = paras['ADAMparas']['learningRate']
adam_learning_rate_decay = paras['ADAMparas']['learningRatedecay']
adam_beta1 = paras['ADAMparas']['beta1']
adam_beta2 = paras['ADAMparas']['beta2']
max_iter = paras['maxIter'] # Maximum num of iterations
report_freq = paras['reportFreq'] # Frequency of reporting
record_freq = paras['recordFreq'] # Frequency of recording
trainset =  paras['trainset']
test_siz = paras['Ntest']
en_nlvl = 4
de_nlvl = 4
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

middle_net = MiddleLayer(in_siz, en_mid_siz, de_mid_siz, False)

if butterfly:
    en_butterfly_net = ButterflyLayer(N, in_siz, en_mid_siz, False,
        channel_siz, en_nlvl, -1, prefixed,
        in_range, en_mid_range,0.45)
    de_butterfly_net = ButterflyLayer(N, de_mid_siz, out_siz, False,
        channel_siz, de_nlvl, 1, prefixed,
        de_mid_range, out_range,0.45)
    
    y_train_en_mid = en_butterfly_net(trainInData)
    y_train_de_mid = middle_net(y_train_en_mid)
    train_output = de_butterfly_net(y_train_de_mid)/N
    x_train_output = train_output[:,::2]
    
    y_test_en_mid = en_butterfly_net(testInData)
    y_test_de_mid = middle_net(y_test_en_mid)
    test_output = de_butterfly_net(y_test_de_mid)/N
    x_test_output = test_output[:,::2]
else:
    en_cnn_net = CNNLayer(N, in_siz, en_mid_siz, 
        channel_siz, en_nlvl, -1, prefixed,0.3)
    de_cnn_net = CNNLayer(N, de_mid_siz, out_siz,
        channel_siz, de_nlvl, 1, prefixed,0.3)
    
    y_train_en_mid = en_cnn_net(trainInData)
    y_train_de_mid = middle_net(y_train_en_mid)
    train_output = de_cnn_net(y_train_de_mid)/N
    x_train_output = train_output[:,::2]
    
    y_test_en_mid = en_cnn_net(testInData)
    y_test_de_mid = middle_net(y_test_en_mid)
    test_output = de_cnn_net(y_test_de_mid)/N
    x_test_output = test_output[:,::2]

learning_rate = tf.train.exponential_decay(adam_learning_rate,
                                           global_steps,100,
                                           adam_learning_rate_decay)
optimizer_adam = tf.train.AdamOptimizer(learning_rate,
        adam_beta1, adam_beta2)

MSE_loss_train_x = tf.reduce_mean(
        tf.squared_difference(trainOutData, x_train_output))
MSE_loss_train_y = tf.reduce_mean(
        tf.squared_difference(trainMidData, y_train_en_mid))


L2_loss_train_x = tf.reduce_mean(tf.divide(tf.sqrt(tf.reduce_sum(tf.squeeze(
            tf.squared_difference(trainOutData, x_train_output)),1)),trainOutNorm))
L2_loss_train_y = tf.reduce_mean(tf.divide(tf.sqrt(tf.reduce_sum(tf.squeeze(
            tf.squared_difference(trainMidData,y_train_en_mid)),1)),trainMidNorm))

MSE_loss_test_x = tf.reduce_mean(
        tf.squared_difference(testOutData, x_test_output))
MSE_loss_test_y = tf.reduce_mean(
        tf.squared_difference(testMidData, y_test_en_mid))


L2_loss_test_x = tf.reduce_mean(tf.divide(tf.sqrt(tf.reduce_sum(tf.squeeze(
            tf.squared_difference(testOutData, x_test_output)),1)),testOutNorm))
L2_loss_test_y = tf.reduce_mean(tf.divide(tf.sqrt(tf.reduce_sum(tf.squeeze(
            tf.squared_difference(testMidData,y_test_en_mid)),1)),testMidNorm))

train_step = optimizer_adam.minimize(MSE_loss_train_x,global_step=global_steps)

# Initialize Variables
init = tf.global_variables_initializer()
print("Total Num Paras:  %6d" % ( np.sum( [np.prod(v.get_shape().as_list())
    for v in tf.trainable_variables()]) ))
    
saver = tf.train.Saver()   
sess.run(init)
MODEL_SAVE_PATH = "train_model/"
MODEL_NAME = "noise_"+str(noise) +"_"+str(prefixed)+"_"+str(butterfly)+"_model"

for it in range(max_iter):
    xdata,ydata,udata,xnorm,ynorm,unorm,nnorm,nrel = gen_ete_denoise_data(
                                            freqmag,freqidx,batch_siz,sig,noise)
    train_dict = {trainInData: udata, trainInNorm: unorm,
                  trainMidData: ydata, trainMidNorm: ynorm,
                  trainOutData: xdata, trainOutNorm: xnorm}
    if it % report_freq == 0:
        [y_L2_loss,y_loss,x_L2_loss,x_loss] = sess.run(
                    [L2_loss_train_y, MSE_loss_train_y,
                     L2_loss_train_x, MSE_loss_train_x],
                    feed_dict=train_dict)
        print('step(s): %d'%it)
        print('y_mse:%10e,y_l2:%10e'%(y_loss, y_L2_loss))
        print('x_mse:%10e,x_l2:%10e'%(x_loss, x_L2_loss))

    sess.run(train_step, feed_dict=train_dict)
saver.save(sess, MODEL_SAVE_PATH + MODEL_NAME+".ckpt")
x_test,y_test,u_test,xnorm_test,ynorm_test,unorm_test,nnorm,nrel = gen_ede_denoise_data(
                                            freqmag,freqidx,test_siz,sig,noise)
print(nrel)
test_dict = {testInData: u_test, testInNorm: unorm_test,
             testMidData: y_test, testMidNorm: ynorm_test,
             testOutData: x_test, testOutNorm: xnorm_test}
print(np.mean(xnorm))
print(np.mean(ynorm))
print(np.mean(unorm))
print(np.mean(nnorm))
[y_L2_loss,y_loss,x_L2_loss,x_loss] = sess.run(
                    [L2_loss_test_y, MSE_loss_test_y,
                     L2_loss_test_x, MSE_loss_test_x],
                    feed_dict=test_dict)
print("test error:")
print('y_mse:%10e,y_l2:%10e'%(y_loss, y_L2_loss))
print('u_mse:%10e,u_l2:%10e'%(x_loss, x_L2_loss))