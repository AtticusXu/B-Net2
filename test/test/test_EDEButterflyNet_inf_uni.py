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
from gen_dft_data import gen_ede_uni_data
from Bi_ButterflyLayer import ButterflyLayer
from middle_layer import MiddleLayer


json_file = open('paras.json')
paras = json.load(json_file)

N = 32
in_siz = 64
en_mid_siz = 16
de_mid_siz = min(in_siz,en_mid_siz*2)
out_siz = 64
in_range = np.float32([0,1])
en_mid_range = np.float32([0,en_mid_siz/in_siz])
de_mid_range = np.float32([-0.25,0.25])
out_range = np.float32([0,1])
sig = paras['sigma']
freqidx = range(en_mid_siz//2)
freqmag = np.fft.ifftshift(gaussianfun(np.arange(-N//2,N//2),
                                       [3],[0.1]))
#freqmag = np.ones((N))￼
freqmag[N//2] = 0

#=========================================================
#----- Parameters Setup

prefixed = True

channel_siz = 12 # Num of interp pts on each dim


batch_siz = paras['batchSize'] # Batch size during traning


adam_learning_rate = paras['ADAMparas']['learningRate']
adam_learning_rate_decay = paras['ADAMparas']['learningRatedecay']
adam_beta1 = paras['ADAMparas']['beta1']
adam_beta2 = paras['ADAMparas']['beta2']

max_iter = paras['maxIter'] # Maximum num of iterations
test_batch_siz = paras['Ntest']
report_freq = paras['reportFreq'] # Frequency of reporting
record_freq = paras['recordFreq'] # Frequency of recording
#----- Self-adjusted Parameters of BNet
# Num of levels of the BF struct, must be a even num
en_nlvl = 5
de_nlvl = 5

print("======== Parameters =========")
print("Batch Size:       %6d" % (batch_siz))
print("Channel Size:     %6d" % (channel_siz))
print("ADAM LR:          %10ef" % (adam_learning_rate))
print("ADAM LR decay:    %6.4f" % (adam_learning_rate_decay))
print("ADAM Beta1:       %6.4f" % (adam_beta1))
print("ADAM Beta2:       %6.4f" % (adam_beta2))
print("Max Iter:         %6d" % (max_iter))
print("en_Num Levels:       %6d" % (en_nlvl))
print("de_Num Levels:       %6d" % (de_nlvl))
print("Prefix Coef:      %6r" % (prefixed))
print("In Range:        (%6.2f, %6.2f)" % (in_range[0], in_range[1]))
print("en_Mid Range:    (%6.2f, %6.2f)" % (en_mid_range[0], en_mid_range[1]))
print("de_Mid Range:    (%6.2f, %6.2f)" % (de_mid_range[0], de_mid_range[1]))
print("Out Range:       (%6.2f, %6.2f)" % (out_range[0], out_range[1]))


#=========================================================
#----- Variable Preparation
sess = tf.Session()

trainInData = tf.placeholder(tf.float32, shape=(batch_siz,in_siz,1),
        name="trainInData")
trainMidData = tf.placeholder(tf.float32, shape=(batch_siz,en_mid_siz,1),
        name="trainInData")
trainOutData = tf.placeholder(tf.float32, shape=(batch_siz,out_siz,1),
        name="trainOutData")
trainInNorm = tf.placeholder(tf.float32, shape=(batch_siz),
        name="trainNorm")
trainMidNorm = tf.placeholder(tf.float32, shape=(batch_siz),
        name="trainNorm")

global_steps=tf.Variable(0, trainable=False)
#=========================================================
#----- Training Preparation
en_butterfly_net = ButterflyLayer(N, in_siz, en_mid_siz, False,
        channel_siz, en_nlvl, -1, prefixed,
        in_range, en_mid_range)

middle_layer = MiddleLayer(in_siz, en_mid_siz, prefixed = 1, std = 0.35)

de_butterfly_net = ButterflyLayer(N, de_mid_siz, out_siz,False,
            channel_siz , de_nlvl, 1, prefixed,
             de_mid_range, out_range)

learning_rate = tf.train.exponential_decay(adam_learning_rate,
                                           global_steps,100,
                                           adam_learning_rate_decay)
optimizer_adam = tf.train.AdamOptimizer(learning_rate,
        adam_beta1, adam_beta2)

y_train_output = en_butterfly_net(trainInData)

mid_train_output = middle_layer(y_train_output)

f_train_output = de_butterfly_net(mid_train_output)
f_train_output = tf.divide(f_train_output,N)

MSE_loss_train_y = tf.reduce_mean(
        tf.squared_difference(trainMidData, y_train_output))

L2_loss_train_y = tf.reduce_mean(tf.divide(tf.sqrt(tf.reduce_sum(tf.squeeze(
        tf.squared_difference(trainMidData, y_train_output)),1)),trainMidNorm))

MSE_loss_train_f = tf.reduce_mean(
        tf.squared_difference(trainOutData, f_train_output))

L2_loss_train_f = tf.reduce_mean(tf.divide(tf.sqrt(tf.reduce_sum(tf.squeeze(
        tf.squared_difference(trainOutData, f_train_output)),1)),trainInNorm))

train_step = optimizer_adam.minimize(MSE_loss_train_f,global_step=global_steps)

# Initialize Variables
init = tf.global_variables_initializer()

print("Total Num Paras:  %6d" % ( np.sum( [np.prod(v.get_shape().as_list())
    for v in tf.trainable_variables()]) ))

#=========================================================
#----- Step by Step Training

sess.run(init)
err_list = np.zeros((max_iter//record_freq,2))
epochs = np.linspace(0,max_iter,max_iter//record_freq)
for it in range(max_iter):
    x_train,y_train,x_norm,y_norm,y = gen_ede_uni_data(freqmag,freqidx,batch_siz,sig)
    train_dict = {trainInData: x_train,trainMidData: y_train, trainOutData: x_train,
                  trainInNorm: x_norm,trainMidNorm: y_norm}
    if it % report_freq == 0:
        [ytrain,y_loss,f_loss] = sess.run(
            [y_train_output,L2_loss_train_y,L2_loss_train_f], feed_dict=train_dict)
        print("Iter # %6d yLoss: %10e. fLoss:%10e." % (it, y_loss,f_loss))
    if it % record_freq == 0:
        err_list[it//record_freq,0] = y_loss
        err_list[it//record_freq,1] = f_loss
    sess.run(train_step, feed_dict=train_dict)  

[dh,f] = sess.run([trainInData[0,:,0],f_train_output[0,:,0]],
                     feed_dict=train_dict)
dh_r = dh[::2]
dh_f = np.fft.fft(dh_r)
f_r = f[::2]
f_f = np.fft.fft(f_r)

print(dh_f)
print(f_f)
#[x,eh,m] = sess.run([trainMidData[0,:,0],y_train_output[0,:,0],
#                  mid_train_output[0,:,0]], feed_dict=train_dict)
fig = plt.figure(0,figsize=(10,8))
plt.plot(np.linspace(0,in_siz,in_siz),dh,'r',label = 'input')
plt.plot(np.linspace(0,in_siz,in_siz),f,'b',label = 'output')

#fig = plt.figure(1,figsize=(10,8))
#plt.plot(np.linspace(0,en_mid_siz,en_mid_siz),x,'r')
#plt.plot(np.linspace(0,en_mid_siz,en_mid_siz),eh,'b')
#plt.plot(np.linspace(0,de_mid_siz,de_mid_siz),m,'g')
#plt.plot(epochs, err_list[:,1], 'b', label = 'f_Train Error')
#plt.plot(epochs, err_list[:,0], 'r', label = 'Y_Train Error')
plt.legend() # 添加图例
#plt.savefig("EDE_FFT_Train_Error_"+ str(prefixed)+".png")