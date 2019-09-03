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
from ButterflyLayer import ButterflyLayer
from Inv_ButterflyLayer import InvButterflyLayer

json_file = open('paras.json')
paras = json.load(json_file)

N = 32
in_siz = 32
en_mid_siz = 32
de_mid_siz = min(in_siz,en_mid_siz)
out_siz = 64
in_range = np.float32([0,1])
en_mid_range = np.float32([0,en_mid_siz//2])
de_mid_range = np.float32([0,1])
out_range = np.float32([0,32])
sig = paras['sigma']
freqidx = range(en_mid_siz//2)
freqmag = np.fft.ifftshift(gaussianfun(np.arange(-N//2,N//2),
                                       [2],[0.1]))
#freqmag = np.ones((N))￼
freqmag[N//2] = 0

#=========================================================
#----- Parameters Setup

prefixed = True

channel_siz = 32 # Num of interp pts on each dim


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
de_nlvl = 4

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
trainOutData = tf.placeholder(tf.float32, shape=(batch_siz,out_siz//2,1),
        name="trainOutData")
trainInNorm = tf.placeholder(tf.float32, shape=(batch_siz),
        name="trainNorm")
trainMidNorm = tf.placeholder(tf.float32, shape=(batch_siz),
        name="trainNorm")

global_steps=tf.Variable(0, trainable=False)
#=========================================================
#----- Training Preparation
butterfly_net = ButterflyLayer(in_siz, en_mid_siz, False,
        channel_siz, en_nlvl, prefixed,
        in_range, en_mid_range)

inv_butterfly_net = InvButterflyLayer(de_mid_siz, out_siz,
            channel_siz , de_nlvl , prefixed,
             de_mid_range, out_range)
learning_rate = tf.train.exponential_decay(adam_learning_rate,
                                           global_steps,100,
                                           adam_learning_rate_decay)
optimizer_adam = tf.train.AdamOptimizer(learning_rate,
        adam_beta1, adam_beta2)

y_train_output = butterfly_net(trainInData)

f_train_output,en_mid,idata = inv_butterfly_net(trainMidData)

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

[x,f] = sess.run([trainInData[0,:,0],f_train_output[0,:,0]], feed_dict=train_dict)
[m1,m2,i] = sess.run([trainMidData[0,:,0],en_mid[0,:],idata[0,:,0]], feed_dict=train_dict)
print(m1)
print(m2)
print(y[0])
print(ytrain[0,:,0])
err_list = np.log10(err_list)
fig = plt.figure(0,figsize=(10,8))
plt.plot(np.linspace(0,in_siz,in_siz),x,'r')
plt.plot(np.linspace(0,in_siz,in_siz),f,'b')
plt.plot(np.linspace(0,in_siz,in_siz),i,'g')

#plt.plot(epochs, err_list[:,1], 'b', label = 'f_Train Error')
#plt.plot(epochs, err_list[:,0], 'r', label = 'Y_Train Error')
#plt.legend() # 添加图例
#plt.savefig("EDE_FFT_Train_Error_"+ str(prefixed)+".png")