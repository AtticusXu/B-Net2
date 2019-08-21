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
from gen_dft_data import gen_uni_data
from ButterflyLayer import ButterflyLayer

json_file = open('paras.json')
paras = json.load(json_file)

N = paras['inputSize']

in_siz = paras['inputSize']
out_siz = paras['outputSize']
in_range = np.float32([0,1])
out_range = np.float32([0,out_siz//2])
sig = paras['sigma']
freqidx = range(out_siz//2)
freqmag_test = np.fft.ifftshift(gaussianfun(np.arange(-N//2,N//2),
                                       [0],[sig]))
freqmag_test[N//2] = 0
freqmag = np.zeros(N)
for i in range(out_siz//2):
    freqmag[i]=0.05
    freqmag[-i]=0.05
freqmag[N//2] = 0

#=========================================================
#----- Parameters Setup

prefixed = paras['prefixed']

channel_siz = paras['channelSize'] # Num of interp pts on each dim


batch_siz = paras['batchSize'] # Batch size during traning
channel_siz = paras['channelSize'] # Num of interp pts on each dim

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
nlvl = paras['nlvl']

print("======== Parameters =========")
print("Batch Size:       %6d" % (batch_siz))
print("Channel Size:     %6d" % (channel_siz))
print("ADAM LR:          %10ef" % (adam_learning_rate))
print("ADAM LR decay:    %6.4f" % (adam_learning_rate_decay))
print("ADAM Beta1:       %6.4f" % (adam_beta1))
print("ADAM Beta2:       %6.4f" % (adam_beta2))
print("Max Iter:         %6d" % (max_iter))
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
trainNorm = tf.placeholder(tf.float32, shape=(batch_siz),
        name="trainNorm")
testInData = tf.placeholder(tf.float32, shape=(test_batch_siz,in_siz,1),
        name="trainInData")
testOutData = tf.placeholder(tf.float32, shape=(test_batch_siz,out_siz,1),
        name="trainOutData")
testNorm = tf.placeholder(tf.float32, shape=(test_batch_siz),
        name="testNorm")
global_steps=tf.Variable(0, trainable=False)
#=========================================================
#----- Training Preparation
butterfly_net = ButterflyLayer(in_siz, out_siz, False,
        channel_siz, nlvl, prefixed,
        in_range, out_range)

learning_rate = tf.train.exponential_decay(adam_learning_rate,
                                           global_steps,100,
                                           adam_learning_rate_decay)
optimizer_adam = tf.train.AdamOptimizer(learning_rate,
        adam_beta1, adam_beta2)

y_train_output = butterfly_net(trainInData)
y_test_output = butterfly_net(testInData)

MSE_loss_train = tf.reduce_mean(
        tf.squared_difference(trainOutData, y_train_output))

L2_loss_train = tf.reduce_mean(tf.divide(tf.sqrt(tf.reduce_sum(tf.squeeze(
        tf.squared_difference(trainOutData, y_train_output)),1)),trainNorm))


Sqr_loss_train_K = tf.reduce_mean(tf.squeeze(tf.squared_difference(
        trainOutData, y_train_output)),0)

y_norm_train = tf.reduce_mean(trainNorm)

L2_loss_test = tf.reduce_mean(tf.divide(tf.sqrt(tf.reduce_sum(tf.squeeze(
        tf.squared_difference(testOutData, y_test_output)),1)),testNorm))

L2_loss_test_list = tf.divide(tf.sqrt(tf.reduce_sum(tf.squeeze(
        tf.squared_difference(testOutData, y_test_output)),1)),testNorm)

Sqr_loss_test_K = tf.reduce_mean(tf.squeeze(
        tf.squared_difference(testOutData, y_test_output)),0)

Sqr_test_norm_K = tf.reduce_mean(tf.squeeze(tf.square(testOutData)),0)

train_step = optimizer_adam.minimize(MSE_loss_train,global_step=global_steps)

# Initialize Variables
init = tf.global_variables_initializer()

print("Total Num Paras:  %6d" % ( np.sum( [np.prod(v.get_shape().as_list())
    for v in tf.trainable_variables()]) ))

#=========================================================
#----- Step by Step Training
S=2
K_list = np.zeros((S,out_siz,max_iter//record_freq)) 
err_list = np.zeros((S,max_iter//record_freq))
epochs = np.linspace(0,max_iter,max_iter//record_freq)
test_loss_klist = np.zeros((S,out_siz//2))
for s in range(S):
    sess.run(init)
    
    for it in range(max_iter):
        x_train,y_train,y_norm = gen_uni_data(freqmag,freqidx,batch_siz,sig)
        train_dict = {trainInData: x_train, trainOutData: y_train,
                  trainNorm: y_norm}
        if it % report_freq == 0:
            temp_train_loss = sess.run(L2_loss_train, feed_dict=train_dict)
            print("Iter # %6d: Train Loss: %10e." % (it,temp_train_loss))
        if it % record_freq == 0:
            K_loss = sess.run(Sqr_loss_train_K,feed_dict=train_dict)
            err_list[s,it//record_freq] = temp_train_loss
            K_list[s,:,it//record_freq] = K_loss
        sess.run(train_step, feed_dict=train_dict)

    x_test,y_test,y_norm = gen_uni_data(freqmag,freqidx,test_batch_siz,sig)
    test_dict = {testInData: x_test, testOutData: y_test,
                      testNorm: y_norm}
    [test_loss, test_loss_list, test_loss_k,K_norm] = sess.run(
        [L2_loss_test, L2_loss_test_list, Sqr_loss_test_K,Sqr_test_norm_K],
        feed_dict=test_dict)
    print("Test Loss: %10e." % (test_loss))

    for i in range(out_siz//2):
        K_norm[i] = np.sqrt(K_norm[2*i] + K_norm[2*i+1])
        K_list[s,i,:] = np.sqrt(K_list[s,2*i,:] + K_list[s,2*i+1,:])\
                        /K_norm[i]
        test_loss_klist[s,i] = np.sqrt(test_loss_k[2*i] + test_loss_k[2*i+1])\
                                /K_norm[i]
    
    print(test_loss_k[0:out_siz//2])

err_list = np.log10(err_list)
K_list = np.log10(K_list)
print(test_loss_klist)
print(K_norm[0:out_siz//2])

for k in range(out_siz//2):
    fig = plt.figure(k,figsize=(10,8))
    for s in range(S):
        plt.plot(epochs, K_list[s,k], 'r', label = 'k = '+ str(k)+')')
    plt.title('FFT_Training Error Plot(k='+ str(k)+')')
    plt.savefig("FFT_Train_Error_"+ str(prefixed)+"_"+str(k)+".png" )

sess.close()
