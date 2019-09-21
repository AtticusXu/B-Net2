import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
sys.path.insert(0,"../src")
sys.path.insert(0,"../src/data_gen")
sys.path.insert(0,"../test/train_model")
from pathlib import Path
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio
import tensorflow as tf
import json
from gaussianfun import gaussianfun
from gen_dft_data import gen_energy_uni_data
from ButterflyLayer import ButterflyLayer
from CNNLayer import CNNLayer
tf.reset_default_graph()
json_file = open('paras.json')
paras = json.load(json_file)
sig = 2
N = paras['inputSize']
Ntest = paras['Ntest']
in_siz = paras['inputSize']
out_siz = paras['outputSize']
in_range = np.float32([0,1])
out_range = np.float32([0,out_siz//2])
energy_calc_siz = paras['energyCalcSize']
freqidx = range(out_siz//2)
freqmag = np.zeros((5*out_siz//2-3,N))
for i in range(5*out_siz//2-4):
    freqmag[i] = np.fft.ifftshift(gaussianfun(np.arange(-N//2,N//2),
                                       [i/5],[sig]))
    freqmag[i,N//2] = 0
for i in range(out_siz//2):  
    freqmag[5*out_siz//2-4,i]=0.05
    freqmag[5*out_siz//2-4,-i]=0.05
K = np.zeros(out_siz)
for i in range(out_siz//2):
    K[-2*i-2] = 2**(-1-i)
    K[-2*i-1] = 2**(-1-i)
print(K)

#=========================================================
#----- Parameters Setup

prefixed = paras['prefixed']

#----- Tunable Parameters of BNet
batch_siz = paras['Ntest'] # Batch size during traning
channel_siz = paras['channelSize'] # Num of interp pts on each dim

adam_learning_rate = paras['ADAMparas']['learningRate']
adam_beta1 = paras['ADAMparas']['beta1']
adam_beta2 = paras['ADAMparas']['beta2']

max_iter = paras['maxIter'] # Maximum num of iterations
report_freq = paras['reportFreq'] # Frequency of reporting

#----- Self-adjusted Parameters of BNet
# Num of levels of the BF struct, must be a even num
nlvl = paras["nlvl"]
# Filter size for the input and output




print("======== Parameters =========")
print("Batch Size:       %6d" % (batch_siz))
print("Channel Size:     %6d" % (channel_siz))
print("ADAM LR:          %6.4f" % (adam_learning_rate))
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

testInData = tf.placeholder(tf.float32, shape=(batch_siz,in_siz,1),
        name="trainInData")
testOutData = tf.placeholder(tf.float32, shape=(batch_siz,1),
        name="trainOutData")
testMidData = tf.placeholder(tf.float32, shape=(batch_siz,out_siz,1),
        name="trainMidData")
testNorm = tf.placeholder(tf.float32, shape=(batch_siz),
        name="trainNorm")

#=========================================================
#----- Training Preparation
#butterfly_net = ButterflyLayer(in_siz, out_siz,False,
#        channel_siz, nlvl, prefixed,in_range, out_range)

CNN_net = CNNLayer(in_siz, out_siz,False,3,2,
        channel_siz, nlvl, prefixed)

mid_output = CNN_net(testInData)
if energy_calc_siz == 'sqr':
    if prefixed:
        denseVec = tf.Variable(np.float32(K))
    else:
        denseVec = tf.Variable(tf.random_normal([out_siz],0,0.5))
    tmpVar = tf.multiply(tf.reshape(mid_output,[-1,out_siz]),
            denseVec)
    tmpVar = tf.reduce_sum( tf.square( tmpVar ), 1)
    y_output = tf.reshape(tmpVar,[-1,1])
else:
    denseMat1 = tf.Variable(tf.random_normal([out_siz,energy_calc_siz]))
    bias1 = tf.Variable(tf.random_normal([energy_calc_siz]))
    denseMat2 =  tf.Variable(tf.random_normal([energy_calc_siz,1]))
    tmpVar = tf.matmul(tf.reshape(mid_output,[-1,out_siz]),
            denseMat1)
    tmpVar = tf.nn.relu( tf.nn.bias_add(tmpVar,bias1))
    y_output = tf.reshape(tf.matmul(tmpVar,denseMat2),[-1,1,1])
    
loss_test = tf.reduce_mean(tf.squared_difference(testOutData, y_output))

#L2_loss_test = tf.reduce_mean(tf.divide(tf.sqrt(tf.squared_difference(
#        testOutData,y_output)),testOutData))
L2_loss_test =tf.divide(tf.sqrt(loss_test), tf.reduce_mean(testNorm))
L2_loss_mid = tf.reduce_mean(tf.divide(tf.sqrt(tf.reduce_sum(tf.squeeze(
        tf.squared_difference(mid_output, testMidData)),1)),testNorm))

loss_mid =  tf.reduce_mean(tf.squared_difference(mid_output,
    testMidData))

# Initialize Variables
init = tf.global_variables_initializer()

print("Total Num Paras:  %6d" % ( np.sum( [np.prod(v.get_shape().as_list())
    for v in tf.trainable_variables()]) ))

#=========================================================
#----- Step by Step Training
saver = tf.train.Saver()
sess.run(init)
MODEL_SAVE_PATH = "train_model_energy/"
MODEL_NAME = "cnn_"+str(prefixed)+"_-1_h_model"
mk_test_loss_l2_list = np.zeros((5*out_siz//2-3))
mk_non_loss_l2_list = np.zeros((5*out_siz//2-3))
#for j in range(5*out_siz//2-3):
#    rand_x,rand_h,rand_y,ynorm = gen_energy_uni_data(
#            freqmag[j],freqidx,K,batch_siz,sig)
#
#    train_dict = {testInData: rand_x, testOutData: rand_y,
#                  testMidData: rand_h, testNorm: ynorm}
#    [temp_test_loss,temp_mid,mid] = sess.run(
#             [L2_loss_test,L2_loss_mid,mid_output],feed_dict=train_dict)
#    mk_non_loss_l2_list[j] = temp_test_loss
#np.save('train_model_energy/'+MODEL_NAME+'_mk_non_loss_list',
#        mk_non_loss_l2_list)
saver.restore(sess, MODEL_SAVE_PATH + MODEL_NAME+".ckpt")
for j in range(5*out_siz//2-3):
    rand_x,rand_h,rand_y,ynorm = gen_energy_uni_data(
            freqmag[j],freqidx,K,batch_siz,sig)

    train_dict = {testInData: rand_x, testOutData: rand_y,
                 testMidData: rand_h, testNorm: ynorm}
    [temp_test_loss,temp_mid,mid] = sess.run(
             [L2_loss_test,L2_loss_mid,mid_output],feed_dict=train_dict)
    mk_test_loss_l2_list[j] = temp_test_loss

print(mk_test_loss_l2_list)
np.save('train_model_energy/'+MODEL_NAME+'_mk_test_loss_list', 
            mk_test_loss_l2_list)
    