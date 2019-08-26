import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
sys.path.insert(0,"../src")
sys.path.insert(0,"../src/data_gen")
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
from gen_dft_data import gen_uni_data
from CNNLayer import CNNLayer

json_file = open('paras.json')
paras = json.load(json_file)

N = paras['inputSize']
sig = paras['sigma']
in_siz = paras['inputSize']
out_siz = paras['outputSize']
in_range = np.float32([0,1])
out_range = np.float32([0,out_siz//2])

freqidx = range(out_siz//2)
freqmag = np.zeros((5*out_siz//2-3,N))
for i in range(5*out_siz//2-4):
    freqmag[i] = np.fft.ifftshift(gaussianfun(np.arange(-N//2,N//2),
                                       [i/5],[sig]))
    freqmag[i,N//2] = 0
for i in range(out_siz//2):  
    freqmag[5*out_siz//2-4,i]=0.05
    freqmag[5*out_siz//2-4,-i]=0.05
print(freqmag[3])
#=========================================================
#----- Parameters Setup


nlvl = paras['nlvl']# Num of levels of the BF struct
klvl = paras['klvl']
alph = paras['alpha']
prefixed = paras['prefixed']

channel_siz = paras['channelSize'] # Num of interp pts on each dim

batch_siz = paras['batchSize'] # Batch size during traning

adam_learning_rate = paras['ADAMparas']['learningRate']
adam_learning_rate_decay = paras['ADAMparas']['learningRatedecay']
adam_beta1 = paras['ADAMparas']['beta1']
adam_beta2 = paras['ADAMparas']['beta2']

max_iter = paras['maxIter'] # Maximum num of iterations
test_batch_siz = paras['Ntest']
report_freq = paras['reportFreq'] # Frequency of reporting
record_freq = paras['recordFreq'] # Frequency of recording

print("======== Parameters =========")
print("Batch Size:       %6d" % (batch_siz))
print("Channel Size:     %6d" % (channel_siz))
print("Alpha:            %6d" % (alph))
print("ADAM LR:          %10e" % (adam_learning_rate))
print("ADAM LR decay:    %6.4f" % (adam_learning_rate_decay))
print("ADAM Beta1:       %6.4f" % (adam_beta1))
print("ADAM Beta2:       %6.4f" % (adam_beta2))
print("Max Iter:         %6d" % (max_iter))
print("Num Levels:       %6d" % (nlvl))
print("K Levels:         %6d" % (klvl))
print("Prefix Coef:      %6r" % (prefixed))
print("In Range:        (%6.2f, %6.2f)" % (in_range[0], in_range[1]))
print("Out Range:       (%6.2f, %6.2f)" % (out_range[0], out_range[1]))

def train():
    #=========================================================
    #----- Variable Preparation
    sess = tf.Session()
    
    trainInData = tf.placeholder(tf.float32, shape=(batch_siz,in_siz,1),
                                 name="trainInData")
    trainOutData = tf.placeholder(tf.float32, shape=(batch_siz,out_siz,1),
                                  name="trainOutData")
    trainNorm = tf.placeholder(tf.float32, shape=(batch_siz),
                               name="trainNorm")

    global_steps=tf.Variable(0, trainable=False)
    #=========================================================
    #----- Training Preparation
    CNN_net = CNNLayer(in_siz, out_siz, False, klvl, alph,
                       channel_siz, nlvl, prefixed)
    
    learning_rate = tf.train.exponential_decay(adam_learning_rate,
                                               global_steps,100,
                                               adam_learning_rate_decay)
    optimizer_adam = tf.train.AdamOptimizer(learning_rate,
                                                adam_beta1, adam_beta2)
    
    y_train_output = CNN_net(trainInData)
    
    MSE_loss_train = tf.reduce_mean(
            tf.squared_difference(trainOutData, y_train_output))
    
    L2_loss_train = tf.reduce_mean(tf.divide(tf.sqrt(tf.reduce_sum(tf.squeeze(
            tf.squared_difference(trainOutData, y_train_output)),1)),trainNorm))
    
    Sqr_loss_train_K = tf.reduce_mean(tf.squeeze(tf.squared_difference(
            trainOutData, y_train_output)),0)
    
    
    y_norm_train = tf.reduce_mean(trainNorm)

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
    for s in range(S):
        saver = tf.train.Saver()
        sess.run(init)
        MODEL_SAVE_PATH = "train_model/"
        MODEL_NAME = "cnn_"+str(prefixed)+"_"+str(s)+"_model"
        for it in range(max_iter):
            x_train,y_train,y_norm = gen_uni_data(freqmag[0],freqidx,batch_siz,sig)
            train_dict = {trainInData: x_train, trainOutData: y_train,
                          trainNorm: y_norm}
           if it % report_freq == 0:
                [temp_train_loss,y_norm] = sess.run(
                       [L2_loss_train,y_norm_train], feed_dict=train_dict)
                print("Iter # %6d: Train Loss: %10e.%10e" % (it,
                      temp_train_loss,y_norm))
            if it % record_freq == 0:
                K_loss = sess.run(Sqr_loss_train_K,feed_dict=train_dict)
                err_list[s,it//record_freq] = temp_train_loss
                K_list[s,:,it//record_freq] = K_loss
            sess.run(train_step, feed_dict=train_dict)
        
        saver.save(sess, MODEL_SAVE_PATH + MODEL_NAME+".ckpt")
    np.save('train_model/cnn_err_list_'+str(prefixed), err_list)
    np.save('train_model/cnn_K_list_'+str(prefixed), K_list)  
    sess.close()


def main(argv=None):
    tf.reset_default_graph()
    train()


if __name__ == '__main__':
    main()
