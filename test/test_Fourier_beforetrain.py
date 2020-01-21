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

from gaussianfun import gaussianfun
from gen_dft_data import gen_degree_data
from ButterflyLayer import ButterflyLayer

N = 1024
Ntrain = N
in_siz = N
out_siz = 256
in_range = np.float32([0,1])
out_range = np.float32([0,out_siz//2])
prefixed = 'true'
x_train,y_train = gen_degree_data(in_siz,in_range,out_siz,out_range)
err= np.zeros([4,8])
for r in range (3,7):
    for l in range (5,13):

    
        #=========================================================
        #----- Parameters Setup

        
    
        channel_siz = 4*r # Num of interp pts on each dim
        nlvl = l
        # Filter size for the input and output


        print("======== Parameters =========")
        print("Channel Size:     %6d" % (channel_siz))
        print("Num Levels:       %6d" % (nlvl))
        print("Prefix Coef:      %6r" % (prefixed))
        print("In Range:        (%6.2f, %6.2f)" % (in_range[0], in_range[1]))
        print("Out Range:       (%6.2f, %6.2f)" % (out_range[0], out_range[1]))
        tf.reset_default_graph() 
        #=========================================================
        #----- Variable Preparation
        sess = tf.Session()

        trainInData = tf.placeholder(tf.float32, shape=(N,in_siz,1),
                                 name="trainInData")
        trainOutData = tf.placeholder(tf.float32, shape=(N,out_siz,1),
                                  name="trainOutData")
    
        #=========================================================
        #----- Training Preparation
        butterfly_net = ButterflyLayer(in_siz, out_siz,
                                   channel_siz, nlvl, prefixed,
                                   in_range, out_range)
        
        y_train_output = butterfly_net(tf.convert_to_tensor(x_train))
      
        # Initialize Variables
        init = tf.global_variables_initializer()
    
        print("Total Num Paras:  %6d" % ( np.sum( [np.prod(v.get_shape().as_list())
            for v in tf.trainable_variables()]) ))
        
        #=========================================================
        #----- Step by Step Training
    
        sess.run(init)
        train_dict = {trainInData: x_train, trainOutData: y_train}
        y_train_output = sess.run(y_train_output,feed_dict=train_dict)
        
        p = np.inf
    
        B_r=np.zeros([in_siz, out_siz//2])
        B_i=np.zeros([in_siz, out_siz//2])
        F_r=np.zeros([in_siz, out_siz//2])
        F_i=np.zeros([in_siz, out_siz//2])
        for i in range(0,in_siz):
            for j in range(0,out_siz//2):
                B_r[i][j] = y_train_output[i][2*j][0] 
                B_i[i][j] = y_train_output[i][2*j+1][0]
                F_r[i][j] = y_train[i][2*j][0] 
                F_i[i][j] = y_train[i][2*j+1][0]
        
        B = B_r + B_i *1j
        F = F_r + F_i *1j

        loss_train = np.linalg.norm((F-B),p)
               
        norm_F = np.linalg.norm((F),p)
        
        print("Loss: %6d, norm: %6d"%(loss_train,norm_F))


        err[r-3][l-5] = loss_train/norm_F
        
        sess.close()
print(err)