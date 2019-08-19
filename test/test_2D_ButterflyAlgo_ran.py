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
from gaussianfun import gaussianfun2D
from gen_dft_data import gen_2D_gaussian_data
from gen_dft_data import gen_2D_zero_data
from ButterflyLayer2D import ButterflyLayer2D

json_file = open('2Dparas.json')
paras = json.load(json_file)

N = paras['inputSize']
Ntrain = 1
in_siz = paras['inputSize']
out_siz = paras['outputSize']
sig = paras['sigma']
in_range = np.float32([[0,1],[0,1]])
out_range = np.float32([[0,out_siz],[0,out_siz]])

freqmag = gaussianfun2D(np.arange(-N//2,N//2),np.arange(-N//2,N//2),
                                         [0,0],[sig,sig],0)

#=========================================================
#----- Parameters Setup

prefixed = paras['prefixed']

channel_siz = 3 # Num of interp pts on each dim

x_train,y_train = gen_2D_gaussian_data(freqmag,out_siz, Ntrain,sig)
np.save('tftmp/x_train',x_train)
np.save('tftmp/y_train',y_train)

#----- Self-adjusted Parameters of BNet
# Num of levels of the BF struct, must be a even num
nlvl = paras['nlvl']
klvl = paras['klvl']
# Filter size for the input and output


print("======== Parameters =========")
print("Channel Size:     %6d" % (channel_siz))
print("Num Levels:       %6d" % (nlvl))
print("K   Levels:       %6d" % (klvl))
print("Prefix Coef:      %6r" % (prefixed))
print("In Range:        (%6.2f, %6.2f)*(%6.2f, %6.2f)" 
      % (in_range[0][0], in_range[0][1], in_range[1][0], in_range[1][1]))
print("Out Range:       (%6.2f, %6.2f)*(%6.2f, %6.2f)" 
      % (out_range[0][0], out_range[0][1], out_range[1][0], out_range[1][1]))

#=========================================================
#----- Variable Preparation
sess = tf.Session()

trainInData = tf.placeholder(tf.float32, shape=(Ntrain,in_siz,in_siz,1),
        name="trainInData")
trainOutData = tf.placeholder(tf.float32, shape=(Ntrain,out_siz,out_siz,2),
        name="trainOutData")

#=========================================================
#----- Training Preparation
butterfly_net = ButterflyLayer2D(in_siz, in_siz, out_siz, out_siz,
        channel_siz, nlvl, klvl, prefixed,
        in_range, out_range)

y_train_output = butterfly_net(tf.cast(tf.convert_to_tensor(x_train), tf.float32))

# Initialize Variables
init = tf.global_variables_initializer()

print("Total Num Paras:  %6d" % ( np.sum( [np.prod(v.get_shape().as_list())
    for v in tf.trainable_variables()]) ))

#=========================================================
#----- Step by Step Training

sess.run(init)
train_dict = {trainInData: x_train, trainOutData: y_train}
y_train_output = sess.run(y_train_output,feed_dict=train_dict)
print(y_train_output)



sess.close()
