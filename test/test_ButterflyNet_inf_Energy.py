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
from matplotlib import pyplot as plt
import json
tf.reset_default_graph() 
from gen_dft_data import gen_energy_data
from ButterflyLayer import ButterflyLayer

json_file = open('paras.json')
paras = json.load(json_file)

N = paras['inputSize']
Ntest = paras['Ntest']
in_siz = paras['inputSize']
out_siz = paras['outputSize']
in_range = np.float32([0,1])
out_range = np.float32([0,out_siz//2])
energy_calc_siz = paras['energyCalcSize']

#=========================================================
#----- Parameters Setup

prefixed = paras['prefixed']

#----- Tunable Parameters of BNet
batch_siz = paras['batchSize'] # Batch size during traning
channel_siz = paras['channelSize'] # Num of interp pts on each dim

adam_learning_rate = paras['ADAMparas']['learningRate']
adam_beta1 = paras['ADAMparas']['beta1']
adam_beta2 = paras['ADAMparas']['beta2']

max_iter = paras['maxIter'] # Maximum num of iterations
report_freq = paras['reportFreq'] # Frequency of reporting

#----- Self-adjusted Parameters of BNet
# Num of levels of the BF struct, must be a even num
nlvl = 2*math.floor(math.log(min(in_siz,out_siz//2),2)/2)
# Filter size for the input and output
in_filter_siz = in_siz//2**nlvl
out_filter_siz = out_siz//2**nlvl

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

trainInData = tf.placeholder(tf.float32, shape=(batch_siz,in_siz,1),
        name="trainInData")
trainOutData = tf.placeholder(tf.float32, shape=(batch_siz,1,1),
        name="trainOutData")
testInData = tf.placeholder(tf.float32, shape=(1,in_siz,1),
        name="testInData")
testOutData = tf.placeholder(tf.float32, shape=(1,1,1),
        name="testOutData")

#=========================================================
#----- Training Preparation
butterfly_net = ButterflyLayer(in_siz, out_siz,
        channel_siz, nlvl, prefixed,
        in_range, out_range)

optimizer_adam = tf.train.AdamOptimizer(adam_learning_rate,
        adam_beta1, adam_beta2)

if energy_calc_siz == 'sqr':
    denseVec = tf.Variable(tf.random_normal([out_siz]))
    tmpVar = tf.multiply(
            tf.reshape(butterfly_net(trainInData),[-1,out_siz]),
            denseVec)
    tmpVar = tf.reduce_sum( tf.square( tmpVar ), 1)
    y_output = tf.reshape(tmpVar,[-1,1,1])
else:
    denseMat1 = tf.Variable(tf.random_normal([out_siz,energy_calc_siz]))
    bias1 = tf.Variable(tf.random_normal([energy_calc_siz]))
    denseMat2 =  tf.Variable(tf.random_normal([energy_calc_siz,1]))
    tmpVar = tf.matmul(tf.reshape(butterfly_net(trainInData),[-1,out_siz]),
            denseMat1)
    tmpVar = tf.nn.relu( tf.nn.bias_add(tmpVar,bias1))
    y_output = tf.reshape(tf.matmul(tmpVar,denseMat2),[-1,1,1])

loss_train = tf.reduce_mean(tf.squared_difference(trainOutData,
    y_output))

if energy_calc_siz == 'sqr':
    tmpVar = tf.multiply(
            tf.reshape(butterfly_net(testInData),[-1,out_siz]),
            denseVec)
    tmpVar = tf.reduce_sum( tf.square( tmpVar ), 1)
    y_test_output = tf.reshape(tmpVar,[-1,1,1])
else:
    tmpVar = tf.matmul(tf.reshape(butterfly_net(testInData),[-1,out_siz]),
            denseMat1)
    tmpVar = tf.nn.relu( tf.nn.bias_add(tmpVar,bias1))
    y_test_output = tf.reshape(tf.matmul(tmpVar,denseMat2),[-1,1,1])

loss_test = tf.reduce_mean(tf.squared_difference(testOutData,
    y_test_output))

train_step = optimizer_adam.minimize(loss_train)

# Initialize Variables
init = tf.global_variables_initializer()

print("Total Num Paras:  %6d" % ( np.sum( [np.prod(v.get_shape().as_list())
    for v in tf.trainable_variables()]) ))

#=========================================================
#----- Step by Step Training
sess.run(init)

for it in range(max_iter):
    rand_x,rand_y = gen_energy_data(N,batch_siz)
    train_dict = {trainInData: rand_x, trainOutData: rand_y}
    if it % report_freq == 0:
        temp_train_loss = sess.run(loss_train,feed_dict=train_dict)
        print("Iter # %6d: Train Loss: %10e." % (it+1,temp_train_loss))
        sys.stdout.flush()
    sess.run(train_step, feed_dict=train_dict)

# ========= Testing ============
x_test_data_file = Path("./tftmp/x_test_data.npy")
y_test_data_file = Path("./tftmp/y_test_data.npy")
if x_test_data_file.exists() & y_test_data_file.exists():
    x_test_data = np.load(x_test_data_file)
    y_test_data = np.load(y_test_data_file)
else:
    x_test_data,y_test_data = gen_energy_data(N,Ntest)
    np.save(x_test_data_file,x_test_data)
    np.save(y_test_data_file,y_test_data)

test_ys = []
for it in range(Ntest):
    rand_x = x_test_data[it,:,:].reshape((1,-1,1))
    rand_y = y_test_data[it,:,:].reshape((1,-1,1))
    test_dict = {testInData: rand_x, testOutData: rand_y}
    temp_test_loss = sess.run(loss_test,feed_dict=test_dict)
    temp_test_y = sess.run(y_test_output, feed_dict=test_dict)
    test_ys.append(temp_test_y)
    print("Iter # %6d: Test Loss: %10e." % (it+1,temp_test_loss))
    sys.stdout.flush()

if energy_calc_siz == 'sqr':
    np.save("./tftmp/test_BNet_Energy_sqr.npy",test_ys)
else:
    np.save("./tftmp/test_BNet_Energy_"+str(energy_calc_siz)+".npy",test_ys)

sess.close()