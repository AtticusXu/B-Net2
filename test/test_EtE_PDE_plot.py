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
    freqmag[i] = 1
freqmag[N//2] = 0
N_0 = 2**10
a = np.ones(N_0+1)
m = N_0//4
for j in range(N_0+1):
    if (j-m//2)%(2*m) < m:
        a[j] = 10
b = 10**4
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
test_siz = 1
en_nlvl = 4
de_nlvl = 4
beta = 10**3
dir_mat = DirSineElliptic(a[::(2**10//N)],N)
dir_mat_train = np.tile(dir_mat,(batch_siz,1,1))
dir_mat_test = np.tile(dir_mat,(test_siz,1,1))

MODEL_SAVE_PATH = "train_model_equations/"
MODEL_NAME_sb = "equations_True_True_True_model"
MODEL_NAME_sc = "equations_True_True_False_model"
MODEL_NAME_rb = "equations_True_False_True_model"
MODEL_NAME_rc = "equations_True_False_False_model"


f_test,y_test,u_test,fnorm_test,ynorm_test,unorm_test = gen_ede_Ell_data(
            test_siz,freqidx,freqmag,a)

np.save(MODEL_SAVE_PATH+'f_test',f_test)
np.save(MODEL_SAVE_PATH+'y_test',y_test)
np.save(MODEL_SAVE_PATH+'u_test',u_test)
np.save(MODEL_SAVE_PATH+'fnorm_test',fnorm_test)
np.save(MODEL_SAVE_PATH+'ynorm_test',ynorm_test)
np.save(MODEL_SAVE_PATH+'unorm_test',unorm_test)


f_test = np.load(MODEL_SAVE_PATH+'f_test.npy')
y_test = np.load(MODEL_SAVE_PATH+'y_test.npy')
u_test = np.load(MODEL_SAVE_PATH+'u_test.npy')
fnorm_test = np.load(MODEL_SAVE_PATH+'fnorm_test.npy')
ynorm_test = np.load(MODEL_SAVE_PATH+'ynorm_test.npy')
unorm_test = np.load(MODEL_SAVE_PATH+'unorm_test.npy')
#=========================================================
#----- Variable Preparation

init = tf.global_variables_initializer()
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



test_dict = {testInData: f_test, testInNorm: [fnorm_test],
             testMidData: y_test, testMidNorm: [ynorm_test],
             testOutData: u_test, testOutNorm: [unorm_test]}

middle_net = MiddleLayer(in_siz, en_mid_siz, de_mid_siz, True,
                         a[::(2**10//N)], True, std = 0.1)

en_butterfly_net = ButterflyLayer(2*N, in_siz, en_mid_siz, False,
        channel_siz, en_nlvl, -1, True,
        in_range, en_mid_range,0.45)
de_butterfly_net = ButterflyLayer(N, de_mid_siz, out_siz, False,
        channel_siz, de_nlvl, 1, True,
        de_mid_range, out_range,0.45)
    
y_test_en_mid = en_butterfly_net(testInData)
y_test_de_mid = middle_net(y_test_en_mid)
y_test_output = de_butterfly_net(y_test_de_mid)/N
y_test_output_i = -y_test_output[:,1::2,:]



sess = tf.Session()
saver = tf.train.Saver()   
sess.run(init)
saver.restore(sess, MODEL_SAVE_PATH + MODEL_NAME_sb+".ckpt")
net_test_sb = sess.run(y_test_output_i,feed_dict=test_dict)
f_net_sb = np.reshape(net_test_sb,[input_size//2])
sess.close()
tf.reset_default_graph() 
#=========================================================
init = tf.global_variables_initializer()
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



test_dict = {testInData: f_test, testInNorm: [fnorm_test],
             testMidData: y_test, testMidNorm: [ynorm_test],
             testOutData: u_test, testOutNorm: [unorm_test]}

middle_net = MiddleLayer(in_siz, en_mid_siz, de_mid_siz, True,
                         a[::(2**10//N)], False, std = 0.1)


en_butterfly_net = ButterflyLayer(2*N, in_siz, en_mid_siz, False,
        channel_siz, en_nlvl, -1, False,
        in_range, en_mid_range,0.45)
de_butterfly_net = ButterflyLayer(N, de_mid_siz, out_siz, False,
        channel_siz, de_nlvl, 1, False,
        de_mid_range, out_range,0.45)
    
y_test_en_mid = en_butterfly_net(testInData)
y_test_de_mid = middle_net(y_test_en_mid)
y_test_output = de_butterfly_net(y_test_de_mid)/N
y_test_output_i = -y_test_output[:,1::2,:]



sess = tf.Session()
saver = tf.train.Saver()   
sess.run(init)
saver.restore(sess, MODEL_SAVE_PATH + MODEL_NAME_rb+".ckpt")
net_test_rb = sess.run(y_test_output_i,feed_dict=test_dict)
f_net_rb = np.reshape(net_test_rb,[input_size//2])
sess.close()
tf.reset_default_graph() 
#=========================================================
init = tf.global_variables_initializer()
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



test_dict = {testInData: f_test, testInNorm: [fnorm_test],
             testMidData: y_test, testMidNorm: [ynorm_test],
             testOutData: u_test, testOutNorm: [unorm_test]}

middle_net = MiddleLayer(in_siz, en_mid_siz, de_mid_siz, True,
                         a[::(2**10//N)], True, std = 0.1)
en_cnn_net = CNNLayer(2*N, in_siz, en_mid_siz, 
        channel_siz, en_nlvl, -1, True,0.3)
de_cnn_net = CNNLayer(N, de_mid_siz, out_siz,
        channel_siz, de_nlvl, 1, True,0.3)
    
    
y_test_en_mid = en_cnn_net(testInData)
y_test_de_mid = middle_net(y_test_en_mid)
y_test_output = de_cnn_net(y_test_de_mid)/N
y_test_output_i = -y_test_output[:,1::2,:]

sess = tf.Session()
saver = tf.train.Saver()  
sess.run(init)
saver.restore(sess, MODEL_SAVE_PATH + MODEL_NAME_sc+".ckpt")
net_test_sc = sess.run(y_test_output_i,feed_dict=test_dict)
f_net_sc = np.reshape(net_test_sc,[input_size//2])
sess.close()
tf.reset_default_graph() 
#=========================================================
init = tf.global_variables_initializer()
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



test_dict = {testInData: f_test, testInNorm: [fnorm_test],
             testMidData: y_test, testMidNorm: [ynorm_test],
             testOutData: u_test, testOutNorm: [unorm_test]}

middle_net = MiddleLayer(in_siz, en_mid_siz, de_mid_siz, True,
                         a[::(2**10//N)], False, std = 0.1)
en_cnn_net = CNNLayer(2*N, in_siz, en_mid_siz, 
        channel_siz, en_nlvl, -1, False,0.3)
de_cnn_net = CNNLayer(N, de_mid_siz, out_siz,
        channel_siz, de_nlvl, 1, False,0.3)
    
    
y_test_en_mid = en_cnn_net(testInData)
y_test_de_mid = middle_net(y_test_en_mid)
y_test_output = de_cnn_net(y_test_de_mid)/N
y_test_output_i = -y_test_output[:,1::2,:]

sess = tf.Session()
saver = tf.train.Saver()  
sess.run(init)
saver.restore(sess, MODEL_SAVE_PATH + MODEL_NAME_rc+".ckpt")
net_test_rc = sess.run(y_test_output_i,feed_dict=test_dict)
f_net_rc = np.reshape(net_test_rc,[input_size//2])
sess.close()




u_true = np.reshape(100*u_test,[input_size//2])
k = np.arange(0,1,1/64)
f = plt.figure(figsize=(16,8))

p1 = plt.subplot(121,aspect=1/0.4)
p2 = plt.subplot(122,aspect=0.1/0.04)

p1.plot(k,u_true,'g',label='u',linewidth=0.5)

p1.plot(k,100*f_net_rc,'b',label='$CNN-rand$',linewidth=0.5)
p1.plot(k,100*f_net_rb,'y',label='$BNet2-rand$',linewidth=0.5)
p1.plot(k,100*f_net_sc,'*c',label='$CNN-FT$',markersize=1)
p1.plot(k,100*f_net_sb,'.m',label='$BNet2-FT$',markersize=1)

p2.plot(k,u_true,'g',label='u',linewidth=1)

p2.plot(k,100*f_net_rc,'b',label='CNN-rand',linewidth=1)
p2.plot(k,100*f_net_rb,'y',label='BNet2-rand',linewidth=1)
p2.plot(k,100*f_net_sc,'*c',label='CNN-FT',markersize=10)
p2.plot(k,100*f_net_sb,'.m',label='BNet2-FT',markersize=10)

la = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 25,
}

p1.axis([0.0,1.0,-0.3,0.08])
p1.set_ylabel("u",la )
p1.set_xlabel("t",la )
p1.grid(True)

p1.tick_params(labelsize=18)
p2.axis([0.78,0.9,-0.286,-0.24])
p2.set_ylabel("u",la )
p2.set_xlabel("t",la )
p2.grid(True)
p2.legend(prop=la)
p2.tick_params(labelsize=18)
tx0 = 0.78
tx1 = 0.9
ty0 = -0.286
ty1 = -0.24
sx = [tx0,tx1,tx1,tx0,tx0]
sy = [ty0,ty0,ty1,ty1,ty0]
p1.plot(sx,sy,"green")
plt.tight_layout()
plt.savefig("linear_ode.png")

