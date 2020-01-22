import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
sys.path.insert(0,"../src")
sys.path.insert(0,"../src/data_gen")
sys.path.insert(0,"../test/train_model")
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
from gen_dft_data import gen_ede_deblur_data, gen_ede_denoise_data
from Bi_ButterflyLayer import ButterflyLayer
from Bi_CNNLayer import CNNLayer
from middle_layer import MiddleLayer

json_file = open('paras.json')
paras = json.load(json_file)
butterfly = paras['butterfly']
input_size = paras['inputSize']
output_size = paras['outputSize']
prefixed = paras['prefixed']
N = input_size
blur_sig = 3
noise = 0.002
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
test_siz = 1
en_nlvl = 4
de_nlvl = 4

MODEL_SAVE_PATH = "train_model_de/"
MODEL_NOISE = "noise_0.002_"
MODEL_NAME_sb = "True_True_model"
MODEL_NAME_sc = "True_False_model"
MODEL_NAME_rb = "False_True_model"
MODEL_NAME_rc = "False_False_model"

x_test,y_test,u_test,xnorm_test,ynorm_test,unorm_test,nnorm,brel = gen_ede_denoise_data(
                                            freqmag,freqidx,test_siz,sig,noise)
np.save(MODEL_SAVE_PATH+MODEL_NOISE+'x_test',x_test)
np.save(MODEL_SAVE_PATH+MODEL_NOISE+'y_test',y_test)
np.save(MODEL_SAVE_PATH+MODEL_NOISE+'u_test',u_test)
np.save(MODEL_SAVE_PATH+MODEL_NOISE+'xnorm_test',xnorm_test)
np.save(MODEL_SAVE_PATH+MODEL_NOISE+'ynorm_test',ynorm_test)
np.save(MODEL_SAVE_PATH+MODEL_NOISE+'unorm_test',unorm_test)
print(brel)

x_test = np.load(MODEL_SAVE_PATH+MODEL_NOISE+'x_test.npy')
y_test = np.load(MODEL_SAVE_PATH+MODEL_NOISE+'y_test.npy')
u_test = np.load(MODEL_SAVE_PATH+MODEL_NOISE+'u_test.npy')
xnorm_test = np.load(MODEL_SAVE_PATH+MODEL_NOISE+'xnorm_test.npy')
ynorm_test = np.load(MODEL_SAVE_PATH+MODEL_NOISE+'ynorm_test.npy')
unorm_test = np.load(MODEL_SAVE_PATH+MODEL_NOISE+'unorm_test.npy')


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



test_dict = {testInData: u_test, testInNorm: [unorm_test],
             testMidData: y_test, testMidNorm: [ynorm_test],
             testOutData: x_test, testOutNorm: [xnorm_test]}

middle_net = MiddleLayer(in_siz, en_mid_siz, de_mid_siz, False)


en_butterfly_net = ButterflyLayer(N, in_siz, en_mid_siz, False,
        channel_siz, en_nlvl, -1, True,
        in_range, en_mid_range,0.45)
de_butterfly_net = ButterflyLayer(N, de_mid_siz, out_siz, False,
        channel_siz, de_nlvl, 1, True,
        de_mid_range, out_range,0.45)
    
y_butterfly_en_mid = en_butterfly_net(testInData)
y_butterfly_de_mid = middle_net(y_butterfly_en_mid)
butterfly_output = de_butterfly_net(y_butterfly_de_mid)/N
x_butterfly_output = butterfly_output[:,::2]


sess = tf.Session()
saver = tf.train.Saver()   
sess.run(init)

saver.restore(sess, MODEL_SAVE_PATH +MODEL_NOISE+ MODEL_NAME_sb+".ckpt")
net_test_sb = sess.run(x_butterfly_output,feed_dict=test_dict)
f_net_sb = np.reshape(net_test_sb,[input_size])
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



test_dict = {testInData: u_test, testInNorm: [unorm_test],
             testMidData: y_test, testMidNorm: [ynorm_test],
             testOutData: x_test, testOutNorm: [xnorm_test]}
middle_net = MiddleLayer(in_siz, en_mid_siz, de_mid_siz, False)


en_butterfly_net = ButterflyLayer(N, in_siz, en_mid_siz, False,
        channel_siz, en_nlvl, -1, False,
        in_range, en_mid_range,0.45)
de_butterfly_net = ButterflyLayer(N, de_mid_siz, out_siz, False,
        channel_siz, de_nlvl, 1, False,
        de_mid_range, out_range,0.45)

y_butterfly_en_mid = en_butterfly_net(testInData)
y_butterfly_de_mid = middle_net(y_butterfly_en_mid)
butterfly_output = de_butterfly_net(y_butterfly_de_mid)/N
x_butterfly_output = butterfly_output[:,::2]

sess = tf.Session()
saver = tf.train.Saver()   
sess.run(init)

saver.restore(sess, MODEL_SAVE_PATH +MODEL_NOISE + MODEL_NAME_rb+".ckpt")
net_test_rb = sess.run(x_butterfly_output,feed_dict=test_dict)
f_net_rb = np.reshape(net_test_rb,[input_size])
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



test_dict = {testInData: u_test, testInNorm: [unorm_test],
             testMidData: y_test, testMidNorm: [ynorm_test],
             testOutData: x_test, testOutNorm: [xnorm_test]}
middle_net = MiddleLayer(in_siz, en_mid_siz, de_mid_siz, False)
en_cnn_net = CNNLayer(N, in_siz, en_mid_siz, 
        channel_siz, en_nlvl, -1, True,0.3)
de_cnn_net = CNNLayer(N, de_mid_siz, out_siz,
        channel_siz, de_nlvl, 1, True,0.3)
    
    
y_cnn_en_mid = en_cnn_net(testInData)
y_cnn_de_mid = middle_net(y_cnn_en_mid)
cnn_output = de_cnn_net(y_cnn_de_mid)/N
x_cnn_output = cnn_output[:,::2]
sess = tf.Session()
saver = tf.train.Saver()  
sess.run(init)

saver.restore(sess, MODEL_SAVE_PATH+MODEL_NOISE + MODEL_NAME_sc+".ckpt")
net_test_sc = sess.run(x_cnn_output,feed_dict=test_dict)
f_net_sc = np.reshape(net_test_sc,[input_size])
sess.close()
sess = tf.Session()
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



test_dict = {testInData: u_test, testInNorm: [unorm_test],
             testMidData: y_test, testMidNorm: [ynorm_test],
             testOutData: x_test, testOutNorm: [xnorm_test]}
middle_net = MiddleLayer(in_siz, en_mid_siz, de_mid_siz, False)
en_cnn_net = CNNLayer(N, in_siz, en_mid_siz, 
        channel_siz, en_nlvl, -1, False,0.3)
de_cnn_net = CNNLayer(N, de_mid_siz, out_siz,
        channel_siz, de_nlvl, 1, False,0.3)
    
    
y_cnn_en_mid = en_cnn_net(testInData)
y_cnn_de_mid = middle_net(y_cnn_en_mid)
cnn_output = de_cnn_net(y_cnn_de_mid)/N
x_cnn_output = cnn_output[:,::2]
sess = tf.Session()
saver = tf.train.Saver()  
sess.run(init)

saver.restore(sess, MODEL_SAVE_PATH+MODEL_NOISE + MODEL_NAME_rc+".ckpt")
net_test_rc = sess.run(x_cnn_output,feed_dict=test_dict)
f_net_rc = np.reshape(net_test_rc,[input_size])
sess.close()


f_noise = np.reshape(u_test[:,::2],[input_size])
f_true = np.reshape(x_test,[input_size])
k = np.arange(0,1,1/128)
f = plt.figure(figsize=(16,8))

p1 = plt.subplot(121,aspect=1/0.4)
p2 = plt.subplot(122,aspect=0.1/0.05)

p1.plot(k,f_true,'g',label='f',linewidth=0.5)
p1.plot(k,f_noise,'r',label='f_noise',linewidth=0.5)
p1.plot(k,f_net_rc,'b',label='ran_ini CNN',linewidth=0.5)
p1.plot(k,f_net_rb,'y',label='ran_ini csCNN',linewidth=0.5)
p1.plot(k,f_net_sc,'*c',label='str_ini CNN',markersize=1)
p1.plot(k,f_net_sb,'.m',label='str_ini csCNN',markersize=1)

p2.plot(k,f_true,'g',label='f',linewidth=1)
p2.plot(k,f_noise,'r',label='f_noise',linewidth=1)
p2.plot(k,f_net_rc,'b-',label='CNN-rand',linewidth=1)
p2.plot(k,f_net_rb,'y',label='BNet2-rand',linewidth=1)
p2.plot(k,f_net_sc,'*c',label='CNN-FT',markersize=10)
p2.plot(k,f_net_sb,'.m',label='BNet2-FT',markersize=10)

la = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 22,
}


p1.axis([-0.0,1.0,-0.12,0.18])
p1.set_ylabel("f",la)
p1.set_xlabel("t",la)
p1.grid(True)
p1.tick_params(labelsize=20)

p2.axis([0.72,0.82,-0.1105,-0.073])
p2.set_ylabel("f",la)
p2.set_xlabel("t",la)
p2.grid(True)
p2.legend(prop=la)
p2.tick_params(labelsize=20)
tx0 = 0.72
tx1 = 0.82
ty0 = -0.1105
ty1 = -0.073
sx = [tx0,tx1,tx1,tx0,tx0]
sy = [ty0,ty0,ty1,ty1,ty0]
p1.plot(sx,sy,"green")
plt.tight_layout()
plt.savefig("denoise.eps")
