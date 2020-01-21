import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
sys.path.insert(0,"../src")
sys.path.insert(0,"../src/data_gen")
import numpy as np
import tensorflow as tf
import json
tf.reset_default_graph() 
from gen_dft_data import gen_energy_uni_data
from CNNLayer import CNNLayer
from ButterflyLayer import ButterflyLayer
from gaussianfun import gaussianfun

tf.reset_default_graph()
json_file = open('paras.json')
paras = json.load(json_file)
#=========================================================
#----- Parameters Setup
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

DW = np.fft.fftfreq(N)
DW[DW==0] = np.inf
DW = 1/DW/N
K = np.zeros(out_siz)
for i in range(out_siz//2):
    K[2*i] = DW[i]
    K[2*i+1] = DW[i]
#----- Tunable Parameters of Net
Butterfly = paras['butterfly']
prefixed = paras['prefixed']
batch_siz = paras['batchSize'] # Batch size during traning
channel_siz = paras['channelSize'] # Num of interp pts on each dim

adam_learning_rate = paras['ADAMparas']['learningRate']
adam_learning_rate_decay = paras['ADAMparas']['learningRatedecay']
adam_beta1 = paras['ADAMparas']['beta1']
adam_beta2 = paras['ADAMparas']['beta2']

max_iter = paras['maxIter'] # Maximum num of iterations
report_freq = paras['reportFreq'] # Frequency of reporting
test_batch_siz = paras['Ntest']
nlvl = paras["nlvl"]
klvl = paras['klvl']
alph = paras['alpha']



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

#=========================================================
#----- Variable Preparation
sess = tf.Session()

trainInData = tf.placeholder(tf.float32, shape=(batch_siz,in_siz,1),
        name="trainInData")
trainOutData = tf.placeholder(tf.float32, shape=(batch_siz,1),
        name="trainOutData")
trainMidData = tf.placeholder(tf.float32, shape=(batch_siz,out_siz,1),
        name="trainMidData")
trainNorm = tf.placeholder(tf.float32, shape=(batch_siz),
        name="trainNorm")
testInData = tf.placeholder(tf.float32, shape=(test_batch_siz,in_siz,1),
        name="testInData")
testOutData = tf.placeholder(tf.float32, shape=(test_batch_siz,1),
        name="testOutData")
testMidData = tf.placeholder(tf.float32, shape=(test_batch_siz,out_siz,1),
        name="testMidData")
testNorm = tf.placeholder(tf.float32, shape=(test_batch_siz),
        name="testNorm")
global_steps=tf.Variable(0, trainable=False)
#=========================================================
#----- Training Preparation
if Butterfly:
    butterfly_net = ButterflyLayer(in_siz, out_siz,False,
                                   channel_siz, nlvl, prefixed,
                                   in_range, out_range)
    mid_output_train = butterfly_net(trainInData)
    mid_output_test = butterfly_net(testInData)
    net_name = "fft_"
else:
    
    CNN_net = CNNLayer(in_siz, out_siz,False,klvl,alph,
                       channel_siz, nlvl, prefixed)
    mid_output_train = CNN_net(trainInData)
    mid_output_test = CNN_net(testInData)
    net_name = "cnn_"

learning_rate = tf.train.exponential_decay(adam_learning_rate,
                                           global_steps,100,
                                           adam_learning_rate_decay)
optimizer_adam = tf.train.AdamOptimizer(adam_learning_rate,
        adam_beta1, adam_beta2)


if energy_calc_siz == 'sqr':
    if prefixed:
        denseVec = tf.Variable(np.float32(K))
    else:
        denseVec = tf.Variable(tf.random_normal([out_siz],0,0.5))
    tmpVar_train = tf.multiply(tf.reshape(mid_output_train,[-1,out_siz]),
            denseVec)
    tmpVar_train = tf.reduce_sum( tf.square( tmpVar_train ), 1)
    y_output_train = tf.reshape(tmpVar_train,[-1,1])
    tmpVar_test = tf.multiply(tf.reshape(mid_output_test,[-1,out_siz]),
            denseVec)
    tmpVar_test = tf.reduce_sum( tf.square( tmpVar_test ), 1)
    y_output_test = tf.reshape(tmpVar_test,[-1,1])
else:
    denseMat1 = tf.Variable(tf.random_normal([out_siz,energy_calc_siz]))
    bias1 = tf.Variable(tf.random_normal([energy_calc_siz]))
    denseMat2 =  tf.Variable(tf.random_normal([energy_calc_siz,1]))
    tmpVar = tf.matmul(tf.reshape(mid_output_train,[-1,out_siz]),
            denseMat1)
    tmpVar = tf.nn.relu( tf.nn.bias_add(tmpVar,bias1))
    y_output = tf.reshape(tf.matmul(tmpVar,denseMat2),[-1,1,1])

L2_loss_train = tf.reduce_mean(tf.divide(tf.sqrt(tf.squared_difference(trainOutData,
    y_output_train)),trainOutData))

loss_train = tf.reduce_mean(tf.squared_difference(trainOutData, y_output_train))

L2_loss_mid_train = tf.reduce_mean(tf.divide(tf.sqrt(tf.reduce_sum(tf.squeeze(
        tf.squared_difference(mid_output_train, trainMidData)),1)),trainNorm))

loss_mid_train =  tf.reduce_mean(tf.squared_difference(mid_output_train,
    trainMidData))

L2_loss_test = tf.reduce_mean(tf.divide(tf.sqrt(tf.squared_difference(testOutData,
    y_output_test)),testOutData))

loss_test = tf.reduce_mean(tf.squared_difference(testOutData, y_output_test))

L2_loss_mid_test = tf.reduce_mean(tf.divide(tf.sqrt(tf.reduce_sum(tf.squeeze(
        tf.squared_difference(mid_output_test, testMidData)),1)),testNorm))

loss_mid_test =  tf.reduce_mean(tf.squared_difference(mid_output_test,
    testMidData))

train_step = optimizer_adam.minimize(loss_train,global_step=global_steps)

# Initialize Variables
init = tf.global_variables_initializer()

print("Total Num Paras:  %6d" % ( np.sum( [np.prod(v.get_shape().as_list())
    for v in tf.trainable_variables()]) ))

#=========================================================
#----- Step by Step Training
saver = tf.train.Saver()
sess.run(init)
MODEL_SAVE_PATH = "train_model_energy/"
MODEL_NAME = net_name+str(prefixed)+"_0_p_model"
for it in range(max_iter):
    rand_x,rand_h,rand_y,ynorm = gen_energy_uni_data(freqmag[0],freqidx,K,batch_siz,sig)

    train_dict = {trainInData: rand_x, trainOutData: rand_y,
                  trainMidData: rand_h, trainNorm: ynorm}
    if it % report_freq == 0:
        [temp_train,temp_train_l2,temp_mid,temp_mid_l2] = sess.run([
                                                loss_train,L2_loss_train,
                                                loss_mid_train,L2_loss_mid_train],
                                    feed_dict=train_dict)
        print("Iter # %6d: Mid loss: %10e Train Loss: %10e." % (it,temp_mid_l2,temp_train_l2))

    sess.run(train_step, feed_dict=train_dict)
saver.save(sess, MODEL_SAVE_PATH + MODEL_NAME+".ckpt")
# ========= Testing ============
test_x,test_h,test_y,test_ynorm = gen_energy_uni_data(freqmag[0],freqidx,K,test_batch_siz,sig)
test_dict = {testInData: test_x, testOutData: test_y,
                  testMidData: test_h, testNorm: test_ynorm}
[test,test_l2,test_mid,test_mid_l2] = sess.run([
                                                loss_test,L2_loss_test,
                                                loss_mid_test,L2_loss_mid_test],
                                    feed_dict=test_dict)
print("test: Mid loss: %10e Train Loss: %10e." % (test_mid_l2,test_l2))

sess.close()