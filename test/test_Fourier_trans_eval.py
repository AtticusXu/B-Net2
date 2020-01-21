import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
sys.path.insert(0,"../src")
sys.path.insert(0,"../src/data_gen")
sys.path.insert(0,"../test/train_model")
import numpy as np

import tensorflow as tf
import json
from gaussianfun import gaussianfun
from gen_dft_data import gen_uni_data
from CNNLayer import CNNLayer
from ButterflyLayer import ButterflyLayer

json_file = open('paras.json')
paras = json.load(json_file)
#=========================================================
#----- Parameters Setup
S=20
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
    
#----- Self-adjusted Parameters of Net
Butterfly = paras['butterfly']
nlvl = paras['nlvl']# Num of levels of the BF struct
klvl = paras['klvl']
alph = paras['alpha']
prefixed = paras['prefixed']

channel_siz = paras['channelSize'] # Num of interp pts on each dim

batch_siz = paras['batchSize'] # Batch size during traning

test_batch_siz = paras['Ntest']


print("======== Parameters =========")
print("Batch Size:       %6d" % (batch_siz))
print("Channel Size:     %6d" % (channel_siz))
print("Alpha:            %6d" % (alph))
print("Num Levels:       %6d" % (nlvl))
print("K Levels:         %6d" % (klvl))
print("Prefix Coef:      %6r" % (prefixed))
print("In Range:        (%6.2f, %6.2f)" % (in_range[0], in_range[1]))
print("Out Range:       (%6.2f, %6.2f)" % (out_range[0], out_range[1]))


def evaluate():
    testInData = tf.placeholder(tf.float32, shape=(test_batch_siz,in_siz,1),
                                name="trainInData")
    testOutData = tf.placeholder(tf.float32, shape=(test_batch_siz,out_siz,1),
                                 name="trainOutData")
    testNorm = tf.placeholder(tf.float32, shape=(test_batch_siz),
                              name="testNorm")
    if Butterfly:
        butterfly_net = ButterflyLayer(in_siz, out_siz, False,
                                       channel_siz, nlvl, prefixed,
                                       in_range, out_range)
    
        y_test_output = butterfly_net(testInData)
        net_name = "fft_"
    else:
        
        CNN_net = CNNLayer(in_siz, out_siz, False, klvl, alph,
                           channel_siz, nlvl, prefixed)
    
        y_test_output = CNN_net(testInData)
        net_name = "cnn_"
    
    L2_loss_test = tf.reduce_mean(tf.divide(tf.sqrt(tf.reduce_sum(tf.squeeze(
            tf.squared_difference(testOutData, y_test_output)),1)),testNorm))
    
    MSE_loss_test = tf.reduce_mean(
        tf.squared_difference(testOutData, y_test_output))

    y_norm_test = tf.reduce_mean(tf.sqrt(tf.reduce_sum(
        tf.square(testOutData),1)))

    
    init = tf.global_variables_initializer()
    
    sess = tf.Session()
    saver = tf.train.Saver()
    
    mk_test_loss_list = np.zeros((5*out_siz//2-3,S))
    mk_test_loss_l2_list = np.zeros((5*out_siz//2-3,S))
    for s in range(S):
        sess.run(init)
        MODEL_SAVE_PATH = "train_model/"
        MODEL_NAME = net_name+str(prefixed)+"_"+str(s)+"_model"
        saver.restore(sess, MODEL_SAVE_PATH + MODEL_NAME+".ckpt")
        for j in range(5*out_siz//2-3):
            x_test,y_test,y_norm = gen_uni_data(freqmag[j],
                                                freqidx,test_batch_siz,sig)
            test_dict = {testInData: x_test, testOutData: y_test,
                         testNorm: y_norm}
            [test_loss, test_norm, test_z_loss_k] = sess.run(
            [MSE_loss_test, y_norm_test, L2_loss_test],
            feed_dict=test_dict)
        
            mk_test_loss_list[j,s] = test_loss
            mk_test_loss_l2_list[j,s] = np.sqrt(test_loss)/test_norm

    print(mk_test_loss_list)
    np.save('train_model/'+net_name+'mk_test_loss_list_'+str(prefixed), 
            mk_test_loss_list)
    np.save('train_model/'+net_name+'mk_test_loss_l2_list_'+str(prefixed), 
            mk_test_loss_l2_list)

def main(argv=None):
    tf.reset_default_graph()
    evaluate()

if __name__ == '__main__':
    main()