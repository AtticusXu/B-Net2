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
import multi_test_ButterflyNet_inf_uni
from gaussianfun import gaussianfun
from gen_dft_data import gen_uni_data
from ButterflyLayer import ButterflyLayer

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


def evaluate():
    testInData = tf.placeholder(tf.float32, shape=(test_batch_siz,in_siz,1),
                                name="trainInData")
    testOutData = tf.placeholder(tf.float32, shape=(test_batch_siz,out_siz,1),
                                 name="trainOutData")
    testNorm = tf.placeholder(tf.float32, shape=(test_batch_siz),
                              name="testNorm")
    butterfly_net = ButterflyLayer(in_siz, out_siz, False,
        channel_siz, nlvl, prefixed,
        in_range, out_range)
    
    y_test_output = butterfly_net(testInData)
    
    L2_loss_test = tf.reduce_mean(tf.divide(tf.sqrt(tf.reduce_sum(tf.squeeze(
            tf.squared_difference(testOutData, y_test_output)),1)),testNorm))
    
    L2_loss_test_list = tf.divide(tf.sqrt(tf.reduce_sum(tf.squeeze(
            tf.squared_difference(testOutData, y_test_output)),1)),testNorm)
    
    Sqr_loss_test_K = tf.reduce_mean(tf.squeeze(
            tf.squared_difference(testOutData, y_test_output)),0)
    
    Sqr_test_norm_K = tf.reduce_mean(tf.squeeze(tf.square(testOutData)),0)
    
    init = tf.global_variables_initializer()
    
    sess = tf.Session()
    saver = tf.train.Saver()
    S=2
    mk_test_loss_list = np.zeros((5*out_siz//2-3,S))
    epochs = np.linspace(0,max_iter,max_iter//record_freq)
    mk_test_loss_klist = np.zeros((S,5*out_siz//2-3,out_siz//2))
    for s in range(S):
        sess.run(init)
        MODEL_SAVE_PATH = "train_model/"
        MODEL_NAME = "fft_"+str(prefixed)+"_"+str(s)+"_model"
        saver.restore(sess, MODEL_SAVE_PATH + MODEL_NAME+".ckpt")
        for j in range(5*out_siz//2-3):
            x_test,y_test,y_norm = gen_uni_data(freqmag[j],
                                                freqidx,test_batch_siz,sig)
            test_dict = {testInData: x_test, testOutData: y_test,
                         testNorm: y_norm}
            [test_loss, test_loss_list, test_loss_k,K_norm] = sess.run(
            [L2_loss_test, L2_loss_test_list, Sqr_loss_test_K,Sqr_test_norm_K],
            feed_dict=test_dict)
        
            mk_test_loss_list[j,s] = test_loss
            #print("Test Loss: %10e." % (test_loss))
            
            for i in range(8):
                K_norm[i] = np.sqrt(K_norm[2*i] + K_norm[2*i+1])
            #   K_list[s,i,:] = np.sqrt(K_list[s,2*i,:] + K_list[s,2*i+1,:])\
            #                   /K_norm[i]
                mk_test_loss_klist[s,j,i] = np.sqrt(test_loss_k[2*i] + 
                                  test_loss_k[2*i+1])/K_norm[i]
            
        #err_list = np.log10(err_list)
        #K_list = np.log10(K_list)
        #mk_test_loss_list = np.mean(np.log10(mk_test_loss_list),axis = 1)
        #print(test_loss_klist)
        #print(K_norm[0:out_siz//2])
    print(mk_test_loss_list)
    np.save('train_model/fft_mk_test_loss_list_'+str(prefixed)+"_"+str(s), 
            mk_test_loss_list)
    np.save('train_model/fft_mk_test_loss_klist_'+str(prefixed)+"_"+str(s), 
            mk_test_loss_klist)
    #for k in range(out_siz//2):
    #    fig = plt.figure(k,figsize=(10,8))
    #    for s in range(S):
    #        plt.plot(epochs, K_list[s,k], 'r', label = 'k = '+ str(k)+')')
    #    plt.title('CNN_Training Error Plot(k='+ str(k)+')')
    #    plt.savefig("CNN_Train_Error_"+ str(prefixed)+"_"+str(k)+".png" )
def main(argv=None):
    tf.reset_default_graph()
    evaluate()

if __name__ == '__main__':
    main()
    
