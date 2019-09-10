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
import multi_test_CNNNet_inf_uni
from gaussianfun import gaussianfun
from gen_dft_data import gen_degree_data
from CNNLayer import CNNLayer
from ButterflyLayer import ButterflyLayer
json_file = open('paras.json')
paras = json.load(json_file)

N = paras['inputSize']
sig = paras['sigma']
in_siz = paras['inputSize']
out_siz = paras['outputSize']
in_range = np.float32([0,1])
out_range = np.float32([0,out_siz//2])

    
a = np.ones((1,out_siz))
a[0,0]=1
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
    testInData = tf.placeholder(tf.float32, shape=(in_siz,in_siz,1),
                                name="trainInData")
    testOutData = tf.placeholder(tf.float32, shape=(in_siz,out_siz,1),
                                 name="trainOutData")
    #CNN_net = CNNLayer(in_siz, out_siz, False, klvl, alph,
    #                   channel_siz, nlvl, prefixed)
    butterfly_net = ButterflyLayer(in_siz, out_siz, False,
        channel_siz, nlvl, prefixed,
        in_range, out_range)
    
    y_test_output = butterfly_net(testInData)
    #y_test_output = CNN_net(testInData)
    
    init = tf.global_variables_initializer()
    
    sess = tf.Session()
    saver = tf.train.Saver()
    S=20
    q = np.zeros((S+1,out_siz))
    v = np.empty((4,S+1,in_siz))
    x_test,y_test = gen_degree_data(in_siz,in_range,out_siz,out_range)
    for k in range(S+1):
        sess.run(init)
        if k!=S:
            MODEL_SAVE_PATH = "train_model_a-1_3/"
            MODEL_NAME = "fft_"+str(prefixed)+"_"+str(k)+"_model"
            saver.restore(sess, MODEL_SAVE_PATH + MODEL_NAME+".ckpt")
        
        test_dict = {testInData: x_test, testOutData: y_test}
        y_output = sess.run(y_test_output,feed_dict=test_dict)
        
        B=np.zeros([out_siz, in_siz])

        F=np.zeros([out_siz, in_siz])

        for i in range(0,out_siz):
            for j in range(0,in_siz):
                B[i][j] = y_output[j][i][0] 
                F[i][j] = y_test[j][i][0]

        R = F-B
        U, s, V = np.linalg.svd(R, full_matrices=True)
        q[k] = s
        for i in range(4):
            v[i,k] = V[:,i]

    print(q)

    va = np.mean(v[0,0:S],0)
    vo = v[0,S]
    print(np.mean(va))
    print(np.mean(vo))
    u = np.fft.fft(va,in_siz)
    uo = np.fft.fft(vo,in_siz)
    #u_0 = np.fft.fft(v[0,S,:],in_siz,0)
    #print(u_0)
    #vi = np.fft.ifft(va,in_siz,0)
    print(uo)
    e = np.sqrt(np.square(u.real)+np.square(u.imag))
    eo = np.sqrt(np.square(uo.real)+np.square(uo.imag))
    qa = np.mean(q[0:S],0)
    print(e)
    print(eo)
    print(qa)
    print(q[S])
    np.save('3_Kenergy',eo)
    
def main(argv=None):
    tf.reset_default_graph()
    evaluate()

if __name__ == '__main__':
    main()