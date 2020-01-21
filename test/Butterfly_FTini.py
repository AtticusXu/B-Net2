import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
sys.path.insert(0,"../src")
sys.path.insert(0,"../src/data_gen")
import numpy as np
import tensorflow as tf
import json
tf.reset_default_graph() 
from gaussianfun import gaussianfun
from gen_dft_data import gen_uni_data
from ButterflyLayer import ButterflyLayer

json_file = open('paras.json')
paras = json.load(json_file)

#=========================================================
#----- Parameters Setup
N = paras['inputSize']
Ntest = paras['Ntest']
in_siz = paras['inputSize']
out_siz = paras['outputSize']
in_range = np.float32([0,1])
out_range = np.float32([0,out_siz//2])
sig = paras['sigma']
freqidx = range(out_siz//2)
freqmag = np.fft.ifftshift(gaussianfun(np.arange(-N//2,N//2),
                                       [0],[sig]))
freqmag[N//2] = 0

#----- Self-adjusted Parameters of Net
nlvl = paras['nlvl']
prefixed = paras['prefixed']
channel_siz = paras['channelSize'] # Num of interp pts on each dim


print("======== Parameters =========")
print("Channel Size:     %6d" % (channel_siz))
print("Num Levels:       %6d" % (nlvl))
print("Prefix Coef:      %6r" % (prefixed))
print("In Range:        (%6.2f, %6.2f)" % (in_range[0], in_range[1]))
print("Out Range:       (%6.2f, %6.2f)" % (out_range[0], out_range[1]))


x_test,y_test,y_norm = gen_uni_data(freqmag,freqidx,Ntest,sig)
#=========================================================
#----- Variable Preparation
sess = tf.Session()

testInData = tf.placeholder(tf.float32, shape=(Ntest,in_siz,1),
        name="testInData")
testOutData = tf.placeholder(tf.float32, shape=(Ntest,out_siz,1),
        name="testOutData")

#=========================================================
#----- Training Preparation
butterfly_net = ButterflyLayer(in_siz, out_siz,True,
        channel_siz, nlvl, prefixed,
        in_range, out_range)

y_test_output = butterfly_net(tf.convert_to_tensor(x_test))

loss_test = tf.reduce_mean(
        tf.squared_difference(y_test, y_test_output))

# Initialize Variables
init = tf.global_variables_initializer()

print("Total Num Paras:  %6d" % ( np.sum( [np.prod(v.get_shape().as_list())
    for v in tf.trainable_variables()]) ))

#=========================================================
#----- Step by Step Training

sess.run(init)
test_dict = {testInData: x_test, testOutData: y_test}
test_loss = sess.run(loss_test,feed_dict=test_dict)
print("Test Loss: %10e." % (test_loss))

for n in tf.global_variables():
    np.save('tftmp/'+n.name.split(':')[0], n.eval(session=sess))
    print(n.name.split(':')[0] + ' saved')

sess.close()
