import math
import numpy as np
import tensorflow as tf

from LagrangeMat import LagrangeMat
from ODE_matrix import Inv_net_SineElliptic
class MiddleLayer(tf.keras.layers.Layer):
    #==================================================================
    # Initialize parameters in the layer
    def __init__(self, in_siz, en_mid_siz, de_mid_siz, sine = True,
                 a=[1],prefixed = False, std = 0.5):
        super(MiddleLayer, self).__init__()
        self.in_siz         = in_siz
        self.en_mid_siz     = en_mid_siz
        self.de_mid_siz     = de_mid_siz
        self.std            = std        
        self.sine           = sine
        self.prefixed       = prefixed
        self.a              = a
        if self.prefixed:
            self.buildSineInverse()            
        else:
            self.buildrandom()

            
            
    
    def call(self, in_data):
        
        in_siz = self.in_siz
        en_mid_siz =  self.en_mid_siz
        de_mid_siz = self.de_mid_siz
        sine = self.sine
        half = min(en_mid_siz//2,in_siz//2)
        if sine:
            en_mid_data = tf.reshape(in_data,(np.size(in_data,0),
                                              en_mid_siz//2,2))
            en_mid_data = tf.reshape(-en_mid_data[:,:,1],(np.size(in_data,0),
                                              en_mid_siz//2,1))
            de_mid_data_r = np.reshape([], (0, de_mid_siz))
            for i in range(np.size(in_data,0)):
                tmpVar = en_mid_data[i]
                tmpVar = -tf.matmul(self.mid_DenseVar_relu,tmpVar)
                tmpVar = tf.reshape(tmpVar,(1,de_mid_siz))
                de_mid_data_r = tf.concat([de_mid_data_r, tmpVar], axis=0)
            de_mid_data_r = tf.nn.relu(tf.nn.bias_add(de_mid_data_r, self.mid_Bias))

            de_mid_data_r = tf.add(-de_mid_data_r[:,de_mid_siz//2:],
                                   de_mid_data_r[:,:de_mid_siz//2])
            de_mid_data_r = tf.reshape(de_mid_data_r,
                                       (np.size(in_data,0), de_mid_siz//2, 1))
            de_mid_data_i = np.zeros((np.size(in_data,0),de_mid_siz//2,1))
            de_mid_data = tf.reshape(tf.concat((de_mid_data_r,de_mid_data_i),2),
                                     (np.size(in_data,0), de_mid_siz, 1))
            return(de_mid_data)
            
        else:
            en_mid_data = tf.reshape(in_data,(np.size(in_data,0),
                                              en_mid_siz//2,2))
            en_mid_data_r = en_mid_data[:,:,0]
            en_mid_data_i = en_mid_data[:,:,1]
            en_mid_data_r = tf.reshape(tf.concat([np.zeros((np.size(in_data,0),1)),
                                    en_mid_data_r[:,half-1:0:-1],
                                    en_mid_data_r[:,0:half]],1),
                                    (np.size(in_data,0),de_mid_siz//2,1))
            en_mid_data_i = tf.reshape(tf.concat([np.zeros((np.size(in_data,0),1)),
                                    -en_mid_data_i[:,half-1:0:-1],
                                    en_mid_data_i[:,0:half]],1),
                                    (np.size(in_data,0),de_mid_siz//2,1))
            de_mid_data = tf.reshape(tf.concat([en_mid_data_r, en_mid_data_i],2),
                                     (np.size(in_data,0),de_mid_siz,1))
            
            return(de_mid_data)
        
            
        
    def buildrandom(self):
        std = self.std
        en_mid_siz = self.en_mid_siz
        de_mid_siz = self.de_mid_siz
        # Mid Layer
        self.mid_DenseVar_relu = tf.Variable(tf.random_normal(
                [de_mid_siz,en_mid_siz//2],0,std),name = "Dense_mid_ran")
        self.mid_Bias = tf.Variable(tf.zeros([de_mid_siz]),
                                        name = "Bias_mid_str")

        
    def buildidentity(self):
        de_mid_siz = self.de_mid_siz
        mat = np.ones((1,de_mid_siz//2,2))
        self.mid_DenseVar = tf.Variable(mat.astype(np.float32),
                                        name = "Dense_mid_str")
        tf.summary.histogram("Dense_mid_str", self.mid_DenseVar)
        

    
    def buildSineInverse(self):
        en_mid_siz = self.en_mid_siz
        de_mid_siz = self.de_mid_siz
        in_siz = self.in_siz
        N = in_siz//4
        K1 = en_mid_siz//2
        K2 = de_mid_siz//2
        a = self.a
        mat = Inv_net_SineElliptic(a,N)
        mat = mat[:K2,:K1]
        mat_relu = np.empty((2*K2,K1))
        mat_relu[:K2,:] = mat
        mat_relu[K2:,:] = -mat
        b_relu = np.zeros(2*K2)
        self.mid_DenseVar_relu = tf.Variable(mat_relu.astype(np.float32),
                                        name = "Dense_mid_str")
        self.mid_Bias = tf.Variable(b_relu.astype(np.float32),
                                        name = "Bias_mid_str")
        
        