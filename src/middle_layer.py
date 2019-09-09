import math
import numpy as np
import tensorflow as tf

from LagrangeMat import LagrangeMat

class MiddleLayer(tf.keras.layers.Layer):
    #==================================================================
    # Initialize parameters in the layer
    def __init__(self, in_siz, en_mid_siz, prefixed = 0, std = 0.5):
        super(MiddleLayer, self).__init__()
        self.in_siz         = in_siz
        self.en_mid_siz     = en_mid_siz
        self.std            = std
        self.de_mid_siz     = min(in_siz,en_mid_siz*2)

        if prefixed == 0:
            self.buildrandom()
        elif prefixed == 1:
            self.buildidentity()
    
    def call(self, in_data):
        in_siz = self.in_siz
        en_mid_siz =  self.en_mid_siz
        de_mid_siz = self.de_mid_siz
        half = min(en_mid_siz//2,in_siz//4)
        
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
        en_mid_data = tf.concat([en_mid_data_r, en_mid_data_i],2)
        de_mid_data = tf.multiply(en_mid_data,self.mid_DenseVar)
        de_mid_data = tf.reshape(de_mid_data,(np.size(in_data,0),de_mid_siz,1))
        return(de_mid_data)
        
    def buildrandom(self):
        std = self.std
        de_mid_siz = self.de_mid_siz
        # Mid Layer
        self.mid_DenseVar = tf.Variable(tf.random_normal(
                [1,de_mid_siz//2,2],0,std),name = "Dense_mid_ran")
        tf.summary.histogram("Dense_mid_ran", self.mid_DenseVar)
        
    def buildidentity(self):
        de_mid_siz = self.de_mid_siz
        mat = np.ones((1,de_mid_siz//2,2))
        self.mid_DenseVar = tf.Variable(mat.astype(np.float32),
                                        name = "Dense_mid_str")
        tf.summary.histogram("Dense_mid_str", self.mid_DenseVar)

        