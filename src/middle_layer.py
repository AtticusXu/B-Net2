import math
import numpy as np
import tensorflow as tf

from LagrangeMat import LagrangeMat

class MiddleLayer(tf.keras.layers.Layer):
    #==================================================================
    # Initialize parameters in the layer
    def __init__(self, in_siz, en_mid_siz, a, sine = True, prefixed = 0, std = 0.5):
        super(MiddleLayer, self).__init__()
        self.in_siz         = in_siz
        self.en_mid_siz     = en_mid_siz
        self.std            = std
        self.de_mid_siz     = min(in_siz,en_mid_siz*2)
        self.sine           = sine
        self.prefixed       = prefixed
        self.a              = a
        if self.prefixed == 0:
            self.buildrandom()
        elif self.prefixed == 1:
            self.buildidentity()
        elif self.prefixed == 2:
            self.buildSineInverse()
            
    
    def call(self, in_data):
        
        in_siz = self.in_siz
        en_mid_siz =  self.en_mid_siz
        de_mid_siz = self.de_mid_siz
        prefixed = self.prefixed
        sine = self.sine
        half = min(en_mid_siz//2,in_siz//4)
        if sine:
            en_mid_data = tf.reshape(in_data,(np.size(in_data,0),
                                              en_mid_siz//2,2))
            en_mid_data = -en_mid_data[:,:,1]
            
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
            en_mid_data = tf.concat([en_mid_data_r, en_mid_data_i],2)
        
        if prefixed == 1:
            de_mid_data = tf.multiply(en_mid_data,self.mid_DenseVar)
            de_mid_data = tf.reshape(de_mid_data,(np.size(in_data,0),de_mid_siz,1))
            return(de_mid_data)
        elif sine:
            de_mid_data_i = np.reshape([], (0, en_mid_siz//2,1))
            for i in range(np.size(in_data,0)):
                tmpVar = en_mid_data[i]
                tmpVar = tf.matmul(self.mid_DenseVar,tmpVar)
                tmpVar = tf.reshape(tmpVar,(1,de_mid_siz,1))
                de_mid_data_i = tf.concat([de_mid_data_i, tmpVar], axis=0)
            de_mid_data_r = np.zeros_like(de_mid_data_i)
            de_mid_data = tf.reshape(tf.concat([de_mid_data_r,de_mid_data_i],2),
                                     (np.size(in_data,0), en_mid_siz, 1))
            return(de_mid_data)
        else:
            en_mid_data = tf.reshape(en_mid_data,
                                     (np.size(in_data,0),de_mid_siz,1))
            de_mid_data = np.reshape([], (0, de_mid_siz,1))
            for i in range(np.size(in_data,0)):
                tmpVar = en_mid_data[i]
                tmpVar = tf.matmul(self.mid_DenseVar,tmpVar)
                tmpVar = tf.reshape(tmpVar,(1,de_mid_siz,1))
                de_mid_data = tf.concat([de_mid_data, tmpVar], axis=0)
            return(de_mid_data)
        
    def buildrandom(self):
        std = self.std
        de_mid_siz = self.de_mid_siz
        # Mid Layer
        self.mid_DenseVar = tf.Variable(tf.random_normal(
                [de_mid_siz,de_mid_siz],0,std),name = "Dense_mid_ran")
        tf.summary.histogram("Dense_mid_ran", self.mid_DenseVar)
        
    def buildidentity(self):
        de_mid_siz = self.de_mid_siz
        mat = np.ones((1,de_mid_siz//2,2))
        self.mid_DenseVar = tf.Variable(mat.astype(np.float32),
                                        name = "Dense_mid_str")
        tf.summary.histogram("Dense_mid_str", self.mid_DenseVar)
        
    def buildWinverse(self):
        de_mid_siz = self.de_mid_siz
        a = self.a
        # Mid Layer
        k = np.zeros(de_mid_siz)
        for i in range(de_mid_siz//2):
            k[2*i] = i-de_mid_siz//4
            k[2*i+1] = i-de_mid_siz//4
        mat1 = np.empty((de_mid_siz,de_mid_siz))
        for i in range(0,de_mid_siz//2):
            for j in range(0,de_mid_siz//2):
                p = (i-j + de_mid_siz//4)%(de_mid_siz//2)
                mat1[2*i,2*j] = -a[p*2]*k[p*2]*k[2*j]
                mat1[2*i,2*j+1] = a[p*2+1]*k[p*2+1]*k[2*j+1]
                mat1[2*i+1,2*j] = -a[p*2+1]*k[p*2+1]*k[2*j]
                mat1[2*i+1,2*j+1] = -a[p*2]*k[p*2]*k[2*j+1]
        #print(mat1[0])
        mat2 = np.empty((de_mid_siz,de_mid_siz))
        for i in range(0,de_mid_siz//2):
            for j in range(0,de_mid_siz//2):
                p = (i-j + de_mid_siz//4)%(de_mid_siz//2)
                mat2[2*i,2*j] = -a[p*2]*k[2*j]**2
                mat2[2*i,2*j+1] = a[p*2+1]*k[2*j+1]**2
                mat2[2*i+1,2*j] = -a[p*2+1]*k[2*j]**2
                mat2[2*i+1,2*j+1] = -a[p*2]*k[2*j+1]**2
        #print(mat2[0])
        mat = mat1 + mat2
        #print(mat)
        #mat =np.linalg.inv(mat)
        self.mid_DenseVar = tf.Variable(mat.astype(np.float32),
                                        name = "Dense_mid_Win")
        tf.summary.histogram("Dense_mid_ran", self.mid_DenseVar)
    
    def buildSineInverse(self):
        en_mid_siz = self.en_mid_siz
        N = en_mid_siz//2
        a = self.a
        mat_a = np.zeros(N,N)
        mat_k = np.zeros(N,N)
        mat_C = np.empty(N,N)
        mat_I = np.zeros(N,N)
        for i in range(N):
            for j in range(N):
                mat_a[i][i] = a[i]
                mat_k[i][i] = i * np.pi/N
                mat_C[i][j] = np.cos(i*j*np.pi/N)
        mat = np.matmul(mat_k,mat_C*2/N)
        mat = np.matmul(mat,mat_a)
        mat = np.matmul(mat,mat_C)
        mat = np.matmul(mat,mat_k)
        mat = np.linalg.inv(mat[1:][1:])
        
        mat_I[1:][1:] = mat
        self.mid_DenseVar = tf.Variable(mat_I.astype(np.float32),
                                        name = "Dense_mid_Win")
        
        