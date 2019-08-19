import math
import numpy as np
import tensorflow as tf

class CNNLayer(tf.keras.layers.Layer):
    #==================================================================
    # Initialize parameters in the layer
    def __init__(self, in_siz, out_siz, prep, klvl, alph = 2,
            channel_siz = 8, nlvl = -1, prefixed = True):
        super(CNNLayer, self).__init__()
        self.in_siz         = in_siz
        self.out_siz        = out_siz
        self.channel_siz    = channel_siz
        self.alph           = alph
        self.nlvl           = nlvl
        self.klvl           = min(nlvl,klvl)
        self.in_filter_siz  = in_siz // 2**nlvl
        self.out_filter_siz = max(out_siz // alph**self.klvl,2)
        self.prep           = prep
        if prefixed:
            self.buildButterflyCNN()
        else:
            self.buildRand()


    #==================================================================
    # Forward structure in the layer
    def call(self, in_data):
        # Preparation Layer
        InInterp = tf.nn.conv1d(in_data, self.InFilterVar,
                stride=self.in_filter_siz, padding='VALID')
        InInterp = tf.nn.relu(tf.nn.bias_add(InInterp, self.InBiasVar))
        
        # ell Layer
        tfVars = []
        tfVars.append(InInterp)

        for lvl in range(1,self.klvl+1):
            Var = tf.nn.conv1d(tfVars[lvl-1],
                    self.FilterVars[lvl],
                    stride=2, padding='VALID')
            Var = tf.nn.relu(tf.nn.bias_add(Var,
                    self.BiasVars[lvl]))
            tfVars.append(Var)
            
        for lvl in range(self.klvl+1,self.nlvl+1):
            Var = tf.nn.conv1d(tfVars[lvl-1],
                    self.FilterVars[lvl],
                    stride=2, padding='VALID')
            Var = tf.nn.relu(tf.nn.bias_add(Var,
                    self.BiasVars[lvl]))
            tfVars.append(Var)

        # Feature Layer
        OutInterp = np.reshape([], (np.size(in_data,0), 0,
                self.out_filter_siz))
        spa = int(max((self.alph**self.klvl)/(self.out_siz//2),1))
        
        for itk in range(0, min(self.alph**self.klvl,(self.out_siz//2))):
            tmpVar = tfVars[self.nlvl]\
                [:,0,spa*itk*self.channel_siz:spa*(itk+1)*self.channel_siz]
            tmpVar = tf.matmul(tmpVar,self.FeaDenseVars[itk])
            tmpVar = tf.reshape(tmpVar,
                    (np.size(in_data,0),1,self.out_filter_siz))
            OutInterp = tf.concat([OutInterp, tmpVar], axis=1)
        out_data = tf.reshape(OutInterp,shape=(np.size(in_data,0),
            self.out_siz,1))
        return(out_data)

    #==================================================================
    # Initialize variables in the layer
    def buildRand(self):
        std = self.alph**(-np.sqrt(self.klvl)/2)
        
        # Preparation Layer
        if self.prep:
            mat = np.load('tftmp/Filter_In.npy')
            self.InFilterVar = tf.Variable( mat.astype(np.float32),
                                           name="Filter_In" )
            mat = np.load('tftmp/Bias_In.npy')
            self.InBiasVar = tf.Variable( mat.astype(np.float32),
                                         name="Bias_In" )
        else:
            self.InFilterVar = tf.Variable( tf.random_normal(
                    [self.in_filter_siz, 1, self.channel_siz],0,std),
                        name="Filter_In" )
            self.InBiasVar = tf.Variable( tf.zeros([self.channel_siz]),
                                         name="Bias_In" )
        
        # ell Layer
        self.FilterVars = []
        self.BiasVars = []
        self.FilterVars.append([])
        self.BiasVars.append([])
        for lvl in range(1,self.klvl+1):
            varLabel = "LVL_%02d" % (lvl)
            filterVar = tf.Variable(
                    tf.random_normal([2,self.alph**(lvl-1)*self.channel_siz,
                        self.alph**lvl*self.channel_siz],0,std),
                    name="Filter_"+varLabel )
            biasVar = tf.Variable(tf.zeros([self.alph**lvl*self.channel_siz]),
                    name="Bias_"+varLabel )
            self.FilterVars.append(filterVar)
            self.BiasVars.append(biasVar)
        
        for lvl in range(self.klvl+1,self.nlvl+1):
            varLabel = "LVL_%02d" % (lvl)
            filterVar = tf.Variable(
                    tf.random_normal(\
                    [2,self.alph**(self.klvl)*self.channel_siz,
                    self.alph**self.klvl*self.channel_siz],0,std),
                    name="Filter_"+varLabel )
            biasVar = tf.Variable(tf.zeros(\
                    [self.alph**self.klvl*self.channel_siz]),
                    name="Bias_"+varLabel )
            self.FilterVars.append(filterVar)
            self.BiasVars.append(biasVar)

        # Feature Layer
        self.FeaDenseVars = []
        spa = int(max((self.alph**self.klvl)/(self.out_siz//2),1))

        for itk in range(0,self.out_siz//2):
            varLabel = "Filter_Out_%04d" % (itk)
            denseVar = tf.Variable(
                    tf.random_normal([spa * self.channel_siz,
                         self.out_filter_siz],0,std),name=varLabel)

            self.FeaDenseVars.append(denseVar)

    #==================================================================
    # Initialize variables with coeffs in BF in the layer
    def buildButterflyCNN(self):

        #----------------
        # Setup initial interpolation weights
        mat = np.load('tftmp/Filter_In.npy')
        self.InFilterVar = tf.Variable( mat.astype(np.float32),
                name="Filter_In" )
        mat = np.load('tftmp/Bias_In.npy')
        self.InBiasVar = tf.Variable( mat.astype(np.float32),
                name="Bias_In" )

        #----------------
        # Setup right factor interpolation weights
        self.FilterVars = []
        self.BiasVars = []
        self.FilterVars.append([])
        self.BiasVars.append([])
        for lvl in range(1,self.klvl+1):
            bigmatf = np.zeros((2, 2**(lvl-1)*self.channel_siz,
                    2**lvl*self.channel_siz))
            bigmatb = np.zeros((2**lvl*self.channel_siz))
            for itk in range(0,2**lvl):
                varLabel = "LVL_%02d_%04d" % (lvl, itk)
                offset1 = itk//2*self.channel_siz
                offset2 = itk*self.channel_siz
                siz = self.channel_siz
                mat = np.load('tftmp/Filter_'+varLabel+'.npy')
                bigmatf[:, offset1:offset1+siz,
                        offset2:offset2+siz] = mat
                mat = np.load('tftmp/Bias_'+varLabel+'.npy')
                bigmatb[offset2:offset2+siz] = mat
            varLabel = "LVL_%02d" % (lvl)
            filterVar = tf.Variable( bigmatf.astype(np.float32),
                    name="Filter_"+varLabel )
            biasVar = tf.Variable( bigmatb.astype(np.float32),
                    name="Bias_"+varLabel )
            self.FilterVars.append(filterVar)
            self.BiasVars.append(biasVar)
            
        for lvl in range(self.klvl+1,self.nlvl+1):
            bigmatf = np.zeros((2, 2**(self.klvl)*self.channel_siz,
                    2**self.klvl*self.channel_siz))
            bigmatb = np.zeros((2**self.klvl*self.channel_siz))
            for itk in range(0,2**self.klvl):
                varLabel = "LVL_%02d_%04d" % (lvl, itk*(2**(lvl-self.klvl)))
                offset = itk*self.channel_siz
                siz = self.channel_siz
                mat = np.load('tftmp/Filter_'+varLabel+'.npy')
                bigmatf[:, offset:offset+siz,
                        offset:offset+siz] = mat
                mat = np.load('tftmp/Bias_'+varLabel+'.npy')
                bigmatb[offset:offset+siz] = mat
            varLabel = "LVL_%02d" % (lvl)
            filterVar = tf.Variable( bigmatf.astype(np.float32),
                    name="Filter_"+varLabel )
            biasVar = tf.Variable( bigmatb.astype(np.float32),
                    name="Bias_"+varLabel )
            self.FilterVars.append(filterVar)
            self.BiasVars.append(biasVar)

        #----------------
        # Setup final interpolation weights
        self.FeaDenseVars = []
        for itk in range(0,2**self.klvl):
            varLabel = "%04d" % (itk)
            mat = np.load('tftmp/Filter_Out_'+varLabel+'.npy')
            denseVar = tf.Variable( mat.astype(np.float32),
                name="Filter_Out" )
            self.FeaDenseVars.append(denseVar)

        

