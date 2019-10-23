import math
import numpy as np
import tensorflow as tf

class CNNLayer(tf.keras.layers.Layer):
    #==================================================================
    # Initialize parameters in the layer
    def __init__(self, N, in_siz, out_siz, channel_siz = 8, nlvl = -1,
                 sig = -1, prefixed = False):
        super(CNNLayer, self).__init__()
        self.N              = N
        self.in_siz         = in_siz
        self.out_siz        = out_siz
        self.channel_siz    = channel_siz
        self.nlvl           = nlvl
        self.k1lvl          = max(np.floor(
                                nlvl-np.log2(in_siz/2)).astype('int'), 0)
        self.k2lvl          = min(np.floor(
                                np.log2(out_siz/2)).astype('int'), nlvl)
        self.sig            = sig

        self.in_filter_siz  = max(2, in_siz // 2**nlvl)
        self.out_filter_siz = max(2, out_siz // 2**nlvl)

        if prefixed:
            self.buildButterflyCNN()
        else:
            self.buildRand()


    #==================================================================
    # Forward structure in the layer
    def call(self, in_data):

        batch_siz      = np.size(in_data,0)
        nlvl           = self.nlvl
        k1lvl          = self.k1lvl
        k2lvl          = self.k2lvl
        in_siz         = self.in_siz
        out_siz        = self.out_siz
        channel_siz    = self.channel_siz
        in_filter_siz  = self.in_filter_siz
        out_filter_siz = self.out_filter_siz
        

        # Preparation Layer
        InInterp = tf.nn.conv1d(in_data, self.InFilterVar,
                stride=in_filter_siz, padding='VALID')
        InInterp = tf.nn.relu(tf.nn.bias_add(InInterp, self.InBiasVar))
        
        # ell Layer
        tfVars = []
        tfVars.append(InInterp)
        
        for lvl in range(1,k1lvl+1):
            Var = tf.nn.conv1d(tfVars[lvl-1],
                    self.FilterVars[lvl],
                    stride=1, padding='VALID')
            Var = tf.nn.relu(tf.nn.bias_add(Var,
                    self.BiasVars[lvl]))
            tfVars.append(Var)
        
        for lvl in range(k1lvl+1,nlvl+1):
            Var = tf.nn.conv1d(tfVars[lvl-1],
                    self.FilterVars[lvl],
                    stride=2, padding='VALID')
            Var = tf.nn.relu(tf.nn.bias_add(Var,
                    self.BiasVars[lvl]))
            tfVars.append(Var)
            
            
        # Feature Layer     
        out_data = np.reshape([], (batch_siz, 0, out_filter_siz))
        for itk in range(0,2**k2lvl):
            tmpVar = tfVars[nlvl][:,0,itk*channel_siz:(itk+1)*channel_siz]
            tmpVar = tf.matmul(tmpVar,self.FeaDenseVars[itk])
            tmpVar = tf.reshape(tmpVar,(-1,1,out_filter_siz))
            out_data = tf.concat([out_data, tmpVar], axis=1)
        out_data = tf.reshape(out_data,(batch_siz,-1,1))
        return(out_data)

    #==================================================================
    # Initialize variables in the layer
    def buildRand(self):
        
        std = 0.4
        nlvl           = self.nlvl
        k1lvl          = self.k1lvl
        k2lvl          = self.k2lvl
        channel_siz    = self.channel_siz
        in_filter_siz  = self.in_filter_siz
        out_filter_siz = self.out_filter_siz

        #----------------
        # Setup preparation layer weights
        
        self.InFilterVar = tf.Variable( tf.random_normal(
            [in_filter_siz, 1, channel_siz],0,std), name="Filter_In_ran")
        self.InBiasVar = tf.Variable( tf.zeros([channel_siz]),
                                         name="Bias_In_ran" )
        tf.summary.histogram("Filter_In_ran", self.InFilterVar)
        tf.summary.histogram("Bias_In_ran", self.InBiasVar)
        # ell Layer
        self.FilterVars = []
        self.BiasVars = []
        self.FilterVars.append([])
        self.BiasVars.append([])
        
        for lvl in range(1,k1lvl+1):
            varLabel = "LVL_%02d_ran" % (lvl)
            filterVar = tf.Variable(
                        tf.random_normal([1,2**(lvl-1)*channel_siz,
                            2**lvl*channel_siz],0,std),
                        name="Filter_"+varLabel )
            biasVar = tf.Variable(tf.zeros([2**lvl*channel_siz]),
                        name="Bias_"+varLabel )
            tf.summary.histogram("Filter_"+varLabel, filterVar)
            tf.summary.histogram("Bias_"+varLabel, biasVar)
            self.FilterVars.append(filterVar)
            self.BiasVars.append(biasVar)
            
        for lvl in range(k1lvl+1,k2lvl+1):
            varLabel = "LVL_%02d_ran" % (lvl)
            filterVar = tf.Variable(
                        tf.random_normal([2,2**(lvl-1)*channel_siz,
                            2**lvl*channel_siz],0,std),
                        name="Filter_"+varLabel )
            biasVar = tf.Variable(tf.zeros([2**lvl*channel_siz]),
                        name="Bias_"+varLabel )
            tf.summary.histogram("Filter_"+varLabel, filterVar)
            tf.summary.histogram("Bias_"+varLabel, biasVar)
            self.FilterVars.append(filterVar)
            self.BiasVars.append(biasVar)
        
        for lvl in range(k2lvl+1,nlvl+1):
            varLabel = "LVL_%02d_ran" % (lvl)
            filterVar = tf.Variable(
                        tf.random_normal([2,2**k2lvl*channel_siz,
                            2**k2lvl*channel_siz],0,std),
                        name="Filter_"+varLabel )
            biasVar = tf.Variable(tf.zeros([2**k2lvl*channel_siz]),
                        name="Bias_"+varLabel )
            tf.summary.histogram("Filter_"+varLabel, filterVar)
            tf.summary.histogram("Bias_"+varLabel, biasVar)
            self.FilterVars.append(filterVar)
            self.BiasVars.append(biasVar)

        # Feature Layer
        self.FeaDenseVars = []
        for itk in range(0,2**k2lvl):
            varLabel = "Filter_Out_%04d_ran" % (itk)
            denseVar = tf.Variable(
                    tf.random_normal([channel_siz,
                        out_filter_siz],0,std),name=varLabel)
            tf.summary.histogram(varLabel, denseVar)
            self.FeaDenseVars.append(denseVar)



    #==================================================================
    # Initialize variables with coeffs in BF in the layer
    def buildButterflyCNN(self):

        N              = self.N
        channel_siz    = self.channel_siz
        nlvl           = self.nlvl
        k1lvl          = self.k1lvl
        k2lvl          = self.k2lvl
        in_filter_siz  = self.in_filter_siz
        out_filter_siz = self.out_filter_siz
        sig            = self.sig

        if sig ==-1:
            coder = 'en_'
        else:
            coder = 'de_'
        #----------------
        # Setup initial interpolation weights
        mat = np.load('tftmp/'+coder+'Filter_In_str.npy')
        self.InFilterVar = tf.Variable( mat.astype(np.float32),
                name=coder+"Filter_In_str" )
        mat = np.load('tftmp/'+coder+'Bias_In_str.npy')
        self.InBiasVar = tf.Variable( mat.astype(np.float32),
                name=coder+"Bias_In_str" )
        tf.summary.histogram(coder+"Filter_In_str", self.InFilterVar)
        tf.summary.histogram(coder+"Bias_In_str", self.InBiasVar)

        #----------------
        # Setup ell layer weights
        #----------------
        # Setup right factor interpolation weights
        self.FilterVars = []
        self.BiasVars = []
        self.FilterVars.append([])
        self.BiasVars.append([])
        for lvl in range(1,k1lvl+1):
            bigmatf = np.zeros((1, 2**(lvl-1)*channel_siz,
                    2**lvl*channel_siz))
            bigmatb = np.zeros((2**lvl*channel_siz))
            for itk in range(0,2**lvl):
                varLabel = "LVL_%02d_%04d_str" % (lvl, itk)
                offset1 = itk//2*channel_siz
                offset2 = itk*channel_siz
                siz = channel_siz
                mat = np.load('tftmp/'+coder+'Filter_'+varLabel+'.npy')
                bigmatf[:, offset1:offset1+siz,
                        offset2:offset2+siz] = mat
                mat = np.load('tftmp/'+coder+'Bias_'+varLabel+'.npy')
                bigmatb[offset2:offset2+siz] = mat
            varLabel = "LVL_%02d_str" % (lvl)
            filterVar = tf.Variable( bigmatf.astype(np.float32),
                    name=coder+"Filter_"+varLabel )
            biasVar = tf.Variable( bigmatb.astype(np.float32),
                    name=coder+"Bias_"+varLabel )
            tf.summary.histogram(coder+"Filter_"+varLabel, filterVar)
            tf.summary.histogram(coder+"Bias_"+varLabel, biasVar)
            self.FilterVars.append(filterVar)
            self.BiasVars.append(biasVar)

        for lvl in range(k1lvl+1,k2lvl+1):
            bigmatf = np.zeros((2, 2**(lvl-1)*channel_siz,
                    2**lvl*channel_siz))
            bigmatb = np.zeros((2**lvl*channel_siz))
            for itk in range(0,2**lvl):
                varLabel = "LVL_%02d_%04d_str" % (lvl, itk)
                offset1 = itk//2*channel_siz
                offset2 = itk*channel_siz
                siz = channel_siz
                mat = np.load('tftmp/'+coder+'Filter_'+varLabel+'.npy')
                bigmatf[:, offset1:offset1+siz,
                        offset2:offset2+siz] = mat
                mat = np.load('tftmp/'+coder+'Bias_'+varLabel+'.npy')
                bigmatb[offset2:offset2+siz] = mat
            varLabel = "LVL_%02d_str" % (lvl)
            filterVar = tf.Variable( bigmatf.astype(np.float32),
                    name=coder+"Filter_"+varLabel )
            biasVar = tf.Variable( bigmatb.astype(np.float32),
                    name=coder+"Bias_"+varLabel )
            tf.summary.histogram(coder+"Filter_"+varLabel, filterVar)
            tf.summary.histogram(coder+"Bias_"+varLabel, biasVar)
            self.FilterVars.append(filterVar)
            self.BiasVars.append(biasVar)
        
        for lvl in range(k2lvl+1,nlvl+1):
            bigmatf = np.zeros((2, 2**k2lvl*channel_siz, 2**k2lvl*channel_siz))
            bigmatb = np.zeros((2**k2lvl*channel_siz))
            for itk in range(0,2**k2lvl):
                varLabel = "LVL_%02d_%04d_str" % (lvl, itk*(2**(lvl-k2lvl)))
                offset = itk*channel_siz
                siz = channel_siz
                mat = np.load('tftmp/'+coder+'Filter_'+varLabel+'.npy')
                bigmatf[:, offset:offset+siz,
                        offset:offset+siz] = mat
                mat = np.load('tftmp/'+coder+'Bias_'+varLabel+'.npy')
                bigmatb[offset:offset+siz] = mat
            varLabel = "LVL_%02d_str" % (lvl)
            filterVar = tf.Variable( bigmatf.astype(np.float32),
                    name=coder+"Filter_"+varLabel )
            biasVar = tf.Variable( bigmatb.astype(np.float32),
                    name=coder+"Bias_"+varLabel )
            tf.summary.histogram(coder+"Filter_"+varLabel, filterVar)
            tf.summary.histogram(coder+"Bias_"+varLabel, biasVar)
            self.FilterVars.append(filterVar)
            self.BiasVars.append(biasVar)
            
        self.FeaDenseVars = []
        for itk in range(0,2**k2lvl):
            varLabel = "%04d_str" % (itk)
            mat = np.load('tftmp/'+coder+'Filter_Out_'+varLabel+'.npy')
            denseVar = tf.Variable( mat.astype(np.float32),
                name=coder+"Filter_Out_str" )
            tf.summary.histogram(varLabel, denseVar)
            self.FeaDenseVars.append(denseVar)