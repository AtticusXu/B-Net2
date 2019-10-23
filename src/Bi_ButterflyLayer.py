import math
import numpy as np
import tensorflow as tf

from LagrangeMat import LagrangeMat

class ButterflyLayer(tf.keras.layers.Layer):
    #==================================================================
    # Initialize parameters in the layer
    def __init__(self, N, in_siz, out_siz, prep,
            channel_siz = 8, nlvl = -1, sig = -1, prefixed = False,
            in_range = [0, 1], out_range = [0, 1]):
        super(ButterflyLayer, self).__init__()
        self.N              = N
        self.in_siz         = in_siz
        self.out_siz        = out_siz
        self.channel_siz    = channel_siz
        self.nlvl           = nlvl
        self.k1lvl          = max(np.floor(
                                nlvl-np.log2(in_siz/2)).astype('int'), 0)
        self.k2lvl          = min(np.floor(
                                np.log2(out_siz/2)).astype('int'), nlvl)
        self.in_range       = in_range
        self.out_range      = out_range
        self.sig            = sig
        self.prep           = prep

        self.in_filter_siz  = max(2, in_siz // 2**nlvl)
        self.out_filter_siz = max(2, out_siz // 2**nlvl)

        if prefixed:
            self.buildButterfly()
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
        in_filter_siz  = self.in_filter_siz
        out_filter_siz = self.out_filter_siz
        

        # Preparation Layer
        InInterp = tf.nn.conv1d(in_data, self.InFilterVar,
                stride=in_filter_siz, padding='VALID')
        InInterp = tf.nn.relu(tf.nn.bias_add(InInterp, self.InBiasVar))
        
        # ell Layer
        tfVars = []
        tmpVars = []
        tmpVars.append(InInterp)
        tfVars.append(list(tmpVars))
        
        for lvl in range(1,k1lvl+1):
            tmpVars = []
            for itk in range(0,2**lvl):
                Var = tf.nn.conv1d(tfVars[lvl-1][itk//2],
                    self.FilterVars[lvl][itk],
                    stride=1, padding='VALID')
                Var = tf.nn.relu(tf.nn.bias_add(Var,
                    self.BiasVars[lvl][itk]))
                tmpVars.append(Var)
            tfVars.append(list(tmpVars))
        
        for lvl in range(k1lvl+1,k2lvl+1):
            tmpVars = []

            for itk in range(0,2**lvl):
                Var = tf.nn.conv1d(tfVars[lvl-1][itk//2],
                    self.FilterVars[lvl][itk],
                    stride=2, padding='VALID')
                Var = tf.nn.relu(tf.nn.bias_add(Var,
                    self.BiasVars[lvl][itk]))
                tmpVars.append(Var)
            tfVars.append(list(tmpVars))
            
        for lvl in range(k2lvl+1,nlvl+1):
            tmpVars = []
            for itk in range(0,2**k2lvl):
                Var = tf.nn.conv1d(tfVars[lvl-1][itk],
                    self.FilterVars[lvl][itk],
                    stride=2, padding='VALID')
                Var = tf.nn.relu(tf.nn.bias_add(Var,
                    self.BiasVars[lvl][itk]))
                tmpVars.append(Var)
            tfVars.append(list(tmpVars))
            
        # Feature Layer
        
        out_data = np.reshape([], (batch_siz, 0, out_filter_siz))
        for itk in range(0,2**k2lvl):
            tmpVar = tfVars[nlvl][itk][:,0,:]
            tmpVar = tf.matmul(tmpVar,self.FeaDenseVars[itk])
            tmpVar = tf.reshape(tmpVar,(-1,1,out_filter_siz))
            out_data = tf.concat([out_data, tmpVar], axis=1)
        out_data = tf.reshape(out_data,(batch_siz,-1,1))
        return(out_data)

    #==================================================================
    # Initialize variables in the layer
    def buildRand(self):
        
        std = 0.5
        nlvl           = self.nlvl
        k1lvl          = self.k1lvl
        k2lvl          = self.k2lvl
        channel_siz    = self.channel_siz
        in_filter_siz  = self.in_filter_siz
        out_filter_siz = self.out_filter_siz
        in_range       = self.in_range
        out_range      = self.out_range
        #----------------
        # Setup preparation layer weights
        
        
        
        if self.prep:
            NG = int(channel_siz/4)
            ChebNodes = (np.cos(np.array(range(2*NG-1,0,-2))/2/NG*math.pi) +
                         1)/2
            xNodes = np.arange(0,1,1.0/in_filter_siz)
            LMat = LagrangeMat(ChebNodes,xNodes)                         
            mat = np.empty((in_filter_siz,1,channel_siz))
            kcen = np.mean(out_range)
            xlen = (in_range[1] - in_range[0])/2**nlvl
            for it in range(0,NG):
                KVal = np.exp(-2*math.pi*1j*kcen*(xNodes-ChebNodes[it])*xlen)
                LVec = np.squeeze(LMat[:,it])
                mat[:,0,4*it]   =  np.multiply(KVal.real,LVec)
                mat[:,0,4*it+1] =  np.multiply(KVal.imag,LVec)
                mat[:,0,4*it+2] = -np.multiply(KVal.real,LVec)
                mat[:,0,4*it+3] = -np.multiply(KVal.imag,LVec)

            self.InFilterVar = tf.Variable( mat.astype(np.float32),
                name="Filter_In" ,trainable=False )
            self.InBiasVar = tf.Variable( tf.zeros([channel_siz]),
                name="Bias_In" ,trainable=False )
        else:
            self.InFilterVar = tf.Variable( tf.random_normal(
            [in_filter_siz, 1, channel_siz],0,std), name="Filter_In_ran")
            self.InBiasVar = tf.Variable( tf.zeros([channel_siz]),
                                         name="Bias_In_ran" )
            tf.summary.histogram("Filter_In_ran", self.InFilterVar)
            tf.summary.histogram("Bias_In_ran", self.InBiasVar)
        # ell Layer
        self.FilterVars = []
        self.BiasVars = []
        self.FilterVars.append(list([]))
        self.BiasVars.append(list([]))
        
        for lvl in range(1,k1lvl+1):
            tmpFilterVars = []
            tmpBiasVars = []
            for itk in range(0,2**lvl):
                varLabel = "LVL_%02d_%04d_ran" % (lvl, itk)
                filterVar = tf.Variable(
                        tf.random_normal([1,channel_siz,
                            channel_siz],0,std),
                        name="Filter_"+varLabel )
                biasVar = tf.Variable(tf.zeros([channel_siz]),
                        name="Bias_"+varLabel )
                tf.summary.histogram("Filter_"+varLabel, filterVar)
                tf.summary.histogram("Bias_"+varLabel, biasVar)
                tmpFilterVars.append(filterVar)
                tmpBiasVars.append(biasVar)
            self.FilterVars.append(list(tmpFilterVars))
            self.BiasVars.append(list(tmpBiasVars))
            
        for lvl in range(k1lvl+1,k2lvl+1):
            tmpFilterVars = []
            tmpBiasVars = []
            for itk in range(0,2**lvl):
                varLabel = "LVL_%02d_%04d_ran" % (lvl, itk)
                filterVar = tf.Variable(
                        tf.random_normal([2,channel_siz,
                            channel_siz],0,std),
                        name="Filter_"+varLabel )
                biasVar = tf.Variable(tf.zeros([channel_siz]),
                        name="Bias_"+varLabel )
                tf.summary.histogram("Filter_"+varLabel, filterVar)
                tf.summary.histogram("Bias_"+varLabel, biasVar)
                tmpFilterVars.append(filterVar)
                tmpBiasVars.append(biasVar)
            self.FilterVars.append(list(tmpFilterVars))
            self.BiasVars.append(list(tmpBiasVars))
        
        for lvl in range(k2lvl+1,nlvl+1):
            tmpFilterVars = []
            tmpBiasVars = []
            for itk in range(0,2**k2lvl):
                varLabel = "LVL_%02d_%04d_ran" % (lvl, itk*(2**(lvl-k2lvl)))
                filterVar = tf.Variable(
                        tf.random_normal([2,channel_siz,
                            channel_siz],0,std),
                        name="Filter_"+varLabel )
                biasVar = tf.Variable(tf.zeros([channel_siz]),
                        name="Bias_"+varLabel )
                tf.summary.histogram("Filter_"+varLabel, filterVar)
                tf.summary.histogram("Bias_"+varLabel, biasVar)
                tmpFilterVars.append(filterVar)
                tmpBiasVars.append(biasVar)
            self.FilterVars.append(list(tmpFilterVars))
            self.BiasVars.append(list(tmpBiasVars))

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
    def buildButterfly(self):

        N              = self.N
        sig            = self.sig
        channel_siz    = self.channel_siz
        nlvl           = self.nlvl
        k1lvl          = self.k1lvl
        k2lvl          = self.k2lvl
        in_filter_siz  = self.in_filter_siz
        out_filter_siz = self.out_filter_siz
        in_range       = self.in_range
        out_range      = self.out_range
        if sig ==-1:
            coder = 'en_'
        else:
            coder = 'de_'

        NG = channel_siz//4
        ChebNodes = (np.cos(np.array(range(2*NG-1,0,-2))/2/NG*math.pi) \
                + 1)/2
        xNodes = np.arange(0,1,2/in_filter_siz)
        LMat = LagrangeMat(ChebNodes,xNodes)

        #----------------
        # Setup preparation layer weights
        mat = np.empty((in_filter_siz,1,channel_siz))
        kcen = (out_range[1] - out_range[0])*0.5 + out_range[0]
        xlen = (in_range[1] - in_range[0])/2**self.nlvl
        for it in range(0,NG):
            KVal = np.exp(sig*2*math.pi*1j*N \
                    * kcen * (xNodes - ChebNodes[it]) * xlen)
            LVec = np.squeeze(LMat[:,it])
            idx = range(0,in_filter_siz,2) 
            mat[idx,0,4*it  ] =  np.multiply(KVal.real,LVec)
            mat[idx,0,4*it+1] =  np.multiply(KVal.imag,LVec)
            mat[idx,0,4*it+2] = -np.multiply(KVal.real,LVec)
            mat[idx,0,4*it+3] = -np.multiply(KVal.imag,LVec)
            idx = range(1,in_filter_siz,2) 
            mat[idx,0,4*it  ] = -np.multiply(KVal.imag,LVec)
            mat[idx,0,4*it+1] =  np.multiply(KVal.real,LVec)
            mat[idx,0,4*it+2] =  np.multiply(KVal.imag,LVec)
            mat[idx,0,4*it+3] = -np.multiply(KVal.real,LVec)

        self.InFilterVar = tf.Variable( mat.astype(np.float32),
                name=coder+"Filter_In_str" )
        self.InBiasVar = tf.Variable( tf.zeros([channel_siz]),
                name=coder+"Bias_In_str" )
        tf.summary.histogram(coder+"Filter_In_str", self.InFilterVar)
        tf.summary.histogram(coder+"Bias_In_str", self.InBiasVar)

        #----------------
        # Setup ell layer weights
        x1Nodes = ChebNodes/2
        x2Nodes = ChebNodes/2 + 1/2
        LMat1 = LagrangeMat(ChebNodes,x1Nodes)
        LMat2 = LagrangeMat(ChebNodes,x2Nodes)

        self.FilterVars = []
        self.BiasVars = []
        self.FilterVars.append(list([]))
        self.BiasVars.append(list([]))
        
        for lvl in range(1,k1lvl+1):
            tmpFilterVars = []
            tmpBiasVars = []
            for itk in range(0,2**lvl):
                varLabel = "LVL_%02d_%04d_str" % (lvl, itk)

                mat = np.empty((1, channel_siz, channel_siz))
                kcen = (out_range[1] - out_range[0])/2**lvl \
                        * (itk + 0.5) + out_range[0]
                xlen = (in_range[1] - in_range[0])/2**(nlvl-lvl)

                for it in range(0,NG):
                    KVal = np.exp( sig*2*math.pi*1j*N * kcen \
                            * (x1Nodes-ChebNodes[it]) * xlen)
                    LVec = np.squeeze(LMat1[:,it])
                    mat[0,range(0,4*NG,4),4*it  ] = \
                            np.multiply(KVal.real,LVec)
                    mat[0,range(1,4*NG,4),4*it  ] = \
                            - np.multiply(KVal.imag,LVec)
                    mat[0,range(2,4*NG,4),4*it  ] = \
                            - np.multiply(KVal.real,LVec)
                    mat[0,range(3,4*NG,4),4*it  ] = \
                            np.multiply(KVal.imag,LVec)
                    mat[0,range(0,4*NG,4),4*it+1] = \
                            np.multiply(KVal.imag,LVec)
                    mat[0,range(1,4*NG,4),4*it+1] = \
                            np.multiply(KVal.real,LVec)
                    mat[0,range(2,4*NG,4),4*it+1] = \
                            - np.multiply(KVal.imag,LVec)
                    mat[0,range(3,4*NG,4),4*it+1] = \
                            - np.multiply(KVal.real,LVec)
                    mat[0,:,(4*it+2,4*it+3)] = - mat[0,:,(4*it,4*it+1)]

                filterVar = tf.Variable( mat.astype(np.float32),
                        name=coder+"Filter_"+varLabel )
                biasVar = tf.Variable(tf.zeros([channel_siz]),
                        name=coder+"Bias_"+varLabel )
                tf.summary.histogram(coder+"Filter_"+varLabel, filterVar)
                tf.summary.histogram(coder+"Bias_"+varLabel, biasVar)
                tmpFilterVars.append(filterVar)
                tmpBiasVars.append(biasVar)
            self.FilterVars.append(list(tmpFilterVars))
            self.BiasVars.append(list(tmpBiasVars))
            
        for lvl in range(k1lvl+1,k2lvl+1):
            tmpFilterVars = []
            tmpBiasVars = []
            for itk in range(0,2**lvl):
                varLabel = "LVL_%02d_%04d_str" % (lvl, itk)

                mat = np.empty((2, channel_siz, channel_siz))
                kcen = (out_range[1] - out_range[0])/2**lvl \
                        * (itk + 0.5) + out_range[0]
                xlen = (in_range[1] - in_range[0])/2**(nlvl-lvl)

                for it in range(0,NG):
                    KVal = np.exp( sig*2*math.pi*1j*N * kcen \
                            * (x1Nodes-ChebNodes[it]) * xlen)
                    LVec = np.squeeze(LMat1[:,it])
                    mat[0,range(0,4*NG,4),4*it  ] = \
                            np.multiply(KVal.real,LVec)
                    mat[0,range(1,4*NG,4),4*it  ] = \
                            - np.multiply(KVal.imag,LVec)
                    mat[0,range(2,4*NG,4),4*it  ] = \
                            - np.multiply(KVal.real,LVec)
                    mat[0,range(3,4*NG,4),4*it  ] = \
                            np.multiply(KVal.imag,LVec)
                    mat[0,range(0,4*NG,4),4*it+1] = \
                            np.multiply(KVal.imag,LVec)
                    mat[0,range(1,4*NG,4),4*it+1] = \
                            np.multiply(KVal.real,LVec)
                    mat[0,range(2,4*NG,4),4*it+1] = \
                            - np.multiply(KVal.imag,LVec)
                    mat[0,range(3,4*NG,4),4*it+1] = \
                            - np.multiply(KVal.real,LVec)
                    mat[0,:,(4*it+2,4*it+3)] = - mat[0,:,(4*it,4*it+1)]

                    KVal = np.exp( sig*2*math.pi*1j*N * kcen \
                            * (x2Nodes-ChebNodes[it]) * xlen)
                    LVec = np.squeeze(LMat2[:,it])
                    mat[1,range(0,4*NG,4),4*it  ] = \
                            np.multiply(KVal.real,LVec)
                    mat[1,range(1,4*NG,4),4*it  ] = \
                            - np.multiply(KVal.imag,LVec)
                    mat[1,range(2,4*NG,4),4*it  ] = \
                            - np.multiply(KVal.real,LVec)
                    mat[1,range(3,4*NG,4),4*it  ] = \
                            np.multiply(KVal.imag,LVec)
                    mat[1,range(0,4*NG,4),4*it+1] = \
                            np.multiply(KVal.imag,LVec)
                    mat[1,range(1,4*NG,4),4*it+1] = \
                            np.multiply(KVal.real,LVec)
                    mat[1,range(2,4*NG,4),4*it+1] = \
                            - np.multiply(KVal.imag,LVec)
                    mat[1,range(3,4*NG,4),4*it+1] = \
                            - np.multiply(KVal.real,LVec)
                    mat[1,:,(4*it+2,4*it+3)] = - mat[1,:,(4*it,4*it+1)]

                filterVar = tf.Variable( mat.astype(np.float32),
                        name=coder+"Filter_"+varLabel )
                biasVar = tf.Variable(tf.zeros([channel_siz]),
                        name=coder+"Bias_"+varLabel )
                tf.summary.histogram(coder+"Filter_"+varLabel, filterVar)
                tf.summary.histogram(coder+"Bias_"+varLabel, biasVar)
                tmpFilterVars.append(filterVar)
                tmpBiasVars.append(biasVar)
            self.FilterVars.append(list(tmpFilterVars))
            self.BiasVars.append(list(tmpBiasVars))
        
        for lvl in range(k2lvl+1,nlvl+1):
            tmpFilterVars = []
            tmpBiasVars = []
            for itk in range(0,2**k2lvl):
                varLabel = "LVL_%02d_%04d_str" % (lvl, itk*(2**(lvl-k2lvl)))

                mat = np.empty((2, channel_siz, channel_siz))
                kcen = (out_range[1] - out_range[0])/2**lvl \
                        *(itk*(2**(lvl-k2lvl))+0.5) + out_range[0]
                xlen = (in_range[1] - in_range[0])/2**(nlvl-lvl)
                for it in range(0,NG):
                    KVal = np.exp( sig*2*math.pi*1j*N * kcen *
                            (x1Nodes-ChebNodes[it]) * xlen)
                    LVec = np.squeeze(LMat1[:,it])
                    mat[0,range(0,4*NG,4),4*it  ] = \
                            np.multiply(KVal.real,LVec)
                    mat[0,range(1,4*NG,4),4*it  ] = \
                            - np.multiply(KVal.imag,LVec)
                    mat[0,range(2,4*NG,4),4*it  ] = \
                            - np.multiply(KVal.real,LVec)
                    mat[0,range(3,4*NG,4),4*it  ] = \
                            np.multiply(KVal.imag,LVec)
                    mat[0,range(0,4*NG,4),4*it+1] = \
                            np.multiply(KVal.imag,LVec)
                    mat[0,range(1,4*NG,4),4*it+1] = \
                            np.multiply(KVal.real,LVec)
                    mat[0,range(2,4*NG,4),4*it+1] = \
                            - np.multiply(KVal.imag,LVec)
                    mat[0,range(3,4*NG,4),4*it+1] = \
                            - np.multiply(KVal.real,LVec)
                    mat[0,:,(4*it+2,4*it+3)] = - mat[0,:,(4*it,4*it+1)]

                    KVal = np.exp( sig*2*math.pi*1j *N* kcen *
                            (x2Nodes-ChebNodes[it]) * xlen)
                    LVec = np.squeeze(LMat2[:,it])
                    mat[1,range(0,4*NG,4),4*it  ] = \
                            np.multiply(KVal.real,LVec)
                    mat[1,range(1,4*NG,4),4*it  ] = \
                            - np.multiply(KVal.imag,LVec)
                    mat[1,range(2,4*NG,4),4*it  ] = \
                            - np.multiply(KVal.real,LVec)
                    mat[1,range(3,4*NG,4),4*it  ] = \
                            np.multiply(KVal.imag,LVec)
                    mat[1,range(0,4*NG,4),4*it+1] = \
                            np.multiply(KVal.imag,LVec)
                    mat[1,range(1,4*NG,4),4*it+1] = \
                            np.multiply(KVal.real,LVec)
                    mat[1,range(2,4*NG,4),4*it+1] = \
                            - np.multiply(KVal.imag,LVec)
                    mat[1,range(3,4*NG,4),4*it+1] = \
                            - np.multiply(KVal.real,LVec)
                    mat[1,:,(4*it+2,4*it+3)] = - mat[1,:,(4*it,4*it+1)]

                filterVar = tf.Variable( mat.astype(np.float32),
                        name=coder+"Filter_"+varLabel )
                biasVar = tf.Variable(tf.zeros([channel_siz]),
                        name=coder+"Bias_"+varLabel )
                tf.summary.histogram(coder+"Filter_"+varLabel, filterVar)
                tf.summary.histogram(coder+"Bias_"+varLabel, biasVar)
                tmpFilterVars.append(filterVar)
                tmpBiasVars.append(biasVar)
            self.FilterVars.append(list(tmpFilterVars))
            self.BiasVars.append(list(tmpBiasVars))
        #----------------
        # Setup feature layer weights 
        self.FeaDenseVars = []
        
        for itk in range(0,2**k2lvl):
            varLabel = "Filter_Out_%04d_str" % (itk)
            mat = np.empty((channel_siz, out_filter_siz))
            klen = (out_range[1] - out_range[0])/2**nlvl
            koff = klen*itk*(2**(nlvl-k2lvl)) + out_range[0]
            kNodes = np.arange(0,1,2/out_filter_siz)*klen + koff
            xlen = in_range[1] - in_range[0]
            xoff = in_range[0]
            xNodes = ChebNodes*xlen + xoff

            for iti in range(0,NG):
                for itj in range(0,out_filter_siz//2):
                    KVal = np.exp( sig*2*math.pi*1j*N
                            *kNodes[itj]*xNodes[iti])
                    mat[4*iti  ,2*itj  ] =   KVal.real
                    mat[4*iti+1,2*itj  ] = - KVal.imag
                    mat[4*iti+2,2*itj  ] = - KVal.real
                    mat[4*iti+3,2*itj  ] =   KVal.imag
                    mat[4*iti  ,2*itj+1] =   KVal.imag
                    mat[4*iti+1,2*itj+1] =   KVal.real
                    mat[4*iti+2,2*itj+1] = - KVal.imag
                    mat[4*iti+3,2*itj+1] = - KVal.real

            denseVar = tf.Variable( mat.astype(np.float32),
                        name=coder+varLabel )
            tf.summary.histogram(varLabel, denseVar)
            self.FeaDenseVars.append(denseVar)