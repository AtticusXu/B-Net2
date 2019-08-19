import math
import numpy as np
import tensorflow as tf

from LagrangeMat import LagrangeMat

class ButterflyLayer(tf.keras.layers.Layer):
    #==================================================================
    # Initialize parameters in the layer
    def __init__(self, in_siz, out_siz, prep,
            channel_siz = 8, nlvl = -1, prefixed = False,
            in_range = [], out_range = []):
        super(ButterflyLayer, self).__init__()
        self.in_siz         = in_siz
        self.out_siz        = out_siz
        #TODO: set the default values based on in_siz and out_siz
        self.channel_siz    = channel_siz
        self.nlvl           = nlvl
        self.klvl           = min(np.floor(np.log2(out_siz/2)).astype('int'), nlvl)
        self.in_filter_siz  = in_siz // 2**nlvl
        self.out_filter_siz = out_siz // 2**self.klvl
        self.in_range       = in_range
        self.out_range      = out_range
        self.prep           = prep
        if prefixed:
            self.buildButterfly()
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
        tmpVars = []
        tmpVars.append(InInterp)
        tfVars.append(list(tmpVars))

        for lvl in range(1,self.klvl+1):
            tmpVars = []
            for itk in range(0,2**lvl):
                Var = tf.nn.conv1d(tfVars[lvl-1][itk//2],
                    self.FilterVars[lvl][itk],
                    stride=2, padding='VALID')
                Var = tf.nn.relu(tf.nn.bias_add(Var,
                    self.BiasVars[lvl][itk]))
                tmpVars.append(Var)
            tfVars.append(list(tmpVars))
            
        for lvl in range(self.klvl+1,self.nlvl+1):
            tmpVars = []

            for itk in range(0,2**self.klvl):
                Var = tf.nn.conv1d(tfVars[lvl-1][itk],
                    self.FilterVars[lvl][itk],
                    stride=2, padding='VALID')
                Var = tf.nn.relu(tf.nn.bias_add(Var,
                    self.BiasVars[lvl][itk]))
                tmpVars.append(Var)
            tfVars.append(list(tmpVars))
            

        # Feature Layer
        OutInterp = np.reshape([], (np.size(in_data,0), 0,
                self.out_filter_siz))
        
        for itk in range(0,2**self.klvl):
            tmpVar = tfVars[self.nlvl][itk][:,0,:]
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
        
        std = 0.9
        
        #----------------
        # Setup preparation layer weights
        
        
        
        if self.prep:
            NG = int(self.channel_siz/4)
            ChebNodes = (np.cos(np.array(range(2*NG-1,0,-2))/2/NG*math.pi) +
                         1)/2
            xNodes = np.arange(0,1,1.0/self.in_filter_siz)
            LMat = LagrangeMat(ChebNodes,xNodes)                         
            mat = np.empty((self.in_filter_siz,1,self.channel_siz))
            kcen = np.mean(self.out_range)
            xlen = (self.in_range[1] - self.in_range[0])/2**self.nlvl
            for it in range(0,NG):
                KVal = np.exp(-2*math.pi*1j*kcen*(xNodes-ChebNodes[it])*xlen)
                LVec = np.squeeze(LMat[:,it])
                mat[:,0,4*it]   =  np.multiply(KVal.real,LVec)
                mat[:,0,4*it+1] =  np.multiply(KVal.imag,LVec)
                mat[:,0,4*it+2] = -np.multiply(KVal.real,LVec)
                mat[:,0,4*it+3] = -np.multiply(KVal.imag,LVec)

            self.InFilterVar = tf.Variable( mat.astype(np.float32),
                name="Filter_In" )
            self.InBiasVar = tf.Variable( tf.zeros([self.channel_siz]),
                name="Bias_In" )
        else:
            self.InFilterVar = tf.Variable( tf.random_normal(
            [self.in_filter_siz, 1, self.channel_siz],0,std), name="Filter_In")
            self.InBiasVar = tf.Variable( tf.zeros([self.channel_siz]),
                                         name="Bias_In" )
            
        # ell Layer
        self.FilterVars = []
        self.BiasVars = []
        self.FilterVars.append(list([]))
        self.BiasVars.append(list([]))
        for lvl in range(1,self.klvl+1):
            tmpFilterVars = []
            tmpBiasVars = []
            for itk in range(0,2**lvl):
                varLabel = "LVL_%02d_%04d" % (lvl, itk)
                filterVar = tf.Variable(
                        tf.random_normal([2,self.channel_siz,
                            self.channel_siz],0,std),
                        name="Filter_"+varLabel )
                biasVar = tf.Variable(tf.zeros([self.channel_siz]),
                        name="Bias_"+varLabel )
                tmpFilterVars.append(filterVar)
                tmpBiasVars.append(biasVar)
            self.FilterVars.append(list(tmpFilterVars))
            self.BiasVars.append(list(tmpBiasVars))
        
        for lvl in range(self.klvl+1,self.nlvl+1):
            tmpFilterVars = []
            tmpBiasVars = []
            for itk in range(0,2**self.klvl):
                varLabel = "LVL_%02d_%04d" % (lvl, itk*(2**(lvl-self.klvl)))
                filterVar = tf.Variable(
                        tf.random_normal([2,self.channel_siz,
                            self.channel_siz],0,std),
                        name="Filter_"+varLabel )
                biasVar = tf.Variable(tf.zeros([self.channel_siz]),
                        name="Bias_"+varLabel )
                tmpFilterVars.append(filterVar)
                tmpBiasVars.append(biasVar)
            self.FilterVars.append(list(tmpFilterVars))
            self.BiasVars.append(list(tmpBiasVars))

        # Feature Layer
        self.FeaDenseVars = []
        for itk in range(0,2**self.klvl):
            varLabel = "Filter_Out_%04d" % (itk)
            denseVar = tf.Variable(
                    tf.random_normal([self.channel_siz,
                        self.out_filter_siz],0,std),name=varLabel)

            self.FeaDenseVars.append(denseVar)



    #==================================================================
    # Initialize variables with coeffs in BF in the layer
    def buildButterfly(self):
        

        NG = int(self.channel_siz/4)
        ChebNodes = (np.cos(np.array(range(2*NG-1,0,-2))/2/NG*math.pi) +
                1)/2
        xNodes = np.arange(0,1,1.0/self.in_filter_siz)
        LMat = LagrangeMat(ChebNodes,xNodes)

        #----------------
        # Setup preparation layer weights
        mat = np.empty((self.in_filter_siz,1,self.channel_siz))
        kcen = np.mean(self.out_range)
        xlen = (self.in_range[1] - self.in_range[0])/2**self.nlvl
        for it in range(0,NG):
            KVal = np.exp(-2*math.pi*1j*kcen*(xNodes-ChebNodes[it])*xlen)
            LVec = np.squeeze(LMat[:,it])
            mat[:,0,4*it]   =  np.multiply(KVal.real,LVec)
            mat[:,0,4*it+1] =  np.multiply(KVal.imag,LVec)
            mat[:,0,4*it+2] = -np.multiply(KVal.real,LVec)
            mat[:,0,4*it+3] = -np.multiply(KVal.imag,LVec)

        self.InFilterVar = tf.Variable( mat.astype(np.float32),
                name="Filter_In" )
        self.InBiasVar = tf.Variable( tf.zeros([self.channel_siz]),
                name="Bias_In" )

        #----------------
        # Setup ell layer weights
        ChebNodes = (np.cos(np.array(range(2*NG-1,0,-2))/2/NG*math.pi) +
                1)/2
        x1Nodes = ChebNodes/2
        x2Nodes = ChebNodes/2 + 1/2
        LMat1 = LagrangeMat(ChebNodes,x1Nodes)
        LMat2 = LagrangeMat(ChebNodes,x2Nodes)

        self.FilterVars = []
        self.BiasVars = []
        self.FilterVars.append(list([]))
        self.BiasVars.append(list([]))
        for lvl in range(1,self.klvl+1):
            tmpFilterVars = []
            tmpBiasVars = []
            for itk in range(0,2**lvl):
                varLabel = "LVL_%02d_%04d" % (lvl, itk)

                mat = np.empty((2, self.channel_siz, self.channel_siz))
                kcen = (self.out_range[1] \
                        - self.out_range[0])/2**lvl*(itk+0.5) \
                        + self.out_range[0]
                xlen = (self.in_range[1] - \
                        self.in_range[0])/2**(self.nlvl-lvl)
                for it in range(0,NG):
                    KVal = np.exp( -2*math.pi*1j * kcen *
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

                    KVal = np.exp( -2*math.pi*1j * kcen *
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
                        name="Filter_"+varLabel )
                biasVar = tf.Variable(tf.zeros([self.channel_siz]),
                        name="Bias_"+varLabel )
                tmpFilterVars.append(filterVar)
                tmpBiasVars.append(biasVar)
            self.FilterVars.append(list(tmpFilterVars))
            self.BiasVars.append(list(tmpBiasVars))
            
        
        for lvl in range(self.klvl+1,self.nlvl+1):
            tmpFilterVars = []
            tmpBiasVars = []
            for itk in range(0,2**self.klvl):
                varLabel = "LVL_%02d_%04d" % (lvl, itk*(2**(lvl-self.klvl)))

                mat = np.empty((2, self.channel_siz, self.channel_siz))
                kcen = (self.out_range[1] \
                        - self.out_range[0])/2**lvl*(itk*(2**(lvl-self.klvl))+0.5) \
                        + self.out_range[0]
                xlen = (self.in_range[1] - \
                        self.in_range[0])/2**(self.nlvl-lvl)
                for it in range(0,NG):
                    KVal = np.exp( -2*math.pi*1j * kcen *
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

                    KVal = np.exp( -2*math.pi*1j * kcen *
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
                        name="Filter_"+varLabel )
                biasVar = tf.Variable(tf.zeros([self.channel_siz]),
                        name="Bias_"+varLabel )
                tmpFilterVars.append(filterVar)
                tmpBiasVars.append(biasVar)
            self.FilterVars.append(list(tmpFilterVars))
            self.BiasVars.append(list(tmpBiasVars))
            

        #----------------
        # Setup feature layer weights 
        self.FeaDenseVars = []
        
        for itk in range(0,2**self.klvl):
            varLabel = "Filter_Out_%04d" % (itk)
            mat = np.empty((self.channel_siz, self.out_filter_siz))
            kNodes = np.arange(0,1,2.0/self.out_filter_siz)
            klen = (self.out_range[1] - self.out_range[0])/2**self.nlvl
            koff = klen*itk*(2**(self.nlvl-self.klvl)) + self.out_range[0]
            kNodes = kNodes*klen + koff
            xlen = self.in_range[1] - self.in_range[0]
            xoff = self.in_range[0]
            xNodes = ChebNodes*xlen + xoff

            for iti in range(0,NG):
                for itj in range(0,self.out_filter_siz//2):
                    KVal = np.exp( - 2*math.pi*1j
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
                        name=varLabel )

            self.FeaDenseVars.append(denseVar)

