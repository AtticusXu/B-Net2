import math
import numpy as np
import tensorflow as tf

from LagrangeMat import LagrangeMat

class InvButterflyLayer(tf.keras.layers.Layer):
    #==================================================================
    # Initialize parameters in the layer
    def __init__(self,  mid_siz, out_siz,
            channel_siz = 8, nlvl = -1, prefixed = False,
             mid_range = [], out_range = []):
        super(InvButterflyLayer, self).__init__()
        self.mid_siz        = mid_siz
        self.out_siz        = out_siz
        #TODO: set the default values based on in_siz and out_siz
        self.channel_siz    = channel_siz
        self.nlvl           = nlvl
        self.klvl           = min(np.floor(np.log2(out_siz/2)).astype('int'),nlvl)
        self.mid_filter_siz = mid_siz // 2**self.nlvl
        self.out_filter_siz = out_siz // 2**self.klvl
        self.mid_range      = mid_range
        self.out_range      = out_range

        if prefixed:
            self.buildButterfly()
        else:
            self.buildRand()

        

    #==================================================================
    # Forward structure in the layer
    def call(self, in_data):
        
        
        
        # Mid Layer
        en_mid_data = tf.reshape(in_data,(np.size(in_data,0),
            self.mid_siz,2))
        de_mid_data = tf.multiply(en_mid_data,self.mid_DenseVar)
        de_mid_data = tf.reshape(de_mid_data,(np.size(in_data,0),
            self.mid_siz,2,1))
        
        # de_Preparation Layerd
        de_InInterp = []
        for i in range(2):
            de_InInterp_ir = tf.nn.conv1d(de_mid_data[:,:,i], self.de_InFilterVar,
                stride=self.mid_filter_siz//2, padding='VALID')
            de_InInterp_ir = tf.nn.relu(tf.nn.bias_add(de_InInterp_ir,
                                                  self.de_InBiasVar))
            de_InInterp.append(de_InInterp_ir)
        
        
        # de_ell Layer
        de_tfVars = []
        tmpVars = []
        tmpVars.append(de_InInterp)
        de_tfVars.append(list(tmpVars))
        
        
        for lvl in range(1,self.klvl+1):
            tmpVars = []
            for itk in range(0,2**lvl):
                Var = []
                for i in range(2):
                    Var_ir = tf.nn.conv1d(de_tfVars[lvl-1][itk//2][i],
                                       self.de_FilterVars[lvl][itk],
                                       stride=2, padding='VALID')
                    Var_ir = tf.nn.relu(tf.nn.bias_add(Var_ir,
                                                    self.de_BiasVars[lvl][itk]))
                    Var.append(Var_ir)
                tmpVars.append(Var)
            de_tfVars.append(list(tmpVars))
            
        for lvl in range(self.klvl+1,self.nlvl+1):
            tmpVars = []
            for itk in range(0,2**self.klvl):
                Var = []
                for i in range(2):
                    Var_ir = tf.nn.conv1d(de_tfVars[lvl-1][itk][i],
                                       self.de_FilterVars[lvl][itk],
                                       stride=2, padding='VALID')
                    Var_ir = tf.nn.relu(tf.nn.bias_add(Var,
                                                    self.de_BiasVars[lvl][itk]))
                    Var.append(Var_ir)
                tmpVars.append(Var)
            de_tfVars.append(list(tmpVars))
            
        # de_Feature Layer
        
        OutInterp = []
        OutInterp_r = np.reshape([], (np.size(in_data,0), 0,
                self.out_filter_siz))
        OutInterp_i = np.reshape([], (np.size(in_data,0), 0,
                self.out_filter_siz))
        for itk in range(0,2**self.klvl):
            tmpVar = de_tfVars[self.nlvl][itk][0][:,0,:]
            tmpVar = tf.matmul(tmpVar,self.de_FeaDenseVars[itk])
            tmpVar = tf.reshape(tmpVar,
                    (np.size(in_data,0),1,self.out_filter_siz))
            OutInterp_r = tf.concat([OutInterp_r, tmpVar], axis=1)
            tmpVar = de_tfVars[self.nlvl][itk][1][:,0,:]
            tmpVar = tf.matmul(tmpVar,self.de_FeaDenseVars[itk])
            tmpVar = tf.reshape(tmpVar,
                    (np.size(in_data,0),1,self.out_filter_siz))
            OutInterp_i = tf.concat([OutInterp_i, tmpVar], axis=1)
        OutInterp_r = tf.reshape(OutInterp_r,shape=(np.size(in_data,0),
            self.out_siz//2,2))
        OutInterp_i = tf.reshape(OutInterp_i,shape=(np.size(in_data,0),
            self.out_siz//2,2))
        out_data_r = tf.divide(tf.subtract(
                OutInterp_r[:,:,0],OutInterp_i[:,:,1]),self.mid_siz)
        #out_data_i = tf.add(OutInterp_r[:,:,1],OutInterp_i[:,:,0])
        out_data = tf.reshape(out_data_r,shape=(np.size(in_data,0),
            self.out_siz//2,1))
        return(out_data,OutInterp_r[:,:,1])

    #==================================================================
    # Initialize variables in the layer
    def buildRand(self):
        
        std = 0.5

        # Mid Layer
        self.mid_DenseVar = tf.Variable(tf.random_normal(
                [1,self.mid_siz,2],0,std),name = "Dense_mid")
        
        
        # Setup de_preparation layer weights
        
        self.de_InFilterVar = tf.Variable( tf.random_normal(
        [self.mid_filter_siz, 1, self.channel_siz],0,std), name="Filter_de_In")
        self.de_InBiasVar = tf.Variable( tf.zeros([self.channel_siz]),
                                         name="Bias_de_In" )
            
        # de_ell Layer
        self.de_FilterVars = []
        self.de_BiasVars = []
        self.de_FilterVars.append(list([]))
        self.de_BiasVars.append(list([]))
        for lvl in range(1,self.klvl+1):
            tmpFilterVars = []
            tmpBiasVars = []
            for itk in range(0,2**lvl):
                varLabel = "de_LVL_%02d_%04d" % (lvl, itk)
                filterVar = tf.Variable(
                        tf.random_normal([2,self.channel_siz,
                            self.channel_siz],0,std),
                        name="Filter_"+varLabel )
                biasVar = tf.Variable(tf.zeros([self.channel_siz]),
                        name="Bias_"+varLabel )
                tmpFilterVars.append(filterVar)
                tmpBiasVars.append(biasVar)
            self.de_FilterVars.append(list(tmpFilterVars))
            self.de_BiasVars.append(list(tmpBiasVars))
        
        for lvl in range(self.klvl+1,self.nlvl+1):
            tmpFilterVars = []
            tmpBiasVars = []
            for itk in range(0,2**self.klvl):
                varLabel = "de_LVL_%02d_%04d" % (lvl, itk*(2**(lvl-self.klvl)))
                filterVar = tf.Variable(
                        tf.random_normal([2,self.channel_siz,
                            self.channel_siz],0,std),
                        name="Filter_"+varLabel )
                biasVar = tf.Variable(tf.zeros([self.channel_siz]),
                        name="Bias_"+varLabel )
                tmpFilterVars.append(filterVar)
                tmpBiasVars.append(biasVar)
            self.de_FilterVars.append(list(tmpFilterVars))
            self.de_BiasVars.append(list(tmpBiasVars))

        # de_Feature Layer
        self.de_FeaDenseVars = []
        for itk in range(0,2**self.de_klvl):
            varLabel = "Filter_de_Out_%04d" % (itk)
            denseVar = tf.Variable(
                    tf.random_normal([self.channel_siz,
                        self.out_filter_siz],0,std),name=varLabel)

            self.de_FeaDenseVars.append(denseVar)
        

    #==================================================================
    # Initialize variables with coeffs in BF in the layer
    def buildButterfly(self):
     
        # Setup en_feature layer weights 
        mat = np.ones((1,self.mid_siz,2))
        self.mid_DenseVar = tf.Variable(mat.astype(np.float32),
                                        name = "Dense_mid")
        
        #----------------
        # Setup de_preparation layer weights
        NG = int(self.channel_siz/4)
        ChebNodes = (np.cos(np.array(range(2*NG-1,0,-2))/2/NG*math.pi) +
                1)/2
        xNodes = np.arange(0,1,1.0/self.mid_filter_siz)
        xlen = (self.mid_range[1] - self.mid_range[0])/2**self.nlvl
        LMat = LagrangeMat(ChebNodes,xNodes)
        kcen = np.mean(self.out_range)
        
        mat = np.empty((self.mid_filter_siz,1,self.channel_siz))
        
        for it in range(0,NG):
            KVal = np.exp(2*math.pi*1j*kcen*(xNodes-ChebNodes[it])*xlen)
            LVec = np.squeeze(LMat[:,it])
            mat[:,0,4*it]   =  np.multiply(KVal.real,LVec)
            mat[:,0,4*it+1] =  np.multiply(KVal.imag,LVec)
            mat[:,0,4*it+2] = -np.multiply(KVal.real,LVec)
            mat[:,0,4*it+3] = -np.multiply(KVal.imag,LVec)

        self.de_InFilterVar = tf.Variable( mat.astype(np.float32),
                name="Filter_de_In" )
        self.de_InBiasVar = tf.Variable( tf.zeros([self.channel_siz]),
                name="Bias_de_In" )

        #----------------
        # Setup en_ell layer weights
        ChebNodes = (np.cos(np.array(range(2*NG-1,0,-2))/2/NG*math.pi) +
                1)/2
        x1Nodes = ChebNodes/2
        x2Nodes = ChebNodes/2 + 1/2
        LMat1 = LagrangeMat(ChebNodes,x1Nodes)
        LMat2 = LagrangeMat(ChebNodes,x2Nodes)

        self.de_FilterVars = []
        self.de_BiasVars = []
        self.de_FilterVars.append(list([]))
        self.de_BiasVars.append(list([]))
        for lvl in range(1,self.klvl+1):
            tmpFilterVars = []
            tmpBiasVars = []
            for itk in range(0,2**lvl):
                varLabel = "de_LVL_%02d_%04d" % (lvl, itk)

                mat = np.empty((2, self.channel_siz, self.channel_siz))
                kcen = (self.out_range[1] \
                        - self.out_range[0])/2**lvl*(itk+0.5) \
                        + self.out_range[0]
                xlen = (self.mid_range[1] - \
                        self.mid_range[0])/2**(self.nlvl-lvl)
                for it in range(0,NG):
                    KVal = np.exp(2*math.pi*1j * kcen *
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

                    KVal = np.exp(2*math.pi*1j * kcen *
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
            self.de_FilterVars.append(list(tmpFilterVars))
            self.de_BiasVars.append(list(tmpBiasVars))
            
        
        for lvl in range(self.klvl+1,self.nlvl+1):
            tmpFilterVars = []
            tmpBiasVars = []
            for itk in range(0,2**self.klvl):
                varLabel = "de_LVL_%02d_%04d" % (lvl, itk*(2**(lvl-self.klvl)))

                mat = np.empty((2, self.channel_siz, self.channel_siz))
                kcen = (self.out_range[1] \
                        - self.out_range[0])/2**lvl*(itk*(2**(lvl-self.klvl))+0.5) \
                        + self.out_range[0]
                xlen = (self.mid_range[1] - \
                        self.mid_range[0])/2**(self.nlvl-lvl)
                for it in range(0,NG):
                    KVal = np.exp(2*math.pi*1j * kcen *
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

                    KVal = np.exp(2*math.pi*1j * kcen *
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
            self.de_FilterVars.append(list(tmpFilterVars))
            self.de_BiasVars.append(list(tmpBiasVars))
            

        #----------------
        # Setup en_feature layer weights 
        self.de_FeaDenseVars = []
        
        for itk in range(0,2**self.klvl):
            varLabel = "de_Filter_Out_%04d" % (itk)
            mat = np.empty((self.channel_siz, self.out_filter_siz))
            kNodes = np.arange(0,1,2.0/self.out_filter_siz)
            klen = (self.out_range[1] - self.out_range[0])/2**self.nlvl
            koff = klen*itk*(2**(self.nlvl-self.klvl)) + self.out_range[0]
            kNodes = kNodes*klen + koff
            xlen = self.mid_range[1] - self.mid_range[0]
            xoff = self.mid_range[0]
            xNodes = ChebNodes*xlen + xoff

            for iti in range(0,NG):
                for itj in range(0,self.out_filter_siz//2):
                    KVal = np.exp( 2*math.pi*1j
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

            self.de_FeaDenseVars.append(denseVar)
        
        