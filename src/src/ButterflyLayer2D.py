import math
import numpy as np
import tensorflow as tf

from LagrangeMat import LagrangeMat

class ButterflyLayer2D(tf.keras.layers.Layer):
    #==================================================================
    # Initialize parameters in the layer
    def __init__(self, in_siz_x, in_siz_y, out_siz_u, out_siz_v,
            channel_siz = 3, nlvl = -1, klvl = -1, prefixed = False,
            in_range = [], out_range = []):
        super(ButterflyLayer2D, self).__init__()
        self.in_siz_x         = in_siz_x
        self.in_siz_y         = in_siz_y
        self.out_siz_u        = out_siz_u
        self.out_siz_v        = out_siz_v
        #TODO: set the default values based on in_siz and out_siz
        self.channel_siz      = channel_siz
        self.nlvl             = nlvl
        self.klvl             = klvl
        self.in_filter_siz_x  = in_siz_x // 2**nlvl
        self.in_filter_siz_y  = in_siz_y // 2**nlvl
        self.out_filter_siz_u = out_siz_u // 2**self.klvl
        self.out_filter_siz_v = out_siz_v // 2**self.klvl
        self.in_range         = in_range
        self.out_range        = out_range
        print(self.in_filter_siz_x)
        print(self.out_filter_siz_v)
        if prefixed:
            self.buildButterfly()
        else:
            self.buildRand()

        

    #==================================================================
    # Forward structure in the layer
    def call(self, in_data):
        
        
        # Preparation Layer
        InInterp = tf.nn.conv2d(in_data, self.InFilterVar,
                strides=[1,self.in_filter_siz_x,self.in_filter_siz_y,1],
                padding='VALID')
        InInterp = tf.nn.relu(tf.nn.bias_add(InInterp, self.InBiasVar))
        #in_data[1,2^Lm,2^Lm,1]
        #self.InFilterVar[m,m,1,r^2]
        #stride[1,m,m,1]
        #InInterp[1,2^L,2^L,r^2]
        
        # ell Layer
        tfVars = []
        tmpvVars = []
        tmpuVars = []
        tmpvVars.append(InInterp)
        tmpuVars.append(tmpvVars)
        tfVars.append(list(tmpuVars))

        for lvl in range(1,self.klvl+1):
            tmpuVars = []
            for itu in range(0,2**lvl):
                tmpvVars = []
                for itv in range(0,2**lvl):
                    Var = tf.nn.conv2d(tfVars[lvl-1][itu//2][itv//2],
                                       self.FilterVars[lvl][itu][itv],
                                       strides=[1,2,2,1], padding='VALID')
                    Var = tf.nn.relu(tf.nn.bias_add(Var,
                                       self.BiasVars[lvl][itu][itv]))
                    tmpvVars.append(Var)
                tmpuVars.append(tmpvVars)
            tfVars.append(list(tmpuVars))
            
        for lvl in range(self.klvl+1,self.nlvl+1):
            tmpuVars = []
            for itu in range(0,2**self.klvl):
                tmpvVars = []
                for itv in range(0,2**self.klvl):
                    Var = tf.nn.conv2d(tfVars[lvl-1][itu][itv],
                                       self.FilterVars[lvl][itu][itv],
                                       strides=[1,2,2,1], padding='VALID')
                    Var = tf.nn.relu(tf.nn.bias_add(Var,
                                       self.BiasVars[lvl][itu][itv]))
                    tmpvVars.append(Var)
                tmpuVars.append(tmpvVars)
            tfVars.append(list(tmpuVars))
            

        # Feature Layer
        OutuInterp = np.reshape([], (np.size(in_data,0), 0, 2**self.klvl,
                self.out_filter_siz_u,self.out_filter_siz_v,2))
        
        for itu in range(0,2**self.klvl):
            OutvInterp = np.reshape([], (np.size(in_data,0), 0,
                self.out_filter_siz_u,self.out_filter_siz_v,2))
            for itv in range(0,2**self.klvl):
                tmpVar = tfVars[self.nlvl][itu][itv][:,0,0,:]
                tmpVar_r = tf.matmul(tmpVar,self.FeaDenseVars[itu][itv][0])
                tmpVar_i = tf.matmul(tmpVar,self.FeaDenseVars[itu][itv][1])
                tmpVar_r = tf.reshape(tmpVar_r,
                    (np.size(in_data,0),1,
                     self.out_filter_siz_u,self.out_filter_siz_v,1))
                tmpVar_i = tf.reshape(tmpVar_i,
                    (np.size(in_data,0),1,
                     self.out_filter_siz_u,self.out_filter_siz_v,1))
                tmpVar = tf.concat([tmpVar_r,tmpVar_i],axis=4)
                OutvInterp = tf.concat([OutvInterp, tmpVar], axis=1)
            OutvInterp = tf.reshape(OutvInterp,
                                    (np.size(in_data,0),1,2**self.klvl,
                     self.out_filter_siz_u,self.out_filter_siz_v,2))
            OutuInterp = tf.concat([OutuInterp, OutvInterp], axis=1)
        OutInterp = tf.transpose(OutuInterp,[0,1,3,2,4,5])
        out_data = tf.reshape(OutInterp,shape=(np.size(in_data,0),
            self.out_siz_u,self.out_siz_v,2))
        return(out_data)

    #==================================================================
    # Initialize variables in the layer
    def buildRand(self):
        
        # Preparation Layer
        self.InFilterVar = tf.Variable( tf.random_normal(
            [self.in_filter_siz_x, self.in_filter_siz_y,
            1, 4*np.square(self.channel_siz)]),
            name="Filter_In" )
        self.InBiasVar = tf.Variable( 
            tf.zeros([4*np.square(self.channel_siz)]),
            name="Bias_In" )
        
        # ell Layer
        self.FilterVars = []
        self.BiasVars = []
        self.FilterVars.append(list([]))
        self.BiasVars.append(list([]))
        for lvl in range(1,self.klvl+1):
            tmpuFilterVars = []
            tmpuBiasVars = []
            for itu in range(0,2**lvl):
                tmpvFilterVars = []
                tmpvBiasVars = []
                for itv in range(0,2**lvl):
                    varLabel = "LVL_%02d_%04d_%04d" % (lvl, itu, itv)
                    filterVar = tf.Variable(
                            tf.random_normal([2,2,4*np.square(self.channel_siz),
                                              4*np.square(self.channel_siz)]),
                                name="Filter_"+varLabel )
                    biasVar = tf.Variable(
                            tf.zeros([4*np.square(self.channel_siz)]),
                                name="Bias_"+varLabel )
                    tmpvFilterVars.append(filterVar)
                    tmpvBiasVars.append(biasVar)
                tmpuFilterVars.append(tmpvFilterVars)
                tmpuBiasVars.append(tmpvBiasVars)
            self.FilterVars.append(list(tmpuFilterVars))
            self.BiasVars.append(list(tmpuBiasVars))
        
        for lvl in range(self.klvl+1,self.nlvl+1):
            tmpuFilterVars = []
            tmpuBiasVars = []
            for itu in range(0,2**self.klvl):
                tmpvFilterVars = []
                tmpvBiasVars = []
                for itv in range(0,2**self.klvl):
                    varLabel = "LVL_%02d_%04d_%04d" % (lvl,
                                        itu*(2**(lvl-self.klvl)),
                                        itv*(2**(lvl-self.klvl)))
                    filterVar = tf.Variable(
                            tf.random_normal([2,2,4*np.square(self.channel_siz),
                                              4*np.square(self.channel_siz)]),
                                name="Filter_"+varLabel )
                    biasVar = tf.Variable(
                            tf.zeros([4*np.square(self.channel_siz)]),
                                name="Bias_"+varLabel )
                    tmpvFilterVars.append(filterVar)
                    tmpvBiasVars.append(biasVar)
                tmpuFilterVars.append(tmpvFilterVars)
                tmpuBiasVars.append(tmpvBiasVars)
            self.FilterVars.append(list(tmpuFilterVars))
            self.BiasVars.append(list(tmpuBiasVars))

        # Feature Layer
        self.FeaDenseVars = []
        for itu in range(0,2**self.klvl):
            tmpvdenseVars = []
            for itv in range(0,2**self.klvl):
                tmpirdenseVars = []
                for ir in range(0,2):
                    varLabel = "Filter_Out_%04d_%04d_%04d" % (itu,itv,ir)
                    denseVar = tf.Variable(
                            tf.random_normal([4*np.square(self.channel_siz),
                            self.out_filter_siz_u*self.out_filter_siz_v]),
                                name=varLabel)
                    tmpirdenseVars.append(denseVar)
                tmpvdenseVars.append(tmpirdenseVars)
            self.FeaDenseVars.append(tmpvdenseVars)

 #==================================================================
    # Initialize variables with coeffs in BF in the layer
    def buildButterfly(self):
        
        ChebNodes = (np.cos(np.array(range(2*self.channel_siz-1,0,-2))\
                            /2/self.channel_siz*math.pi) + 1)/2
        xNodes = np.arange(0,1,1.0/self.in_filter_siz_x)
        yNodes = np.arange(0,1,1.0/self.in_filter_siz_y)
        LMat = LagrangeMat(ChebNodes,xNodes)

        #----------------
        # Setup preparation layer weights
        mat = np.empty((self.in_filter_siz_x, self.in_filter_siz_y,
            1, 4*np.square(self.channel_siz)))
        ucen = np.mean(self.out_range[0,:])
        vcen = np.mean(self.out_range[1,:])
        xlen = (self.in_range[0][1] - self.in_range[0][0])/2**self.nlvl
        ylen = (self.in_range[1][1] - self.in_range[1][0])/2**self.nlvl
        for it_x in range(0,self.in_filter_siz_x):
            for it_y in range(0,self.in_filter_siz_y):
                for it_u in range(0,self.channel_siz):
                    for it_v in range(0,self.channel_siz):
                        KVal = np.exp(-2*math.pi*1j* \
                              (ucen*(xNodes[it_x]-ChebNodes[it_u])*xlen + 
                               vcen*(yNodes[it_y]-ChebNodes[it_v])*ylen))
                        LVec_1 = LMat[it_x,it_u]
                        LVec_2 = LMat[it_y,it_v]
                        mat[it_x,it_y,0,4*self.channel_siz*it_u + 4*it_v] \
                           =np.multiply(np.multiply(KVal.real,LVec_1),LVec_2)
                        mat[it_x,it_y,0,4*self.channel_siz*it_u + 4*it_v + 1] \
                           =np.multiply(np.multiply(KVal.imag,LVec_1),LVec_2)
                        mat[it_x,it_y,0,4*self.channel_siz*it_u + 4*it_v + 2] \
                           =-np.multiply(np.multiply(KVal.real,LVec_1),LVec_2)
                        mat[it_x,it_y,0,4*self.channel_siz*it_u + 4*it_v + 3] \
                           =-np.multiply(np.multiply(KVal.imag,LVec_1),LVec_2)

        self.InFilterVar = tf.Variable( mat.astype(np.float32),
                name="Filter_In" )
        self.InBiasVar = tf.Variable(tf.zeros([4*np.square(self.channel_siz)]),
                name="Bias_In" )

        #----------------
        # Setup ell layer weights
        ChebNodes = (np.cos(np.array(range(2*self.channel_siz-1,0,-2))\
                            /2/self.channel_siz*math.pi) + 1)/2
        xy1Nodes = ChebNodes/2
        xy2Nodes = ChebNodes/2 + 1/2
        LMat1 = LagrangeMat(ChebNodes,xy1Nodes)
        LMat2 = LagrangeMat(ChebNodes,xy2Nodes)
        xy1Nodes = np.expand_dims(xy1Nodes,axis=0)
        xy2Nodes = np.expand_dims(xy2Nodes,axis=0)
        xyNodes = np.concatenate((xy1Nodes,xy2Nodes),axis = 0)
        LMat1 = np.expand_dims(LMat1,axis=0)
        LMat2 = np.expand_dims(LMat2,axis=0)
        LMat = np.concatenate((LMat1,LMat2),axis = 0)
        self.FilterVars = []
        self.BiasVars = []
        self.FilterVars.append(list([]))
        self.BiasVars.append(list([]))
        for lvl in range(1,self.klvl+1):
            tmpuFilterVars = []
            tmpuBiasVars = []
            for itu in range(0,2**lvl):
                tmpvFilterVars = []
                tmpvBiasVars = []
                for itv in range(0,2**lvl):
                    varLabel = "LVL_%02d_%04d_%04d" % (lvl, itu, itv)

                    mat = np.empty((2,2,4*np.square(self.channel_siz),
                                              4*np.square(self.channel_siz)))
                    ucen = (self.out_range[0][1] \
                            - self.out_range[0][0])/2**lvl*(itu+0.5) \
                            + self.out_range[0][0]
                    vcen = (self.out_range[1][1] \
                            - self.out_range[1][0])/2**lvl*(itv+0.5) \
                            + self.out_range[1][0]
                    xlen = (self.in_range[0][1] - \
                                self.in_range[0][0])/2**(self.nlvl-lvl)
                    ylen = (self.in_range[1][1] - \
                                self.in_range[1][0])/2**(self.nlvl-lvl)
                    for it_1 in range(0,self.channel_siz):
                        for it_2 in range(0,self.channel_siz):
                            for ot_1 in range(0,self.channel_siz):
                                for ot_2 in range(0,self.channel_siz):
                                    for itx in range(0,2):
                                        for ity in range(0,2):
                                            KVal = np.exp(-2*math.pi*1j * \
                                                (ucen*(xyNodes[itx][it_1]- \
                                                 ChebNodes[ot_1])*xlen + \
                                                 vcen*(xyNodes[ity][it_2]- \
                                                 ChebNodes[ot_2])*ylen))
                                            LVec_1 = LMat[itx][it_1][ot_1]
                                            LVec_2 = LMat[ity][it_2][ot_2]
                                            mat[itx,ity,
                                                4*self.channel_siz*it_1 + 4*it_2,
                                                4*self.channel_siz*ot_1 + 4*ot_2] = \
                                            np.multiply(np.multiply(KVal.real,LVec_1),LVec_2)
                                            mat[itx,ity,
                                                4*self.channel_siz*it_1 + 4*it_2 + 1,
                                                4*self.channel_siz*ot_1 + 4*ot_2] = \
                                            -np.multiply(np.multiply(KVal.imag,LVec_1),LVec_2)
                                            mat[itx,ity,
                                                4*self.channel_siz*it_1 + 4*it_2 + 2,
                                                4*self.channel_siz*ot_1 + 4*ot_2] = \
                                            -np.multiply(np.multiply(KVal.real,LVec_1),LVec_2)
                                            mat[itx,ity,
                                                4*self.channel_siz*it_1 + 4*it_2 + 3,
                                                4*self.channel_siz*ot_1 + 4*ot_2] = \
                                            np.multiply(np.multiply(KVal.imag,LVec_1),LVec_2)
                                            mat[itx,ity,
                                                4*self.channel_siz*it_1 + 4*it_2,
                                                4*self.channel_siz*ot_1 + 4*ot_2 + 1] = \
                                            np.multiply(np.multiply(KVal.imag,LVec_1),LVec_2)
                                            mat[itx,ity,
                                                4*self.channel_siz*it_1 + 4*it_2 + 1,
                                                4*self.channel_siz*ot_1 + 4*ot_2 + 1] = \
                                            np.multiply(np.multiply(KVal.real,LVec_1),LVec_2)
                                            mat[itx,ity,
                                                4*self.channel_siz*it_1 + 4*it_2 + 2,
                                                4*self.channel_siz*ot_1 + 4*ot_2 + 1] = \
                                            -np.multiply(np.multiply(KVal.imag,LVec_1),LVec_2)
                                            mat[itx,ity,
                                                4*self.channel_siz*it_1 + 4*it_2 + 3,
                                                4*self.channel_siz*ot_1 + 4*ot_2 + 1] = \
                                            -np.multiply(np.multiply(KVal.real,LVec_1),LVec_2)

                                            mat[itx,ity,:,(4*self.channel_siz*ot_1 + 4*ot_2 + 2,
                                                           4*self.channel_siz*ot_1 + 4*ot_2 + 3)]= \
                                            -mat[itx,ity,:,(4*self.channel_siz*ot_1 + 4*ot_2,
                                                             4*self.channel_siz*ot_1 + 4*ot_2 + 1)]
                    filterVar = tf.Variable( mat.astype(np.float32),
                        name="Filter_"+varLabel )
                    biasVar = tf.Variable(tf.zeros([4*np.square(self.channel_siz)]),
                        name="Bias_"+varLabel )
                    tmpvFilterVars.append(filterVar)
                    tmpvBiasVars.append(biasVar)
                tmpuFilterVars.append(tmpvFilterVars)
                tmpuBiasVars.append(tmpvBiasVars)
            self.FilterVars.append(list(tmpuFilterVars))
            self.BiasVars.append(list(tmpuBiasVars))
            
        for lvl in range(self.klvl+1,self.nlvl+1):
            tmpuFilterVars = []
            tmpuBiasVars = []
            for itu in range(0,2**lvl):
                tmpvFilterVars = []
                tmpvBiasVars = []
                for itv in range(0,2**lvl):
                    varLabel = "LVL_%02d_%04d_%04d" % (lvl, 
                                itu*(2**(lvl-self.klvl)),
                                itv*(2**(lvl-self.klvl)))

                    mat = np.empty((2,2,4*np.square(self.channel_siz),
                                              4*np.square(self.channel_siz)))
                    ucen = (self.out_range[0][1] \
                            - self.out_range[0][0])/2**lvl*(itu*(2**(lvl-self.klvl))+0.5) \
                            + self.out_range[0][0]
                    vcen = (self.out_range[1][1] \
                            - self.out_range[1][0])/2**lvl*(itv*(2**(lvl-self.klvl))+0.5) \
                            + self.out_range[1][0]
                    xlen = (self.in_range[0][1] - \
                                self.in_range[0][0])/2**(self.nlvl-lvl)
                    ylen = (self.in_range[1][1] - \
                                self.in_range[1][0])/2**(self.nlvl-lvl)
                    for it_1 in range(0,self.channel_siz):
                        for it_2 in range(0,self.channel_siz):
                            for ot_1 in range(0,self.channel_siz):
                                for ot_2 in range(0,self.channel_siz):
                                    for itx in range(0,2):
                                        for ity in range(0,2):
                                            KVal = np.exp(-2*math.pi*1j * \
                                                (ucen*(xyNodes[itx][it_1]- \
                                                 ChebNodes[ot_1])*xlen + \
                                                 vcen*(xyNodes[ity][it_2]- \
                                                 ChebNodes[ot_2])*ylen))
                                            LVec_1 = LMat[itx][it_1][ot_1]
                                            LVec_2 = LMat[ity][it_2][ot_2]
                                            mat[itx,ity,
                                                4*self.channel_siz*it_1 + 4*it_2,
                                                4*self.channel_siz*ot_1 + 4*ot_2] = \
                                            np.multiply(np.multiply(KVal.real,LVec_1),LVec_2)
                                            mat[itx,ity,
                                                4*self.channel_siz*it_1 + 4*it_2 + 1,
                                                4*self.channel_siz*ot_1 + 4*ot_2] = \
                                            -np.multiply(np.multiply(KVal.imag,LVec_1),LVec_2)
                                            mat[itx,ity,
                                                4*self.channel_siz*it_1 + 4*it_2 + 2,
                                                4*self.channel_siz*ot_1 + 4*ot_2] = \
                                            -np.multiply(np.multiply(KVal.real,LVec_1),LVec_2)
                                            mat[itx,ity,
                                                4*self.channel_siz*it_1 + 4*it_2 + 3,
                                                4*self.channel_siz*ot_1 + 4*ot_2] = \
                                            np.multiply(np.multiply(KVal.imag,LVec_1),LVec_2)
                                            mat[itx,ity,
                                                4*self.channel_siz*it_1 + 4*it_2,
                                                4*self.channel_siz*ot_1 + 4*ot_2 + 1] = \
                                            np.multiply(np.multiply(KVal.imag,LVec_1),LVec_2)
                                            mat[itx,ity,
                                                4*self.channel_siz*it_1 + 4*it_2 + 1,
                                                4*self.channel_siz*ot_1 + 4*ot_2 + 1] = \
                                            np.multiply(np.multiply(KVal.real,LVec_1),LVec_2)
                                            mat[itx,ity,
                                                4*self.channel_siz*it_1 + 4*it_2 + 2,
                                                4*self.channel_siz*ot_1 + 4*ot_2 + 1] = \
                                            -np.multiply(np.multiply(KVal.imag,LVec_1),LVec_2)
                                            mat[itx,ity,
                                                4*self.channel_siz*it_1 + 4*it_2 + 3,
                                                4*self.channel_siz*ot_1 + 4*ot_2 + 1] = \
                                            -np.multiply(np.multiply(KVal.real,LVec_1),LVec_2)

                                            mat[itx,ity,:,(4*self.channel_siz*ot_1 + 4*ot_2 + 2,
                                                           4*self.channel_siz*ot_1 + 4*ot_2 + 3)]= \
                                            -mat[itx,ity,:,(4*self.channel_siz*ot_1 + 4*ot_2,
                                                             4*self.channel_siz*ot_1 + 4*ot_2 + 1)]

                    filterVar = tf.Variable( mat.astype(np.float32),
                        name="Filter_"+varLabel )
                    biasVar = tf.Variable(tf.zeros([4*np.square(self.channel_siz)]),
                        name="Bias_"+varLabel )
                    tmpvFilterVars.append(filterVar)
                    tmpvBiasVars.append(biasVar)
                tmpuFilterVars.append(tmpvFilterVars)
                tmpuBiasVars.append(tmpvBiasVars)
            self.FilterVars.append(list(tmpuFilterVars))
            self.BiasVars.append(list(tmpuBiasVars))
            

        #----------------
        # Setup feature layer weights 
        self.FeaDenseVars = []
        
        for itu in range(0,2**self.klvl):
            tmpvdenseVars = []
            for itv in range(0,2**self.klvl):
                tmpirdenseVars = []
                varLabel_r = "Filter_Out_%04d_%04d_%04d" % (itu,itv,0)
                varLabel_i = "Filter_Out_%04d_%04d_%04d" % (itu,itv,1)
                mat_r = np.empty((4*np.square(self.channel_siz),
                        self.out_filter_siz_u*self.out_filter_siz_v))
                mat_i = np.empty((4*np.square(self.channel_siz),
                        self.out_filter_siz_u*self.out_filter_siz_v))
                uNodes = np.arange(0,1,1.0/self.out_filter_siz_u)
                vNodes = np.arange(0,1,1.0/self.out_filter_siz_v)
                ulen = (self.out_range[0][1] - self.out_range[0][0])/2**self.nlvl
                vlen = (self.out_range[1][1] - self.out_range[1][0])/2**self.nlvl
                uoff = ulen*itu*(2**(self.nlvl-self.klvl)) + self.out_range[0][0]
                voff = vlen*itv*(2**(self.nlvl-self.klvl)) + self.out_range[1][0]
                uNodes = uNodes*ulen + uoff
                vNodes = vNodes*vlen + voff
                xlen = self.in_range[0][1] - self.in_range[0][0]
                ylen = self.in_range[1][1] - self.in_range[1][0]
                xoff = self.in_range[0][0]
                yoff = self.in_range[1][0]
                xNodes = ChebNodes*xlen + xoff
                yNodes = ChebNodes*ylen + yoff

                for it_1 in range(0,self.channel_siz):
                    for it_2 in range(0,self.channel_siz):
                        for ot_1 in range(0,self.out_filter_siz_u):
                            for ot_2 in range(0,self.out_filter_siz_v):
                                KVal = np.exp( - 2 * math.pi*1j * \
                                          (uNodes[ot_1]*xNodes[it_1]+\
                                           vNodes[ot_2]*yNodes[it_1]))
                                mat_r[4*self.channel_siz*it_1 + 4*it_2,
                                   ot_1*self.out_filter_siz_v + ot_2]=KVal.real
                                mat_r[4*self.channel_siz*it_1 + 4*it_2 + 1,
                                   ot_1*self.out_filter_siz_v + ot_2]= - KVal.imag
                                mat_r[4*self.channel_siz*it_1 + 4*it_2 + 2,
                                   ot_1*self.out_filter_siz_v + ot_2]= - KVal.real
                                mat_r[4*self.channel_siz*it_1 + 4*it_2 + 3,
                                   ot_1*self.out_filter_siz_v + ot_2]=KVal.imag
                                mat_i[4*self.channel_siz*it_1 + 4*it_2,
                                   ot_1*self.out_filter_siz_v + ot_2]=KVal.imag
                                mat_i[4*self.channel_siz*it_1 + 4*it_2 + 1,
                                   ot_1*self.out_filter_siz_v + ot_2]=KVal.real
                                mat_i[4*self.channel_siz*it_1 + 4*it_2 + 2,
                                   ot_1*self.out_filter_siz_v + ot_2]= - KVal.imag
                                mat_i[4*self.channel_siz*it_1 + 4*it_2 + 2,
                                   ot_1*self.out_filter_siz_v + ot_2]= - KVal.real
                
                denseVar_r = tf.Variable( mat_r.astype(np.float32),
                        name=varLabel_r )
                denseVar_i = tf.Variable( mat_i.astype(np.float32),
                        name=varLabel_i )
                tmpirdenseVars.append(denseVar_r)
                tmpirdenseVars.append(denseVar_i)
                tmpvdenseVars.append(tmpirdenseVars)
            self.FeaDenseVars.append(tmpvdenseVars)



   