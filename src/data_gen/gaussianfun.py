import numpy as np
def gaussianfun(x, mulist, siglist):
    # Lengths of mulist and siglist are assumed to be the same
    len_list = len(mulist)
    gx = np.zeros(x.shape)
    for it in range(len_list):
        mu   = mulist[it]
        sig2 = siglist[it]*siglist[it]
        gx   = gx + np.exp(-np.power(x-mu,2.)/(2*sig2)) \
               / np.sqrt(2*np.pi*sig2) / len_list
    for k in range(len(x)//2):
        gx[k] = gx[-k-1]
        
    return gx

def gaussianfun2D(x,y, mu, sig, r):
    # Lengths of mulist and siglist are assumed to be the same
    gx = np.zeros([len(x),len(y)])
    for i in range(len(x)):
        for j in range(len(y)):
            gx[i][j] = np.exp((np.power((x[i]-mu[0])/sig[0],2.) +
              2*r*(x[i]-mu[0])*(y[j]-mu[1]) + np.power((y[j]-mu[1])/sig[1],2.))
                /(2*(1-r**2))) / (2*np.pi*sig[0]*sig[1]*np.sqrt(1-r**2))
    return gx