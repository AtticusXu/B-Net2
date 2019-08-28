import numpy as np
import math
def gen_uni_data(freqmag,freqidx,siz,sig):
    N = len(freqmag)
    K = len(freqidx)
    a = 6*np.sqrt(math.pi)*sig/math.erf(K/sig)

    freqmag = np.tile(np.reshape(freqmag,[1,N]),(siz,1))
    consty = np.random.uniform(-np.sqrt(a),np.sqrt(a),[siz,1])
    zeroy = np.zeros([siz,1])
    if N % 2 == 0:
        halfy = np.random.uniform(-np.sqrt(a/2),np.sqrt(a/2),[siz,N//2-1])
        realy = np.concatenate((consty,halfy,zeroy,halfy[:,::-1]),axis=1)
        halfy = np.random.uniform(-np.sqrt(a/2),np.sqrt(a/2),[siz,N//2-1])
        imagy = np.concatenate((zeroy,halfy,zeroy,-halfy[:,::-1]),axis=1)
    else:
        halfy = np.random.uniform(-np.sqrt(a/2),np.sqrt(a/2),[siz,N//2])
        realy = np.concatenate((consty,halfy,halfy[:,::-1]),axis=1)
        halfy = np.random.uniform(-np.sqrt(a/2),np.sqrt(a/2),[siz,N//2])
        imagy = np.concatenate((zeroy,halfy,-halfy[:,::-1]),axis=1)

    realy = realy*freqmag
    imagy = imagy*freqmag
    y = realy + imagy*1j
    xdata = np.reshape(np.fft.ifft(y,N,1).real,(siz,N,1),order='F')
    y = np.reshape(np.fft.fft(xdata,N,1),(siz,1,N),order='F')
    realy = y.real[:,:,freqidx]
    imagy = y.imag[:,:,freqidx]
    ydata = np.reshape(np.concatenate((realy,imagy),axis=1),(siz,-1,1),order='F')
    ynorm = np.squeeze(np.linalg.norm(ydata,2,1))
    xdata = np.float32(xdata)
    ydata = np.float32(ydata)
    
    return xdata,ydata,ynorm

def gen_ede_uni_data(freqmag,freqidx,siz,sig):
    N = len(freqmag)
    K = len(freqidx)
    a = 100

    freqmag = np.tile(np.reshape(freqmag,[1,N]),(siz,1))
    consty = np.random.uniform(-np.sqrt(a),np.sqrt(a),[siz,1])
    zeroy = np.zeros([siz,1])
    if N % 2 == 0:
        halfy = np.random.uniform(-np.sqrt(a/2),np.sqrt(a/2),[siz,N//2-1])
        realy = np.concatenate((consty,halfy,zeroy,halfy[:,::-1]),axis=1)
        halfy = np.random.uniform(-np.sqrt(a/2),np.sqrt(a/2),[siz,N//2-1])
        imagy = np.concatenate((zeroy,halfy,zeroy,-halfy[:,::-1]),axis=1)
    else:
        halfy = np.random.uniform(-np.sqrt(a/2),np.sqrt(a/2),[siz,N//2])
        realy = np.concatenate((consty,halfy,halfy[:,::-1]),axis=1)
        halfy = np.random.uniform(-np.sqrt(a/2),np.sqrt(a/2),[siz,N//2])
        imagy = np.concatenate((zeroy,halfy,-halfy[:,::-1]),axis=1)

    realy = realy*freqmag
    imagy = imagy*freqmag
    y = realy + imagy*1j
    xdata = np.reshape(np.fft.ifft(y,N,1).real,(siz,N,1),order='F')
    y = np.reshape(np.fft.fft(xdata,N,1),(siz,1,N),order='F')
    realy = y.real[:,:,freqidx]
    imagy = y.imag[:,:,freqidx]
    ydata = np.reshape(np.concatenate((realy,imagy),axis=1),(siz,-1,1),order='F')
    xnorm = np.squeeze(np.linalg.norm(xdata,2,1))
    ynorm = np.squeeze(np.linalg.norm(ydata,2,1))
    xdata = np.float32(xdata)
    ydata = np.float32(ydata)
    return xdata,ydata,xnorm,ynorm
    
def gen_2D_straight_data(siz_x,siz_y,siz_u,siz_v, N):
    
    xdata = np.zeros([N,siz_x,siz_y,1])
    xdata[0,0,0,0] = 1
    ydata = np.zeros([N,siz_u,siz_v,2])
    for u in range(siz_u):
        for v in range(siz_v):
            for x in range(siz_x):
                for y in range(siz_y):
                    ydata[:,u,v,0] = ydata[:,u,v,0] + \
                    (np.exp(-2*math.pi*1j*(u*x+v*y)/siz_x)*xdata[:,x,y,0]).real
                    ydata[:,u,v,1] = ydata[:,u,v,1] + \
                    (np.exp(-2*math.pi*1j*(u*x+v*y)/siz_x)*xdata[:,x,y,0]).imag
    return xdata,ydata

def gen_2D_gaussian_data(freqmag,out_siz,siz,sig):
    M = freqmag.shape[0]
    N = freqmag.shape[1]
    a = 2**(-20)

    freqmag = np.tile(np.reshape(freqmag,[1,M,N]),(siz,1))

    xdata = np.random.uniform(-np.sqrt(a/2),np.sqrt(a/2),[siz,M,N])
    xdata = xdata * freqmag
    
    ydata = np.fft.fftshift(np.fft.fft2(xdata), axes=(1,2))
    realy = ydata.real
    imagy = ydata.imag
    realy = np.reshape(realy[:,0:out_siz,0:out_siz],(siz,out_siz,out_siz),order='F')
    imagy = np.reshape(imagy[:,0:out_siz,0:out_siz],(siz,out_siz,out_siz),order='F')
    ydata = np.empty((siz,out_siz,out_siz,2))
    ydata[:,:,:,0] = realy
    ydata[:,:,:,1] = imagy
    xdata = np.reshape(xdata,[siz,M,N,1],order='F')
    xdata = np.float32(xdata)
    ydata = np.float32(ydata)
    return xdata,ydata


def gen_degree_data(in_size,in_range,out_size,out_range):
    xdata = np.zeros([in_size,in_size,1])
    ydata = np.zeros([in_size,out_size,1])
    for i in range(0,in_size):
        xdata[i][i][0] = 1
        for j in range(0,out_size//2):
            t = in_range[0]+(in_range[1] - in_range[0])*i/in_size
            k = out_range[0]+(out_range[1] - out_range[0])*j/(out_size//2)
            ydata[i][2*j] = np.exp(-2*math.pi*1j*t*k).real
            ydata[i][2*j+1] = np.exp(-2*math.pi*1j*t*k).imag
    xdata = np.float32(xdata)
    ydata = np.float32(ydata)
    return xdata,ydata
    
def gen_gaussian_data(freqmag,freqidx,siz):
    N = len(freqmag)
    freqmag = np.tile(np.reshape(freqmag,[1,N]),(siz,1))
    consty = np.random.normal(0,np.sqrt(2.0),[siz,1])
    zeroy = np.zeros([siz,1])
    if N % 2 == 0:
        halfy = np.random.normal(0,1,[siz,N//2-1])
        realy = np.concatenate((consty,halfy,zeroy,halfy[:,::-1]),axis=1)
        halfy = np.random.normal(0,1,[siz,N//2-1])
        imagy = np.concatenate((zeroy,halfy,zeroy,-halfy[:,::-1]),axis=1)
    else:
        halfy = np.random.normal(0,1,[siz,N//2])
        realy = np.concatenate((consty,halfy,halfy[:,::-1]),axis=1)
        halfy = np.random.normal(0,1,[siz,N//2])
        imagy = np.concatenate((zeroy,halfy,-halfy[:,::-1]),axis=1)

    realy = realy*freqmag
    imagy = imagy*freqmag
    y = realy + imagy*1j
    xdata = np.reshape(np.fft.ifft(y,N,1).real,(siz,N,1),order='F')
    realy = np.reshape(realy[:,freqidx],(siz,1,-1),order='F')
    imagy = np.reshape(imagy[:,freqidx],(siz,1,-1),order='F')
    ydata = np.reshape(np.concatenate((realy,imagy),axis=1),(siz,-1,1),order='F')
    xdata = np.float32(xdata)
    ydata = np.float32(ydata)
    return xdata,ydata

def gen_energy_data(N,siz):
    DW = np.fft.fftfreq(N)
    DW[DW==0] = np.inf
    DW = 1/DW
    print(DW)
    tmp = np.random.normal(0,1,[siz,N//8])
    xdata = np.fft.irfft(np.fft.rfft(tmp,axis=1),N,1)
    ydata = np.sum(np.absolute(np.multiply(
        np.fft.fft(xdata,axis=1), DW))**2,axis=1)/N**2
    xdata = np.float32(np.reshape(xdata,[siz,N,1]))
    ydata = np.float32(np.reshape(ydata,[siz,1,1]))
    return xdata,ydata

def gen_energy2d_data(N,siz):
    DW = np.fft.fftfreq(N)
    DW[DW==0] = np.inf
    DW = 1/DW
    DW2 = np.outer(DW,DW)
    tmp = np.random.normal(0,1,[siz,N//8,N//8])
    xdata = np.fft.irfft(np.fft.rfft(tmp,axis=1),N,1)
    xdata = np.fft.irfft(np.fft.rfft(xdata,axis=2),N,2)
    ydata = np.sum(np.absolute(np.multiply(
        np.fft.fft2(xdata,axes=(1,2)), DW2))**2,axis=(1,2))/N**4
    xdata = np.float32(np.reshape(xdata,[siz,N,N,1]))
    ydata = np.float32(np.reshape(ydata,[siz,1,1,1]))
    return xdata,ydata
