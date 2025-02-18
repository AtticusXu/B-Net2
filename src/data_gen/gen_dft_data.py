import numpy as np
import math
from ODE_matrix import InvElliptic, InvSineElliptic
from scipy import fftpack,signal
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

def gen_ete_denoise_data(freqmag,freqidx,siz,sig,n):
    N = len(freqmag)
    K = len(freqidx)
    a = 3000
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
    noise = n*np.random.randn(siz,1,N)
    realx = np.reshape(np.fft.ifft(y,N,1).real,(siz,1,N),order='F')
    imagx = np.reshape(np.fft.ifft(y,N,1).imag,(siz,1,N),order='F')
    xdata = np.reshape(realx,(siz,-1,1))
    realu = realx + noise
    udata = np.reshape(np.concatenate((realu,imagx),axis=1),(siz,-1,1),order='F')
    y = np.reshape(y,(siz,1,N),order='F')
    realy = y.real[:,:,freqidx]
    imagy = y.imag[:,:,freqidx]
    ydata = np.reshape(np.concatenate((realy,imagy),axis=1),(siz,-1,1),order='F')
    xnorm = np.squeeze(np.linalg.norm(xdata,2,1))
    ynorm = np.squeeze(np.linalg.norm(ydata,2,1))
    unorm = np.squeeze(np.linalg.norm(udata,2,1))
    nnorm = np.squeeze(np.linalg.norm(noise,2,2))
    n_rel = np.mean(nnorm/xnorm)
    xdata = np.float32(xdata)
    ydata = np.float32(ydata)
    udata = np.float32(udata)
    return xdata,ydata,udata,xnorm,ynorm,unorm,nnorm,n_rel

def gen_ete_deblur_data(freqmag,freqidx,siz,sig,bs):
    N = len(freqmag)
    K = len(freqidx)
    a = 3000
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
    gaus = np.empty(9)
    for i in range(9):
        gaus[i] = np.exp(-np.power(i-4,2.)/(2*bs**2))/np.sqrt(2*np.pi*bs**2)
    
    realx = np.reshape(np.fft.ifft(y,N,1).real,(siz,1,N),order='F')
    imagx = np.reshape(np.fft.ifft(y,N,1).imag,(siz,1,N),order='F')
    xdata = np.reshape(realx,(siz,-1,1))
    realu = np.empty((siz,1,N))
    for j in range(siz):
        realu[j,0] = signal.convolve(realx[j,0], gaus, mode='same')  
    udata = np.reshape(np.concatenate((realu,imagx),axis=1),(siz,-1,1),order='F')
    y = np.reshape(y,(siz,1,N),order='F')
    realy = y.real[:,:,freqidx]
    imagy = y.imag[:,:,freqidx]
    ydata = np.reshape(np.concatenate((realy,imagy),axis=1),(siz,-1,1),order='F')
    xnorm = np.squeeze(np.linalg.norm(xdata,2,1))
    ynorm = np.squeeze(np.linalg.norm(ydata,2,1))
    unorm = np.squeeze(np.linalg.norm(udata,2,1))
    bnorm = np.squeeze(np.linalg.norm(udata[:,::2]-xdata,2,1))
    b_rel = np.mean(bnorm/xnorm)
    xdata = np.float32(xdata)
    ydata = np.float32(ydata)
    udata = np.float32(udata)
    return xdata,ydata,udata,xnorm,ynorm,unorm,b_rel

def gen_ete_Ell_data(siz, freqidx, freqmag, a_0):
    

    N = len(freqmag)
    N_0 = 2**10
    K = len(freqidx)
    l = 1/N
    
    freqmag = np.tile(np.reshape(freqmag,[1,N]),(siz,1))
    y = np.random.uniform(-np.sqrt(l),np.sqrt(l),[siz,N])
    y = y*freqmag*N
    
    zerof = np.zeros([siz,1])
    f = np.concatenate((zerof,fftpack.dst(y[:,1:],1,N_0-1,axis = 1)),axis=1)/2/N
    f_0 = np.reshape(f,(siz,N_0,1))
    f = f_0[:,0::N_0//N,:]
    fdata_r = np.concatenate((f,np.zeros([siz,1,1]),-f[:,-1:0:-1]),axis=1)
    fdata_i = np.zeros_like(fdata_r)
    fdata = np.reshape(np.concatenate((fdata_r,fdata_i),axis = 2),(siz,-1,1))

    ydata_r = np.concatenate((zerof,fftpack.dst(f[:,1:,0],1,N-1,axis = 1)),axis=1)
    ydata_r = np.reshape(ydata_r[:,freqidx],(siz,K,1))
    ydata_i = np.zeros_like(ydata_r)
    ydata = np.reshape(np.concatenate((ydata_r,ydata_i),axis = 2),(siz,-1,1))
    print("ydata")
    print(ydata[0,:,0])
    ynorm = np.squeeze(np.linalg.norm(ydata,2,1))
    
    u_0 = np.zeros((siz,N_0,1))
    mat = InvSineElliptic(a_0,N_0)
    mat = np.tile(mat,(siz,1,1))

    u_0[:,1:,:] = np.matmul(mat, f_0[:,1:])
    u = u_0[:,::N_0//N,:]
    fnorm = np.linalg.norm(np.squeeze(fdata),2)/np.sqrt(2)
    unorm = np.squeeze(np.linalg.norm(u,2,1))
    fdata = np.float32(fdata)
    udata = np.float32(u)
    ydata = np.float32(ydata)
    return fdata, ydata, udata, fnorm, ynorm, unorm

def gen_ede_Nl_Ell_data(siz, freqidx, freqmag, a_0):
    
    #f_r = np.random.uniform(-np.sqrt(3/N),np.sqrt(3/N),[batch_siz,N,1])
    #f_i = np.zeros_like(f_r)
    #f = np.reshape(np.concatenate((f_r,f_i),axis=2),(batch_siz,-1,1),order='C')
    #print(f[0,:,0])
    N = len(freqmag)
    N_0 = 2**10
    K = len(freqidx)
    l = 1/N
    
    freqmag = np.tile(np.reshape(freqmag,[1,N]),(siz,1))
    y = np.random.uniform(-np.sqrt(l),np.sqrt(l),[siz,N])
    #y = np.ones([siz,N])
    y = y*freqmag*N
    #print("y")
    #print(y[0,:])
    zerof = np.zeros([siz,1])
    f = np.concatenate((zerof,fftpack.dst(y[:,1:],1,N_0-1,axis = 1)),axis=1)/2/N
    #f = np.concatenate((zerof,np.random.uniform(-np.sqrt(l),np.sqrt(l),[siz,N_0-1])),axis=1)
    f_0 = np.reshape(f,(siz,N_0,1))
    f = f_0[:,0::N_0//N,:]
    fdata_r = np.concatenate((f,np.zeros([siz,1,1]),-f[:,-1:0:-1]),axis=1)
    fdata_i = np.zeros_like(fdata_r)
    fdata = np.reshape(np.concatenate((fdata_r,fdata_i),axis = 2),(siz,-1,1))
    #print("f")
    #print(fdata[0,:,0])
    ydata_r = np.concatenate((zerof,fftpack.dst(f[:,1:,0],1,N-1,axis = 1)),axis=1)
    ydata_r = np.reshape(ydata_r[:,freqidx],(siz,K,1))
    ydata_i = np.zeros_like(ydata_r)
    ydata = np.reshape(np.concatenate((ydata_r,ydata_i),axis = 2),(siz,-1,1))
    print("ydata")
    print(ydata[0,:,0])
    ynorm = np.squeeze(np.linalg.norm(ydata,2,1))
    
    u_0 = np.zeros((siz,N_0,1))
    mat = InvSineElliptic(a_0,N_0)
    #print(mat.shape)
    mat = np.tile(mat,(siz,1,1))

    u_0[:,1:,:] = np.matmul(mat, f_0[:,1:])
    u = u_0[:,::N_0//N,:]
    #print("u")
    #print(u[0,:,0])
    fnorm = np.squeeze(np.linalg.norm(fdata,2,1))/np.sqrt(2)
    unorm = np.squeeze(np.linalg.norm(u,2,1))
    #print(np.mean(unorm))
    fdata = np.float32(fdata)
    udata = np.float32(u)
    ydata = np.float32(ydata)
    return fdata, ydata, udata, fnorm, ynorm, unorm


def gen_ede_Ell_sine_data(siz, freqidx, freqmag, a):
    
    #f_r = np.random.uniform(-np.sqrt(3/N),np.sqrt(3/N),[batch_siz,N,1])
    #f_i = np.zeros_like(f_r)
    #f = np.reshape(np.concatenate((f_r,f_i),axis=2),(batch_siz,-1,1),order='C')
    #print(f[0,:,0])
    N = len(freqmag)
    K = len(freqidx)
    l = N
    
    freqmag = np.tile(np.reshape(freqmag,[1,N]),(siz,1))
    #y = np.random.uniform(-np.sqrt(l),np.sqrt(l),[siz,N])
    y = np.ones([siz,N])
    y = y*freqmag
    #print("y")
    #print(y[0,:])
    zerof = np.zeros([siz,1])
    f = np.concatenate((zerof,fftpack.dst(y[:,1:],1,N-1,axis = 1)),axis=1)/2/N
    f = np.reshape(f,(siz,N,1))
    print("f")
    print(f[0,:,0])
    ydata = np.concatenate((zerof,fftpack.dst(f[:,1:,0],1,N-1,axis = 1)),axis=1)
    ydata = np.reshape(ydata[:,freqidx],(siz,K,1))
    print("ydata")
    print(ydata[0,:,0])
    ynorm = np.squeeze(np.linalg.norm(ydata,2,1))
    
    u = np.zeros((siz,N,1))
    mat = InvElliptic(a,N)
    mat = np.tile(mat,(siz,1,1))

    u[:,1:] = np.matmul(mat, f[:,1:])
    #print("u")
    #print(u[0,:,0])
    fnorm = np.squeeze(np.linalg.norm(f,2,1))
    unorm = np.squeeze(np.linalg.norm(u,2,1))
    fdata = np.float32(f)
    udata = np.float32(u)
    ydata = np.float32(ydata)
    return fdata, ydata, udata, fnorm, ynorm, unorm    
  

def gen_energy_uni_data(freqmag,freqidx,K_,siz,sig):
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
    K_ = np.reshape(K_,(-1,1))
    edata = np.sum(np.absolute(np.multiply(ydata, K_))**2,axis=1)
    xdata = np.float32(xdata)
    ydata = np.float32(ydata)
    return xdata,ydata,edata,ynorm
 




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
    realx = np.reshape(np.fft.ifft(y,N,1).real,(siz,1,N),order='F')
    imagx = np.reshape(np.fft.ifft(y,N,1).imag,(siz,1,N),order='F')
    xdata = np.reshape(np.concatenate((realx,imagx),axis=1),(siz,-1,1),order='F')
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
    tmp = np.random.normal(0,1,[siz,N//8])
    xdata = np.fft.irfft(np.fft.rfft(tmp,axis=1),N,1)
    realh = np.reshape(np.fft.fft(xdata,axis=1).real,(siz,N),order='F')
    imagh = np.reshape(np.fft.fft(xdata,axis=1).imag,(siz,N),order='F')
    hdata = np.reshape(np.concatenate((realh,imagh),axis=1),(siz,-1,1),order='F')
    ydata = np.sum(np.absolute(np.multiply(
        np.fft.fft(xdata,axis=1), DW))**2,axis=1)/N**2
    xdata = np.float32(np.reshape(xdata,[siz,N,1]))
    ydata = np.float32(np.reshape(ydata,[siz,1,1]))
    DW = np.float32(DW)
    return xdata,ydata,hdata,DW


