
import math
import numpy as np
from scipy.integrate import odeint, solve_bvp, solve_ivp
from scipy import fftpack

def InvElliptic(a,N):
    h = 1/N
    D_1 = np.zeros((N,N))
    D_2 = np.zeros((N,N))
    b = np.zeros(N)
    for i in range(N-1):
        D_1[i,i+1] = 1/(2*h)
        D_1[i+1,i] = -1/(2*h)
        D_2[i,i+1] = 1/(h**2)
        D_2[i+1,i] = 1/(h**2)
        b[i+1] = (a[i+2]-a[i])/(2*h)
    D_1[0][N-1] = -1/(2*h)
    D_1[N-1][0] = 1/(2*h)
    D_2[0][N-1] = 1/(h**2)
    D_2[N-1][0] = 1/(h**2)
    b[0] = (a[1]-a[-1])/(2*h)
    for i in range(N):     
        D_2[i,i] = -2/(h**2)
        D_2[i] = D_2[i]*a[i]
        D_1[i] = D_1[i]*b[i]
    P = D_1 + D_2
    I =np.linalg.inv(P[1:,1:])
    return I


def Inv_2_Elliptic(a,N):
    h = 1/N
    D_1 = np.zeros((N,N))
    P = np.zeros((N,N))
    for i in range(N-1):
        D_1[i,i+1] = 1/(2*h)
        D_1[i+1,i] = -1/(2*h)
    D_1[0][N-1] = -1/(2*h)
    D_1[N-1][0] = 1/(2*h)

    for i in range(N):     
        P[i] = D_1[i]*a[i]
    P = np.matmul(D_1,P)
    I =np.linalg.inv(P[1:,1:])
    return I

def Dir_2_Elliptic(a,N):
    h = 1/N
    D_1 = np.zeros((N,N))
    P = np.zeros((N,N))
    for i in range(N-1):
        D_1[i,i+1] = 1/(2*h)
        D_1[i+1,i] = -1/(2*h)
    D_1[0][N-1] = 1/(2*h)
    D_1[N-1][0] = -1/(2*h)

    for i in range(N):     
        P[i] = D_1[i]*a[i]
    P = np.matmul(D_1,P)
    return P

def Inv_net_SineElliptic(a,N):
    mat_a = np.zeros((N+1,N+1))
    mat_k = np.zeros_like(mat_a)
    mat_C = np.empty_like(mat_a)
    mat_S = np.empty_like(mat_a)
    mat_I = np.zeros((N,N))
    for i in range(N+1):
        mat_a[i][i] = a[i]
        mat_k[i][i] = (i) * np.pi
        for j in range(N+1):
            mat_C[i][j] = np.cos((i)*(j)*np.pi/N)
            mat_S[i][j] = np.sin((i)*(j)*np.pi/N)
        mat_C[i][0] = 1/2
        mat_C[i][N] = ((-1)**i)/2
    mat_k[N][N] = 0
    mat = np.matmul(mat_C*2/N,mat_k)
    mat = np.matmul(mat_a,mat)
    mat = np.matmul(mat_C,mat)
    mat = np.matmul(-mat_k,mat)
    mat = mat[1:-1,1:-1]
    mat_I[1:,1:] = np.linalg.inv(mat)
    #mat_ki = np.linalg.inv(mat_k[1:,1:])
    #mat_I = np.matmul(mat_ki,mat_S[1:,1:]*2/N)
    #mat_I = np.matmul(-mat_ki,mat_I)
    #mat_I = np.matmul(mat_S[1:,1:],mat_I)
    return mat_I

def InvSineElliptic(a,N):
    mat_a = np.zeros((N+1,N+1))
    mat_k = np.zeros_like(mat_a)
    mat_C = np.empty_like(mat_a)
    mat_S = np.empty_like(mat_a)
    mat_I = np.zeros_like(mat_a)
    for i in range(N+1):
        mat_a[i][i] = a[i]
        mat_k[i][i] = (i) * np.pi
        for j in range(N+1):           
            mat_C[i][j] = np.cos((i)*(j)*np.pi/N)
            mat_S[i][j] = np.sin((i)*(j)*np.pi/N)
        mat_C[i][0] = 1/2
        mat_C[i][N] = ((-1)**i)/2
    mat_k[N][N] = 0
    mat = np.matmul(mat_k,mat_S)
    mat = np.matmul(mat_C*2/N,mat)
    mat = np.matmul(mat_a,mat)
    mat = np.matmul(mat_C,mat)
    mat = np.matmul(-mat_k,mat)
    mat = np.matmul(mat_S*2/N,mat)
    mat = mat[1:-1,1:-1]
    mat_I = np.linalg.inv(mat)
    return mat_I


def DirSineElliptic(a,N):
    mat_a = np.zeros((N+1,N+1))
    mat_k = np.zeros_like(mat_a)
    mat_C = np.empty_like(mat_a)
    mat_S = np.empty_like(mat_a)
    #mat_d = np.zeros((N,N))
    for i in range(N+1):
        mat_a[i][i] = a[i]
        mat_k[i][i] = (i) * np.pi
        for j in range(N+1):           
            mat_C[i][j] = np.cos((i)*(j)*np.pi/N)
            mat_S[i][j] = np.sin((i)*(j)*np.pi/N)
        mat_C[i][0] = 1/2
        mat_C[i][N] = ((-1)**i)/2
    mat_k[N][N] = 0
    mat = np.matmul(mat_k,mat_S)
    mat = np.matmul(mat_C*2/N,mat)
    mat = np.matmul(mat_a,mat)
    mat = np.matmul(mat_C,mat)
    mat = np.matmul(-mat_k,mat)
    mat = np.matmul(mat_S*2/N,mat)

    return mat[:-1,:-1]



