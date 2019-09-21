import math
import numpy as np
from scipy.integrate import odeint, solve_bvp, solve_ivp
import matplotlib.pyplot as plt
def InvElliptic(a,N):
    h = 1/N
    D_1 = np.zeros((N-1,N-1))
    D_2 = np.zeros((N-1,N-1))
    b = np.zeros(N-1)
    for i in range(N-2):
        D_1[i,i+1] = 1/(2*h)
        D_1[i+1,i] = -1/(2*h)
        D_2[i,i+1] = 1/(h**2)
        D_2[i+1,i] = 1/(h**2)
    for i in range(N-1):
        b[i] = (a[i+2]-a[i])/(2*h)
        D_2[i,i] = -2/(h**2)
        D_2[i] = D_2[i]*a[i]
        D_1[i] = D_1[i]*b[i]
    P = D_1 + D_2
    I =np.linalg.inv(P)
    return I

def InvSineElliptic(a,N):
    mat_a = np.zeros((N+1,N+1))
    mat_k = np.zeros_like(mat_a)
    mat_C = np.empty_like(mat_a)
    mat_S = np.empty_like(mat_a)
    mat_I = np.zeros_like(mat_a)
    for i in range(N+1):
        for j in range(N+1):
            mat_a[i][i] = a[i]
            mat_k[i][i] = (i) * np.pi
            mat_C[i][j] = np.cos((i)*(j)*np.pi/N)
            mat_S[i][j] = np.sin((i)*(j)*np.pi/N)
        mat_C[i][0] = 1/2
        mat_C[i][N] = ((-1)**i)/2
    mat = np.matmul(mat_k,mat_S)
    #mat = np.matmul(mat_C*2/N,mat)
    #mat = np.matmul(mat_a,mat)
    #mat = np.matmul(mat_C,mat)
    mat = np.matmul(-mat_k,mat)
    mat = np.matmul(mat_S*2/N,mat)
    mat = mat[1:-1,1:-1]
    mat_I = np.linalg.inv(mat)
    #mat_ki = np.linalg.inv(mat_k[1:,1:])
    #mat_I = np.matmul(mat_ki,mat_S[1:,1:]*2/N)
    #mat_I = np.matmul(-mat_ki,mat_I)
    #mat_I = np.matmul(mat_S[1:,1:],mat_I)
    return mat_I


L2_loss = np.ones(6)
Li_loss = np.zeros(6)
Nl = np.zeros(6)
excel = np.zeros((6,4))
for i in range(6):
    N = 2**(i+3)
    Nl[i] = N
    a = np.ones(N+1)
    
    f = np.empty((N+1,1))
    u = np.empty((N+1,1))
    m = np.floor(np.sqrt(N))//2
    a = np.ones(N+1)

    for j in range(N+1):
        #f[j][0] = j/N*(1-j/N)
        #u[j][0] = (j/N)**3/6-(j/N)**4/12-(j/N)/12
        #x = j/N
        f[j][0] = -2
        u[j][0] = j/N*(1-j/N)
        #u[j][0] = -x**2-np.sin(2*np.pi*x)/(4*np.pi**2) +\
        #            x*np.cos(2*np.pi*x)/(2*np.pi)+(1-1/(2*np.pi))*x
        #a[j] = 1/(1+np.sin(2*np.pi*x)/2)
        
        #f[j][0] = -np.sin(j*np.pi/N)
        #u[j][0] = np.sin(j*np.pi/N)/np.pi/np.pi
    S = InvSineElliptic(a,N)
    D = InvElliptic(a,N)
    u_ = np.matmul(S,f[1:-1])

    d = np.reshape(u[1:-1]-u_,N-1)
    excel[i][0] = np.sum(np.square(d))/(N-1)
    excel[i][2] = np.linalg.norm(d,np.inf)
    if i != 0:
        excel[i][1] = np.log(excel[i][0]/excel[i-1][0])/np.log(2)
        excel[i][3] = np.log(excel[i][2]/excel[i-1][2])/np.log(2)
    
print(excel)
#fig = plt.figure(1,figsize=(10,8))
#plt.plot(Nl,np.log10(excel[:,0]),label = 'L_2')
#plt.plot(Nl,np.log10(excel[:,2]),label = 'L_inf')
#plt.legend()