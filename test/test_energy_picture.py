import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
sys.path.insert(0,"../src")
sys.path.insert(0,"../src/data_gen")
sys.path.insert(0,"../test/train_energy")
from pathlib import Path
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio
import tensorflow as tf
import json

k = np.linspace(0,7,36)

cnn_False_0_c_test_loss_list = np.log10(np.load(
        'train_model_energy/cnn_False_0_c_model_mk_test_loss_list.npy'))
fft_False_0_c_test_loss_list = np.log10(np.load(
        'train_model_energy/fft_False_0_c_model_mk_test_loss_list.npy'))
cnn_True_0_c_test_loss_list = np.log10(np.load(
        'train_model_energy/cnn_True_0_c_model_mk_test_loss_list.npy'))
fft_True_0_c_test_loss_list = np.log10(np.load(
        'train_model_energy/fft_True_0_c_model_mk_test_loss_list.npy'))
non_train_c_loss_list = np.log10(np.load(
        'train_model_energy/fft_True_-2_c_model_mk_non_loss_list.npy'))
fig = plt.figure(0,figsize=(10,8))

plt.plot(k,cnn_False_0_c_test_loss_list[0:-1],'r',label='CNN-f')
plt.plot(k,cnn_True_0_c_test_loss_list[0:-1],'r--',label='CNN-t')
plt.plot(k,fft_False_0_c_test_loss_list[0:-1],'b',label='fft-f')
plt.plot(k,fft_True_0_c_test_loss_list[0:-1],'b--',label='fft-t')
plt.plot(k,non_train_c_loss_list[0:-1],'g*',label='non-train')
plt.plot([-0.5],cnn_False_0_c_test_loss_list[-1],'r*')
plt.plot([-0.5],cnn_True_0_c_test_loss_list[-1],'ro')
plt.plot([-0.5],fft_False_0_c_test_loss_list[-1],'b*')
plt.plot([-0.5],fft_True_0_c_test_loss_list[-1],'bo')
plt.title('constant mask K, low frequency training set') 
plt.legend()
plt.savefig("energy_ck_lt.png" )


cnn_False_1_c_test_loss_list = np.log10(np.load(
        'train_model_energy/cnn_False_-1_c_model_mk_test_loss_list.npy'))
fft_False_1_c_test_loss_list = np.log10(np.load(
        'train_model_energy/fft_False_-1_c_model_mk_test_loss_list.npy'))
cnn_True_1_c_test_loss_list = np.log10(np.load(
        'train_model_energy/cnn_True_-1_c_model_mk_test_loss_list.npy'))
fft_True_1_c_test_loss_list = np.log10(np.load(
        'train_model_energy/fft_True_-1_c_model_mk_test_loss_list.npy'))
non_train_c_loss_list = np.log10(np.load(
        'train_model_energy/fft_True_-2_c_model_mk_non_loss_list.npy'))
fig = plt.figure(1,figsize=(10,8))

plt.plot(k,cnn_False_1_c_test_loss_list[0:-1],'r',label='CNN-f')
plt.plot(k,cnn_True_1_c_test_loss_list[0:-1],'r--',label='CNN-t')
plt.plot(k,fft_False_1_c_test_loss_list[0:-1],'b',label='fft-f')
plt.plot(k,fft_True_1_c_test_loss_list[0:-1],'b--',label='fft-t')
plt.plot(k,non_train_c_loss_list[0:-1],'g*',label='non-train')
plt.plot([-0.5],cnn_False_1_c_test_loss_list[-1],'r*')
plt.plot([-0.5],cnn_True_1_c_test_loss_list[-1],'ro')
plt.plot([-0.5],fft_False_1_c_test_loss_list[-1],'b*')
plt.plot([-0.5],fft_True_1_c_test_loss_list[-1],'bo')
plt.title('constant mask K, constant mask training set') 
plt.legend()
plt.savefig("energy_ck_ct.png" )


cnn_False_2_c_test_loss_list = np.log10(np.load(
        'train_model_energy/cnn_False_-2_c_model_mk_test_loss_list.npy'))
fft_False_2_c_test_loss_list = np.log10(np.load(
        'train_model_energy/fft_False_-2_c_model_mk_test_loss_list.npy'))
cnn_True_2_c_test_loss_list = np.log10(np.load(
        'train_model_energy/cnn_True_-2_c_model_mk_test_loss_list.npy'))
fft_True_2_c_test_loss_list = np.log10(np.load(
        'train_model_energy/fft_True_-2_c_model_mk_test_loss_list.npy'))
non_train_c_loss_list = np.log10(np.load(
        'train_model_energy/fft_True_-2_c_model_mk_non_loss_list.npy'))
fig = plt.figure(2,figsize=(10,8))

plt.plot(k,cnn_False_2_c_test_loss_list[0:-1],'r',label='CNN-f')
plt.plot(k,cnn_True_2_c_test_loss_list[0:-1],'r--',label='CNN-t')
plt.plot(k,fft_False_2_c_test_loss_list[0:-1],'b',label='fft-f')
plt.plot(k,fft_True_2_c_test_loss_list[0:-1],'b--',label='fft-t')
plt.plot(k,non_train_c_loss_list[0:-1],'g*',label='non-train')
plt.plot([-0.5],cnn_False_2_c_test_loss_list[-1],'r*')
plt.plot([-0.5],cnn_True_2_c_test_loss_list[-1],'ro')
plt.plot([-0.5],fft_False_2_c_test_loss_list[-1],'b*')
plt.plot([-0.5],fft_True_2_c_test_loss_list[-1],'bo')
plt.title('constant mask K, high frequency training set') 
plt.legend()
plt.savefig("energy_ck_ht.png" )




cnn_False_0_l_test_loss_list = np.log10(np.load(
        'train_model_energy/cnn_False_0_l_model_mk_test_loss_list.npy'))
fft_False_0_l_test_loss_list = np.log10(np.load(
        'train_model_energy/fft_False_0_l_model_mk_test_loss_list.npy'))
cnn_True_0_l_test_loss_list = np.log10(np.load(
        'train_model_energy/cnn_True_0_l_model_mk_test_loss_list.npy'))
fft_True_0_l_test_loss_list = np.log10(np.load(
        'train_model_energy/fft_True_0_l_model_mk_test_loss_list.npy'))
non_train_l_loss_list = np.log10(np.load(
        'train_model_energy/fft_True_-2_l_model_mk_non_loss_list.npy'))
fig = plt.figure(3,figsize=(10,8))

plt.plot(k,cnn_False_0_l_test_loss_list[0:-1],'r',label='CNN-f')
plt.plot(k,cnn_True_0_l_test_loss_list[0:-1],'r--',label='CNN-t')
plt.plot(k,fft_False_0_l_test_loss_list[0:-1],'b',label='fft-f')
plt.plot(k,fft_True_0_l_test_loss_list[0:-1],'b--',label='fft-t')
plt.plot(k,non_train_l_loss_list[0:-1],'g*',label='non-train')
plt.plot([-0.5],cnn_False_0_l_test_loss_list[-1],'r*')
plt.plot([-0.5],cnn_True_0_l_test_loss_list[-1],'ro')
plt.plot([-0.5],fft_False_0_l_test_loss_list[-1],'b*')
plt.plot([-0.5],fft_True_0_l_test_loss_list[-1],'bo')
plt.title('low frequency K, low frequency training set') 
plt.legend()
plt.savefig("energy_lk_lt.png" )


cnn_False_1_l_test_loss_list = np.log10(np.load(
        'train_model_energy/cnn_False_-1_l_model_mk_test_loss_list.npy'))
fft_False_1_l_test_loss_list = np.log10(np.load(
        'train_model_energy/fft_False_-1_l_model_mk_test_loss_list.npy'))
cnn_True_1_l_test_loss_list = np.log10(np.load(
        'train_model_energy/cnn_True_-1_l_model_mk_test_loss_list.npy'))
fft_True_1_l_test_loss_list = np.log10(np.load(
        'train_model_energy/fft_True_-1_l_model_mk_test_loss_list.npy'))

fig = plt.figure(4,figsize=(10,8))

plt.plot(k,cnn_False_1_l_test_loss_list[0:-1],'r',label='CNN-f')
plt.plot(k,cnn_True_1_l_test_loss_list[0:-1],'r--',label='CNN-t')
plt.plot(k,fft_False_1_l_test_loss_list[0:-1],'b',label='fft-f')
plt.plot(k,fft_True_1_l_test_loss_list[0:-1],'b--',label='fft-t')
plt.plot(k,non_train_l_loss_list[0:-1],'g*',label='non-train')
plt.plot([-0.5],cnn_False_1_l_test_loss_list[-1],'r*')
plt.plot([-0.5],cnn_True_1_l_test_loss_list[-1],'ro')
plt.plot([-0.5],fft_False_1_l_test_loss_list[-1],'b*')
plt.plot([-0.5],fft_True_1_l_test_loss_list[-1],'bo')
plt.title('low frequency K, constant mask training set') 
plt.legend()
plt.savefig("energy_lk_ct.png" )


cnn_False_2_l_test_loss_list = np.log10(np.load(
        'train_model_energy/cnn_False_-2_l_model_mk_test_loss_list.npy'))
fft_False_2_l_test_loss_list = np.log10(np.load(
        'train_model_energy/fft_False_-2_l_model_mk_test_loss_list.npy'))
cnn_True_2_l_test_loss_list = np.log10(np.load(
        'train_model_energy/cnn_True_-2_l_model_mk_test_loss_list.npy'))
fft_True_2_l_test_loss_list = np.log10(np.load(
        'train_model_energy/fft_True_-2_l_model_mk_test_loss_list.npy'))

fig = plt.figure(5,figsize=(10,8))

plt.plot(k,cnn_False_2_l_test_loss_list[0:-1],'r',label='CNN-f')
plt.plot(k,cnn_True_2_l_test_loss_list[0:-1],'r--',label='CNN-t')
plt.plot(k,fft_False_2_l_test_loss_list[0:-1],'b',label='fft-f')
plt.plot(k,fft_True_2_l_test_loss_list[0:-1],'b--',label='fft-t')
plt.plot(k,non_train_l_loss_list[0:-1],'g*',label='non-train')
plt.plot([-0.5],cnn_False_2_l_test_loss_list[-1],'r*')
plt.plot([-0.5],cnn_True_2_l_test_loss_list[-1],'ro')
plt.plot([-0.5],fft_False_2_l_test_loss_list[-1],'b*')
plt.plot([-0.5],fft_True_2_l_test_loss_list[-1],'bo')
plt.title('low frequency K, high frequency training set') 
plt.legend()
plt.savefig("energy_lk_ht.png" )




cnn_False_0_h_test_loss_list = np.log10(np.load(
        'train_model_energy/cnn_False_0_h_model_mk_test_loss_list.npy'))
fft_False_0_h_test_loss_list = np.log10(np.load(
        'train_model_energy/fft_False_0_h_model_mk_test_loss_list.npy'))
cnn_True_0_h_test_loss_list = np.log10(np.load(
        'train_model_energy/cnn_True_0_h_model_mk_test_loss_list.npy'))
fft_True_0_h_test_loss_list = np.log10(np.load(
        'train_model_energy/fft_True_0_h_model_mk_test_loss_list.npy'))
non_train_h_loss_list = np.log10(np.load(
        'train_model_energy/fft_True_-2_h_model_mk_non_loss_list.npy'))
fig = plt.figure(6,figsize=(10,8))

plt.plot(k,cnn_False_0_h_test_loss_list[0:-1],'r',label='CNN-f')
plt.plot(k,cnn_True_0_h_test_loss_list[0:-1],'r--',label='CNN-t')
plt.plot(k,fft_False_0_h_test_loss_list[0:-1],'b',label='fft-f')
plt.plot(k,fft_True_0_h_test_loss_list[0:-1],'b--',label='fft-t')
plt.plot(k,non_train_h_loss_list[0:-1],'g*',label='non-train')
plt.plot([-0.5],cnn_False_0_h_test_loss_list[-1],'r*')
plt.plot([-0.5],cnn_True_0_h_test_loss_list[-1],'ro')
plt.plot([-0.5],fft_False_0_h_test_loss_list[-1],'b*')
plt.plot([-0.5],fft_True_0_h_test_loss_list[-1],'bo')
plt.title('high frequency K, low frequency training set') 
plt.legend()
plt.savefig("energy_hk_lt.png" )


cnn_False_1_h_test_loss_list = np.log10(np.load(
        'train_model_energy/cnn_False_-1_h_model_mk_test_loss_list.npy'))
fft_False_1_h_test_loss_list = np.log10(np.load(
        'train_model_energy/fft_False_-1_h_model_mk_test_loss_list.npy'))
cnn_True_1_h_test_loss_list = np.log10(np.load(
        'train_model_energy/cnn_True_-1_h_model_mk_test_loss_list.npy'))
fft_True_1_h_test_loss_list = np.log10(np.load(
        'train_model_energy/fft_True_-1_h_model_mk_test_loss_list.npy'))

fig = plt.figure(7,figsize=(10,8))

plt.plot(k,cnn_False_1_h_test_loss_list[0:-1],'r',label='CNN-f')
plt.plot(k,cnn_True_1_h_test_loss_list[0:-1],'r--',label='CNN-t')
plt.plot(k,fft_False_1_h_test_loss_list[0:-1],'b',label='fft-f')
plt.plot(k,fft_True_1_h_test_loss_list[0:-1],'b--',label='fft-t')
plt.plot(k,non_train_h_loss_list[0:-1],'g*',label='non-train')
plt.plot([-0.5],cnn_False_1_h_test_loss_list[-1],'r*')
plt.plot([-0.5],cnn_True_1_h_test_loss_list[-1],'ro')
plt.plot([-0.5],fft_False_1_h_test_loss_list[-1],'b*')
plt.plot([-0.5],fft_True_1_h_test_loss_list[-1],'bo')
plt.title('high frequency K, constant mask training set') 
plt.legend()
plt.savefig("energy_hk_ct.png" )


cnn_False_2_h_test_loss_list = np.log10(np.load(
        'train_model_energy/cnn_False_-2_h_model_mk_test_loss_list.npy'))
fft_False_2_h_test_loss_list = np.log10(np.load(
        'train_model_energy/fft_False_-2_h_model_mk_test_loss_list.npy'))
cnn_True_2_h_test_loss_list = np.log10(np.load(
        'train_model_energy/cnn_True_-2_h_model_mk_test_loss_list.npy'))
fft_True_2_h_test_loss_list = np.log10(np.load(
        'train_model_energy/fft_True_-2_h_model_mk_test_loss_list.npy'))

fig = plt.figure(8,figsize=(10,8))

plt.plot(k,cnn_False_2_h_test_loss_list[0:-1],'r',label='CNN-f')
plt.plot(k,cnn_True_2_h_test_loss_list[0:-1],'r--',label='CNN-t')
plt.plot(k,fft_False_2_h_test_loss_list[0:-1],'b',label='fft-f')
plt.plot(k,fft_True_2_h_test_loss_list[0:-1],'b--',label='fft-t')
plt.plot(k,non_train_h_loss_list[0:-1],'g*',label='non-train')
plt.plot([-0.5],cnn_False_2_h_test_loss_list[-1],'r*')
plt.plot([-0.5],cnn_True_2_h_test_loss_list[-1],'ro')
plt.plot([-0.5],fft_False_2_h_test_loss_list[-1],'b*')
plt.plot([-0.5],fft_True_2_h_test_loss_list[-1],'bo')
plt.title('high frequency K, high frequency training set') 
plt.legend()
plt.savefig("energy_hk_ht.png" )
