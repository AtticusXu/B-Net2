import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
sys.path.insert(0,"../src")
sys.path.insert(0,"../src/data_gen")
sys.path.insert(0,"../test/train_model")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

MODEL_SAVE_PATH = "train_model_a_-1_0-7/"
k = np.linspace(0,7,36)
k_c = np.linspace(-1,7,41)
fft_mk_test_loss_list_t = np.load(MODEL_SAVE_PATH+'fft_mk_test_loss_list_True.npy')
fft_mk_test_loss_list_f = np.load(MODEL_SAVE_PATH+'fft_mk_test_loss_list_False.npy')
cnn_mk_test_loss_list_t = np.load(MODEL_SAVE_PATH+'cnn_mk_test_loss_list_True.npy')
cnn_mk_test_loss_list_f = np.load(MODEL_SAVE_PATH+'cnn_mk_test_loss_list_False.npy')

fft_mk_test_loss_list_t = np.load(MODEL_SAVE_PATH+'fft_mk_test_loss_l2_list_True.npy')
fft_mk_test_loss_list_f = np.load(MODEL_SAVE_PATH+'fft_mk_test_loss_l2_list_False.npy')
cnn_mk_test_loss_list_t = np.load(MODEL_SAVE_PATH+'cnn_mk_test_loss_l2_list_True.npy')
cnn_mk_test_loss_list_f = np.load(MODEL_SAVE_PATH+'cnn_mk_test_loss_l2_list_False.npy')

adv = -1

print(10**np.mean(np.log10(fft_mk_test_loss_list_t[adv][:-1])))
print(10**np.mean(np.log10(fft_mk_test_loss_list_f[adv][:-1])))
print(10**np.mean(np.log10(cnn_mk_test_loss_list_t[adv])))
print(10**np.mean(np.log10(cnn_mk_test_loss_list_f[adv])))

fft_err_list_t = np.load(MODEL_SAVE_PATH+'fft_err_list_True.npy')
fft_err_list_f = np.load(MODEL_SAVE_PATH+'fft_err_list_False.npy')
cnn_err_list_t = np.load(MODEL_SAVE_PATH+'cnn_err_list_True.npy')
cnn_err_list_f = np.load(MODEL_SAVE_PATH+'cnn_err_list_False.npy')

fft_k_list_t = np.load(MODEL_SAVE_PATH+'fft_z_list_True.npy')
fft_k_list_f = np.load(MODEL_SAVE_PATH+'fft_z_list_False.npy')
cnn_k_list_t = np.load(MODEL_SAVE_PATH+'cnn_z_list_True.npy')
cnn_k_list_f = np.load(MODEL_SAVE_PATH+'cnn_z_list_False.npy')



non_train = np.log10(np.load(MODEL_SAVE_PATH+'mk_test_loss_l2_list_nt.npy'))
zero = np.zeros((36))
print(cnn_mk_test_loss_list_t.shape)
fft_mk_test_loss_list_t_mean = np.mean(np.log10(fft_mk_test_loss_list_t[:,0:-1]), axis=1)
cnn_mk_test_loss_list_t_mean = np.mean(np.log10(cnn_mk_test_loss_list_t), axis=1)
fft_mk_test_loss_list_t_ste = np.std(np.log10(fft_mk_test_loss_list_t[:,0:-1]), axis=1)
cnn_mk_test_loss_list_t_ste = np.std(np.log10(cnn_mk_test_loss_list_t), axis=1)
fft_mk_test_loss_list_f_mean = np.mean(np.log10(fft_mk_test_loss_list_f[:,0:-1]), axis=1)
cnn_mk_test_loss_list_f_mean = np.mean(np.log10(cnn_mk_test_loss_list_f), axis=1)
fft_mk_test_loss_list_f_ste = np.std(np.log10(fft_mk_test_loss_list_f[:,0:-1]), axis=1)
cnn_mk_test_loss_list_f_ste = np.std(np.log10(cnn_mk_test_loss_list_f), axis=1)
print(fft_mk_test_loss_list_t_mean)
print(cnn_mk_test_loss_list_t_mean)

fig = plt.figure(3,figsize=(10,8))

plt.errorbar(k,cnn_mk_test_loss_list_f_mean[0:-1],
             yerr=cnn_mk_test_loss_list_f_ste[0:-1],fmt='r',
             ecolor='hotpink',elinewidth=2,capsize=3,label='CNN-rand')
plt.errorbar([-0.5],cnn_mk_test_loss_list_f_mean[-1],
             yerr=cnn_mk_test_loss_list_f_ste[-1],fmt='ro',
             ecolor='hotpink',ms=5,mfc='r',mec='r',
             elinewidth=2,capsize=3)
plt.errorbar(k,fft_mk_test_loss_list_f_mean[0:-1],
             yerr=fft_mk_test_loss_list_f_ste[0:-1],fmt='b',
             ecolor='g',elinewidth=2,capsize=3,label='BNet2-rand')
plt.errorbar([-0.5],fft_mk_test_loss_list_f_mean[-1],
             yerr=fft_mk_test_loss_list_f_ste[-1],fmt='bo',
             ecolor='g',ms=5,mfc='b',mec='b',
             elinewidth=2,capsize=3)

plt.errorbar(k,cnn_mk_test_loss_list_t_mean[0:-1],
             yerr=cnn_mk_test_loss_list_t_ste[0:-1],fmt='orange',
             ecolor='hotpink',elinewidth=2,capsize=3,label='CNN-FT')
plt.errorbar([-0.5],cnn_mk_test_loss_list_t_mean[-1],
             yerr=cnn_mk_test_loss_list_t_ste[-1],fmt='oy',
             ecolor='hotpink',ms=5,mfc='y',mec='y',
             elinewidth=2,capsize=3)
plt.errorbar(k,fft_mk_test_loss_list_t_mean[0:-1],
             yerr=fft_mk_test_loss_list_t_ste[0:-1],fmt='lime',
             ecolor='g',elinewidth=2,capsize=3,label='BNet2-FT')
plt.errorbar([-0.5],fft_mk_test_loss_list_t_mean[-1],
             yerr=fft_mk_test_loss_list_t_ste[-1],fmt='oc',
             ecolor='g',ms=5,mfc='c',mec='c',
             elinewidth=2,capsize=3)

print(10**non_train[-1])
plt.plot(k,non_train[0:-1],'y--')
plt.plot([-0.5],non_train[-1],'yo')
plt.plot([-0.5],non_train[-1],'r')
la = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 27,
}
plt.legend(prop=la)
plt.xlabel('$G_{mean}$',la)
plt.ylabel('log(error)',la)
plt.tick_params(labelsize=25)
plt.show()
#plt.savefig("trans_error_a_02_0-7.eps" ,rasterized=True)

#fft_mk_test_loss_klist_t_mean = np.mean(np.log10(fft_mk_test_loss_klist_t), axis=0)
print(cnn_mk_test_loss_list_t_ste)