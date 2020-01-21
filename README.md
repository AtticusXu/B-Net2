# Butterfly-Net2
The code are for the paper:

[Butterfly-Net2: Simplified Butterfly-Net and Fourier Transform Initialization](https://arxiv.org/abs/1912.04154)<br />

## Invironment
All code were developed on Windows 10 and tested on CentOS 7 with Python 3.6, and were implemented by Tensorflow 1.13

## Experiment

all the experiments are done via code in the folder 'test', while other sources are in the folder 'src'.

### 4.1.1. APPROXIMATION POWER BEFORE TRAINING
```
python test_Fourier_beforetrain.py
``` 
will get the relative error of BNet2 before training in figure 2; for other situation, just change the value of N,K,l,r.

### 4.1.2. APPROXIMATION POWER AFTER TRAINING

First run Butterfly_FTini.py to get the FT initialization, then test_Fourier_aftertrain.py can show the process of training and the result of testing. The four different networks and other hyper-parameters can be changed in paras.json .

### 4.1.3. TRANSFER LEARNING CAPABILITY
test_Fourier_trans_eval.py  test_Fourier_trans_train.py

### 4.2.1. ENERGY OF LAPLACE OPERATOR           
test_Energy.py    

### 4.2.2(3). END-TO-END ELLIPTIC PDE SOLVER     
EtE_Butterfly_FTini.py    ETE_PDE_setgen.py   test_EtE_PDE.py    

### 4.3. Denoising and Deblurring of 1D Signals  
EtE_Butterfly_FTini.py                      test_denoise.py & test_deblur.py    

