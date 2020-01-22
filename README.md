# Butterfly-Net2
The code is for the paper:

[Butterfly-Net2: Simplified Butterfly-Net and Fourier Transform Initialization](https://arxiv.org/abs/1912.04154)<br />

## Environment
All code was developed on Windows 10 and tested on CentOS 7 with Python 3.6, and was implemented by Tensorflow 1.13

## Experiment

all the experiments are done via code in the folder 'test', while other sources are in the folder 'src'.

### 4.1.1. APPROXIMATION POWER BEFORE TRAINING

testing: test_Fourier_beforetrain.py/>
plot: test_Fourier_beforetrain_plot.py />
for other situations, just change the value of N, K, l, r.

### 4.1.2. APPROXIMATION POWER AFTER TRAINING

initialization: Butterfly_FTini.py />
training & testing: test_Fourier_aftertrain.py />
The four different networks and other hyper-parameters can be changed in paras.json.

### 4.1.3. TRANSFER LEARNING CAPABILITY

initialization: Butterfly_FTini.py />
training: test_Fourier_trans_train.py/>
testing: test_Fourier_trans_eval.py />
plot: test_Fourier_trans_plot.py 

### 4.2.1. ENERGY OF LAPLACE OPERATOR           

initialization: Butterfly_FTini.py />
training & testing: test_Energy.py 

### 4.2.2(3). END-TO-END ELLIPTIC PDE SOLVER

initialization: EtE_Butterfly_FTini.py & EtE_PDE_setgen.py/>
training & testing: test_EtE_PDE.py/>
plot: test_EtE_PDE_plot.py

### 4.3. Denoising and Deblurring of 1D Signals  
initialization: EtE_Butterfly_FTini.py/>
training & testing: test_denoise.py & test_deblur.py/>
plot:  test_denoise_plot.py & test_deblur_plot.py

