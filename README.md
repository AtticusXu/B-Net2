# Butterfly-Net2
The code are for the paper:

[Butterfly-Net2: Simplified Butterfly-Net and Fourier Transform Initialization](https://arxiv.org/abs/1912.04154)<br />

## Invironment
All code were developed on Windows 10 and tested on CentOS 7 with Python 3.6, and were implemented by Tensorflow 1.13

## Experiment

| Experiment                                   | initialization                             | training & testing                                                       |
| ---                                          | ---                                        | ---                                            | ---                       |
| 4.1.1. APPROXIMATION POWER BEFORE TRAINING   | /                                          | test_Fourier_beforetrain.py                                               |
| 4.1.2. APPROXIMATION POWER AFTER TRAINING    | Butterfly_FTini.py                         | test_Fourier_aftertrain.py                                               |
| 4.1.3. TRANSFER LEARNING CAPABILITY          |  /                                         | test_Fourier_trans_train.py & test_Fourier_trans_eval.py  |
| 4.2.1. ENERGY OF LAPLACE OPERATOR            |  /                                         | test_Energy.py                                                           |
| 4.2.2(3). END-TO-END ELLIPTIC PDE SOLVER     | EtE_Butterfly_FTini.py & ETE_PDE_setgen.py | test_EtE_PDE.py                                                           |
| 4.3. Denoising and Deblurring of 1D Signals  | EtE_Butterfly_FTini.py                     | test_denoise.py & test_deblur.py                                         |


| Model                                     | example of config file    | training                    | testing          | plot             |
| ---                                       | ---                       | ---                         | ---              | ---              |
| Convection-Diffusion Equations            | checkpoint/linpde.yaml    | learn_variantcoelinear2d.py | linpdetest.py    | linpdeplot.py    |
| Diffusion Equations with Nonlinear Source | checkpoint/nonlinpde.yaml | learn_singlenonlinear2d.py  | nonlinpdetest.py | nonlinpdeplot.py |
