""" 
240807
Experiment 1: The graph-guided fused lasso
\min_{x,y} F(x,y) = f(x) + g(y) := 1/n*\frac{1}{1+exp(b_{i}a_{i}^{T}x)} + \lambda_{1}*\|y\|_{1}
subject to Ax-y = 0
"""

# library import
import sys
print(sys.path)
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import math
import time

from utilize.eval_functions import *
from utilize.import_data import *
from methods.SADMM import *
from methods.Spider_B_ADMM import *
from methods.SVRG_ADMM import *
from methods.hybrid_sarah_ADMM import *
from methods.hybrid_SARAH_double import *


lambda_ = 0.5
rho = 50
alpha = 1.5

#get data
file = 'splice_test.t'     # test
file2 = 'splice_train.t'    # train 

#Ready for training
sparse_A, LPADMM_param = covsel( lambda_, rho, alpha, main, file2)
#Loading the test data
x_test, y_test = test_Data(file)
LPADMM_param['x_test'] = x_test 
LPADMM_param['y_test'] = y_test  


#set initial and parameters
np.random.seed(42) 
x0 = np.random.normal(0, 0.1, (LPADMM_param['d'], 1))
# x0 = 0.6*np.ones( (LPADMM_param['d'],1) )
y0 = np.random.normal(0, 0.1, (LPADMM_param['d'], 1))
lamb0 = np.random.normal(0, 0.1, (LPADMM_param['d'], 1))
B = -np.eye(LPADMM_param['d'])
c = np.zeros ( (LPADMM_param['d'], 1 ) )
LPADMM_param['b_batch'] = 50
LPADMM_param['epoch'] = 10
L = LPADMM_param['L']
LPADMM_param['lambda1'] = 1.2 * 1e-4 # 1 / LPADMM_param['N'] # 1 / LPADMM_param['N']




eta_sadmm = 1
eta_prime = 1.0  
LPADMM_param['epoch'] = 20 
LPADMM_param['b_batch'] = int( LPADMM_param['N']**(1/2) )
LPADMM_param['rho'] = 0.1
SADMM_output2 = SADMM(LPADMM_param,
                        x0,
                        y0,
                        lamb0,
                        eta_sadmm,
                        eta_prime, 
                        B,
                        c,
                        x_update_2,
                        y_update_1,
                        linear_cons_1,
                        sto_grad_1,
                        FullGradEval_1,
                        FuncF_Eval_1,
                        predict,
                        chole_inver2)










### Our Hybrid SADMM Single
LPADMM_param['init_grad'] = int( LPADMM_param['N']**(1/2) ) #Tran:grad batch
LPADMM_param['epoch'] = 20
LPADMM_param['max_inner'] = int( ( LPADMM_param['epoch']* LPADMM_param['N']-LPADMM_param['init_grad'] ) / ( 3*LPADMM_param['b_batch'] ) )
LPADMM_param['b_batch'] = int( LPADMM_param['N']**(1/4) ) 
hsto_alpha = 1 - np.sqrt( LPADMM_param['b_batch'] / ( LPADMM_param['init_grad']*LPADMM_param['max_inner'] ) )
LPADMM_param['rho'] = 10*(L**2)*( LPADMM_param['init_grad']*LPADMM_param['max_inner'] )**0.25
hsto_eta = 0.1*LPADMM_param['rho']
LPADMM_param['eta_y'] = (L**2) * LPADMM_param['rho']
hy_sarah_output = hybrid_sarah_ADMM(LPADMM_param,
                                    x0,
                                    y0,
                                    lamb0,
                                    hsto_eta,
                                    B,
                                    c,
                                    x_update_2,
                                    y_update_1,
                                    linear_cons_1,
                                    LPADMM_param['N'],
                                    sto_grad_1,
                                    hsto_alpha,
                                    FullGradEval_1,
                                    FuncF_Eval_1,
                                    grad_diff_1,
                                    predict,
                                    chole_inver2)




### Our Hybrid SADMM Double
LPADMM_param['init_grad'] = int( LPADMM_param['N']**(1/2) )
LPADMM_param['b_batch'] = int( LPADMM_param['N']**(1/2) ) 
LPADMM_param['max_inner'] = int( LPADMM_param['N']/ LPADMM_param['b_batch']) 
hsto_alpha = 1 - np.sqrt( LPADMM_param['b_batch'] / ( LPADMM_param['init_grad']*LPADMM_param['max_inner'] ) )
LPADMM_param['rho'] = 10*(L**2)*( LPADMM_param['init_grad']*LPADMM_param['max_inner'] )**0.25
LPADMM_param['epoch'] =  20
LPADMM_param['rho'] = 1.5 #1.0 #0.1 #1.0 #1.1 #1.2
hsto_eta = 0.1*LPADMM_param['rho']
LPADMM_param['eta_y'] = (L**2) * LPADMM_param['rho'] 
hsto_alpha_dou = 1 - np.sqrt( LPADMM_param['b_batch'] / ( LPADMM_param['init_grad']*LPADMM_param['max_inner'] ) )
#0.9, 0.99,0.97
hy_sarah_output_double = hybrid_SARAH_double(LPADMM_param,
                                    x0,
                                    y0,
                                    lamb0,
                                    hsto_eta,
                                    B,
                                    c,
                                    x_update_2,
                                    y_update_1,
                                    linear_cons_1,
                                    LPADMM_param['N'],
                                    sto_grad_1,
                                    hsto_alpha_dou,
                                    FullGradEval_1,
                                    FuncF_Eval_1,
                                    grad_diff_1,
                                    predict,
                                    chole_inver2)



#===============================================================================================================================
# spdb alg
eta_spdb = 2*L
LPADMM_param['rho'] = 0.2
LPADMM_param['b_batch'] = int( LPADMM_param['N']**(1/2) )
LPADMM_param['epoch'] = 20
Spider_B_output = Spider_B_ADMM(LPADMM_param,
                                x0,
                                y0,
                                lamb0,
                                eta_spdb,
                                B,
                                c,
                                x_update_2,
                                y_update_1,
                                linear_cons_1,
                                LPADMM_param['N'],
                                sto_grad_1,
                                FullGradEval_1,
                                FuncF_Eval_1,
                                grad_diff_1,
                                predict,
                                chole_inver2)





eta_svrg = 3*L
LPADMM_param['b_batch'] = int( LPADMM_param['N']**(2/3) )
LPADMM_param['epoch'] = 20
LPADMM_param['rho'] = 0.6 # 1.0
SVRG_output = SVRG_ADMM(LPADMM_param,
                        x0,
                        y0,
                        lamb0,
                        eta_svrg,
                        B,
                        c,
                        x_update_2,
                        y_update_1,
                        linear_cons_1,
                        LPADMM_param['N'],
                        sto_grad_1,
                        FullGradEval_1,
                        FuncF_Eval_1,
                        grad_diff_1,
                        predict,
                        chole_inver2)





#===============================================================================================================================
# plot the single loop algs


# 设置颜色和线型
color1 = '#287885'
color3 = '#9AC9DB'
color4 = [0.45, 0.35, 0.6]  
color5 = '#F8AC8C'
color7 = 'C82423'
color6 = [0.28, 0.75, 0.76]  #

linestyle3 = '--'  
linestyle1 = '-.'    

#### loal data
SADMM_acc = np.load('SADMM_l1_acc_splice_250813.npy')
SAGA_acc = np.load('SAGA_l1_acc_splice_250813.npy')
hy_single_acc = np.load('hysingle_l1_acc_splice_250813.npy')
# hy_single_acc = [x+2 for x in hy_single_acc]
hy_double_acc = np.load('hydouble_l1_acc_splice_250813.npy')
SVRG_acc = np.load('SVRG_l1_acc_splice_250813.npy')
SPIDER_acc = np.load('Spider_l1_acc_splice_250813.npy')


acc_values = [ SADMM_acc[-1], SAGA_acc[-1], hy_single_acc[-1],
                hy_double_acc[-1], SVRG_acc[-1], SPIDER_acc[-1]
                ]


# 找到最大值的下标
Lowest = max(acc_values)
index = acc_values.index(max(acc_values))

algorithm = ['SADMM', 'SAGA-ADMM', 'Hy-SADMM', 'Hy-SADMM-RS', 'SVRG-ADMM', 'SPIDER-ADMM']
algori_name = algorithm[index]
print(f'The algorithm with the largest accuracy value is: {algori_name}.')
print('SADMM len:', len(SADMM_acc),
        '\nHy-SADMM len:', len(hy_single_acc),
        '\nHy-SADMM-RS len:', len(hy_double_acc),
        '\nSAGA-SADMM len:', len(SAGA_acc),
        '\nSPIDER-ADMM len:', len(SPIDER_acc),
        '\nSVRG-ADMM len:', len(SVRG_acc)
        )     


#===============================================================================================================================
# plot loss 

SADMM_obj = np.load('SADMM_l1_obj_splice_250813.npy')
SAGA_ADMM_obj = np.load('SAGA_l1_obj_splice_250813.npy')
hy_single_obj = np.load('hysingle_l1_obj_splice_250813_2.npy')
# hy_single_acc = [x+2 for x in hy_single_acc]
hy_double_obj = np.load('hydouble_l1_obj_splice_250813.npy')
SVRG_obj = np.load('SVRG_l1_obj_splice_250813.npy')
Spider_obj = np.load('Spider_l1_obj_splice_250813.npy')

obj_values = [ SADMM_obj[-1], hy_double_obj[-1],  SAGA_ADMM_obj[-1], hy_single_obj[-1],
              Spider_obj[-1], SVRG_obj[-1]]
Lowest = min(obj_values)
index = obj_values.index(min(obj_values))
algorithm = ['SADMM', 'Hy-SADMM-RS', 'SAGA-ADMM', 'Hy-SADMM', 'SPIDER-SADMM', 'SVRG-ADMM']
algori_name = algorithm[index]
print(f'The algorithm with the smallest objective value is: {algori_name}.')

