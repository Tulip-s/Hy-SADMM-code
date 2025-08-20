""" 
method_general_SADMM

The algorithm is usd to solve: The graph-guided fused lasso
\min_{x,y} F(x,y) = f(x) + g(y) := 1/n*\frac{1}{1+exp(b_{i}a_{i}^{T}x)} + \lambda_{1}*\|y\|_{1}
subject to Ax-y = 0
"""

# library import
import numpy as np
import math
import time

# from utils import *


"""
general Stochastic ADMM algorithm 
"""

def SADMM(param,
            x0,
            y0,
            lamb0,
            eta,
            eta_prime, 
            B,
            c,
            x_update,
            y_update,
            linear_cons,
            sto_grad,
            FullGradEval,
            FuncF_Eval,
            predict,
            chole_inver):

    """
    parameters
    @param param    				    : a dict of params
    @param x0    				        : initial point x
    @param y0    				        : initial point y
    @param lamb0    				    : initial point lambda
    @param eta    				        : 1 / learning rate
    @param eta_prime    				: used for diminishing learning rate scheme
    @param batch_b       		        : batch size of computing stochastic gradient
    @param y_update                     : function pointer for update y
    @param linear_cons_1                : the linear constraint Ax+By-c
    @param sto_grad                     : function pointer for stochastic grad of f
    @param FullGradEval 		        : function pointer for gradient of f
    @param FuncF_Eval                   : function pointer to compute objective value of \f$ f(x)+g(y) \f$
    """  
    
    d = param['d']
    N = param['N']
    max_num_epoch = param['epoch']
    delta = param['delta']
    rho = param['rho']
    lambda_1 = param['lambda1']
    gamma = param['gamma']
    sparse_A = param['sparse_A']
    AS = sparse_A
    Hessian = param['Hessian']
    AT = param['AT']
    A = param['A']
    L = param['L']

    HessianA_s = param['HessianA_s']
    AST = param['sparse_AT']
    b_batch = param['b_batch']
    print_interval = 10 # 50

    # acc_interval = 40
    x_test = param['x_test']
    y_test = param['y_test']

    
    print( 'Classic Stochastic ADMM ...' )
    print(
    '{message:{fill}{align}{width}}'.format(message='',fill='=',align='^',width=99,),'\n',
    '{message:{fill}{align}{width}}'.format(message='rho',fill=' ',align='^',width=10,),'|',
    '{message:{fill}{align}{width}}'.format(message='N',fill=' ',align='^',width=10,),'|',
    '{message:{fill}{align}{width}}'.format(message='d',fill=' ',align='^',width=10,),'|',
    '{message:{fill}{align}{width}}'.format(message='batch_size',fill=' ',align='^',width=20,),'\n'
    '{message:{fill}{align}{width}}'.format(message='',fill='-',align='^',width=99,)
    )
    print(
        '{:^11.2e}'.format(rho),'|',
        '{:^10.2f}'.format(N),'|',
        '{:^10.2f}'.format(d),'|',
        '{:^19d}'.format(b_batch),
    ) 

    
    x = x0
    y = y0
    w = lamb0
    num_grad, num_epoch = 0, 0
    iter = 0

    acc_list = [predict(x_test, y_test, x)]


    obj = [FuncF_Eval(A, x, lambda_1, N, AS)]
    total_time = 0
    norm_c = [ np.linalg.norm( linear_cons(AS, -np.eye(d), 0, x, y), 2 ) ]
    grad_norm = [ np.linalg.norm( FullGradEval(A, AT, x, N), 2 ) ] 

    "Add cholesky"
    L = chole_inver(eta, rho*HessianA_s, d)    


    print(
    '{message:{fill}{align}{width}}'.format(message='',fill='=',align='^',width=99,),'\n',
    '{message:{fill}{align}{width}}'.format(message='iter',fill=' ',align='^',width=10,),'|',
    '{message:{fill}{align}{width}}'.format(message='obj',fill=' ',align='^',width=10,),'|',
    '{message:{fill}{align}{width}}'.format(message='grad norm',fill=' ',align='^',width=10,),'|',
    '{message:{fill}{align}{width}}'.format(message='dual gap',fill=' ',align='^',width=20,),'\n'
    '{message:{fill}{align}{width}}'.format(message='',fill='-',align='^',width=99,)
    )
    while num_epoch < max_num_epoch:
        tic = time.time()
        iter += 1
        rho = gamma * rho
        rho_r = 1/rho
        #y-update
        y = y_update(rho, AS, x, y, w, lambda_1)
        
        #x-update
        grad = sto_grad(N, b_batch, A, x)
        eta_cur = eta * ( 1 + eta_prime*(iter//N) )
        x_old =x

        "invese case update"
        qx = rho * AST.dot(y + w*rho_r) + x * eta - grad 
        x = x_update( qx, L )


        num_grad += b_batch
        num_epoch = num_grad / N
      
        u = linear_cons(AS, B, c, x, y)
        w = w - rho * u
        norm_c.append(np.linalg.norm(u, 2))
        max_tol = np.max(np.abs(u))
        itertime = time.time() - tic
        total_time = total_time +itertime

        # if max_tol < delta:
        #     break

        obj.append(FuncF_Eval(A, x, lambda_1, N, AS))
        grad_norm.append(np.linalg.norm( FullGradEval(A, AT, x, N), 2 ))

        acc_cur = predict(x_test, y_test, x)
        acc_list.append(acc_cur)

        if iter % print_interval == print_interval-1:
            print(
            '{:^16.4f}'.format(iter),'|',
            '{:^15.3e}'.format(obj[-1]),'|',
            '{:^15.3e}'.format(grad_norm[-1]),'|',
            '{:^15.3e}'.format(norm_c[-1]),'|',
            '{:^15.3e}'.format(acc_cur),
            )
    
    output = {
        'iter': iter,
        'objective': obj[-1],
        'beta': x,
        'total time': total_time,
        'obj_list': obj,
        'dual_gap_norm': norm_c,
        'grad_norm': grad_norm,
        'acc_list': acc_list
    }

    return output



















