""" 
method_hybrid_SARAH_double

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
general single loop hybrid SARAH algorithm 
"""

def hybrid_SARAH_double(param,
                        x0,
                        y0,
                        lamb0,
                        eta,
                        B,
                        c,
                        x_update,
                        y_update,
                        linear_cons,
                        batch_M,
                        sto_grad,
                        alpha,
                        FullGradEval,
                        FuncF_Eval,
                        grad_diff,
                        predict,
                        chole_inver):

    """
    parameters
    @param param    				    : a dict of params
    @param x0    				        : initial point x
    @param y0    				        : initial point y
    @param lamb0    				    : initial point lambda
    @param batch_b       		        : batch size of computing stochastic gradient
    @param y_update                     : function pointer for update y
    @param linear_cons                  : the linear constraint Ax+By-c
    @param batch_M       		        : batch size of computing v0 
    @param sto_grad                     : function pointer for stochastic grad of f
    @param alpha    			        : hybrid parameter for stochastic gradients
    @param FullGradEval 		        : function pointer for gradient of f
    @param FuncF_Eval                   : function pointer to compute objective value of \f$ f(x)+g(y) \f$
    @param grad_diff                    : function pointer to compute gradient difference of loss function
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
    print_interval = 20

    init_grad = param['init_grad']
    max_inner = param['max_inner']

    x_test = param['x_test']
    y_test = param['y_test']     
    
    print( 'New hybrid sto ADMM ...' )
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

    
    num_grad, num_epoch = 0, 0
    x = x0
    x_old = x0
    y = y0
    w = lamb0
    iter_num = 0
    acc_list = [predict(x_test, y_test, x)]

    obj = [FuncF_Eval(A, x, lambda_1, N, AS)]
    total_time = 0
    norm_c = [ np.linalg.norm( linear_cons(AS, -np.eye(d), 0, x, y), 2 ) ]
    v = sto_grad(N, init_grad, A, x)
    grad_norm = [ np.linalg.norm( v, 2 ) ] 

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
        v = sto_grad(N, init_grad, A, x)
        num_grad += init_grad
        num_epoch += init_grad / N

        # Inner Loop
        for iter in range( 1,max_inner ):
            rho = gamma * rho
            rho_r = 1/rho

            #x-update
            diff = grad_diff(N, b_batch, A, x, x_old)
            v_sarah = diff + v
            sto_g = sto_grad(N, b_batch, A, x)
            v = alpha*v_sarah + (1-alpha)*sto_g
            x_old = x

            "invese case update"
            qx = rho * AST.dot(y + w*rho_r) + x * eta - v  
            x = x_update( qx, L )
            num_grad += 3 * b_batch
            num_epoch = num_grad / N

            #y-update
            y = y_update(rho, AS, x, y, w, lambda_1)        
        
            #dual-update      
            u = linear_cons(AS, B, c, x, y)
            w = w - rho * u
            norm_c.append(np.linalg.norm(u, 2))
            max_tol = np.max(np.abs(u))
            itertime = time.time() - tic
            total_time = total_time +itertime

            iter_num += 1

            # if max_tol < delta:
            #     break

            obj.append(FuncF_Eval(A, x, lambda_1, N, AS))
            grad_norm.append(np.linalg.norm( FullGradEval(A, AT, x, N), 2 ))

            acc_cur = predict(x_test, y_test, x)
            acc_list.append(acc_cur)


            if iter_num % print_interval == print_interval-1:
                print(
                '{:^16.4f}'.format(iter_num),'|',
                '{:^15.3e}'.format(obj[-1]),'|',
                '{:^15.3e}'.format(grad_norm[-1]),'|'
                '{:^15.3e}'.format(norm_c[-1]),
                '{:^15.3e}'.format(acc_cur),
                )                
    
    output = {
        'iter': iter_num,
        'objective': obj[-1],
        'beta': x,
        'total time': total_time,
        'obj_list': obj,
        'dual_gap_norm': norm_c,
        'grad_norm': grad_norm,
        'acc_list': acc_list
    }

    return output



















