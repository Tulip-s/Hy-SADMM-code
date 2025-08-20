"""
function for inexact hybrid stochastic ADMM for composite optimization problem
"""

# import package
import numpy as np
from scipy.linalg import cholesky, solve_triangular



"""
functions for loss 1
"""
def FuncF_Eval_1(A, x, lambda_, N, AS):
    """
    compute the objective of 1/n*\frac{1}{1+exp(b_{i}a_{i}^{T}x)} + \lambda_*\|y\|_{1}
    subject to AS*x-y = 0

    """
    term1 = np.sum(1. / (1 + np.exp(np.dot(A, x)))) / N
    term2 = lambda_ * np.linalg.norm(np.dot(AS, x), 1)
    objective = term1 + term2
    return objective


def prox_l1_norm( w, lamb ):
	"""! Compute the proximal operator of the \f$\ell_1\f$ - norm

	\f$ prox_{\lambda \|.\|_1} = {arg\min_x}\left\{\|.\|_1^2 + \frac{1}{2\lambda}\|x - w\|^2\right\} \f$
	
	Parameters
	---------- 
	@param w : input vector
	@param lamb : penalty paramemeter
	    
	Returns
	---------- 
	@retval : perform soft - thresholding on input vector
	"""
	return np.sign( w ) * np.maximum( np.abs( w ) - lamb, 0 )



"""
for SCAD penalty
"""
def fun_val_SCAD_g(y, kappa, c):
    """
    compute the SCAD penalty function value g(x) =  sum_{i=1}^{n}p_k(|y_i|)
    parameters
    @param kappa and c: parameters of SCAD function, kappa > 0, c > 2
    """

    y_a = np.abs(y)
    SCAD_y = np.where(y_a <= kappa, kappa * y_a, np.where(y_a > c * kappa, 0.5 * (c + 1) * (kappa ** 2),
                                                         (-(y_a ** 2) + 2 * c * kappa * y_a - kappa ** 2) / (2 * (c - 1))))
    return np.sum(SCAD_y)

def y_update_scad(qk, v, c, kappa):
    """
    For y update
    """
    qk_a = np.abs(qk)
    
    y_new = np.zeros_like(qk)
    
    mask1 = qk_a <= (1 + v)*kappa
    y_new[mask1] = np.sign(qk[mask1]) * np.maximum(qk_a[mask1] - kappa*v, 0)
    
    mask2 = qk_a > c*kappa
    y_new[mask2] = qk[mask2]
    
    mask3 = np.logical_and(~mask1, ~mask2)
    y_new[mask3] = ((c - 1) * qk[mask3] - np.sign(qk[mask3]) * c * kappa * v) / (c - 1 - v)
    
    return y_new



def FuncF_Eval_scad(A, x, lambda_, N, AS, kappa, c):
    """
    compute the objective of 1/n*\frac{1}{1+exp(b_{i}a_{i}^{T}x)} + \lambda_*SCAD(y)
    subject to AS*x-y = 0

    """
    term1 = np.sum(1. / (1 + np.exp(np.dot(A, x)))) / N
    
    term2 = lambda_ * fun_val_SCAD_g(np.dot(AS, x), kappa, c)
    objective = term1 + term2
    return objective    




def FullGradEval_1(A, AT, x_s, N):
    exp_Ax_s = np.exp(np.dot(A, x_s))
    gradient = -np.dot(AT, 1 / ((1 + exp_Ax_s) * ( 1 + 1/exp_Ax_s ))) / N 
    return gradient



def grad_diff_1(N, b_batch, A, x1, x2):
    #generate a random variable
    index = np.random.choice(N, b_batch, replace=False)
    s1 = A[index, :]
    exp_s1_x1 = np.exp(np.dot(s1, x1))
    p_1 = -np.dot(s1.T, 1 / ((1 + exp_s1_x1) * (1 + 1/exp_s1_x1 ))) / b_batch  

    exp_s1_x2 = np.exp(np.dot(s1, x2))
    p_2 = -np.dot(s1.T, 1 / ((1 + exp_s1_x2) * (1 + 1/exp_s1_x2 ))) / b_batch  

    grad_diff = p_1 - p_2
    return grad_diff

def sto_grad_1(N, b_batch, A, x): 
    index = np.random.choice(N, b_batch, replace=False)
    s = A[index, :]
    exp_s_x = np.exp(np.dot(s, x))
    p = -np.dot(s.T, 1 / ((1 + exp_s_x)* ( 1 + 1/exp_s_x ) )) / b_batch  

    return p


def linear_cons_1(A, B, c, x, y):
    return np.dot(A, x) + np.dot(B, y) - c

def y_update_1(rho, AS, x, y, w, lambda_):
    rho_r = 1 / rho

    # update y
    y_mid = AS.dot(x) - rho_r * w
    y = prox_l1_norm(y_mid, lambda_ * rho_r)
    return y




def x_update_1( x, rho, HessianA_s, AST, y, w, eta, grad_f):
    x_mid = AST.dot(y + w / rho)
    x = x - (1 / eta) * (grad_f + rho * HessianA_s.dot(x) - rho * x_mid)
    return x


"""For matrix inverse"""


def x_update_2( qx, L):
    x_mid = solve_triangular(L, qx, lower=True)
    x = solve_triangular(L.T, x_mid, lower=False)
    return x



def chole_inver2(eta, beta_ATA, nx):
    "Compute \frac{1}{\beta}\left(A^{\top} A+\frac{1}{\beta \eta_1 I_{k x}}\right)"
    diag_term = eta * np.eye(nx)
    M = beta_ATA + diag_term
    L = cholesky(M, lower=True)
    return L



# for accuracy computation
def predict(x_test, y_test, beta):
    z = np.dot(x_test, beta)
    prob = 1 / (1 + np.exp(-z))  
    y_test = y_test.reshape( prob.shape )
    
    prob[prob > 0.5] = 1
    prob[prob < 0.5] = -1
    
    acc = np.sum(y_test == prob) / len(y_test)
    
    return acc















