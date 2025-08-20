""" 
example_2 for hybrid SADMM algorithm: The graph-guided fused lasso
\min_{x,y} F(x,y) = f(x) + g(y) := 1/n*\frac{1}{1+exp(b_{i}a_{i}^{T}x)} + \lambda_{1}*\|y\|_{1}
subject to Ax-y = 0
"""

import numpy as np
import os
import time
from sklearn import preprocessing
import pandas as pd
import scipy.linalg
from sklearn.datasets import load_svmlight_file

def read_csv_data(file):
    df = pd.read_csv(file)
    ## frist line is label
    y = df['0']
    x = df.drop(['0'], axis=1)
    x_train = np.asarray(x)
    y_train = np.asarray(y)
    return x_train, y_train


def read_libsvm_data(file):
    data = load_svmlight_file(file)
    x_train = data[0].toarray()
    y_train = data[1]
    return x_train, y_train


def train_data(path):
    # x_train'shape is (N,d)， y_trian‘s shape is N
    x_train, y_train = read_libsvm_data(path)
    N, d = x_train.shape
    # zscore = preprocessing.StandardScaler()
    zscore = preprocessing.StandardScaler()
    x_train = zscore.fit_transform(x_train)
    A = (x_train.T * y_train).T
    Hessian = np.dot(A.T, A)
    eig, _ = np.linalg.eig(Hessian)
    L = max(eig)
    AT = A.T
    kappa = 1
    epsilon = 0.1
    ## make parameter dic
    LPADMM_param = {}
    LPADMM_param['d'] = d
    LPADMM_param['N'] = N
    LPADMM_param['maxiter'] = 1000
    LPADMM_param['epsilon'] = epsilon
    LPADMM_param['delta'] = 1e-5
    LPADMM_param['rho'] = 6
    LPADMM_param['gamma'] = 1.00
    LPADMM_param['kappa'] = kappa
    LPADMM_param['Hessian'] = Hessian
    LPADMM_param['A'] = A
    LPADMM_param['AT'] = AT
    # LPADMM_param['L'] = L
    LPADMM_param['L_data_A'] = L
    LPADMM_param['L'] = (2+np.sqrt(3))*(1+np.sqrt(3))/(3 + np.sqrt(3))**3

    LPADMM_param['lambda1'] = 1e-7
    LPADMM_param['x_train'] = x_train
    
    return A, LPADMM_param


def test_Data(filename):
    path2 = r'/Users/'
    path = path2 + filename
    x_test, y_test = read_libsvm_data(path)
    N, d = x_test.shape
    # zscore = preprocessing.StandardScaler()
    zscore = preprocessing.StandardScaler()
    x_test = zscore.fit_transform(x_test)
    return x_test, y_test


def main(file2):
    path2 = r'/Users/'
    filename2 = path2 + file2
    A, LPADMM_param = train_data(filename2)
    return A, LPADMM_param









def covsel( lambda_, rho, alpha, main, file2):
    A, LPADMM_param = main(file2)
    D = LPADMM_param['x_train']
    QUIET = False
    MAX_ITER = 1000
    ABSTOL = 1e-4
    RELTOL = 1e-2
    S = np.cov(D, rowvar=False)
    n = S.shape[0]

    X = np.zeros((n, n))
    Z = np.zeros((n, n))
    U = np.zeros((n, n))

    if not QUIET:
        print(f'{"iter":<3}\t{"r norm":<10}\t{"eps pri":<10}\t{"s norm":<10}\t{"eps dual":<10}\t{"objective":<10}')

    for k in range(1, MAX_ITER+1):
        # x-update
        L_1, Q = np.linalg.eigh(rho * (Z - U) - S)
        # es = np.diag(L_1)
        es = L_1
        xi = (es + np.sqrt(es**2 + 4 * rho)) / (2 * rho)
        X = Q @ np.diag(xi) @ Q.T

        # z-update with relaxation
        Zold = Z
        X_hat = alpha * X + (1 - alpha) * Zold
        Z = shrinkage(X_hat + U, lambda_ / rho)

        U = U + (X_hat - Z)

        # diagnostics, reporting, termination checks
        objval = objective(S, X, Z, lambda_)

        r_norm = np.linalg.norm(X - Z, 'fro')
        s_norm = np.linalg.norm(-rho * (Z - Zold), 'fro')

        eps_pri = np.sqrt(n * n) * ABSTOL + RELTOL * max(np.linalg.norm(X, 'fro'), np.linalg.norm(Z, 'fro'))
        eps_dual = np.sqrt(n * n) * ABSTOL + RELTOL * np.linalg.norm(rho * U, 'fro')

        if not QUIET:
            print(f'{k:<3}\t{r_norm:<10.4f}\t{eps_pri:<10.4f}\t{s_norm:<10.4f}\t{eps_dual:<10.4f}\t{objval:.2f}')

    sparse_A = X
    sparse_AT = X.T
    
    LPADMM_param['HessianA_s'] = np.dot(sparse_AT, sparse_A)
    LPADMM_param['sparse_A'] = sparse_A
    LPADMM_param['sparse_AT'] = sparse_AT

    return sparse_A, LPADMM_param

def shrinkage(X, kappa):
    return np.maximum(0, X - kappa) - np.maximum(0, -X - kappa)

def objective(S, X, Z, lambda_):
    return np.trace(np.dot(S, X)) - np.log(np.linalg.det(X)) + lambda_ * np.linalg.norm(Z, 1)

lambda_ = 0.5
rho = 50
alpha = 1.5


