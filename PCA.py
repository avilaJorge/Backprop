# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:16:31 2020

@author: Jorge
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from numpy import linalg as LA

DIFF_THRESH = 1e-8
ZERO_THRESH = 1e-12
INFO_TXT = '[INFO]: '
# DEBUG = True
DEBUG = False

def sanity_check(A, W, Av, k):
    """
    Sanity check code for PCA

    :param A:  Matrix of images as column vectors
    :param W:  eigenvalues
    :param Av: Matrix of eigenvectors for large covariance matrix N^2xN^2
    :param k:  Number of principal components
    :return:   N/A
    """
    plot = None
    print("--- Sanity Check ---\n")
    for i in range(k):
        eig_val = W[i]
        eig_val_sqrt = np.sqrt(eig_val)
        eig_vec = Av[:,i]/LA.norm(Av[:,i])
        
        print(INFO_TXT + 'Checking eigenvector ' + str(i))
        
        proj = np.matmul(A.T, eig_vec)
        
        assert(np.mean(proj) <= ZERO_THRESH)
            
        assert(np.absolute(np.std(proj) - eig_val_sqrt) <= DIFF_THRESH)
        
        proj_2 = proj/eig_val_sqrt
        
        assert(np.mean(proj_2) <= ZERO_THRESH)
        
        assert(np.absolute(np.std(proj_2) - 1) <= DIFF_THRESH)
            
        print(INFO_TXT + 'OK')

def PCA(data, k, mean=None):
    """
    
    Returns PCs as column vectors of returned matrix and the mean image

    Parameters
    ----------
    imgs : images as returned by load_data
    k : number of PCs to return
    mean_image : [optional] Mean image of data as a dx1 vector. The default is None.

    Returns
    -------
    v_star: Normalized eigenvectors for larger covariance matrix (N^2xN^2)
    Psi : mean image (dx1)

    """
    # Flatten images and create matrix A = [Phi_1 ... Phi_i ... Phi_M]
    A = np.array(data).T
    M = A.shape[1]

    assert(k <= M)

    # Compute mean image (if necessary) and center all images
    Psi = mean
    if mean is None:
        Psi = A.sum(axis=1)/M
    A = np.array([np.subtract(Phi_i, Psi) for Phi_i in A.T]).T
    
    # Create alternate (smaller) form of covariance matrix
    C = np.matmul(A.T, A)
    C /= M
    
    # Get the eigenvalues/eigenvectors and sort by descending eigenvalues
    w, v = LA.eig(C)
    sorted_indices = w.argsort()[::-1]
    w = w[sorted_indices]
    v = v[:, sorted_indices]
    
    # Compute eigenvectors of much later covariance matrix.
    Av = np.matmul(A,v)
    
    if (DEBUG): sanity_check(A, w, Av, k)
    
    v_star = []
    for i in range(k):
        v_star_buffer = Av[:,i]/(LA.norm(Av[:,i])*np.sqrt(w[i]))
        v_star.append(v_star_buffer)
    
    v_star = np.array(v_star).T
    return v_star, Psi
