import numpy as np
from scipy.integrate import solve_ivp
from fdasrsf import curve_functions as cf
from FrenetFDA.utils.Lie_group.SE3_utils import SE3
from scipy.linalg import block_diag
# import multiprocessing
import time
import signal



def find_best_rotation(X1, X2, allow_reflection=False):
    """
    This function calculates the best rotation between two Euclidean curves using procustes rigid alignment: such that X1_i approx R @ X2_i

    :param X1: numpy ndarray of shape (M,dim) of M samples in R^dim
    :param X2: numpy ndarray of shape (M,dim) of M samples in R^dim
    :param allow_reflection: bool indicating if reflection is allowed (i.e. if the determinant of the optimalrotation can be -1)

    """
    if X1.ndim != 2 or X2.ndim != 2:
        raise Exception("This only supports curves of shape (M,n) for n dimensions and M samples")

    N, dim = X1.shape
    A = X1.T@X2
    U, s, Vh = np.linalg.svd(A)
    S = np.eye(dim)
    if np.linalg.det(A) < 0 and not allow_reflection:
        S[-1, -1] = -1 # the last entry of the matrix becomes -1
    R = U@S@Vh 
    X2new = (R @ X2.T).T

    return X2new, R


def centering(X):
    N = X.shape[0]
    cent = -cf.calculatecentroid(X.T)
    X = X + np.tile(cent,(N,1))
    return X

def Euclidean_dist_cent_rot(Y, X):
    Y_cent = centering(Y)
    X_cent = centering(X)
    new_X, R = find_best_rotation(Y_cent, X_cent)
    dist = np.linalg.norm((Y_cent - new_X))**2
    return dist 