import numpy as np
from scipy.integrate import solve_ivp
from fdasrsf import curve_functions as cf

def solve_FrenetSerret_ODE_SE(theta, t_eval, Z0=None,  method='Radau'):
    N, dim = theta(t_eval).shape
    if Z0 is None:
        Z0 = np.eye(dim+2)
    if method=='Radau':
        rho = np.zeros((dim+1,1))
        rho[0,0] = 1
        A_theta = lambda s: np.vstack((np.hstack(((- np.diag(theta(s), 1) + np.diag(theta(s), -1)), rho)), np.zeros(dim+2)))
        ode = lambda s,Z: (np.matmul(Z.reshape(dim+2,dim+2),A_theta(s))).flatten()
        sol = solve_ivp(ode, t_span=(t_eval[0], t_eval[-1]), y0=Z0.flatten(), t_eval=t_eval,  method='Radau')
        Z = sol.y.reshape(dim+2,dim+2,len(t_eval))
        Z = np.moveaxis(Z, -1, 0)
    elif method=='Linearized':
        n = dim+1
        X0 = Z0[:n,n]
        Q0 = Z0[:n,:n]
        A_theta = lambda s:  np.diag(theta(s), 1) - np.diag(theta(s), -1)
        A22 = lambda t: np.kron(A_theta(t), np.eye(n))
        A11 = np.zeros((n,n))
        A21 = np.zeros((n*n,n))
        A12 = np.concatenate((np.eye(n), np.zeros((n,n*(n-1)))), axis=1)
        Az  = lambda t: np.concatenate((np.concatenate((A11, A12), axis=1), np.concatenate((A21, A22(t)), axis=1)))
        Z0 = np.concatenate((X0, np.concatenate(([Q0[:,i] for i in range(n)]))))
        ode_func = lambda t,z: np.matmul(Az(t),z)
        sol = solve_ivp(ode_func, t_span=(t_eval[0], t_eval[-1]), y0=Z0, t_eval=t_eval)
        sol_Z = sol.y
        """ Reshape Z """
        sol_Z = np.reshape(sol_Z, (dim+2,dim+1,-1))
        sol_Z = np.moveaxis(sol_Z, (0,1,2),(2,1,0))
        Z = np.zeros((len(t_eval), dim+2, dim+2))
        Z[:,dim+1, dim+1] = np.ones(len(t_eval))
        Z[:,:dim+1,:dim+1] = sol_Z[:,:,1:dim+2]
        Z[:,:dim+1,dim+1] = sol_Z[:,:,0]
    else:
        raise Exception('Invalide method name, choose between: Linearized or Radau')
    return Z


def solve_FrenetSerret_ODE_SO(theta, t_eval, Q0=None, method='Radau'):
    N, dim = theta(t_eval).shape
    if Q0 is None:
        Q0 = np.eye(dim+1)
    if method=='Radau':
        A_theta = lambda s: - np.diag(theta(s), 1) + np.diag(theta(s), -1)
        ode = lambda s,Q: (np.matmul(Q.reshape(dim+1,dim+1),A_theta(s))).flatten()
        sol = solve_ivp(ode, t_span=(t_eval[0], t_eval[-1]), y0=Q0.flatten(), t_eval=t_eval,  method='Radau')
        Q = sol.y.reshape(dim+1,dim+1,len(t_eval))
        Q = np.moveaxis(Q, -1, 0)
    return Q


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