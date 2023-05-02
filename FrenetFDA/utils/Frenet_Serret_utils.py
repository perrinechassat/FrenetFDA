import numpy as np
from scipy.integrate import solve_ivp
from fdasrsf import curve_functions as cf
from FrenetFDA.utils.Lie_group.SE3_utils import SE3
from scipy.linalg import block_diag


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

def Euclidean_dist_cent_rot(Y, X):
    Y_cent = centering(Y)
    X_cent = centering(X)
    new_X, R = find_best_rotation(Y_cent, X_cent)
    dist = np.linalg.norm((Y_cent - new_X))**2
    return dist 


def solve_FrenetSerret_SDE_SE3(theta, Sigma, L, t_eval, Z0=None):
    """
        Solve the Frenet-Serret SDE $ dZ(s) = Z(s)(w_\theta(s)ds + LdB(s))^\wedge $ 
    """
    N, dim = theta(t_eval).shape
    if Z0 is None:
        Z0 = np.eye(dim+2)
    Z = np.zeros((N, dim+2, dim+2))
    Z[0] = Z0
    for i in range(1,N):
        delta_t = t_eval[i]-t_eval[i-1]
        Z[i] = Z[i-1]@SE3.exp(delta_t*np.array([theta(t_eval[i-1])[1], 0, theta(t_eval[i-1])[0], 1, 0, 0]) + np.sqrt(delta_t)*L @ np.random.multivariate_normal(np.zeros(2), Sigma(t_eval[i-1])))
    return Z 


def generate_Frenet_state_GP(theta, Sigma, L, t_eval, mu0, P0, method='covariance'):
    mu_Z = solve_FrenetSerret_ODE_SE(theta, t_eval, Z0=mu0)
    if method=='covariance':
        P_mat_full, P = solve_FrenetSerret_SDE_full_cov_matrix(theta, Sigma, L, t_eval, P0)
        xi_arr = np.random.multivariate_normal(mean=np.zeros(len(P_mat_full)), cov=P_mat_full)
        xi_arr = np.reshape(xi_arr, (len(t_eval),P0.shape[0]))
        Z = np.zeros((len(t_eval),mu0.shape[0],mu0.shape[0]))
        for i in range(len(t_eval)):
            Z[i] = mu_Z[i]@SE3.exp(-xi_arr[i])
    else:
        xi_arr = solve_FrenetSerret_SDE_linearized(theta, Sigma, L, t_eval, P0) 
        Z = np.zeros((len(t_eval),mu0.shape[0],mu0.shape[0]))
        for i in range(len(t_eval)):
            Z[i] = mu_Z[i]@SE3.exp(-xi_arr[i])
    return Z



def solve_FrenetSerret_SDE_linearized(theta, Sigma, L, t_eval, P0):
    xi0 = np.random.multivariate_normal(np.zeros(P0.shape[0]), P0)
    N = len(t_eval)
    xi = np.zeros((N,xi0.shape[0]))
    xi[0] = xi0
    for i in range(1,N):
        delta_t = t_eval[i]-t_eval[i-1]
        xi[i] = -delta_t*SE3.Ad(np.array([theta(t_eval[i-1])[1], 0, theta(t_eval[i-1])[0], 1, 0, 0]))@xi[i-1] - np.sqrt(delta_t)*L@np.random.multivariate_normal(np.zeros(2), Sigma(t_eval[i-1]))
    return xi


def solve_FrenetSerret_SDE_cov_matrix(theta, Sigma, L, t_eval, P0):
    dim_g = P0.shape[0]
    F =  lambda s: -SE3.Ad(np.array([theta(s)[1], 0*s, theta(s)[0], 1+0*s, 0*s, 0*s]))
    ode_func = lambda t,p: (np.matmul(F(t),p.reshape(dim_g, dim_g)) + np.matmul(p.reshape(dim_g, dim_g),F(t).T) + L @ Sigma(t) @ L.T).flatten()
    sol = solve_ivp(ode_func, t_span=(t_eval[0], t_eval[-1]), y0=P0.flatten(), t_eval=t_eval) #, method='Radau')
    P = sol.y.reshape(dim_g, dim_g, len(t_eval))
    P = np.moveaxis(P, 2, 0)
    return P 


def solve_FrenetSerret_SDE_full_cov_matrix(theta, Sigma, L, t_eval, P0):
    """
        Attention only in dimension 3.

    """
    N = len(t_eval)
    N, dim_theta = theta(t_eval).shape
    n = dim_theta+1
    dim_g = int((1/2)*n*(n+1))
    rho = np.zeros((n,1))
    rho[0,0] = 1
    A_theta = lambda s: np.vstack((np.hstack(((- np.diag(theta(s), 1) + np.diag(theta(s), -1)), rho)), np.zeros(n+1)))
    ode_U = lambda u,p: (np.matmul(-SE3.Ad(SE3.vee(A_theta(u))),p.reshape(dim_g,dim_g))).flatten()
    sol_U = solve_ivp(ode_U, t_span=(0,1), y0=np.eye(dim_g).flatten(), t_eval=t_eval)
    U_sol_phi = sol_U.y.reshape(dim_g,dim_g,len(t_eval))
    C = np.zeros((dim_g*N,dim_g*N))
    for i in range(N):
        for j in range(i+1):
            if i==j:
                C[i*dim_g:(i+1)*dim_g,j*dim_g:(j+1)*dim_g] = np.eye(dim_g)
            else:
                C[i*dim_g:(i+1)*dim_g,j*dim_g:(j+1)*dim_g] = U_sol_phi[:,:,i]@np.linalg.inv(U_sol_phi[:,:,j])   
    W_appox = np.zeros(((N),dim_g,dim_g))
    W_appox[0] = P0
    for i in range(1,N):
        K_pts = 5
        x = np.linspace(t_eval[i-1],t_eval[i],K_pts)
        w_to_int = np.zeros((dim_g,dim_g,K_pts))
        for k in range(K_pts):
            phi = SE3.Ad_group(SE3.exp(-(t_eval[i]-x[k])*SE3.vee(A_theta((t_eval[i]+x[k])/2))))
            w_to_int[:,:,k] = phi @ L @ Sigma(x[k]) @ L.T @ phi.T
        W_appox[i] = np.trapz(w_to_int, x=x)
    W_mat = block_diag(*W_appox)
    P_mat = C@W_mat@C.T
    P = np.reshape(P_mat, (len(t_eval),dim_g,len(t_eval),dim_g))
    P = np.moveaxis(P, (1,2), (2,1))
    return P_mat, P


