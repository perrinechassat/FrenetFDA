import sys
sys.path.insert(1, '../')
from FrenetFDA.utils.Frenet_Serret_utils import *
from FrenetFDA.utils.smoothing_utils import *
from FrenetFDA.processing_Euclidean_curve.unified_estimate_Frenet_state_space.EM_Frenet_state_space import FrenetStateSpace
from FrenetFDA.processing_Euclidean_curve.unified_estimate_Frenet_state_space.iek_filter_smoother_Frenet_state import IEKFilterSmootherFrenetState
from FrenetFDA.processing_Euclidean_curve.preprocessing import *
from FrenetFDA.processing_Euclidean_curve.estimate_Frenet_path import GramSchmidtOrthogonalization, ConstrainedLocalPolynomialRegression
from FrenetFDA.processing_Euclidean_curve.estimate_Frenet_curvatures import ExtrinsicFormulas
from pickle import *
import time
import os.path
import dill as pickle

""" 

Simulation code to compare the different methods of initialization. 
    
    Methods:
        1. Local polynomial smoothing of Y_{1,...,N}: X, X', X'', X''', 
           + Extrinsic formulas of Frenet curvatures, 
           + GS(X'(0), X''(0), X'''(0)).
        2. Local polynomial smoothing of Y_{1,...,N}: X, X', X'', X''',
           + GS(X', X'', X''') --> Q
           + ODE approximation to obtain the Frenet curvatures.
        3. Local polynomial smoothing of Y_{1,...,N}: X, X', X'', X''',
           + GS(X', X'', X''') --> Q
           + local ODE approximation to obtain the Frenet curvatures.
        4. Constrained local polynomial smoothing of Y_{1,...,N}: X, Q
           + ODE approximation to obtain the Frenet curvatures.
        5. Constrained local polynomial smoothing of Y_{1,...,N}: X, Q
           + local ODE approximation to obtain the Frenet curvatures.

"""


def one_step_simu(theta, P0, mu0, Gamma, Sigma, N, time, arc_length, bandwidth_grid, nb_basis_grid, reg_param_grid, reg_param_EM, max_iter, tol):

   L = np.zeros((6,2))
   L[0,1], L[2,0] = 1, 1

   xi0 = np.random.multivariate_normal(np.zeros(6), P0)
   Z0 = mu0 @ SE3.exp(-xi0)
   Z = solve_FrenetSerret_SDE_SE3(theta, Sigma, L, arc_length, Z0=Z0)
   Q = Z[:,:3,:3]
   X = Z[:,:3,3]

   Y = X + np.random.multivariate_normal(np.zeros(3), Gamma, size=(len(X)))

   derivatives, h_opt = compute_derivatives(Y, time, deg=3, h=None, CV_optimization_h={"flag":True, "h_grid":bandwidth_grid, "K":20})
   grid_arc_s, L, arc_s, arc_s_dot = compute_arc_length(Y, time, smooth=True, smoothing_param=h_opt)

   Gamma_hat = ((Y - derivatives[0]).T @ (Y - derivatives[0]))/N

   h_opt, nb_basis_opt, regularization_parameter_opt, CV_error_tab = ExtrinsicFormulas(Y, time, grid_arc_s, deg_polynomial=3).grid_search_optimization_hyperparameters(bandwidth_grid, nb_basis_grid, reg_param_grid, method='2', n_splits=20)
   Basis_extrins = ExtrinsicFormulas(Y, time, grid_arc_s, deg_polynomial=3).Bspline_smooth_estimates(h_opt, nb_basis_opt, regularization_parameter=regularization_parameter_opt)

   Z_GS, Q_GS, X_GS = GramSchmidtOrthogonalization(Y, grid_arc_s, deg=3).fit(h_opt) 
   mu0_hat = Z_GS[0]

   mu_Z_theta_extrins = solve_FrenetSerret_ODE_SE(Basis_extrins.evaluate, grid_arc_s, Z0=mu0_hat)
   Sigma_hat = np.zeros((N, 2, 2))
   for i in range(N):
      L = np.zeros((6,2))
      L[0,1], L[2,0] = 1, 1
      xi = -SE3.log(np.linalg.inv(mu_Z_theta_extrins[i])@Z_GS[i])
      Sigma_hat[i] = L.T @ xi[:,np.newaxis] @ xi[np.newaxis,:] @ L
   sig = np.sqrt((np.mean(Sigma_hat[:,0,0]) + np.mean(Sigma_hat[:,1,1]))/2)
   Sigma_hat = lambda s: sig**2*np.array([[1+0*s, 0*s], [0*s, 1+0*s]])
   
   P0_hat = sig**2*np.eye(6)
   
   FS_statespace = FrenetStateSpace(arc_length, Y)
   FS_statespace.expectation_maximization(tol, max_iter, nb_basis=15, regularization_parameter_list=reg_param_EM, init_params = {"W":Gamma_hat, "coefs":Basis_extrins.coefs, "mu0":mu0_hat, "Sigma":Sigma_hat, "P0":P0_hat}, init_states = None, method='autre', model_Sigma='single_constant')

   return Z, Y, grid_arc_s, Gamma_hat, Basis_extrins, mu0_hat, P0_hat, sig, FS_statespace


time_init = time.time()


""" Definition of the true parameters """

## Theta 
curv = lambda s : 2*np.cos(2*np.pi*s) + 5
tors = lambda s : 2*np.sin(2*np.pi*s) + 1
def theta(s):
    if isinstance(s, int) or isinstance(s, float):
        return np.array([curv(s), tors(s)])
    elif isinstance(s, np.ndarray):
        return np.vstack((curv(s), tors(s))).T
    else:
        raise ValueError('Variable is not a float, a int or a NumPy array.')
    

## Gamma
gamma = 0.001
Gamma = gamma**2*np.eye(3)


## Sigma ? 
sigma_1 = 0.001
sigma_2 = 0.001
Sigma = lambda s: np.array([[sigma_1**2 + 0*s, 0*s],[0*s, sigma_2**2 + 0*s]])


## mu_0 and P_0
mu0 = np.eye(4)
P0 = 0.001**2*np.eye(6)

## time
N = 200
time_grid = np.linspace(0,1,N)
def warping(s,a):
    if np.abs(a)<1e-15:
        return s
    else:
        return (np.exp(a*s) - 1)/(np.exp(a) - 1)
arc_length = time_grid # warping(time, 2)

## bandwidth grid of parameters
bandwidth_grid = np.array([0.05, 0.1, 0.12, 0.15, 0.17, 0.2, 0.22, 0.25, 0.3])
nb_basis_grid = np.array([15])
reg_param_grid = np.array([1e-05,1e-04,1e-03,1e-02,1e-01])


## Param EM
max_iter = 100
tol = 1e-3
reg_param_EM = np.array([[1e-05,1e-05], [1e-04,1e-04], [1e-03,1e-03], [1e-02,1e-02], [1e-01,1e-01]])
reg_param_EM = np.array(np.meshgrid(*reg_param_EM.T)).reshape((2,-1))
reg_param_EM = np.moveaxis(reg_param_EM, 0,1)




""" Number of simulations """
N_simu = 100


arr_Z = np.empty((N_simu), dtype=object)
arr_Y = np.empty((N_simu), dtype=object)
arr_arc_length =  np.empty((N_simu), dtype=object)
arr_Gamma_hat = np.empty((N_simu), dtype=object)
arr_basis_extrins = np.empty((N_simu), dtype=object)
arr_mu0_hat = np.empty((N_simu), dtype=object)
arr_P0_hat = np.empty((N_simu), dtype=object)
arr_sig0_hat = np.empty((N_simu), dtype=object)
arr_FS_statespace = np.empty((N_simu), dtype=object)


out = Parallel(n_jobs=-1)(delayed(one_step_simu)(theta, P0, mu0, Gamma, Sigma, N, time_grid, arc_length, bandwidth_grid, nb_basis_grid, reg_param_grid, reg_param_EM, max_iter, tol) for k in range(N_simu))


for k in range(N_simu):
   arr_Z[k] = out[k][0]
   arr_Y[k] = out[k][1]
   arr_arc_length[k] = out[k][2]
   arr_Gamma_hat[k] = out[k][3]
   arr_basis_extrins[k] = out[k][4]
   arr_mu0_hat[k] = out[k][5]
   arr_P0_hat[k] = out[k][6]
   arr_sig0_hat[k] = out[k][7]
   arr_FS_statespace[k] = out[k][8]


time_end = time.time()
duration = time_end - time_init

### Sauvegarde

filename = "results/initialization_extrinsic_formulas_01"

dic = {"arr_Z": arr_Z, "arr_Y": arr_Y, "arr_arc_length": arr_arc_length, "arr_Gamma_hat": arr_Gamma_hat, "arr_basis_extrins": arr_basis_extrins, 
       "arr_mu0_hat": arr_mu0_hat, "arr_P0_hat": arr_P0_hat, "arr_sig0_hat": arr_sig0_hat, "arr_FS_statespace" : arr_FS_statespace,
       "duration": duration, "nb_iterations": N_simu,
       "P0": P0, "mu0": mu0, "theta":theta, "Gamma":Gamma, "Sigma":Sigma, "reg_param_EM":reg_param_EM, "max_iter":max_iter, "tol":tol, "N":N,  "time":time_grid, "arc_length": arc_length, 
       "bandwidth_grid" : bandwidth_grid, "nb_basis_grid":nb_basis_grid, "reg_param_grid": reg_param_grid}

if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()







# for k in range(N_simu):

#    xi0 = np.random.multivariate_normal(np.zeros(6), P0)
#    Z0 = mu0 @ SE3.exp(-xi0)
#    Z = solve_FrenetSerret_SDE_SE3(theta, Sigma, L, arc_length, Z0=Z0)
#    Q = Z[:,:3,:3]
#    X = Z[:,:3,3]
#    arr_Z[k] = Z

#    Y = X + np.random.multivariate_normal(np.zeros(3), Gamma, size=(len(X)))
#    arr_Y[k] = Y

#    grid_arc_s, L, arc_s, arc_s_dot = compute_arc_length(Y, time, smooth=True, CV_optimization={"flag":True, "h_grid":bandwidth_grid, "K":20})
#    arr_arc_length[k] = grid_arc_s

#    h_opt, err_h = GramSchmidtOrthogonalization(Y, grid_arc_s, deg=3).grid_search_CV_optimization_bandwidth(bandwidth_grid=bandwidth_grid, K_split=20)
#    Z_GS, Q_GS, X_GS = GramSchmidtOrthogonalization(Y, grid_arc_s, deg=3).fit(h_opt) 

#    Gamma_hat_GS = ((Y - X_GS).T @ (Y - X_GS))/N
#    arr_Gamma_hat[k] = Gamma_hat_GS

#    h_opt, nb_basis_opt, regularization_parameter_opt, CV_error_tab = ExtrinsicFormulas(Y, time, grid_arc_s, deg_polynomial=3).grid_search_optimization_hyperparameters(bandwidth_grid, nb_basis_grid, reg_param_grid, method='2', n_splits=20)
#    Basis_extrins = ExtrinsicFormulas(Y, time, grid_arc_s, deg_polynomial=3).Bspline_smooth_estimates(h_opt, nb_basis_opt, regularization_parameter=regularization_parameter_opt)
#    arr_basis_extrins[k] = Basis_extrins

#    mu0_hat = Z_GS[0]
#    arr_mu0_hat[k] = mu0_hat

#    mu_Z_theta_extrins = solve_FrenetSerret_ODE_SE(Basis_extrins.evaluate, grid_arc_s, Z0=mu0_hat)
#    Sigma_hat = np.zeros((len(N), 2, 2))
#    for i in range(len(N)):
#       xi = -SE3.log(np.linalg.inv(mu_Z_theta_extrins[i])@Z_GS[i])
#       Sigma_hat[i] = L.T @ xi[:,np.newaxis] @ xi[np.newaxis,:] @ L
#    sig = np.sqrt((np.mean(Sigma_hat[:,0,0]) + np.mean(Sigma_hat[:,1,1]))/2)
#    arr_sig0_hat[k] = sig
#    Sigma_hat = lambda s: sig**2*np.array([[1+0*s, 0*s], [0*s, 1+0*s]])
   
#    P0 = sig**2*np.eye(6)
#    arr_P0_hat[k] = P0
   
#    FS_statespace = FrenetStateSpace(arc_length, Y)
#    FS_statespace.expectation_maximization(tol, max_iter, nb_basis=15, regularization_parameter_list=reg_param_EM, init_params = {"W":arr_Gamma_hat[k], "coefs":arr_basis_extrins[k].coefs, "mu0":arr_mu0_hat[k], "Sigma":Sigma_hat, "P0":P0}, init_states = None, method='autre', model_Sigma='single_constant')
#    arr_FS_statespace[k] = FS_statespace








