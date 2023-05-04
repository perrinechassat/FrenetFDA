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
import os
import dill as pickle
from simulations_tools import * 
from tqdm import tqdm

""" 

Scenario 1: Simulation code to compare the different methods of initialization. 
    
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

directory = r"results/scenario_1/model_02"
filename_base = "results/scenario_1/model_02/"

current_directory = os.getcwd()
final_directory = os.path.join(current_directory, directory)
if not os.path.exists(final_directory):
   os.makedirs(final_directory)


""" Definition of the true parameters """

## Theta 
def theta(s):
   curv = lambda s : 2*np.cos(2*np.pi*s) + 5
   tors = lambda s : 2*np.sin(2*np.pi*s) + 1
   if isinstance(s, int) or isinstance(s, float):
      return np.array([curv(s), tors(s)])
   elif isinstance(s, np.ndarray):
      return np.vstack((curv(s), tors(s))).T
   else:
      raise ValueError('Variable is not a float, a int or a NumPy array.')
   
arc_length_fct = lambda s: s
# def warping(s,a):
#     if np.abs(a)<1e-15:
#         return s
#     else:
#         return (np.exp(a*s) - 1)/(np.exp(a) - 1)    

## Gamma
gamma = 0.001
Gamma = gamma**2*np.eye(3)

## Sigma ? 
# sigma_1 = 0.001
# sigma_2 = 0.001
# Sigma = lambda s: np.array([[sigma_1**2 + 0*s, 0*s],[0*s, sigma_2**2 + 0*s]])
Sigma = None 

## mu_0 and P_0
mu0 = np.eye(4)
P0 = 0.001**2*np.eye(6)

## number of samples and basis fct
N = 200
nb_basis = 15

## grid of parameters
bandwidth_grid_init = np.array([0.05, 0.1, 0.12, 0.15, 0.17, 0.2, 0.25])
reg_param_grid_init = np.array([1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01])

## Param EM
max_iter = 200
tol = 1e-3
reg_param_grid_EM = np.array([[1e-06,1e-06], [1e-05,1e-05], [1e-04,1e-04], [1e-03,1e-03], [1e-02,1e-02], [1e-01,1e-01]])
reg_param_grid_EM = np.array(np.meshgrid(*reg_param_grid_EM.T)).reshape((2,-1))
reg_param_grid_EM = np.moveaxis(reg_param_grid_EM, 0,1)


""" Number of simulations """
N_simu = 100


# filename = filename_base + "model"
# dic = {"nb_iterations_simu": N_simu, "P0": P0, "mu0": mu0, "theta":theta, "Gamma":Gamma, "Sigma":Sigma, "reg_param_grid_EM":reg_param_grid_EM, "max_iter":max_iter, "tol":tol, "N":N, 
#        "arc_length_fct": arc_length_fct, "bandwidth_grid_init" : bandwidth_grid_init, "nb_basis":nb_basis, "reg_param_grid_init": reg_param_grid_init}
# if os.path.isfile(filename):
#    print("Le fichier ", filename, " existe déjà.")
#    filename = filename + '_bis'
# fil = open(filename,"xb")
# pickle.dump(dic,fil)
# fil.close()



""" S1.1: LP + GS + Extrinsic formulas """

# print('--------------------- Start scenario 1.1 ---------------------')

# time_init = time.time()

# with tqdm(total=N_simu) as pbar:
#    res_S1_1 = Parallel(n_jobs=50)(delayed(scenario_1_1)(theta, Sigma, mu0, P0, Gamma, N, arc_length_fct, nb_basis, bandwidth_grid_init, reg_param_grid_init, reg_param_grid_EM, max_iter, tol) for k in range(N_simu))
#    pbar.update()

# # arr_FS_statespace_S1_1 = np.empty((N_simu), dtype=object)
# # for k in range(N_simu):
# #    print('iteration:', k)
# #    arr_FS_statespace_S1_1[k] = scenario_1_1(theta, Sigma, mu0, P0, Gamma, N, arc_length_fct, nb_basis, bandwidth_grid_init, reg_param_grid_init, reg_param_grid_EM, max_iter, tol)

# time_end = time.time()
# duration = time_end - time_init

# filename = filename_base + "scenario_1_1"

# dic = {"results_S1_1":res_S1_1}

# if os.path.isfile(filename):
#    print("Le fichier ", filename, " existe déjà.")
#    filename = filename + '_bis'
# fil = open(filename,"xb")
# pickle.dump(dic,fil)
# fil.close()

# print('End of scenario 1.1: time spent', duration, 'seconds. \n')





""" S1.2: LP + GS + Approx ODE """

# print('--------------------- Start scenario 1.2 ---------------------')

# time_init = time.time()

# with tqdm(total=N_simu) as pbar:
#    res_S1_2 = Parallel(n_jobs=50)(delayed(scenario_1_2)(theta, Sigma, mu0, P0, Gamma, N, arc_length_fct, nb_basis, bandwidth_grid_init, reg_param_grid_init, reg_param_grid_EM, max_iter, tol) for k in range(N_simu))
#    pbar.update()

# time_end = time.time()
# duration = time_end - time_init

# filename = filename_base + "scenario_1_2"

# dic = {"results_S1_2":res_S1_2}

# if os.path.isfile(filename):
#    print("Le fichier ", filename, " existe déjà.")
#    filename = filename + '_bis'
# fil = open(filename,"xb")
# pickle.dump(dic,fil)
# fil.close()

# print('End of scenario 1.2: time spent', duration, 'seconds. \n')




""" S1.3: LP + GS + Local Approx ODE """

print('--------------------- Start of scenario 1.3 ---------------------')

time_init = time.time()

with tqdm(total=N_simu) as pbar:
   res_S1_3 = Parallel(n_jobs=50)(delayed(scenario_1_3)(theta, Sigma, mu0, P0, Gamma, N, arc_length_fct, nb_basis, bandwidth_grid_init, reg_param_grid_init, reg_param_grid_EM, max_iter, tol) for k in range(N_simu))
   pbar.update()

time_end = time.time()
duration = time_end - time_init

filename = filename_base + "scenario_1_3"

dic = {"results_S1_3":res_S1_3}

if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()

print('End of scenario 1.3: time spent', duration, 'seconds. \n')




""" S1.4: CLP + Approx ODE """

print('--------------------- Start scenario 1.4 ---------------------')

time_init = time.time()

with tqdm(total=N_simu) as pbar:
   res_S1_4 = Parallel(n_jobs=50)(delayed(scenario_1_4)(theta, Sigma, mu0, P0, Gamma, N, arc_length_fct, nb_basis, bandwidth_grid_init, reg_param_grid_init, reg_param_grid_EM, max_iter, tol) for k in range(N_simu))
   pbar.update()

time_end = time.time()
duration = time_end - time_init

filename = filename_base + "scenario_1_4"

dic = {"results_S1_4":res_S1_4}

if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()

print('End of scenario 1.4: time spent', duration, 'seconds. \n')




""" S1.5: CLP + Local Approx ODE """

print('--------------------- Start scenario 1.5 ---------------------')

time_init = time.time()

with tqdm(total=N_simu) as pbar:
   res_S1_5 = Parallel(n_jobs=50)(delayed(scenario_1_5)(theta, Sigma, mu0, P0, Gamma, N, arc_length_fct, nb_basis, bandwidth_grid_init, reg_param_grid_init, reg_param_grid_EM, max_iter, tol) for k in range(N_simu))
   pbar.update()

time_end = time.time()
duration = time_end - time_init

filename = filename_base + "scenario_1_5"

dic = {"results_S1_5":res_S1_5}

if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()

print('End of scenario 1.5: time spent', duration, 'seconds. \n')











# ### Sauvegarde

# filename = "results/initialization_extrinsic_formulas_01"

# dic = {"arr_Z": arr_Z, "arr_Y": arr_Y, "arr_arc_length": arr_arc_length, "arr_Gamma_hat": arr_Gamma_hat, "arr_basis_extrins": arr_basis_extrins, 
#        "arr_mu0_hat": arr_mu0_hat, "arr_P0_hat": arr_P0_hat, "arr_sig0_hat": arr_sig0_hat, "arr_FS_statespace" : arr_FS_statespace,
#        "duration": duration, "nb_iterations": N_simu,
#        "P0": P0, "mu0": mu0, "theta":theta, "Gamma":Gamma, "Sigma":Sigma, "reg_param_EM":reg_param_EM, "max_iter":max_iter, "tol":tol, "N":N,  "time":time_grid, "arc_length": arc_length, 
#        "bandwidth_grid" : bandwidth_grid, "nb_basis_grid":nb_basis_grid, "reg_param_grid": reg_param_grid}

# if os.path.isfile(filename):
#    print("Le fichier ", filename, " existe déjà.")
#    filename = filename + '_bis'
# fil = open(filename,"xb")
# pickle.dump(dic,fil)
# fil.close()







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








