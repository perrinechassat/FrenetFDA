import sys
sys.path.insert(1, '../')
from FrenetFDA.utils.Frenet_Serret_utils import *
from FrenetFDA.utils.smoothing_utils import *
from FrenetFDA.processing_Euclidean_curve.unified_estimate_Frenet_state_space.EM_Frenet_state_space_CV import FrenetStateSpaceCV_global, bayesian_CV_optimization_regularization_parameter
from FrenetFDA.processing_Euclidean_curve.unified_estimate_Frenet_state_space.iek_filter_smoother_Frenet_state import IEKFilterSmootherFrenetState
from FrenetFDA.processing_Euclidean_curve.preprocessing import *
from FrenetFDA.processing_Euclidean_curve.estimate_Frenet_path import GramSchmidtOrthogonalization, ConstrainedLocalPolynomialRegression
from FrenetFDA.processing_Euclidean_curve.estimate_Frenet_curvatures import ExtrinsicFormulas
from pickle import *
import time 
import os.path
import os
import dill as pickle
from tqdm import tqdm
from simulation_utils import *


# def influence_N_Gamma(theta, arc_length, N, Gamma, mu0, P0, nb_basis, sigma_init, P0_init, noise_init_theta, grid_bandwidth, max_iter_EM, tol_EM, grid_lambda, n_splits_CV):

#     Z, Y, init_coefs, grid_arc_length, init_Gamma, init_mu0 = init_from_true_param(theta, arc_length, N, Gamma, mu0, P0, nb_basis, noise_init_theta, grid_bandwidth)
#     FS_statespace, score_lambda_matrix  = simulation_step(Y, grid_arc_length, init_coefs, init_Gamma, init_mu0, P0_init, sigma_init, nb_basis, max_iter_EM, tol_EM, grid_lambda, n_splits_CV)  

#     return FS_statespace, Z, Y, score_lambda_matrix



""" MODEL for simulations """


def theta(s):
    curv = lambda s : 2*np.cos(2*np.pi*s) + 5
    tors = lambda s : 2*np.sin(2*np.pi*s) + 1
    if isinstance(s, int) or isinstance(s, float):
        return np.array([curv(s), tors(s)])
    elif isinstance(s, np.ndarray):
        return np.vstack((curv(s), tors(s))).T
    else:
        raise ValueError('Variable is not a float, a int or a NumPy array.')


P0 = 0.01**2*np.eye(6)
mu0 = np.eye(4) 
arc_length_fct = lambda s: s
n_MC = 80
noise_init_theta = 1
tol_EM = 0.1
max_iter_EM = 200
nb_basis = 15
n_splits_CV = 5
grid_bandwidth = np.array([0.05, 0.1, 0.12, 0.15, 0.17, 0.2, 0.25])
n_call_bayopt = 20
bounds_lambda = ((1e-09, 1e-06), (1e-09, 1e-06))
N = 100
gamma = 0.001
grid_time = np.linspace(0,1,N)
arc_length = grid_time 
mu_Z = solve_FrenetSerret_ODE_SE(theta, arc_length, mu0)
Gamma = gamma**2*np.eye(3)

tab_Z_init = []
tab_Y = []
tab_init_coefs = []
tab_init_grid = []
tab_init_Gamma = []
tab_init_mu0 = []  
for k in range(n_MC):
    Z, Y, init_coefs, grid_arc_length, init_Gamma, init_mu0 = init_from_true_param(theta, arc_length, N, Gamma, mu0, P0, nb_basis, noise_init_theta, grid_bandwidth)
    tab_Z_init.append(Z)
    tab_Y.append(Y)
    tab_init_coefs.append(init_coefs)
    tab_init_grid.append(grid_arc_length)
    tab_init_Gamma.append(init_Gamma)
    tab_init_mu0.append(init_mu0)


directory = r"results/influence_sigma_init/model_02/"
filename_base = "results/influence_sigma_init/model_02/"

current_directory = os.getcwd()
final_directory = os.path.join(current_directory, directory)
if not os.path.exists(final_directory):
   os.makedirs(final_directory)

filename = filename_base + "model"
dic = {"nb_iterations_simu": n_MC, "P0": P0, "mu0": mu0, "theta":theta, "bounds_lambda":bounds_lambda, "max_iter":max_iter_EM, "tol":tol_EM, "N":N, "Gamma":Gamma, "mu_Z":mu_Z, 
       "arc_length_fct": arc_length_fct, "nb_basis":nb_basis, 'n_splits_CV':n_splits_CV, "grid_bandwidth":grid_bandwidth, "n_call_bayopt":n_call_bayopt, 
       "tab_Z_init":tab_Z_init, "tab_Y":tab_Y, "tab_init_coefs":tab_init_coefs, "tab_init_grid":tab_init_grid, "tab_init_Gamma":tab_init_Gamma, "tab_init_mu0":tab_init_mu0}
if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()



""" _______________________ Simulation 1: \sigma_init = 0.001 _______________________ """

print('--------------------- Simulation n°1: \sigma_init = 0.001 ---------------------')

time_init = time.time()

sigma_init = 0.001
Sigma_init = lambda s: sigma_init**2*np.array([[1 + 0*s, 0*s],[0*s, 1 + 0*s]]) 
P0_init = sigma_init**2*np.eye(6)

with tqdm(total=n_MC) as pbar:
   res = Parallel(n_jobs=n_MC)(delayed(bayesian_CV_optimization_regularization_parameter)(n_CV=n_splits_CV, n_call_bayopt=n_call_bayopt, lambda_bounds=bounds_lambda, grid_obs=tab_init_grid[k], Y_obs=tab_Y[k], tol=tol_EM, max_iter=max_iter_EM, nb_basis=nb_basis, 
                                                                                          init_params={"Gamma":tab_init_Gamma[k], "coefs":tab_init_coefs[k], "mu0":tab_init_mu0[k], "Sigma":Sigma_init, "P0":P0_init}) for k in range(n_MC))
   pbar.update()

time_end = time.time()
duration = time_end - time_init

filename = filename_base + "simu_1"

dic = {"results":res, "duration":duration, "P0_init":P0_init, "sigma_init":sigma_init}

if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()


print('--------------------- End Simulation n°1 ---------------------')




""" _______________________ Simulation 2: \sigma_init = 0.01 _______________________ """

print('--------------------- Simulation n°2: \sigma_init = 0.01 ---------------------')

time_init = time.time()

sigma_init = 0.01
Sigma_init = lambda s: sigma_init**2*np.array([[1 + 0*s, 0*s],[0*s, 1 + 0*s]]) 
P0_init = sigma_init**2*np.eye(6)

with tqdm(total=n_MC) as pbar:
   res = Parallel(n_jobs=n_MC)(delayed(bayesian_CV_optimization_regularization_parameter)(n_CV=n_splits_CV, n_call_bayopt=n_call_bayopt, lambda_bounds=bounds_lambda, grid_obs=tab_init_grid[k], Y_obs=tab_Y[k], tol=tol_EM, max_iter=max_iter_EM, nb_basis=nb_basis, 
                                                                                          init_params={"Gamma":tab_init_Gamma[k], "coefs":tab_init_coefs[k], "mu0":tab_init_mu0[k], "Sigma":Sigma_init, "P0":P0_init}) for k in range(n_MC))
   pbar.update()

time_end = time.time()
duration = time_end - time_init

filename = filename_base + "simu_2"

dic = {"results":res, "duration":duration, "P0_init":P0_init, "sigma_init":sigma_init}

if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()


print('--------------------- End Simulation n°2 ---------------------')




""" _______________________ Simulation 3: \sigma_init = 0.1 _______________________ """

print('--------------------- Simulation n°3: \sigma_init = 0.1 ---------------------')

time_init = time.time()

sigma_init = 0.1
Sigma_init = lambda s: sigma_init**2*np.array([[1 + 0*s, 0*s],[0*s, 1 + 0*s]]) 
P0_init = sigma_init**2*np.eye(6)

with tqdm(total=n_MC) as pbar:
   res = Parallel(n_jobs=n_MC)(delayed(bayesian_CV_optimization_regularization_parameter)(n_CV=n_splits_CV, n_call_bayopt=n_call_bayopt, lambda_bounds=bounds_lambda, grid_obs=tab_init_grid[k], Y_obs=tab_Y[k], tol=tol_EM, max_iter=max_iter_EM, nb_basis=nb_basis, 
                                                                                          init_params={"Gamma":tab_init_Gamma[k], "coefs":tab_init_coefs[k], "mu0":tab_init_mu0[k], "Sigma":Sigma_init, "P0":P0_init}) for k in range(n_MC))
   pbar.update()

time_end = time.time()
duration = time_end - time_init

filename = filename_base + "simu_3"

dic = {"results":res, "duration":duration, "P0_init":P0_init, "sigma_init":sigma_init}

if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()


print('--------------------- End Simulation n°3 ---------------------')













