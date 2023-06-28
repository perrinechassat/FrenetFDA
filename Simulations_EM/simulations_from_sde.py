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
from sklearn.gaussian_process.kernels import Matern



def simu_from_sde(theta, arc_length, N, Gamma, mu0, P0, nb_basis, sigma_init, noise_init_theta, grid_bandwidth, max_iter_EM, tol_EM, bounds_lambda, n_call_bayopt, n_splits_CV, kernel):

    init_Sigma = lambda s: sigma_init**2*np.array([[1 + 0*s, 0*s],[0*s, 1 + 0*s]]) 
    P0_init = sigma_init**2*np.eye(6)

    Z, Y, init_coefs, grid_arc_length, init_Gamma, init_mu0, noise_theta = init_from_true_param_sde(theta, arc_length, N, Gamma, mu0, P0, nb_basis, noise_init_theta, grid_bandwidth, kernel)
    FS_statespace, res_bayopt = bayesian_CV_optimization_regularization_parameter(n_CV=n_splits_CV, n_call_bayopt=n_call_bayopt, lambda_bounds=bounds_lambda, grid_obs=grid_arc_length, Y_obs=Y, tol=tol_EM, max_iter=max_iter_EM, 
                                                                                  nb_basis=nb_basis, init_params={"Gamma":init_Gamma, "coefs":init_coefs, "mu0":init_mu0, "Sigma":init_Sigma, "P0":P0_init})
    
    return FS_statespace, Z, Y, res_bayopt, noise_theta



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
sigma_init = 0.05
N = 100
gamma = 0.001
grid_time = np.linspace(0,1,N)
arc_length = arc_length_fct(grid_time) 
Gamma = gamma**2*np.eye(3)
mu_Z = solve_FrenetSerret_ODE_SE(theta, arc_length, mu0)

directory = r"results/simulation_from_sde/model_01/"
filename_base = "results/simulation_from_sde/model_01/"

current_directory = os.getcwd()
final_directory = os.path.join(current_directory, directory)
if not os.path.exists(final_directory):
   os.makedirs(final_directory)


filename = filename_base + "model"
dic = {"nb_iterations_simu": n_MC, "P0": P0, "mu0": mu0, "theta":theta, "max_iter":max_iter_EM, "tol":tol_EM, "n_call_bayopt" : n_call_bayopt, "bounds_lambda": bounds_lambda,
       "arc_length_fct": arc_length_fct, "nb_basis":nb_basis, "sigma_init":sigma_init, 'n_splits_CV':n_splits_CV, "grid_bandwidth":grid_bandwidth, "N": N, "Gamma":Gamma, "mu_Z":mu_Z}
if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()



""" _______________________ Simulation 1: nu=0.001 _______________________ """

print('--------------------- Simulation n°1: nu=0.001 ---------------------')

time_init = time.time()

kernel = Matern(length_scale=0.1, nu=0.001)

with tqdm(total=n_MC) as pbar:
   res = Parallel(n_jobs=n_MC)(delayed(simu_from_sde)(theta, arc_length, N, Gamma, mu0, P0, nb_basis, sigma_init, noise_init_theta, grid_bandwidth, max_iter_EM, tol_EM, bounds_lambda, n_call_bayopt, n_splits_CV, kernel) for k in range(n_MC))
   pbar.update()

time_end = time.time()
duration = time_end - time_init

filename = filename_base + "simu_1"

dic = {"results":res, "kernel":kernel, "duration":duration}

if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()


print('--------------------- End Simulation n°1 ---------------------')




""" _______________________ Simulation 2: nu=0.0001 _______________________ """

print('--------------------- Simulation n°2: nu=0.0001 ---------------------')

time_init = time.time()

kernel = Matern(length_scale=0.1, nu=0.0001)

with tqdm(total=n_MC) as pbar:
   res = Parallel(n_jobs=n_MC)(delayed(simu_from_sde)(theta, arc_length, N, Gamma, mu0, P0, nb_basis, sigma_init, noise_init_theta, grid_bandwidth, max_iter_EM, tol_EM, bounds_lambda, n_call_bayopt, n_splits_CV, kernel) for k in range(n_MC))
   pbar.update()

time_end = time.time()
duration = time_end - time_init

filename = filename_base + "simu_2"

dic = {"results":res, "kernel":kernel, "duration":duration}

if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()


print('--------------------- End Simulation n°2 ---------------------')