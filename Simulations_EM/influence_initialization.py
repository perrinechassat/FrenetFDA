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


def EM_init_extrins(theta, arc_length_fct, N, Gamma, mu0, P0, nb_basis, sigma_init, grid_bandwidth, max_iter_EM, tol_EM, bounds_lambda_EM, n_call_bayopt_EM, n_splits_CV_EM, bounds_h_init, bounds_lbda_init, n_call_bayopt_init):

    init_Sigma = lambda s: sigma_init**2*np.array([[1 + 0*s, 0*s],[0*s, 1 + 0*s]]) 
    P0_init = sigma_init**2*np.eye(6)
    grid_time = np.linspace(0,1,N)
    arc_length = arc_length_fct(grid_time, np.random.normal(0,1.5))
    mu_Z = solve_FrenetSerret_ODE_SE(theta, arc_length, mu0)

    Y, Z, Z_hat, init_coefs, grid_arc_length, init_Gamma, init_mu0 = init_extrins(theta, arc_length, N, Gamma, mu0, P0, nb_basis, grid_bandwidth, bounds_h_init, bounds_lbda_init, n_call_bayopt_init)
    FS_statespace, res_bayopt = bayesian_CV_optimization_regularization_parameter(n_CV=n_splits_CV_EM, n_call_bayopt=n_call_bayopt_EM, lambda_bounds=bounds_lambda_EM, grid_obs=grid_arc_length, Y_obs=Y, tol=tol_EM, max_iter=max_iter_EM, 
                                                                                  nb_basis=nb_basis, init_params={"Gamma":init_Gamma, "coefs":init_coefs, "mu0":init_mu0, "Sigma":init_Sigma, "P0":P0_init})
    
    return FS_statespace, Z, Y, res_bayopt, Z_hat, mu_Z



def EM_init_GS_LS(theta, arc_length_fct, N, Gamma, mu0, P0, nb_basis, sigma_init, grid_bandwidth, max_iter_EM, tol_EM, bounds_lambda_EM, n_call_bayopt_EM, n_splits_CV_EM, bounds_h_init, bounds_lbda_init, n_call_bayopt_init):

    init_Sigma = lambda s: sigma_init**2*np.array([[1 + 0*s, 0*s],[0*s, 1 + 0*s]]) 
    P0_init = sigma_init**2*np.eye(6)
    grid_time = np.linspace(0,1,N)
    arc_length = arc_length_fct(grid_time, np.random.normal(0,1.5))
    mu_Z = solve_FrenetSerret_ODE_SE(theta, arc_length, mu0)

    Y, Z, Z_hat, init_coefs, grid_arc_length, init_Gamma, init_mu0 = init_GS_LeastSquare(theta, arc_length, N, Gamma, mu0, P0, nb_basis, grid_bandwidth, bounds_h_init, bounds_lbda_init, n_call_bayopt_init)
    FS_statespace, res_bayopt = bayesian_CV_optimization_regularization_parameter(n_CV=n_splits_CV_EM, n_call_bayopt=n_call_bayopt_EM, lambda_bounds=bounds_lambda_EM, grid_obs=grid_arc_length, Y_obs=Y, tol=tol_EM, max_iter=max_iter_EM, 
                                                                                  nb_basis=nb_basis, init_params={"Gamma":init_Gamma, "coefs":init_coefs, "mu0":init_mu0, "Sigma":init_Sigma, "P0":P0_init})
    
    return FS_statespace, Z, Y, res_bayopt, Z_hat, mu_Z



def EM_init_CLP_LS(theta, arc_length_fct, N, Gamma, mu0, P0, nb_basis, sigma_init, grid_bandwidth, max_iter_EM, tol_EM, bounds_lambda_EM, n_call_bayopt_EM, n_splits_CV_EM, bounds_h_init, bounds_lbda_init, n_call_bayopt_init):

    init_Sigma = lambda s: sigma_init**2*np.array([[1 + 0*s, 0*s],[0*s, 1 + 0*s]]) 
    P0_init = sigma_init**2*np.eye(6)
    grid_time = np.linspace(0,1,N)
    arc_length = arc_length_fct(grid_time, np.random.normal(0,1.5))
    mu_Z = solve_FrenetSerret_ODE_SE(theta, arc_length, mu0)

    Y, Z, Z_hat, init_coefs, grid_arc_length, init_Gamma, init_mu0 = init_CLP_LeastSquare(theta, arc_length, N, Gamma, mu0, P0, nb_basis, grid_bandwidth, bounds_h_init, bounds_lbda_init, n_call_bayopt_init)
    FS_statespace, res_bayopt = bayesian_CV_optimization_regularization_parameter(n_CV=n_splits_CV_EM, n_call_bayopt=n_call_bayopt_EM, lambda_bounds=bounds_lambda_EM, grid_obs=grid_arc_length, Y_obs=Y, tol=tol_EM, max_iter=max_iter_EM, 
                                                                                  nb_basis=nb_basis, init_params={"Gamma":init_Gamma, "coefs":init_coefs, "mu0":init_mu0, "Sigma":init_Sigma, "P0":P0_init})
    
    return FS_statespace, Z, Y, res_bayopt, Z_hat, mu_Z



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
    
def arc_length_fct(s, a):
    if abs(a) < 1e-04:
        return s
    else:
        return (np.exp(a*s) - 1)/(np.exp(a) - 1)

P0 = 0.01**2*np.eye(6)
mu0 = np.eye(4) 
N = 100
gamma = 0.001
Gamma = gamma**2*np.eye(3)

n_MC = 80
tol_EM = 0.1
max_iter_EM = 200
nb_basis = 15
n_splits_CV_EM = 5
grid_bandwidth = np.array([0.05, 0.1, 0.12, 0.15, 0.17, 0.2, 0.25])
n_call_bayopt_EM = 20
bounds_lambda_EM = ((1e-09, 1e-05), (1e-09, 1e-05))
sigma_init = 0.05

bounds_lambda_init = np.array([[1e-09, 1e-03], [1e-09, 1e-03]])
bounds_h_init = np.array([0.05, 0.25])
n_call_bayopt_init = 100

directory = r"results/influence_initialization/model_01/"
filename_base = "results/influence_initialization/model_01/"

current_directory = os.getcwd()
final_directory = os.path.join(current_directory, directory)
if not os.path.exists(final_directory):
   os.makedirs(final_directory)


filename = filename_base + "model"
dic = {"nb_iterations_simu": n_MC, "P0": P0, "mu0": mu0, "theta":theta, "max_iter":max_iter_EM, "tol":tol_EM, "n_call_bayopt_EM" : n_call_bayopt_EM, "bounds_lambda_EM": bounds_lambda_EM,
       "arc_length_fct": arc_length_fct, "nb_basis":nb_basis, "sigma_init":sigma_init, 'n_splits_CV_EM':n_splits_CV_EM, "grid_bandwidth":grid_bandwidth, "bounds_lambda_init": bounds_lambda_init, 
       "bounds_h_init": bounds_h_init, "n_call_bayopt_init": n_call_bayopt_init, "N":N, "Gamma": Gamma}
if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()



""" _______________________ Simulation 1: LP + Extrins _______________________ """

print('--------------------- Simulation n°1: LP + Extrins ---------------------')

time_init = time.time()

with tqdm(total=n_MC) as pbar:
   res = Parallel(n_jobs=n_MC)(delayed(EM_init_extrins)(theta, arc_length_fct, N, Gamma, mu0, P0, nb_basis, sigma_init, grid_bandwidth, max_iter_EM, tol_EM, bounds_lambda_EM, n_call_bayopt_EM, n_splits_CV_EM, bounds_h_init, bounds_lambda_init, n_call_bayopt_init) for k in range(n_MC))
   pbar.update()

time_end = time.time()
duration = time_end - time_init

filename = filename_base + "simu_1"

dic = {"results":res, "duration":duration}

if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()


print('--------------------- End Simulation n°1 ---------------------')




""" _______________________ Simulation 2: LP + GS + LS _______________________ """

print('--------------------- Simulation n°2: LP + GS + LS ---------------------')

time_init = time.time()


with tqdm(total=n_MC) as pbar:
   res = Parallel(n_jobs=n_MC)(delayed(EM_init_GS_LS)(theta, arc_length_fct, N, Gamma, mu0, P0, nb_basis, sigma_init, grid_bandwidth, max_iter_EM, tol_EM, bounds_lambda_EM, n_call_bayopt_EM, n_splits_CV_EM, bounds_h_init, bounds_lambda_init, n_call_bayopt_init) for k in range(n_MC))
   pbar.update()

time_end = time.time()
duration = time_end - time_init

filename = filename_base + "simu_2"

dic = {"results":res, "duration":duration}

if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()

print('--------------------- End Simulation n°2 ---------------------')





""" _______________________ Simulation 3: CLP + LS _______________________ """

print('--------------------- Simulation n°3: CLP + LS ---------------------')

time_init = time.time()


with tqdm(total=n_MC) as pbar:
   res = Parallel(n_jobs=n_MC)(delayed(EM_init_CLP_LS)(theta, arc_length_fct, N, Gamma, mu0, P0, nb_basis, sigma_init, grid_bandwidth, max_iter_EM, tol_EM, bounds_lambda_EM, n_call_bayopt_EM, n_splits_CV_EM, bounds_h_init, bounds_lambda_init, n_call_bayopt_init) for k in range(n_MC))
   pbar.update()

time_end = time.time()
duration = time_end - time_init

filename = filename_base + "simu_3"

dic = {"results":res, "duration":duration}

if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()

print('--------------------- End Simulation n°3 ---------------------')
