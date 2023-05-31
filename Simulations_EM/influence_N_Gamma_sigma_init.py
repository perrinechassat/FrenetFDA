import sys
sys.path.insert(1, '../')
from FrenetFDA.utils.Frenet_Serret_utils import *
from FrenetFDA.utils.smoothing_utils import *
from FrenetFDA.processing_Euclidean_curve.unified_estimate_Frenet_state_space.EM_Frenet_state_space_CV import FrenetStateSpaceCV_global
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



# """ MODEL for simulations """


# def theta(s):
#     curv = lambda s : 2*np.cos(2*np.pi*s) + 5
#     tors = lambda s : 2*np.sin(2*np.pi*s) + 1
#     if isinstance(s, int) or isinstance(s, float):
#         return np.array([curv(s), tors(s)])
#     elif isinstance(s, np.ndarray):
#         return np.vstack((curv(s), tors(s))).T
#     else:
#         raise ValueError('Variable is not a float, a int or a NumPy array.')

# P0 = 0.01**2*np.eye(6)
# mu0 = np.eye(4) 
# arc_length_fct = lambda s: s
# n_MC = 80
# grid_lambda = np.logspace(-6, -2, 5)
# noise_init_theta = 1
# tol_EM = 0.1
# max_iter_EM = 100
# nb_basis = 15
# P0_init = 0.01**2*np.eye(6)
# sigma_init = 0.03
# n_splits_CV = 5
# grid_bandwidth = np.array([0.05, 0.1, 0.12, 0.15, 0.17, 0.2, 0.25])

# directory = r"results/influence_sample_size_noise/model_01/"
# filename_base = "results/influence_sample_size_noise/model_01/"

# current_directory = os.getcwd()
# final_directory = os.path.join(current_directory, directory)
# if not os.path.exists(final_directory):
#    os.makedirs(final_directory)


# filename = filename_base + "model"
# dic = {"nb_iterations_simu": n_MC, "P0": P0, "mu0": mu0, "theta":theta, "grid_lambda":grid_lambda, "max_iter":max_iter_EM, "tol":tol_EM,
#        "arc_length_fct": arc_length_fct, "nb_basis":nb_basis, "sigma_init":sigma_init, 'n_splits_CV':n_splits_CV, "grid_bandwidth":grid_bandwidth, "P0_init":P0_init}
# if os.path.isfile(filename):
#    print("Le fichier ", filename, " existe déjà.")
#    filename = filename + '_bis'
# fil = open(filename,"xb")
# pickle.dump(dic,fil)
# fil.close()



# """ _______________________ Simulation 1: N = 200, \gamma = 0.001 _______________________ """

# print('--------------------- Simulation n°1: N = 200, \gamma = 0.001 ---------------------')

# time_init = time.time()

# N = 200
# gamma = 0.001

# grid_time = np.linspace(0,1,N)
# arc_length = grid_time 
# mu_Z = solve_FrenetSerret_ODE_SE(theta, arc_length, mu0)
# Gamma = gamma**2*np.eye(3)


# with tqdm(total=n_MC) as pbar:
#    res = Parallel(n_jobs=n_MC)(delayed(influence_N_Gamma)(theta, arc_length, N, Gamma, mu0, P0, nb_basis, sigma_init, P0_init, noise_init_theta, grid_bandwidth, max_iter_EM, tol_EM, grid_lambda, n_splits_CV) for k in range(n_MC))
#    pbar.update()

# time_end = time.time()
# duration = time_end - time_init

# filename = filename_base + "simu_1"

# dic = {"results":res, "mu_Z": mu_Z, "N":N, "gamma":gamma, "duration":duration}

# if os.path.isfile(filename):
#    print("Le fichier ", filename, " existe déjà.")
#    filename = filename + '_bis'
# fil = open(filename,"xb")
# pickle.dump(dic,fil)
# fil.close()


# print('--------------------- End Simulation n°1 ---------------------')




# """ _______________________ Simulation 2: N = 200, \gamma = 0.005 _______________________ """

# print('--------------------- Simulation n°2: N = 200, \gamma = 0.005 ---------------------')

# time_init = time.time()

# N = 200
# gamma = 0.005

# grid_time = np.linspace(0,1,N)
# arc_length = grid_time 
# mu_Z = solve_FrenetSerret_ODE_SE(theta, arc_length, mu0)
# Gamma = gamma**2*np.eye(3)


# with tqdm(total=n_MC) as pbar:
#    res = Parallel(n_jobs=n_MC)(delayed(influence_N_Gamma)(theta, arc_length, N, Gamma, mu0, P0, nb_basis, sigma_init, P0_init, noise_init_theta, grid_bandwidth, max_iter_EM, tol_EM, grid_lambda, n_splits_CV) for k in range(n_MC))
#    pbar.update()

# time_end = time.time()
# duration = time_end - time_init

# filename = filename_base + "simu_2"

# dic = {"results":res, "mu_Z": mu_Z, "N":N, "gamma":gamma, "duration":duration}

# if os.path.isfile(filename):
#    print("Le fichier ", filename, " existe déjà.")
#    filename = filename + '_bis'
# fil = open(filename,"xb")
# pickle.dump(dic,fil)
# fil.close()

# print('--------------------- End Simulation n°2 ---------------------')




# """ _______________________ Simulation 3: N = 100, \gamma = 0.001 _______________________ """

# print('--------------------- Simulation n°3: N = 100, \gamma = 0.001 ---------------------')

# time_init = time.time()

# N = 100
# gamma = 0.001

# grid_time = np.linspace(0,1,N)
# arc_length = grid_time 
# mu_Z = solve_FrenetSerret_ODE_SE(theta, arc_length, mu0)
# Gamma = gamma**2*np.eye(3)


# with tqdm(total=n_MC) as pbar:
#    res = Parallel(n_jobs=n_MC)(delayed(influence_N_Gamma)(theta, arc_length, N, Gamma, mu0, P0, nb_basis, sigma_init, P0_init, noise_init_theta, grid_bandwidth, max_iter_EM, tol_EM, grid_lambda, n_splits_CV) for k in range(n_MC))
#    pbar.update()

# time_end = time.time()
# duration = time_end - time_init

# filename = filename_base + "simu_3"

# dic = {"results":res, "mu_Z": mu_Z, "N":N, "gamma":gamma, "duration":duration}

# if os.path.isfile(filename):
#    print("Le fichier ", filename, " existe déjà.")
#    filename = filename + '_bis'
# fil = open(filename,"xb")
# pickle.dump(dic,fil)
# fil.close()


# print('--------------------- End Simulation n°3 ---------------------')




# """ _______________________ Simulation 4: N = 100, \gamma = 0.005 _______________________ """

# print('--------------------- Simulation n°4: N = 100, \gamma = 0.005 ---------------------')

# time_init = time.time()

# N = 100
# gamma = 0.005

# grid_time = np.linspace(0,1,N)
# arc_length = grid_time 
# mu_Z = solve_FrenetSerret_ODE_SE(theta, arc_length, mu0)
# Gamma = gamma**2*np.eye(3)


# with tqdm(total=n_MC) as pbar:
#    res = Parallel(n_jobs=n_MC)(delayed(influence_N_Gamma)(theta, arc_length, N, Gamma, mu0, P0, nb_basis, sigma_init, P0_init, noise_init_theta, grid_bandwidth, max_iter_EM, tol_EM, grid_lambda, n_splits_CV) for k in range(n_MC))
#    pbar.update()

# time_end = time.time()
# duration = time_end - time_init

# filename = filename_base + "simu_4"

# dic = {"results":res, "mu_Z": mu_Z, "N":N, "gamma":gamma, "duration":duration}

# if os.path.isfile(filename):
#    print("Le fichier ", filename, " existe déjà.")
#    filename = filename + '_bis'
# fil = open(filename,"xb")
# pickle.dump(dic,fil)
# fil.close()

# print('--------------------- End Simulation n°4 ---------------------')











def influence_sigma_init(tab_sigma_init, theta, arc_length, N, Gamma, mu0, P0, nb_basis, P0_init, noise_init_theta, grid_bandwidth, max_iter_EM, tol_EM, grid_lambda, n_splits_CV):

    Z, Y, init_coefs, grid_arc_length, init_Gamma, init_mu0 = init_from_true_param(theta, arc_length, N, Gamma, mu0, P0, nb_basis, noise_init_theta, grid_bandwidth)

    tab_FS_statespace = np.empty((len(tab_sigma_init)))
    tab_score_lambda_matrix = np.empty((len(tab_sigma_init)))
    for k in range(len(tab_sigma_init)):
        tab_FS_statespace[k], tab_score_lambda_matrix[k] = simulation_step(Y, grid_arc_length, init_coefs, init_Gamma, init_mu0, P0_init, tab_sigma_init[k], nb_basis, max_iter_EM, tol_EM, grid_lambda, n_splits_CV)  

    return tab_FS_statespace, Z, Y, tab_score_lambda_matrix


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
grid_lambda = np.logspace(-6, -2, 5)
noise_init_theta = 1
tol_EM = 0.1
max_iter_EM = 100
nb_basis = 15
P0_init = 0.01**2*np.eye(6)
sigma_init = 0.03
n_splits_CV = 5
grid_bandwidth = np.array([0.05, 0.1, 0.12, 0.15, 0.17, 0.2, 0.25])
N = 200
gamma = 0.001 
Gamma = gamma**2*np.eye(3)
grid_time = np.linspace(0,1,N)
arc_length = grid_time 
mu_Z = solve_FrenetSerret_ODE_SE(theta, arc_length, mu0)

directory = r"results/influence_sigma_init/model_01/"
filename_base = "results/influence_sigma_init/model_01/"

current_directory = os.getcwd()
final_directory = os.path.join(current_directory, directory)
if not os.path.exists(final_directory):
   os.makedirs(final_directory)


filename = filename_base + "model"
dic = {"nb_iterations_simu": n_MC, "P0": P0, "mu0": mu0, "theta":theta, "grid_lambda":grid_lambda, "max_iter":max_iter_EM, "tol":tol_EM,
       "arc_length_fct": arc_length_fct, "nb_basis":nb_basis, "sigma_init":sigma_init, 'n_splits_CV':n_splits_CV, "grid_bandwidth":grid_bandwidth, "P0_init":P0_init, "Gamma":Gamma, "N":N}
if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()


""" _______________________ Simulation compare init sigma  _______________________ """

print('--------------------- Simulation compare init sigma ---------------------')

time_init = time.time()

tab_sigma_init = np.array([0.01,0.03,0.1])


with tqdm(total=n_MC) as pbar:
   res = Parallel(n_jobs=n_MC)(delayed(influence_sigma_init)(tab_sigma_init, theta, arc_length, N, Gamma, mu0, P0, nb_basis, P0_init, noise_init_theta, grid_bandwidth, max_iter_EM, tol_EM, grid_lambda, n_splits_CV) for k in range(n_MC))
   pbar.update()

time_end = time.time()
duration = time_end - time_init

filename = filename_base + "simu_comp_sigma_init"

dic = {"results":res, "mu_Z": mu_Z, "duration":duration, "tab_sigma_init":tab_sigma_init}

if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()


print('--------------------- End compare init sigma ---------------------')

