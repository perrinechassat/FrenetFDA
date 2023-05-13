import sys
sys.path.insert(1, '../')
from FrenetFDA.utils.Frenet_Serret_utils import *
from FrenetFDA.utils.smoothing_utils import *
from FrenetFDA.processing_Euclidean_curve.unified_estimate_Frenet_state_space.EM_Frenet_state_space import FrenetStateSpace, MLE
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


def simu_test_criterion(sigma, theta, mu0, P0, Gamma, N, arc_length, grid_lambda, noise_init_theta, tol_EM, max_iter_EM, nb_basis):
    
    Sigma = lambda s: sigma**2*np.array([[1 + 0*s, 0*s],[0*s, 1 + 0*s]])
    xi0 = np.random.multivariate_normal(np.zeros(6), P0)
    Z0 = mu0 @ SE3.exp(-xi0)
    mat_L = np.zeros((6,2))
    mat_L[0,1], mat_L[2,0] = 1, 1
    Z = solve_FrenetSerret_SDE_SE3(theta, Sigma, mat_L, arc_length, Z0=Z0)
    Q = Z[:,:3,:3]
    X = Z[:,:3,3]
    Y = X + np.random.multivariate_normal(np.zeros(3), Gamma, size=(len(X)))

    theta_noisy_val = theta(arc_length) + np.random.multivariate_normal(np.zeros(2), noise_init_theta**2*np.eye(2), size=len(arc_length))
    BasisThetaNoisy = VectorBSplineSmoothing(2, 15)
    BasisThetaNoisy.fit(arc_length, theta_noisy_val, regularization_parameter=0.0000001)
    theta_init = BasisThetaNoisy.evaluate
    
    sigma_init = 0.03
    Sigma_hat = lambda s: sigma_init**2*np.array([[1 + 0*s, 0*s],[0*s, 1 + 0*s]])
    P0_hat = sigma_init**2*np.eye(6)

    # EM criterion GCV
    try:
        ####     Run the EM    ####
        FS_statespace_GCV = FrenetStateSpace(arc_length, Y)
        FS_statespace_GCV.expectation_maximization(tol_EM, max_iter_EM, nb_basis=nb_basis, regularization_parameter_list=grid_lambda, 
                                                   init_params = {"Gamma":Gamma, "coefs":BasisThetaNoisy.coefs, "mu0":mu0, "Sigma":Sigma_hat, "P0":P0_hat}, 
                                                   method='autre', model_Sigma='scalar', score_lambda='GCV')
    except:
        FS_statespace_GCV = [theta_init]

    # EM criterion V2
    try:
        ####     Run the EM    ####
        FS_statespace_V = FrenetStateSpace(arc_length, Y)
        FS_statespace_V.expectation_maximization(tol_EM, max_iter_EM, nb_basis=nb_basis, regularization_parameter_list=grid_lambda, 
                                                   init_params = {"Gamma":Gamma, "coefs":BasisThetaNoisy.coefs, "mu0":mu0, "Sigma":Sigma_hat, "P0":P0_hat}, 
                                                   method='autre', model_Sigma='scalar', score_lambda='V2')
    except:
        FS_statespace_V = [theta_init]

    # EM criterion U
    try:
        ####     Run the EM    ####
        FS_statespace_U = FrenetStateSpace(arc_length, Y)
        FS_statespace_U.expectation_maximization(tol_EM, max_iter_EM, nb_basis=nb_basis, regularization_parameter_list=grid_lambda, 
                                                   init_params = {"Gamma":Gamma, "coefs":BasisThetaNoisy.coefs, "mu0":mu0, "Sigma":Sigma_hat, "P0":P0_hat}, 
                                                   method='autre', model_Sigma='scalar', score_lambda='U2')
    except:
        FS_statespace_U = [theta_init]

    # EM criterion MSE_YX
    try:
        ####     Run the EM    ####
        FS_statespace_MSE_YX = FrenetStateSpace(arc_length, Y)
        FS_statespace_MSE_YX.expectation_maximization(tol_EM, max_iter_EM, nb_basis=nb_basis, regularization_parameter_list=grid_lambda, 
                                                   init_params = {"Gamma":Gamma, "coefs":BasisThetaNoisy.coefs, "mu0":mu0, "Sigma":Sigma_hat, "P0":P0_hat}, 
                                                   method='autre', model_Sigma='scalar', score_lambda='MSE_YX')
    except:
        FS_statespace_MSE_YX = [theta_init]

    # EM criterion MSE_YmuX
    try:
        ####     Run the EM    ####
        FS_statespace_MSE_YmuX = FrenetStateSpace(arc_length, Y)
        FS_statespace_MSE_YmuX.expectation_maximization(tol_EM, max_iter_EM, nb_basis=nb_basis, regularization_parameter_list=grid_lambda, 
                                                   init_params = {"Gamma":Gamma, "coefs":BasisThetaNoisy.coefs, "mu0":mu0, "Sigma":Sigma_hat, "P0":P0_hat}, 
                                                   method='autre', model_Sigma='scalar', score_lambda='MSE_YmuX')
    except:
        FS_statespace_MSE_YmuX = [theta_init]

    # EM true MSE
    try:
        ####     Run the EM    ####
        FS_statespace_true_MSE = FrenetStateSpace(arc_length, Y)
        FS_statespace_true_MSE.expectation_maximization(tol_EM, max_iter_EM, nb_basis=nb_basis, regularization_parameter_list=grid_lambda, 
                                                   init_params = {"Gamma":Gamma, "coefs":BasisThetaNoisy.coefs, "mu0":mu0, "Sigma":Sigma_hat, "P0":P0_hat}, 
                                                   method='autre', model_Sigma='scalar', score_lambda='true_MSE', true_theta=theta)
    except:
        FS_statespace_true_MSE = [theta_init]

    return FS_statespace_GCV, FS_statespace_V, FS_statespace_U, FS_statespace_MSE_YX, FS_statespace_MSE_YmuX, FS_statespace_true_MSE
    


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

N = 200
grid_time = np.linspace(0,1,N)
arc_length = grid_time 
mu0 = np.eye(4)
mu_Z = solve_FrenetSerret_ODE_SE(theta, arc_length, mu0)
mu_Q, mu_X = mu_Z[:,:3,:3], mu_Z[:,:3,3]

P0 = 0.01**2*np.eye(6)

gamma = 0.001
Gamma = gamma**2*np.eye(3)

n_MC = 100

directory = r"results/compare_lambda_criterion/model_01/"
filename_base = "results/compare_lambda_criterion/model_01/"

current_directory = os.getcwd()
final_directory = os.path.join(current_directory, directory)
if not os.path.exists(final_directory):
   os.makedirs(final_directory)

grid_lambda = np.logspace(-15, -3, 20)
noise_init_theta = 1
tol_EM = 0.01
max_iter_EM = 300
nb_basis = 15


filename = filename_base + "model"
dic = {"nb_iterations_simu": n_MC, "P0": P0, "mu0": mu0, "theta":theta, "Gamma":Gamma, "grid_lambda":grid_lambda, "max_iter":max_iter_EM, "tol":tol_EM, "N":N, 
       "arc_length": arc_length, "nb_basis":nb_basis}
if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()


""" Simulation 1 : sigma = 0 """

print('--------------------- Simulation n°1: sigma = 0 ---------------------')

time_init = time.time()


sigma = 0

with tqdm(total=n_MC) as pbar:
   res = Parallel(n_jobs=50)(delayed(simu_test_criterion)(sigma, theta, mu0, P0, Gamma, N, arc_length, grid_lambda, noise_init_theta, tol_EM, max_iter_EM, nb_basis) for k in range(n_MC))
   pbar.update()

time_end = time.time()
duration = time_end - time_init

filename = filename_base + "simu_1"

dic = {"results":res, "sigma": sigma}

if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()

print('End of simulation n°1: time spent', duration, 'seconds. \n')




""" Simulation 2 : sigma = 0.001 """

print('--------------------- Simulation n°2: sigma = 0.001 ---------------------')

time_init = time.time()


sigma = 0.001

with tqdm(total=n_MC) as pbar:
   res = Parallel(n_jobs=50)(delayed(simu_test_criterion)(sigma, theta, mu0, P0, Gamma, N, arc_length, grid_lambda, noise_init_theta, tol_EM, max_iter_EM, nb_basis) for k in range(n_MC))
   pbar.update()

time_end = time.time()
duration = time_end - time_init

filename = filename_base + "simu_2"

dic = {"results":res, "sigma": sigma}

if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()

print('End of simulation n°2: time spent', duration, 'seconds. \n')



""" Simulation 3 : sigma = 0.01 """

print('--------------------- Simulation n°3: sigma = 0.01 ---------------------')

time_init = time.time()


sigma = 0.01

with tqdm(total=n_MC) as pbar:
   res = Parallel(n_jobs=50)(delayed(simu_test_criterion)(sigma, theta, mu0, P0, Gamma, N, arc_length, grid_lambda, noise_init_theta, tol_EM, max_iter_EM, nb_basis) for k in range(n_MC))
   pbar.update()

time_end = time.time()
duration = time_end - time_init

filename = filename_base + "simu_3"

dic = {"results":res, "sigma": sigma}

if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()

print('End of simulation n°3: time spent', duration, 'seconds. \n')



""" Simulation 4 : sigma = 0.1 """

print('--------------------- Simulation n°4: sigma = 0.1 ---------------------')

time_init = time.time()


sigma = 0.1

with tqdm(total=n_MC) as pbar:
   res = Parallel(n_jobs=50)(delayed(simu_test_criterion)(sigma, theta, mu0, P0, Gamma, N, arc_length, grid_lambda, noise_init_theta, tol_EM, max_iter_EM, nb_basis) for k in range(n_MC))
   pbar.update()

time_end = time.time()
duration = time_end - time_init

filename = filename_base + "simu_4"

dic = {"results":res, "sigma": sigma}

if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()

print('End of simulation n°4: time spent', duration, 'seconds. \n')