import sys
sys.path.insert(1, '../')
from FrenetFDA.utils.Frenet_Serret_utils import *
from FrenetFDA.utils.smoothing_utils import *
from FrenetFDA.processing_Euclidean_curve.unified_estimate_Frenet_state_space.EM_Frenet_state_space import FrenetStateSpace, MLE
from FrenetFDA.processing_Euclidean_curve.unified_estimate_Frenet_state_space.EM_Frenet_state_space_CV import FrenetStateSpaceCV_byit, FrenetStateSpaceCV_global
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




def simu_opti_lambda_byit(theta, mu0, P0, Gamma, N, arc_length, sigma_init, noise_init_theta, tol_EM, max_iter_EM, nb_basis, grid_lambda, score_type, n_splits_CV):

    xi0 = np.random.multivariate_normal(np.zeros(6), P0)
    Z0 = mu0 @ SE3.exp(-xi0)
    Z = solve_FrenetSerret_ODE_SE(theta, arc_length, Z0=Z0)
    X = Z[:,:3,3]
    Y = X + np.random.multivariate_normal(np.zeros(3), Gamma, size=N)

    theta_noisy_val = theta(arc_length) + np.random.multivariate_normal(np.zeros(2), noise_init_theta**2*np.eye(2), size=len(arc_length))
    BasisThetaNoisy = VectorBSplineSmoothing(2, 15)
    BasisThetaNoisy.fit(arc_length, theta_noisy_val, regularization_parameter=0.0000001)

    Sigma_hat = lambda s: sigma_init**2*np.array([[1 + 0*s, 0*s],[0*s, 1 + 0*s]])
    P0_hat = sigma_init**2*np.eye(6)

    try:
        ###     Run the EM    ####
        FS_statespace = FrenetStateSpaceCV_byit(arc_length, Y)
        FS_statespace.expectation_maximization(tol_EM, max_iter_EM, nb_basis=nb_basis, regularization_parameter_list=grid_lambda, 
                                                init_params = {"Gamma":Gamma, "coefs":BasisThetaNoisy.coefs, "mu0":mu0, "Sigma":Sigma_hat, "P0":P0_hat}, 
                                                method='autre', model_Sigma='scalar', score_lambda=score_type, n_splits_CV=n_splits_CV)
    except:
        FS_statespace = [BasisThetaNoisy.coefs]

    return FS_statespace, Z, Y





def simu_opti_lambda_globally(theta, mu0, P0, Gamma, N, arc_length, sigma_init, noise_init_theta, tol_EM, max_iter_EM, nb_basis, grid_lambda, score_type, n_splits_CV):
    
    xi0 = np.random.multivariate_normal(np.zeros(6), P0)
    Z0 = mu0 @ SE3.exp(-xi0)
    Z = solve_FrenetSerret_ODE_SE(theta, arc_length, Z0=Z0)
    X = Z[:,:3,3]
    Y = X + np.random.multivariate_normal(np.zeros(3), Gamma, size=N)

    theta_noisy_val = theta(arc_length) + np.random.multivariate_normal(np.zeros(2), noise_init_theta**2*np.eye(2), size=len(arc_length))
    BasisThetaNoisy = VectorBSplineSmoothing(2, 15)
    BasisThetaNoisy.fit(arc_length, theta_noisy_val, regularization_parameter=0.0000001)

    Sigma_hat = lambda s: sigma_init**2*np.array([[1 + 0*s, 0*s],[0*s, 1 + 0*s]])
    P0_hat = sigma_init**2*np.eye(6)
  
    K = len(grid_lambda)
    score_lambda_matrix = np.zeros((K,K))

    kf = KFold(n_splits=n_splits_CV, shuffle=True)
    ind_CV = 1
    for train_index, test_index in kf.split(arc_length[1:]):
        print('     --> Start CV step n°', ind_CV)
        Y_train = Y[train_index]
        Y_test = Y[test_index]
        grid_train = np.concatenate((np.array([arc_length[0]]), arc_length[1:][train_index]))
        grid_test = np.concatenate((np.array([arc_length[0]]), arc_length[1:][test_index]))

        for i in range(K):
            for j in range(K):
                lbda = np.array([grid_lambda[i], grid_lambda[j]])

                try:
                    FS_statespace = FrenetStateSpaceCV_global(grid_train, Y_train, bornes_theta=np.array([0,1]))
                    FS_statespace.expectation_maximization(tol_EM, max_iter_EM, nb_basis=nb_basis, regularization_parameter=lbda, 
                                                        init_params = {"Gamma":Gamma, "coefs":BasisThetaNoisy.coefs, "mu0":mu0, "Sigma":Sigma_hat, "P0":P0_hat}, 
                                                        method='autre', model_Sigma='scalar')

                    if score_type=='MSE_YX':
                        Z_reconst = solve_FrenetSerret_SDE_SE3(FS_statespace.theta, FS_statespace.Sigma, FS_statespace.L, grid_test, Z0=FS_statespace.mu0)
                        X_reconst_test = Z_reconst[1:,:3,3]
                        score_lambda_matrix[i,j] = score_lambda_matrix[i,j] + np.linalg.norm(X_reconst_test - Y_test)**2

                    elif score_type=='MSE_YmuX':
                        Z_reconst = solve_FrenetSerret_ODE_SE(FS_statespace.theta, grid_test, Z0=FS_statespace.mu0)
                        X_reconst_test = Z_reconst[1:,:3,3]
                        score_lambda_matrix[i,j] = score_lambda_matrix[i,j] + np.linalg.norm(X_reconst_test - Y_test)**2

                    else: 
                        raise Exception("Invalid term for optimization of lamnda score.")
                except:
                    print('Error for lbda:', lbda,' ind:', i, j)
        ind_CV += 1 

    score_lambda_matrix = score_lambda_matrix/n_splits_CV
    ind = np.squeeze(np.array(np.where(score_lambda_matrix==np.min(score_lambda_matrix))))
    lbda_opt = np.array([grid_lambda[ind[0]], grid_lambda[ind[1]]]) 
    print('Optimal chosen lambda:', lbda_opt)
    try:
        FS_statespace = FrenetStateSpaceCV_global(arc_length, Y[1:], bornes_theta=np.array([0,1]))
        FS_statespace.expectation_maximization(tol_EM, max_iter_EM, nb_basis=nb_basis, regularization_parameter=lbda_opt, 
                                            init_params = {"Gamma":Gamma, "coefs":BasisThetaNoisy.coefs, "mu0":mu0, "Sigma":Sigma_hat, "P0":P0_hat}, 
                                            method='autre', model_Sigma='scalar')
    except:
        FS_statespace = [BasisThetaNoisy.coefs]
    
    return FS_statespace, Z, Y, score_lambda_matrix 
    
    


""" MODEL for simulations """


# def theta(s):
#     curv = lambda s : 2*np.cos(2*np.pi*s) + 5
#     tors = lambda s : 2*np.sin(2*np.pi*s) + 1
#     if isinstance(s, int) or isinstance(s, float):
#         return np.array([curv(s), tors(s)])
#     elif isinstance(s, np.ndarray):
#         return np.vstack((curv(s), tors(s))).T
#     else:
#         raise ValueError('Variable is not a float, a int or a NumPy array.')

# N = 200
# grid_time = np.linspace(0,1,N)
# arc_length = grid_time 
# mu0 = np.eye(4)
# mu_Z = solve_FrenetSerret_ODE_SE(theta, arc_length, mu0)
# mu_Q, mu_X = mu_Z[:,:3,:3], mu_Z[:,:3,3]

# P0 = 0.01**2*np.eye(6)

# gamma = 0.001
# Gamma = gamma**2*np.eye(3)

# n_MC = 80

# directory = r"results/simulation_optimization_lambda/model_01/"
# filename_base = "results/simulation_optimization_lambda/model_01/"

# current_directory = os.getcwd()
# final_directory = os.path.join(current_directory, directory)
# if not os.path.exists(final_directory):
#    os.makedirs(final_directory)

# grid_lambda = np.logspace(-6, -2, 5)
# noise_init_theta = 1
# tol_EM = 0.01
# max_iter_EM = 100
# nb_basis = 15
# sigma_init = 0.03
# n_splits_CV = 5

# filename = filename_base + "model"
# dic = {"nb_iterations_simu": n_MC, "P0": P0, "mu0": mu0, "theta":theta, "Gamma":Gamma, "grid_lambda":grid_lambda, "max_iter":max_iter_EM, "tol":tol_EM, "N":N, 
#        "arc_length": arc_length, "nb_basis":nb_basis, "sigma_init":sigma_init, 'n_splits_CV':n_splits_CV}
# if os.path.isfile(filename):
#    print("Le fichier ", filename, " existe déjà.")
#    filename = filename + '_bis'
# fil = open(filename,"xb")
# pickle.dump(dic,fil)
# fil.close()


# """ Simulation 1 : CV by iteration MSE_YX """

# print('--------------------- Simulation n°1: Optimization of lambda by iteration with MSE_YX ---------------------')

# time_init = time.time()

# score_type = 'MSE_YX'

# with tqdm(total=n_MC) as pbar:
#    res = Parallel(n_jobs=n_MC)(delayed(simu_opti_lambda_byit)(theta, mu0, P0, Gamma, N, arc_length, sigma_init, noise_init_theta, tol_EM, max_iter_EM, nb_basis, grid_lambda, score_type, n_splits_CV) for k in range(n_MC))
#    pbar.update()

# time_end = time.time()
# duration = time_end - time_init

# filename = filename_base + "simu_1_byit_MSE_YX"

# dic = {"results":res, "score_type": score_type}

# if os.path.isfile(filename):
#    print("Le fichier ", filename, " existe déjà.")
#    filename = filename + '_bis'
# fil = open(filename,"xb")
# pickle.dump(dic,fil)
# fil.close()


# print('--------------------- End Simulation n°1 ---------------------')







# """ Simulation 2 : CV globally MSE_YX """

# print('--------------------- Simulation n°2: Optimization of lambda globally with MSE_YX ---------------------')

# time_init = time.time()

# score_type = 'MSE_YX'

# with tqdm(total=n_MC) as pbar:
#    res = Parallel(n_jobs=n_MC)(delayed(simu_opti_lambda_globally)(theta, mu0, P0, Gamma, N, arc_length, sigma_init, noise_init_theta, tol_EM, max_iter_EM, nb_basis, grid_lambda, score_type, n_splits_CV) for k in range(n_MC))
#    pbar.update()

# time_end = time.time()
# duration = time_end - time_init

# filename = filename_base + "simu_2_global_MSE_YX"

# dic = {"results":res, "score_type": score_type}

# if os.path.isfile(filename):
#    print("Le fichier ", filename, " existe déjà.")
#    filename = filename + '_bis'
# fil = open(filename,"xb")
# pickle.dump(dic,fil)
# fil.close()


# print('--------------------- End Simulation n°2 ---------------------')








# """ Simulation 3 : CV by iteration MSE_YmuX """

# print('--------------------- Simulation n°3: Optimization of lambda by iteration with MSE_YmuX ---------------------')

# time_init = time.time()

# score_type = 'MSE_YmuX'

# with tqdm(total=n_MC) as pbar:
#    res = Parallel(n_jobs=n_MC)(delayed(simu_opti_lambda_byit)(theta, mu0, P0, Gamma, N, arc_length, sigma_init, noise_init_theta, tol_EM, max_iter_EM, nb_basis, grid_lambda, score_type, n_splits_CV) for k in range(n_MC))
#    pbar.update()

# time_end = time.time()
# duration = time_end - time_init

# filename = filename_base + "simu_3_byit_MSE_YmuX"

# dic = {"results":res, "score_type": score_type}

# if os.path.isfile(filename):
#    print("Le fichier ", filename, " existe déjà.")
#    filename = filename + '_bis'
# fil = open(filename,"xb")
# pickle.dump(dic,fil)
# fil.close()


# print('--------------------- End Simulation n°1 ---------------------')





# """ Simulation 4 : CV globally MSE_YmuX """

# print('--------------------- Simulation n°4: Optimization of lambda globally with MSE_YmuX ---------------------')

# time_init = time.time()

# score_type = 'MSE_YmuX'

# with tqdm(total=n_MC) as pbar:
#    res = Parallel(n_jobs=n_MC)(delayed(simu_opti_lambda_globally)(theta, mu0, P0, Gamma, N, arc_length, sigma_init, noise_init_theta, tol_EM, max_iter_EM, nb_basis, grid_lambda, score_type, n_splits_CV) for k in range(n_MC))
#    pbar.update()

# time_end = time.time()
# duration = time_end - time_init

# filename = filename_base + "simu_4_global_MSE_YmuX"

# dic = {"results":res, "score_type": score_type}

# if os.path.isfile(filename):
#    print("Le fichier ", filename, " existe déjà.")
#    filename = filename + '_bis'
# fil = open(filename,"xb")
# pickle.dump(dic,fil)
# fil.close()


# print('--------------------- End Simulation n°4 ---------------------')




def step_simu_opt(Y, FS_state_space_init, score_lambda_matrix, sigma_init, tol_EM, max_iter_EM, nb_basis, Gamma, mu0):

    Sigma_hat = lambda s: sigma_init**2*np.array([[1 + 0*s, 0*s],[0*s, 1 + 0*s]])
    P0_hat = sigma_init**2*np.eye(6)

    ind = np.squeeze(np.array(np.where(score_lambda_matrix==np.min(score_lambda_matrix))))
    lbda_opt = np.array([grid_lambda[ind[0]], grid_lambda[ind[1]]]) 
    try:
        FS_statespace = FrenetStateSpaceCV_global(arc_length, Y[1:], bornes_theta=np.array([0,1]))
        FS_statespace.expectation_maximization(tol_EM, max_iter_EM, nb_basis=nb_basis, regularization_parameter=lbda_opt, 
                                            init_params = {"Gamma":Gamma, "coefs":FS_state_space_init.tab_coefs[0], "mu0":mu0, "Sigma":Sigma_hat, "P0":P0_hat}, 
                                            method='autre', model_Sigma='scalar')
    except:
        FS_statespace = [FS_state_space_init.tab_coefs[0]]
    
    return FS_statespace



filename_base = "results/simulation_optimization_lambda/model_01/"
filename = filename_base + "model"
fil = open(filename,"rb")
dic_model = pickle.load(fil)
fil.close()

curv = lambda s : 2*np.cos(2*np.pi*s) + 5
tors = lambda s : 2*np.sin(2*np.pi*s) + 1
N_simu = dic_model["nb_iterations_simu"]
P0, mu0 = dic_model["P0"], dic_model["mu0"]
theta = dic_model["theta"]
Gamma = dic_model["Gamma"]
grid_lambda = dic_model["grid_lambda"]
max_iter, tol = dic_model["max_iter"], dic_model["tol"]
arc_length, nb_basis, N = dic_model["arc_length"], dic_model["nb_basis"], dic_model["N"]
sigma_init = dic_model["sigma_init"]
n_splits_CV = dic_model["n_splits_CV"]

mu_Z = solve_FrenetSerret_ODE_SE(theta, arc_length, mu0)
mu_Q, mu_X = mu_Z[:,:3,:3], mu_Z[:,:3,3]
tol_EM = 0.01
max_iter_EM = 100
nb_basis = 15


""" Simulation 2 : CV globally MSE_YX """

print('--------------------- Simulation n°2: Optimization of lambda globally with MSE_YX ---------------------')

time_init = time.time()

score_type = 'MSE_YX'

filename = filename_base + "simu_2_global_MSE_YX"
fil = open(filename,"rb")
dic_simu_2 = pickle.load(fil)
fil.close()

res_simu = dic_simu_2["results"]
tab_FS_state_space = []
tab_Z_init = []
tab_Y_init = []
tab_score_lambda_matrix = []
tab_error = []
for k in range(N_simu):
    if isinstance(res_simu[k][0], FrenetStateSpaceCV_global):
        tab_FS_state_space.append(res_simu[k][0])
        tab_Z_init.append(res_simu[k][1])
        tab_Y_init.append(res_simu[k][2])
        tab_score_lambda_matrix.append(res_simu[k][3])
    else:
        tab_error.append(res_simu[k])
N_simu_ok = len(tab_FS_state_space)

with tqdm(total=N_simu_ok) as pbar:
   res = Parallel(n_jobs=N_simu_ok)(delayed(step_simu_opt)(tab_Y_init[k], tab_FS_state_space[k], tab_score_lambda_matrix[k], sigma_init, tol_EM, max_iter_EM, nb_basis, Gamma, mu0) for k in range(N_simu_ok))
   pbar.update()

time_end = time.time()
duration = time_end - time_init

filename = filename_base + "simu_2_global_MSE_YX_bis"

dic = {"results":res, "score_type": score_type}

if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()


print('--------------------- End Simulation n°2 ---------------------')





""" Simulation 4 : CV globally MSE_YmuX """

print('--------------------- Simulation n°4: Optimization of lambda globally with MSE_YmuX ---------------------')

time_init = time.time()

score_type = 'MSE_YmuX'

filename = filename_base + "simu_4_global_MSE_YmuX"
fil = open(filename,"rb")
dic_simu_4 = pickle.load(fil)
fil.close()

res_simu = dic_simu_4["results"]
tab_FS_state_space = []
tab_Z_init = []
tab_Y_init = []
tab_score_lambda_matrix = []
tab_error = []
for k in range(N_simu):
    if isinstance(res_simu[k][0], FrenetStateSpaceCV_global):
        tab_FS_state_space.append(res_simu[k][0])
        tab_Z_init.append(res_simu[k][1])
        tab_Y_init.append(res_simu[k][2])
        tab_score_lambda_matrix.append(res_simu[k][3])
    else:
        tab_error.append(res_simu[k])
N_simu_ok = len(tab_FS_state_space)

with tqdm(total=N_simu_ok) as pbar:
   res = Parallel(n_jobs=N_simu_ok)(delayed(step_simu_opt)(tab_Y_init[k], tab_FS_state_space[k], tab_score_lambda_matrix[k], sigma_init, tol_EM, max_iter_EM, nb_basis, Gamma, mu0) for k in range(N_simu_ok))
   pbar.update()

time_end = time.time()
duration = time_end - time_init

filename = filename_base + "simu_4_global_MSE_YmuX_bis"

dic = {"results":res, "score_type": score_type}

if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()


print('--------------------- End Simulation n°4 ---------------------')