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



def simu_test_GCV(sigma, theta, mu0, P0, Gamma, N, arc_length, grid_lambda):
    
    Sigma = lambda s: sigma**2*np.array([[1 + 0*s, 0*s],[0*s, 1 + 0*s]])

    xi0 = np.random.multivariate_normal(np.zeros(6), P0)
    Z0 = mu0 @ SE3.exp(-xi0)

    mat_L = np.zeros((6,2))
    mat_L[0,1], mat_L[2,0] = 1, 1

    Z_sde = solve_FrenetSerret_SDE_SE3(theta, Sigma, mat_L, arc_length, Z0=Z0)
    Q_sde = Z_sde[:,:3,:3]
    X_sde = Z_sde[:,:3,3]

    xi_sde = np.zeros((N,6))
    xi_sde_vec = np.zeros((N*6))
    Z_gp = np.zeros((N,4,4))

    P_mat_full, P = solve_FrenetSerret_SDE_full_cov_matrix(theta, Sigma, mat_L, arc_length, P0)
    xi_gp_vec = np.random.multivariate_normal(mean=np.zeros(len(P_mat_full)), cov=P_mat_full)
    xi_gp = np.reshape(xi_gp_vec, (N,6))
    for j in range(N):
        Z_gp[j] = mu_Z[j]@SE3.exp(-xi_gp[j])
        xi_sde[j] = SE3.log(np.linalg.inv(Z_sde[j])@mu_Z[j])
    xi_sde_vec = np.reshape(xi_sde, (N*6,)) 

    Q_gp = Z_gp[:,:3,:3]
    X_gp = Z_gp[:,:3,3]

    Y_gp = X_gp + np.random.multivariate_normal(np.zeros(3), Gamma, size=(len(X_gp)))
    Y_sde = X_sde + np.random.multivariate_normal(np.zeros(3), Gamma, size=(len(X_sde)))

    MLE_gp = MLE(arc_length, Y_gp, Z_gp)
    MLE_sde = MLE(arc_length, Y_sde, Z_sde)
    mu0_hat_gp, P0_hat_gp, Gamma_hat_gp = MLE_gp.opti_other_param()
    mu0_hat_sde, P0_hat_sde, Gamma_hat_sde = MLE_sde.opti_other_param()

    nb_lbda = len(grid_lambda)
    nb_basis = 15
    MLE_gp.def_model_theta(nb_basis)
    MLE_sde.def_model_theta(nb_basis)

    MSE_theta_gp = np.zeros((nb_lbda,nb_lbda,2))
    MSE_gp = np.zeros((nb_lbda,nb_lbda))
    error_sigma_gp = np.zeros((nb_lbda,nb_lbda))
    GCV_gp, L_gp = np.zeros((nb_lbda,nb_lbda)), np.zeros((nb_lbda,nb_lbda))
    V0_gp, V1_gp, V2_gp = np.zeros((nb_lbda,nb_lbda)), np.zeros((nb_lbda,nb_lbda)), np.zeros((nb_lbda,nb_lbda))
    U0_gp, U1_gp, U2_gp = np.zeros((nb_lbda,nb_lbda)), np.zeros((nb_lbda,nb_lbda)), np.zeros((nb_lbda,nb_lbda))
    theta_hat_gp = np.zeros((nb_lbda, nb_lbda, N-1, 2))
    sigma_hat_gp = np.zeros((nb_lbda, nb_lbda))


    MSE_theta_sde = np.zeros((nb_lbda,nb_lbda,2))
    MSE_sde = np.zeros((nb_lbda,nb_lbda))
    error_sigma_sde = np.zeros((nb_lbda,nb_lbda))
    GCV_sde, L_sde = np.zeros((nb_lbda,nb_lbda)), np.zeros((nb_lbda,nb_lbda))
    V0_sde, V1_sde, V2_sde = np.zeros((nb_lbda,nb_lbda)), np.zeros((nb_lbda,nb_lbda)), np.zeros((nb_lbda,nb_lbda))
    U0_sde, U1_sde, U2_sde = np.zeros((nb_lbda,nb_lbda)), np.zeros((nb_lbda,nb_lbda)), np.zeros((nb_lbda,nb_lbda))
    theta_hat_sde = np.zeros((nb_lbda, nb_lbda, N-1, 2))
    sigma_hat_sde = np.zeros((nb_lbda, nb_lbda))

    for i in range(nb_lbda):
        for j in range(len(grid_lambda)):
            reg_param = np.array([grid_lambda[i], grid_lambda[j]])

            MLE_gp.opti_coefs(reg_param)
            MLE_gp.opti_sigma()
            MSE_theta_gp[i,j], MSE_gp[i,j], error_sigma_gp[i,j] = MLE_gp.compute_true_MSE(theta, sigma)
            GCV_gp[i,j], V0_gp[i,j], V1_gp[i,j], V2_gp[i,j], U0_gp[i,j], U1_gp[i,j], U2_gp[i,j], L_gp[i,j] = MLE_gp.compute_GCV_criteria()
            theta_hat_gp[i,j] = np.reshape(MLE_gp.basis_matrix @ MLE_gp.coefs, (-1,2))
            sigma_hat_gp[i,j] = MLE_gp.sigma

            MLE_sde.opti_coefs(reg_param)
            MLE_sde.opti_sigma()
            MSE_theta_sde[i,j], MSE_sde[i,j], error_sigma_sde[i,j] = MLE_sde.compute_true_MSE(theta, sigma)
            GCV_sde[i,j], V0_sde[i,j], V1_sde[i,j], V2_sde[i,j], U0_sde[i,j], U1_sde[i,j], U2_sde[i,j], L_sde[i,j] = MLE_sde.compute_GCV_criteria()
            theta_hat_sde[i,j] = np.reshape(MLE_sde.basis_matrix @ MLE_sde.coefs, (-1,2))
            sigma_hat_sde[i,j] = MLE_sde.sigma

    dic_gp = {"mu0_hat":mu0_hat_gp, "P0_hat": P0_hat_gp, "Gamma_hat":Gamma_hat_gp, "MSE":MSE_gp, "error_sigma":error_sigma_gp, "theta_hat":theta_hat_gp, "sigma_hat":sigma_hat_gp,
              "GCV":GCV_gp, "L":L_gp, "V0":V0_gp, "V1":V1_gp, "V2":V2_gp, "U0":U0_gp, "U1":U1_gp, "U2":U2_gp}
    dic_sde = {"mu0_hat":mu0_hat_sde, "P0_hat": P0_hat_sde, "Gamma_hat":Gamma_hat_sde, "MSE":MSE_sde, "error_sigma":error_sigma_sde, "theta_hat":theta_hat_sde, "sigma_hat":sigma_hat_sde,
              "GCV":GCV_sde, "L":L_sde, "V0":V0_sde, "V1":V1_sde, "V2":V2_sde, "U0":U0_sde, "U1":U1_sde, "U2":U2_sde}
    
    return dic_gp, dic_sde


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

directory = r"results/test_GCV_criterion/model_01/"
filename_base = "results/test_GCV_criterion/model_01/"

current_directory = os.getcwd()
final_directory = os.path.join(current_directory, directory)
if not os.path.exists(final_directory):
   os.makedirs(final_directory)



""" Simulation 1 : sigma = 0 """

print('--------------------- Simulation n°1: sigma = 0 ---------------------')

time_init = time.time()


sigma = 0
grid_lambda = np.logspace(-15, -3, 30)


with tqdm(total=n_MC) as pbar:
   res = Parallel(n_jobs=50)(delayed(simu_test_GCV)(sigma, theta, mu0, P0, Gamma, N, arc_length, grid_lambda) for k in range(n_MC))
   pbar.update()

time_end = time.time()
duration = time_end - time_init

filename = filename_base + "simu_1"

dic = {"results":res, "sigma": sigma, "grid_lambda": grid_lambda}

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
grid_lambda = np.logspace(-15, -3, 30)


with tqdm(total=n_MC) as pbar:
   res = Parallel(n_jobs=50)(delayed(simu_test_GCV)(sigma, theta, mu0, P0, Gamma, N, arc_length, grid_lambda) for k in range(n_MC))
   pbar.update()

time_end = time.time()
duration = time_end - time_init

filename = filename_base + "simu_2"

dic = {"results":res, "sigma": sigma, "grid_lambda": grid_lambda}

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
grid_lambda = np.logspace(-15, -3, 30)


with tqdm(total=n_MC) as pbar:
   res = Parallel(n_jobs=50)(delayed(simu_test_GCV)(sigma, theta, mu0, P0, Gamma, N, arc_length, grid_lambda) for k in range(n_MC))
   pbar.update()

time_end = time.time()
duration = time_end - time_init

filename = filename_base + "simu_3"

dic = {"results":res, "sigma": sigma, "grid_lambda": grid_lambda}

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
grid_lambda = np.logspace(-15, -3, 30)


with tqdm(total=n_MC) as pbar:
   res = Parallel(n_jobs=50)(delayed(simu_test_GCV)(sigma, theta, mu0, P0, Gamma, N, arc_length, grid_lambda) for k in range(n_MC))
   pbar.update()

time_end = time.time()
duration = time_end - time_init

filename = filename_base + "simu_4"

dic = {"results":res, "sigma": sigma, "grid_lambda": grid_lambda}

if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()

print('End of simulation n°4: time spent', duration, 'seconds. \n')