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
from sklearn.gaussian_process.kernels import Matern


def init_from_true_param_sde(theta, arc_length, N, Gamma, mu0, P0, nb_basis, noise_init_theta, grid_bandwidth, kernel):
    
    ## Definition of the states
    xi0 = np.random.multivariate_normal(np.zeros(6), P0)
    Z0 = mu0 @ SE3.exp(-xi0)

    v_grid = (arc_length[1:]+arc_length[:-1])/2
    V = np.expand_dims(v_grid, 1)
    noise_kappa = np.random.multivariate_normal(mean=np.zeros(N-1), cov=kernel(V,V))
    noise_tau = np.random.multivariate_normal(mean=np.zeros(N-1), cov=kernel(V,V))
    noise_theta = np.stack((noise_kappa, noise_tau), axis=1)
    L = np.array([[0,1],[0,0],[1,0],[0,0],[0,0],[0,0]])
    Z = np.zeros((N, 4, 4))
    Z[0] = Z0
    for i in range(1,N):
        delta_t = arc_length[i]-arc_length[i-1]
        pts = (arc_length[i]+arc_length[i-1])/2
        Z[i] = Z[i-1]@SE3.exp(delta_t*np.array([theta(pts)[1], 0, theta(pts)[0], 1, 0, 0]) + np.sqrt(delta_t)*L @ np.array([noise_kappa[i-1], noise_tau[i-1]]))

    X = Z[:,:3,3]
    Y = X + np.random.multivariate_normal(np.zeros(3), Gamma, size=N)

    ## Initialization of the parameters
    theta_noisy_val = theta(arc_length) + np.random.multivariate_normal(np.zeros(2), noise_init_theta**2*np.eye(2), size=len(arc_length))
    BasisThetaNoisy = VectorBSplineSmoothing(2, nb_basis)
    BasisThetaNoisy.fit(arc_length, theta_noisy_val, regularization_parameter=0.0000001)

    mu0_hat = mu0@SE3.exp(np.random.multivariate_normal(np.zeros(6), 0.001**2*np.eye(6)))
    grid_time = np.linspace(0,1,N)
    derivatives, h_opt = compute_derivatives(Y, grid_time, deg=3, h=None, CV_optimization_h={"flag":True, "h_grid":grid_bandwidth, "K":10})
    grid_arc_s, L, arc_s, arc_s_dot = compute_arc_length(Y, grid_time, smooth=True, smoothing_param=h_opt)
    Gamma_hat = ((Y - derivatives[0]).T @ (Y - derivatives[0]))/N

    return Z, Y, BasisThetaNoisy.coefs, grid_arc_s, Gamma_hat, mu0_hat, noise_theta



def init_from_true_param(theta, arc_length, N, Gamma, mu0, P0, nb_basis, noise_init_theta, grid_bandwidth):
    
    ## Definition of the states
    xi0 = np.random.multivariate_normal(np.zeros(6), P0)
    Z0 = mu0 @ SE3.exp(-xi0)
    Z = solve_FrenetSerret_ODE_SE(theta, arc_length, Z0=Z0)
    X = Z[:,:3,3]
    Y = X + np.random.multivariate_normal(np.zeros(3), Gamma, size=N)

    ## Initialization of the parameters
    theta_noisy_val = theta(arc_length) + np.random.multivariate_normal(np.zeros(2), noise_init_theta**2*np.eye(2), size=len(arc_length))
    BasisThetaNoisy = VectorBSplineSmoothing(2, nb_basis)
    BasisThetaNoisy.fit(arc_length, theta_noisy_val, regularization_parameter=0.0000001)

    mu0_hat = mu0@SE3.exp(np.random.multivariate_normal(np.zeros(6), 0.001**2*np.eye(6)))
    grid_time = np.linspace(0,1,N)
    derivatives, h_opt = compute_derivatives(Y, grid_time, deg=3, h=None, CV_optimization_h={"flag":True, "h_grid":grid_bandwidth, "K":10})
    grid_arc_s, L, arc_s, arc_s_dot = compute_arc_length(Y, grid_time, smooth=True, smoothing_param=h_opt)
    Gamma_hat = ((Y - derivatives[0]).T @ (Y - derivatives[0]))/N

    return Z, Y, BasisThetaNoisy.coefs, grid_arc_s, Gamma_hat, mu0_hat





def simulation_step(Y, grid_arc_length, init_coefs, init_Gamma, init_mu0, init_P0, init_sigma, nb_basis, max_iter_EM, tol_EM, grid_lambda, n_splits_CV):  

    init_Sigma = lambda s: init_sigma**2*np.array([[1 + 0*s, 0*s],[0*s, 1 + 0*s]]) 

    ## CV optimization of lambda
    K = len(grid_lambda)
    score_lambda_matrix = np.zeros((K,K))
    kf = KFold(n_splits=n_splits_CV, shuffle=True)
    ind_CV = 1
    for train_index, test_index in kf.split(grid_arc_length[1:]):
        print('     --> Start CV step n°', ind_CV)
        Y_train = Y[train_index]
        Y_test = Y[test_index]
        grid_train = np.concatenate((np.array([grid_arc_length[0]]), grid_arc_length[1:][train_index]))
        grid_test = np.concatenate((np.array([grid_arc_length[0]]), grid_arc_length[1:][test_index]))

        for i in range(K):
            for j in range(K):
                lbda = np.array([grid_lambda[i], grid_lambda[j]])
                try:
                    FS_statespace = FrenetStateSpaceCV_global(grid_train, Y_train, bornes_theta=np.array([0,1]))
                    FS_statespace.expectation_maximization(tol_EM, max_iter_EM, nb_basis=nb_basis, regularization_parameter=lbda, 
                                                        init_params = {"Gamma":init_Gamma, "coefs":init_coefs, "mu0":init_mu0, "Sigma":init_Sigma, "P0":init_P0}, 
                                                        method='autre', model_Sigma='scalar')
                    
                    # elif score_type=='MSE_YmuX':
                    Z_reconst = solve_FrenetSerret_ODE_SE(FS_statespace.theta, grid_test, Z0=FS_statespace.mu0)
                    X_reconst_test = Z_reconst[1:,:3,3]
                    score_lambda_matrix[i,j] = score_lambda_matrix[i,j] + np.linalg.norm(X_reconst_test - Y_test)**2

                    # if score_type=='MSE_YX':
                    #     Z_reconst = solve_FrenetSerret_SDE_SE3(FS_statespace.theta, FS_statespace.Sigma, FS_statespace.L, grid_test, Z0=FS_statespace.mu0)
                    #     X_reconst_test = Z_reconst[1:,:3,3]
                    #     score_lambda_matrix[i,j] = score_lambda_matrix[i,j] + np.linalg.norm(X_reconst_test - Y_test)**2
                    # else: 
                    #     raise Exception("Invalid term for optimization of lamnda score.")
                except:
                    print('Error for lbda:', lbda,' ind:', i, j)
        ind_CV += 1 

    score_lambda_matrix = score_lambda_matrix/n_splits_CV
    ind = np.squeeze(np.array(np.where(score_lambda_matrix==np.min(score_lambda_matrix))))
    lbda_opt = np.array([grid_lambda[ind[0]], grid_lambda[ind[1]]]) 
    print('Optimal chosen lambda:', lbda_opt)
    try:
        FS_statespace = FrenetStateSpaceCV_global(grid_arc_length, Y[1:], bornes_theta=np.array([0,1]))
        FS_statespace.expectation_maximization(tol_EM, max_iter_EM, nb_basis=nb_basis, regularization_parameter=lbda_opt, 
                                            init_params = {"Gamma":init_Gamma, "coefs":init_coefs, "mu0":init_mu0, "Sigma":init_Sigma, "P0":init_P0}, 
                                            method='autre', model_Sigma='scalar')
    except:
        FS_statespace = [init_coefs]
    
    return FS_statespace, score_lambda_matrix 





