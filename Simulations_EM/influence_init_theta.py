import sys
sys.path.insert(1, '../')
from FrenetFDA.utils.Frenet_Serret_utils import *
from FrenetFDA.utils.smoothing_utils import *
from FrenetFDA.processing_Euclidean_curve.unified_estimate_Frenet_state_space.EM_Frenet_state_space_CV import FrenetStateSpaceCV_global, bayesian_CV_optimization_regularization_parameter, grid_search_CV_optimization_regularization_parameter
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


def EM_from_init_theta(filename_save, filename_simu, sigma_init, n_splits_CV, n_call_bayopt, bounds_lambda, tol_EM, max_iter_EM):

    init_Sigma = lambda s: sigma_init**2*np.array([[1 + 0*s, 0*s],[0*s, 1 + 0*s]]) 
    P0_init = sigma_init**2*np.eye(6)

    filename = filename_simu + '_init_and_Q'
    fil = open(filename,"rb")
    dic_init = pickle.load(fil)
    fil.close()

    Y_tab = dic_init["Y_tab"]
    N = dic_init["N"]
    Gamma = dic_init["gamma"]
    nb_basis = dic_init["nb_basis"]
    Z_tab = dic_init["Z_tab"]
    n_MC = Z_tab.shape[0]
    grid_arc_s_tab = dic_init["grid_arc_s_tab"]
    Z_hat_GS_tab = dic_init["Z_hat_GS_tab"]
    Z_hat_CLP_tab = dic_init["Z_hat_CLP_tab"]

    # ## Extrinsic

    # filename = filename_simu + "_basis_theta_extrins"
    # fil = open(filename,"rb")
    # dic_theta_ext = pickle.load(fil)
    # fil.close()
    # tab_smooth_theta_coefs_ext = []
    # Y_tab_bis = []
    # grid_arc_s_tab_bis = []
    # Gamma_tab = []
    # mu0_tab = []
    # for k in range(n_MC):
    #     if dic_theta_ext["results"][k] is not None:
    #         tab_smooth_theta_coefs_ext.append(dic_theta_ext["results"][k][0])
    #         Y_tab_bis.append(Y_tab[k])
    #         grid_arc_s_tab_bis.append(grid_arc_s_tab[k])
    #         Gamma_tab.append(((Y_tab[k] - Z_hat_GS_tab[k][:,:3,3]).T @ (Y_tab[k] - Z_hat_GS_tab[k][:,:3,3]))/N)
    #         mu0_tab.append(Z_hat_GS_tab[k][0])
    # tab_smooth_theta_coefs_ext = np.array(tab_smooth_theta_coefs_ext)
    # Y_tab_bis = np.array(Y_tab_bis)
    # grid_arc_s_tab_bis = np.array(grid_arc_s_tab_bis)
    # Gamma_tab = np.array(Gamma_tab)
    # N_ext = len(tab_smooth_theta_coefs_ext)

    
    # time_init = time.time()

    # with tqdm(total=N_ext) as pbar:
    #     res = Parallel(n_jobs=N_ext)(delayed(bayesian_CV_optimization_regularization_parameter)(n_CV=n_splits_CV, n_call_bayopt=n_call_bayopt, lambda_bounds=bounds_lambda, grid_obs=grid_arc_s_tab_bis[k], Y_obs=Y_tab_bis[k], tol=tol_EM, max_iter=max_iter_EM, 
    #                                                                               nb_basis=nb_basis, init_params={"Gamma":Gamma_tab[k], "coefs":tab_smooth_theta_coefs_ext[k], "mu0":mu0_tab[k], "Sigma":init_Sigma, "P0":P0_init}) for k in range(N_ext))
    # pbar.update()

    # time_end = time.time()
    # duration = time_end - time_init

    # FS_statespace_tab = []
    # res_bayopt_tab = []
    # for k in range(N_ext):
    #     FS_statespace_tab.append(res[k][0])
    #     res_bayopt_tab.append(res[k][1])


    # filename = filename_save + "_from_extrinsic"

    # dic = {"duration":duration, "FS_statespace_tab":FS_statespace_tab, "res_bayopt_tab":res_bayopt_tab}

    # if os.path.isfile(filename):
    #     print("Le fichier ", filename, " existe déjà.")
    #     filename = filename + '_bis'
    # fil = open(filename,"xb")
    # pickle.dump(dic,fil)
    # fil.close()

    # print('___________________________ End EM on extrinsic ___________________________')


    ### GS 

    filename = filename_simu + "_basis_theta_GS_leastsquares"
    fil = open(filename,"rb")
    dic_theta_GS = pickle.load(fil)
    fil.close()
    tab_smooth_theta_coefs_GS = np.array(dic_theta_GS["tab_smooth_theta_coefs"])
    Gamma_tab = []
    mu0_tab = []
    for k in range(n_MC):
        Gamma_tab.append(((Y_tab[k] - Z_hat_GS_tab[k][:,:3,3]).T @ (Y_tab[k] - Z_hat_GS_tab[k][:,:3,3]))/N)
        mu0_tab.append(Z_hat_GS_tab[k][0])
    
    time_init = time.time()

    with tqdm(total=n_MC) as pbar:
        res = Parallel(n_jobs=n_MC)(delayed(bayesian_CV_optimization_regularization_parameter)(n_CV=n_splits_CV, n_call_bayopt=n_call_bayopt, lambda_bounds=bounds_lambda, grid_obs=grid_arc_s_tab[k], Y_obs=Y_tab[k], tol=tol_EM, max_iter=max_iter_EM, 
                                                                                  nb_basis=nb_basis, init_params={"Gamma":Gamma_tab[k], "coefs":tab_smooth_theta_coefs_GS[k], "mu0":mu0_tab[k], "Sigma":init_Sigma, "P0":P0_init}) for k in range(n_MC))
    pbar.update()

    time_end = time.time()
    duration = time_end - time_init

    FS_statespace_tab = []
    res_bayopt_tab = []
    for k in range(n_MC):
        FS_statespace_tab.append(res[k][0])
        res_bayopt_tab.append(res[k][1])


    filename = filename_save + "_from_GS_log"

    dic = {"duration":duration, "FS_statespace_tab":FS_statespace_tab, "res_bayopt_tab":res_bayopt_tab}

    if os.path.isfile(filename):
        print("Le fichier ", filename, " existe déjà.")
        filename = filename + '_bis'
    fil = open(filename,"xb")
    pickle.dump(dic,fil)
    fil.close()

    print('___________________________ End EM on GS ___________________________')






def EM_from_init_theta_grid_search(filename_save, filename_simu, sigma_init, n_splits_CV, lambda_grids, tol_EM, max_iter_EM):

    init_Sigma = lambda s: sigma_init**2*np.array([[1 + 0*s, 0*s],[0*s, 1 + 0*s]]) 
    P0_init = sigma_init**2*np.eye(6)

    filename = filename_simu + '_init_and_Q'
    fil = open(filename,"rb")
    dic_init = pickle.load(fil)
    fil.close()

    Y_tab = dic_init["Y_tab"]
    N = dic_init["N"]
    Gamma = dic_init["gamma"]
    nb_basis = dic_init["nb_basis"]
    Z_tab = dic_init["Z_tab"]
    n_MC = Z_tab.shape[0]
    grid_arc_s_tab = dic_init["grid_arc_s_tab"]
    Z_hat_GS_tab = dic_init["Z_hat_GS_tab"]
    Z_hat_CLP_tab = dic_init["Z_hat_CLP_tab"]

    ### GS 

    filename = filename_simu + "_basis_theta_GS_leastsquares"
    fil = open(filename,"rb")
    dic_theta_GS = pickle.load(fil)
    fil.close()
    tab_smooth_theta_coefs_GS = np.array(dic_theta_GS["tab_smooth_theta_coefs"])
    Gamma_tab = []
    mu0_tab = []
    for k in range(n_MC):
        Gamma_tab.append(((Y_tab[k] - Z_hat_GS_tab[k][:,:3,3]).T @ (Y_tab[k] - Z_hat_GS_tab[k][:,:3,3]))/N)
        mu0_tab.append(Z_hat_GS_tab[k][0])
    
    time_init = time.time()

    with tqdm(total=n_MC) as pbar:

        res = Parallel(n_jobs=n_MC)(delayed(grid_search_CV_optimization_regularization_parameter)(n_CV=n_splits_CV, lambda_grids=lambda_grids, grid_obs=grid_arc_s_tab[k], Y_obs=Y_tab[k], tol=tol_EM, max_iter=max_iter_EM, 
                                                                            nb_basis=nb_basis, init_params={"Gamma":Gamma_tab[k], "coefs":tab_smooth_theta_coefs_GS[k], "mu0":mu0_tab[k], "Sigma":init_Sigma, "P0":P0_init}) for k in range(n_MC))
        
        # res = Parallel(n_jobs=n_MC)(delayed(bayesian_CV_optimization_regularization_parameter)(n_CV=n_splits_CV, n_call_bayopt=n_call_bayopt, lambda_bounds=bounds_lambda, grid_obs=grid_arc_s_tab[k], Y_obs=Y_tab[k], tol=tol_EM, max_iter=max_iter_EM, 
        #                                                                           nb_basis=nb_basis, init_params={"Gamma":Gamma_tab[k], "coefs":tab_smooth_theta_coefs_GS[k], "mu0":mu0_tab[k], "Sigma":init_Sigma, "P0":P0_init}) for k in range(n_MC))
    pbar.update()

    time_end = time.time()
    duration = time_end - time_init

    FS_statespace_tab = []
    res_gridsearch_tab = []
    for k in range(n_MC):
        FS_statespace_tab.append(res[k][0])
        res_gridsearch_tab.append(res[k][1])


    filename = filename_save + "_from_GS_gridsearch"

    dic = {"duration":duration, "FS_statespace_tab":FS_statespace_tab, "res_gridsearch_tab":res_gridsearch_tab}

    if os.path.isfile(filename):
        print("Le fichier ", filename, " existe déjà.")
        filename = filename + '_bis'
    fil = open(filename,"xb")
    pickle.dump(dic,fil)
    fil.close()

    print('___________________________ End EM on GS ___________________________')

