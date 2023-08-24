import sys
sys.path.insert(1, '../')
from FrenetFDA.utils.Frenet_Serret_utils import *
from FrenetFDA.utils.smoothing_utils import *
from FrenetFDA.processing_Euclidean_curve.unified_estimate_Frenet_state_space.EM_Frenet_state_space_CV import FrenetStateSpaceCV_global, bayesian_CV_optimization_regularization_parameter
from FrenetFDA.processing_Euclidean_curve.unified_estimate_Frenet_state_space.iek_filter_smoother_Frenet_state import IEKFilterSmootherFrenetState
from FrenetFDA.processing_Euclidean_curve.preprocessing import *
from FrenetFDA.processing_Euclidean_curve.estimate_Frenet_path import GramSchmidtOrthogonalization, ConstrainedLocalPolynomialRegression
from FrenetFDA.processing_Euclidean_curve.estimate_Frenet_curvatures import ExtrinsicFormulas
from FrenetFDA.processing_Frenet_path.estimate_Frenet_curvatures import ApproxFrenetODE, LocalApproxFrenetODE
import FrenetFDA.utils.visualization as visu
from pickle import *
import time 
import os.path
import os
import dill as pickle
from tqdm import tqdm
import numpy as np


def init_arclength_Q(Y, n_call_bayopt):

    grid_time = np.linspace(0,1,Y.shape[0])
    step = grid_time[1] 
    bounds_h = [0.05, 0.1]

    bounds_h[0] = np.max((bounds_h[0],step*3))
    ## Init Gamma and s(t)
    derivatives, h_opt = compute_derivatives(Y, grid_time, deg=3, h=None, CV_optimization_h={"flag":True, "h_grid":np.array([bounds_h[0], bounds_h[-1]]), "K":10, "method":'bayesian', "n_call":n_call_bayopt, "verbose":True})
    grid_arc_s, L, arc_s, arc_s_dot = compute_arc_length(Y, grid_time, smooth=True, smoothing_param=h_opt)
    Y_scale = Y/L
    # print('fin arc length')

    ## Z GramSchmidt
    # bounds_h[0] = np.max((bounds_h[0], np.max(grid_arc_s[1:]-grid_arc_s[:-1])))
    # if bounds_h[1] <= bounds_h[0]:
    #     bounds_h[1] = np.min((3*bounds_h[0], 0.3))
    GS_orthog = GramSchmidtOrthogonalization(Y_scale, grid_arc_s, deg=3)
    h_opt = GS_orthog.bayesian_optimization_hyperparameters(n_call_bayopt, h_bounds=bounds_h, verbose=False)
    Z_hat_GS, Q_hat_GS, X_hat_GS = GS_orthog.fit(h_opt) 
    # print('fin Z GramSchmidt')

    return grid_arc_s, L, Y_scale, Z_hat_GS, bounds_h, derivatives



def basis_GS_leastsquares(grid_arc_s, Z_hat_GS, nb_basis, bounds_h, bounds_lbda, n_call_bayopt):
    try:
        local_approx_ode = LocalApproxFrenetODE(grid_arc_s, Z=Z_hat_GS)
        # bounds_h[0] = (bounds_h[0]/3)*5
        # if bounds_h[1] <= bounds_h[0]:
        #     bounds_h[1] = np.min((5*bounds_h[0], 0.2))
        h_opt, lbda_opt, coefs_opt = local_approx_ode.bayesian_optimization_hyperparameters(n_call_bayopt=n_call_bayopt, lambda_bounds=bounds_lbda, h_bounds=bounds_h, nb_basis=nb_basis, n_splits=10, verbose=False, return_coefs=True)
        return [coefs_opt, h_opt, lbda_opt]
    except:
        return None
    


def estimation_GS(Y, n_call_bayopt_der, bounds_lbda, n_call_bayopt_theta):

    grid_arc_s, L, Y_scale, Z_hat_GS, bounds_h, derivatives = init_arclength_Q(Y, n_call_bayopt_der)
    nb_basis = int(np.sqrt(Y.shape[0]))
    res_theta = basis_GS_leastsquares(grid_arc_s, Z_hat_GS, nb_basis, bounds_h, bounds_lbda, n_call_bayopt_theta)

    return grid_arc_s, L, Y_scale, Z_hat_GS, res_theta



def estimation_GS_group(filename_base, list_Y, n_call_bayopt_der, bounds_lbda, n_call_bayopt_theta):

    N_curves = len(list_Y)

    # time_init = time.time()

    # with tqdm(total=N_curves) as pbar:
    #     res = Parallel(n_jobs=N_curves)(delayed(estimation_GS)(list_Y[k], n_call_bayopt_der, bounds_lbda, n_call_bayopt_theta) for k in range(N_curves))
    # pbar.update()

    # time_end = time.time()
    # duration = time_end - time_init

    # tab_grid_arc_s = []
    # tab_L = []
    # tab_Y_scale = []
    # tab_Z_hat_GS = []
    # tab_smooth_theta_coefs = []
    # tab_h_opt = []
    # tab_lbda_opt = []
    # for k in range(N_curves):
    #     tab_grid_arc_s.append(res[k][0])
    #     tab_L.append(res[k][1])
    #     tab_Y_scale.append(res[k][2])
    #     tab_Z_hat_GS.append(res[k][3])
    #     if res[k][4] is not None:
    #         tab_smooth_theta_coefs.append(res[k][4][0])
    #         tab_h_opt.append(res[k][4][1])
    #         tab_lbda_opt.append(res[k][4][2])
    #     else:
    #         tab_smooth_theta_coefs.append(None)
    #         tab_h_opt.append(None)
    #         tab_lbda_opt.append(None)

    time_init = time.time()

    with tqdm(total=N_curves) as pbar:
        res = Parallel(n_jobs=N_curves)(delayed(init_arclength_Q)(list_Y[k], n_call_bayopt_der) for k in range(N_curves))
    pbar.update()

    time_end = time.time()
    duration = time_end - time_init
    print('_____________ END Z_Gram_Schmidt ______________')

    tab_grid_arc_s = []
    tab_L = []
    tab_Y_scale = []
    tab_Z_hat_GS = []
    tab_bounds_h = []
    tab_derivatives = []
    for k in range(N_curves):
        tab_grid_arc_s.append(res[k][0])
        tab_L.append(res[k][1])
        tab_Y_scale.append(res[k][2])
        tab_Z_hat_GS.append(res[k][3])
        tab_bounds_h.append(res[k][4])
        tab_derivatives.append(res[k][5])


    filename = filename_base + "estimations_GS_leastsquares_Z"

    dic = {"duration":duration, "tab_grid_arc_s":tab_grid_arc_s, "tab_L":tab_L, "tab_Y_scale":tab_Y_scale, "tab_Z_hat_GS":tab_Z_hat_GS, "tab_derivatives":tab_derivatives}
           
    if os.path.isfile(filename):
        print("Le fichier ", filename, " existe déjà.")
        filename = filename + '_bis'
    fil = open(filename,"xb")
    pickle.dump(dic,fil)
    fil.close()

    time_init = time.time()

    with tqdm(total=N_curves) as pbar:
        res = Parallel(n_jobs=N_curves)(delayed(basis_GS_leastsquares)(tab_grid_arc_s[k], tab_Z_hat_GS[k], int(np.sqrt(list_Y[k].shape[0])), tab_bounds_h[k], bounds_lbda, n_call_bayopt_theta) for k in range(N_curves))
    pbar.update()

    time_end = time.time()
    duration = time_end - time_init
    print('_____________ END Basis Theta ______________')

    tab_smooth_theta_coefs = []
    tab_h_opt = []
    tab_lbda_opt = []
    for k in range(N_curves):
        if res[k] is not None:
            tab_smooth_theta_coefs.append(res[k][0])
            tab_h_opt.append(res[k][1])
            tab_lbda_opt.append(res[k][2])
        else:
            tab_smooth_theta_coefs.append(None)
            tab_h_opt.append(None)
            tab_lbda_opt.append(None)


    filename = filename_base + "estimations_GS_leastsquares_theta"

    dic = {"duration":duration, "tab_smooth_theta_coefs":tab_smooth_theta_coefs, "tab_h_opt":tab_h_opt, "tab_lbda_opt":tab_lbda_opt, "tab_grid_arc_s":tab_grid_arc_s, "tab_L":tab_L, 
            "tab_Y_scale":tab_Y_scale, "tab_Z_hat_GS":tab_Z_hat_GS}
           
    if os.path.isfile(filename):
        print("Le fichier ", filename, " existe déjà.")
        filename = filename + '_bis'
    fil = open(filename,"xb")
    pickle.dump(dic,fil)
    fil.close()





def EM_from_init_theta(filename_save, filename_simu, sigma_init, n_splits_CV, n_call_bayopt, bounds_lambda, tol_EM, max_iter_EM):

    init_Sigma = lambda s: sigma_init**2*np.array([[1 + 0*s, 0*s],[0*s, 1 + 0*s]]) 
    P0_init = sigma_init**2*np.eye(6)

    filename = filename_simu 
    fil = open(filename,"rb")
    dic_init = pickle.load(fil)
    fil.close()

    tab_Y_scale = dic_init["tab_Y_scale"]
    tab_Z_hat_GS = dic_init["tab_Z_hat_GS"]
    tab_smooth_theta_coefs = dic_init["tab_smooth_theta_coefs"]
    tab_grid_arc_s = dic_init["tab_grid_arc_s"]
    n_curves = len(tab_Y_scale)

    Gamma_tab = []
    mu0_tab = []
    nb_basis_tab = []
    for k in range(n_curves):
        Gamma_tab.append(((tab_Y_scale[k] - tab_Z_hat_GS[k][:,:3,3]).T @ (tab_Y_scale[k] - tab_Z_hat_GS[k][:,:3,3]))/len(tab_Y_scale[k]))
        mu0_tab.append(tab_Z_hat_GS[k][0])
        nb_basis_tab.append(int(np.sqrt(tab_Y_scale[k].shape[0])))

    time_init = time.time()

    with tqdm(total=n_curves) as pbar:
        res = Parallel(n_jobs=n_curves)(delayed(bayesian_CV_optimization_regularization_parameter)(n_CV=n_splits_CV, n_call_bayopt=n_call_bayopt, lambda_bounds=bounds_lambda, grid_obs=tab_grid_arc_s[k], 
                                                                                                   Y_obs=tab_Y_scale[k], tol=tol_EM, max_iter=max_iter_EM, nb_basis=nb_basis_tab[k], 
                                                                                init_params={"Gamma":Gamma_tab[k], "coefs":tab_smooth_theta_coefs[k], "mu0":mu0_tab[k], "Sigma":init_Sigma, "P0":P0_init}) for k in range(n_curves))
    pbar.update()

    time_end = time.time()
    duration = time_end - time_init

    FS_statespace_tab = []
    res_bayopt_tab = []
    for k in range(n_curves):
        FS_statespace_tab.append(res[k][0])
        res_bayopt_tab.append(res[k][1])


    filename = filename_save

    dic = {"duration":duration, "FS_statespace_tab":FS_statespace_tab, "res_bayopt_tab":res_bayopt_tab, "nb_basis_tab":nb_basis_tab, "mu0_tab":mu0_tab, "Gamma_tab":Gamma_tab}

    if os.path.isfile(filename):
        print("Le fichier ", filename, " existe déjà.")
        filename = filename + '_bis'
    fil = open(filename,"xb")
    pickle.dump(dic,fil)
    fil.close()

    print('___________________________ End EM ___________________________')
