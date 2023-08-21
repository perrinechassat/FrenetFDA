import sys
sys.path.insert(1, '../')
from FrenetFDA.utils.Frenet_Serret_utils import *
from FrenetFDA.utils.smoothing_utils import *
from FrenetFDA.processing_Euclidean_curve.unified_estimate_Frenet_state_space.EM_Frenet_state_space_CV import FrenetStateSpaceCV_global
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
    bounds_h = [5*step, 0.15]

    ## Init Gamma and s(t)
    derivatives, h_opt = compute_derivatives(Y, grid_time, deg=3, h=None, CV_optimization_h={"flag":True, "h_grid":np.array([bounds_h[0], bounds_h[-1]]), "K":10, "method":'bayesian', "n_call":n_call_bayopt, "verbose":False})
    grid_arc_s, L, arc_s, arc_s_dot = compute_arc_length(Y, grid_time, smooth=True, smoothing_param=h_opt)
    Y_scale = Y/L
    # print('fin arc length')

    ## Z GramSchmidt
    bounds_h[0] = np.max((bounds_h[0], np.max(grid_arc_s[1:]-grid_arc_s[:-1])))
    if bounds_h[1] <= bounds_h[0]:
        bounds_h[1] = np.min((3*bounds_h[0], 0.3))
    GS_orthog = GramSchmidtOrthogonalization(Y_scale, grid_arc_s, deg=3)
    h_opt = GS_orthog.bayesian_optimization_hyperparameters(n_call_bayopt, h_bounds=bounds_h, verbose=False)
    Z_hat_GS, Q_hat_GS, X_hat_GS = GS_orthog.fit(h_opt) 
    # print('fin Z GramSchmidt')

    return grid_arc_s, L, Y_scale, Z_hat_GS, bounds_h, derivatives



def basis_GS_leastsquares(grid_arc_s, Z_hat_GS, nb_basis, bounds_h, bounds_lbda, n_call_bayopt):
    try:
        local_approx_ode = LocalApproxFrenetODE(grid_arc_s, Z=Z_hat_GS)
        bounds_h[0] = (bounds_h[0]/3)*5
        if bounds_h[1] <= bounds_h[0]:
            bounds_h[1] = np.min((5*bounds_h[0], 0.2))
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


