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
from FrenetFDA.processing_Frenet_path.smoothing import KarcherMeanSmoother, TrackingSmootherLinear
from FrenetFDA.processing_Frenet_path.estimate_Frenet_curvatures import TwoStepEstimatorKarcherMean, TwoStepEstimatorTracking
from FrenetFDA.shape_analysis.statistical_mean_shape import StatisticalMeanShapeV1, StatisticalMeanShapeV2, StatisticalMeanShapeV3
from FrenetFDA.shape_analysis.riemannian_geometries import SRVF, SRC, Frenet_Curvatures
from FrenetFDA.utils.Lie_group.SE3_utils import SE3
from FrenetFDA.utils.Lie_group.SO3_utils import SO3
import FrenetFDA.utils.visualization as visu
from pickle import *
import time 
import os.path
import os
import dill as pickle
from tqdm import tqdm
import numpy as np
from scipy.interpolate import interp1d
import collections



def init_arclength_Q(Y, n_call_bayopt, bounds_h):

    grid_time = np.linspace(0,1,Y.shape[0])

    ## Init Gamma and s(t)
    derivatives, h_opt = compute_derivatives(Y, grid_time, deg=3, h=None, CV_optimization_h={"flag":True, "h_grid":np.array([bounds_h[0], bounds_h[-1]]), "K":10, "method":'bayesian', "n_call":n_call_bayopt, "verbose":False})
    grid_arc_s, L, arc_s, arc_s_dot = compute_arc_length(Y, grid_time, smooth=True, smoothing_param=h_opt)
    Y_scale = Y/L

    ## Z GramSchmidt
    bounds_h[0] = np.max((bounds_h[0], np.max(grid_arc_s[1:]-grid_arc_s[:-1])*3))
    GS_orthog = GramSchmidtOrthogonalization(Y_scale, grid_arc_s, deg=3)
    h_opt = GS_orthog.bayesian_optimization_hyperparameters(n_call_bayopt, h_bounds=bounds_h, verbose=False)
    Z_hat_GS, Q_hat_GS, X_hat_GS = GS_orthog.fit(h_opt) 

    return grid_arc_s, L, Y_scale, Z_hat_GS, bounds_h, derivatives



def basis_GS_leastsquares(grid_arc_s, Z_hat_GS, bounds_lbda, n_call_bayopt):
    try:
        local_approx_ode = LocalApproxFrenetODE(grid_arc_s, Z=Z_hat_GS)
        bounds_h = np.array([np.max(grid_arc_s[1:]-grid_arc_s[:-1])*3, np.min((np.max(grid_arc_s[1:]-grid_arc_s[:-1])*8, 0.1))])
        # print(bounds_h)

        knots = [grid_arc_s[0]]
        grid_bis = grid_arc_s[1:-1]
        for i in range(0,len(grid_bis),4):
            knots.append(grid_bis[i])
        knots.append(grid_arc_s[-1])
        nb_basis = len(knots)+2

        h_opt, lbda_opt, coefs_opt = local_approx_ode.bayesian_optimization_hyperparameters(n_call_bayopt=n_call_bayopt, lambda_bounds=bounds_lbda, h_bounds=bounds_h, nb_basis=nb_basis, n_splits=5, verbose=True, return_coefs=True, knots=knots)
        return [coefs_opt, h_opt, lbda_opt, knots, nb_basis, bounds_h]
    except:
        return None
    

def basis_extrins(Y, bounds_h_der, n_call_bayopt_der, bounds_lbda, n_call_bayopt):

    grid_time = np.linspace(0,1,Y.shape[0])

    ## Init Gamma and s(t)
    derivatives, h_opt = compute_derivatives(Y, grid_time, deg=3, h=None, CV_optimization_h={"flag":True, "h_grid":np.array([bounds_h_der[0], bounds_h_der[-1]]), "K":10, "method":'bayesian', "n_call":n_call_bayopt_der, "verbose":False})
    grid_arc_s, L, arc_s, arc_s_dot = compute_arc_length(Y, grid_time, smooth=True, smoothing_param=h_opt)
    Y_scale = Y/L

    try: 
        # bounds_h = np.array([np.max(grid_arc_s[1:]-grid_arc_s[:-1])*3, np.min((np.max(grid_arc_s[1:]-grid_arc_s[:-1])*8, 0.1))])
        knots = [grid_arc_s[0]]
        grid_bis = grid_arc_s[1:-1]
        for i in range(0,len(grid_bis),4):
            knots.append(grid_bis[i])
        knots.append(grid_arc_s[-1])
        nb_basis = len(knots)+2

        extrins_model_theta = ExtrinsicFormulas(Y_scale, grid_time, grid_arc_s, deg_polynomial=3)
        h_opt, lbda_opt, coefs_opt = extrins_model_theta.bayesian_optimization_hyperparameters(n_call_bayopt=n_call_bayopt, lambda_bounds=bounds_lbda, h_bounds=bounds_h_der, nb_basis=nb_basis, n_splits=10, verbose=False, return_coefs=True, knots=knots)
        return [grid_arc_s, Y_scale, coefs_opt, h_opt, lbda_opt, knots, nb_basis]
    except:
        return [grid_arc_s, Y_scale, None]


def karcher_mean_smoother(grid_arc_s, Q_noisy, bounds_lbda, n_call_bayopt, tol, max_iter):
    try:
        bounds_h = np.array([np.max(grid_arc_s[1:]-grid_arc_s[:-1])*2, np.min((np.max(grid_arc_s[1:]-grid_arc_s[:-1])*5, 0.06))])

        knots = [grid_arc_s[0]]
        grid_bis = grid_arc_s[1:-1]
        for i in range(0,len(grid_bis),4):
            knots.append(grid_bis[i])
        knots.append(grid_arc_s[-1])
        nb_basis = len(knots)+2

        karcher_mean_smoother = TwoStepEstimatorKarcherMean(grid_arc_s, Q_noisy)
        coefs_opt, Q_smooth_opt, nb_iter, h_opt, lbda_opt = karcher_mean_smoother.bayesian_optimization_hyperparameters(n_call_bayopt=n_call_bayopt, lambda_bounds=bounds_lbda, h_bounds=bounds_h, nb_basis=nb_basis, epsilon=tol, max_iter=max_iter, n_splits=5, verbose=False, return_coefs=True, knots=knots)

        return [coefs_opt, Q_smooth_opt, nb_iter, h_opt, lbda_opt,  knots, nb_basis, bounds_h]
    except:
        return [None, None, None, None, None,  knots, nb_basis, bounds_h]



def tracking_smoother(grid_arc_s, Q_noisy, bounds_lbda, bounds_lbda_track, n_call_bayopt, tol, max_iter):
    try:
        bounds_h = np.array([np.max(grid_arc_s[1:]-grid_arc_s[:-1])*3, np.min((np.max(grid_arc_s[1:]-grid_arc_s[:-1])*8, 0.1))])

        knots = [grid_arc_s[0]]
        grid_bis = grid_arc_s[1:-1]
        for i in range(0,len(grid_bis),4):
            knots.append(grid_bis[i])
        knots.append(grid_arc_s[-1])
        nb_basis = len(knots)+2

        tracking_smoother = TwoStepEstimatorTracking(grid_arc_s, Q_noisy)
        coefs_opt, Q_smooth_opt, nb_iter, h_opt, lbda_opt, lbda_track_opt = tracking_smoother.bayesian_optimization_hyperparameters(n_call_bayopt=n_call_bayopt, lambda_track_bounds=bounds_lbda_track, lambda_bounds=bounds_lbda, h_bounds=bounds_h, nb_basis=nb_basis, epsilon=tol, max_iter=max_iter, n_splits=5, verbose=False, return_coefs=True, knots=knots)

        return [coefs_opt, Q_smooth_opt, nb_iter, h_opt, lbda_opt, lbda_track_opt, knots, nb_basis, bounds_h]
    except:
        return [None, None, None, None, None, None, knots, nb_basis, bounds_h]




def estimation_GS_group(filename_base, list_Y, n_call_bayopt_der, bounds_h_der, bounds_lbda, n_call_bayopt_theta):

    N_curves = len(list_Y)

    time_init = time.time()

    with tqdm(total=N_curves) as pbar:
        res = Parallel(n_jobs=N_curves)(delayed(init_arclength_Q)(list_Y[k], n_call_bayopt_der, bounds_h_der) for k in range(N_curves))
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

    filename = filename_base + "_Z_GS"

    dic = {"duration":duration, "tab_grid_arc_s":tab_grid_arc_s, "tab_L":tab_L, "tab_Y_scale":tab_Y_scale, "tab_Z_hat_GS":tab_Z_hat_GS, "tab_derivatives":tab_derivatives}
           
    if os.path.isfile(filename):
        print("Le fichier ", filename, " existe déjà.")
        filename = filename + '_bis'
    fil = open(filename,"xb")
    pickle.dump(dic,fil)
    fil.close()

    time_init = time.time()

    with tqdm(total=N_curves) as pbar:
        res = Parallel(n_jobs=N_curves)(delayed(basis_GS_leastsquares)(tab_grid_arc_s[k], tab_Z_hat_GS[k], bounds_lbda, n_call_bayopt_theta) for k in range(N_curves))
    pbar.update()

    time_end = time.time()
    duration = time_end - time_init

    print('_____________ END Basis Theta ______________')

    tab_smooth_theta_coefs = []
    tab_h_opt = []
    tab_lbda_opt = []
    tab_nb_basis = []
    tab_knots = []
    tab_h_bounds = []
    for k in range(N_curves):
        if res[k] is not None:
            tab_smooth_theta_coefs.append(res[k][0])
            tab_h_opt.append(res[k][1])
            tab_lbda_opt.append(res[k][2])
            tab_knots.append(res[k][3])
            tab_nb_basis.append(res[k][4])
            tab_h_bounds.append(res[k][5])
        else:
            tab_smooth_theta_coefs.append(None)
            tab_h_opt.append(None)
            tab_lbda_opt.append(None)
            tab_knots.append(None)
            tab_nb_basis.append(None)
            tab_h_bounds.append(None)

    filename = filename_base + "_theta"

    dic = {"duration":duration, "tab_smooth_theta_coefs":tab_smooth_theta_coefs, "tab_h_opt":tab_h_opt, "tab_lbda_opt":tab_lbda_opt, 
           "tab_grid_arc_s":tab_grid_arc_s, "tab_L":tab_L, "tab_Y_scale":tab_Y_scale, "tab_Z_hat_GS":tab_Z_hat_GS, "tab_knots":tab_knots, "tab_nb_basis":tab_nb_basis, "tab_h_bounds":tab_h_bounds}
           
    if os.path.isfile(filename):
        print("Le fichier ", filename, " existe déjà.")
        filename = filename + '_bis'
    fil = open(filename,"xb")
    pickle.dump(dic,fil)
    fil.close()

    return 




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
    tab_nb_basis = dic_init["tab_nb_basis"]
    tab_knots = dic_init["tab_knots"]
    n_curves = len(tab_Y_scale)

    Gamma_tab = []
    mu0_tab = []
    # nb_basis_tab = []
    for k in range(n_curves):
        Gamma_tab.append(((tab_Y_scale[k] - tab_Z_hat_GS[k][:,:3,3]).T @ (tab_Y_scale[k] - tab_Z_hat_GS[k][:,:3,3]))/len(tab_Y_scale[k]))
        mu0_tab.append(tab_Z_hat_GS[k][0])
        # nb_basis_tab.append(int(np.sqrt(tab_Y_scale[k].shape[0])))

    time_init = time.time()

    with tqdm(total=n_curves) as pbar:
        res = Parallel(n_jobs=n_curves)(delayed(bayesian_CV_optimization_regularization_parameter)(n_CV=n_splits_CV, n_call_bayopt=n_call_bayopt, lambda_bounds=bounds_lambda, grid_obs=tab_grid_arc_s[k], 
                                                                    Y_obs=tab_Y_scale[k], tol=tol_EM, max_iter=max_iter_EM, nb_basis=tab_nb_basis[k], knots=tab_knots[k], 
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

    dic = {"duration":duration, "FS_statespace_tab":FS_statespace_tab, "res_bayopt_tab":res_bayopt_tab, "nb_basis_tab":tab_nb_basis, "mu0_tab":mu0_tab, "Gamma_tab":Gamma_tab, "knots_tab":tab_knots}

    if os.path.isfile(filename):
        print("Le fichier ", filename, " existe déjà.")
        filename = filename + '_bis'
    fil = open(filename,"xb")
    pickle.dump(dic,fil)
    fil.close()

    print('___________________________ End EM ___________________________')

    return 




def estimation_extrinsic_formulas(filename_base, list_Y, n_call_bayopt_der, bounds_h_der, bounds_lbda, n_call_bayopt_theta):
    
    N_curves = len(list_Y)

    time_init = time.time()

    with tqdm(total=N_curves) as pbar:
        res = Parallel(n_jobs=N_curves)(delayed(basis_extrins)(list_Y[k], bounds_h_der, n_call_bayopt_der, bounds_lbda, n_call_bayopt_theta) for k in range(N_curves))
    pbar.update()

    time_end = time.time()
    duration = time_end - time_init
    print('_____________ END Extrins ______________')

    tab_grid_arc_s = []
    tab_Y_scale = []
    tab_smooth_theta_coefs = []
    tab_h_opt = []
    tab_lbda_opt = []
    tab_nb_basis = []
    tab_knots = []
    for k in range(N_curves):
        tab_grid_arc_s.append(res[k][0])
        tab_Y_scale.append(res[k][1])
        if res[k][2] is not None:
            tab_smooth_theta_coefs.append(res[k][2])
            tab_h_opt.append(res[k][3])
            tab_lbda_opt.append(res[k][4])
            tab_knots.append(res[k][5])
            tab_nb_basis.append(res[k][6])
        else:
            tab_smooth_theta_coefs.append(None)
            tab_h_opt.append(None)
            tab_lbda_opt.append(None)
            tab_knots.append(None)
            tab_nb_basis.append(None)

    filename = filename_base

    dic = {"duration":duration, "tab_grid_arc_s":tab_grid_arc_s, "tab_Y_scale":tab_Y_scale, "tab_smooth_theta_coefs":tab_smooth_theta_coefs, 
           "tab_h_opt":tab_h_opt, "tab_lbda_opt":tab_lbda_opt, "tab_knots":tab_knots, "tab_nb_basis":tab_nb_basis}
           
    if os.path.isfile(filename):
        print("Le fichier ", filename, " existe déjà.")
        filename = filename + '_bis'
    fil = open(filename,"xb")
    pickle.dump(dic,fil)
    fil.close()

    return 





def estimation_iteration_Karcher_mean(filename_base, filename_simu, bounds_lbda, n_call_bayopt, tol, max_iter):

    filename = filename_simu 
    fil = open(filename,"rb")
    dic_init = pickle.load(fil)
    fil.close()

    tab_Y_scale = dic_init["tab_Y_scale"]
    tab_Z_hat_GS = dic_init["tab_Z_hat_GS"]
    tab_grid_arc_s = dic_init["tab_grid_arc_s"]
    N_curves = len(tab_Y_scale)
    print(N_curves)

    time_init = time.time()

    res = Parallel(n_jobs=N_curves)(delayed(karcher_mean_smoother)(tab_grid_arc_s[k], tab_Z_hat_GS[k][:,:3,:3], bounds_lbda, n_call_bayopt, tol, max_iter) for k in range(N_curves))

    time_end = time.time()
    duration = time_end - time_init
    
    tab_nb_iter = []
    tab_Q_smooth = []
    tab_smooth_theta_coefs = []
    tab_h_opt = []
    tab_lbda_opt = []
    tab_nb_basis = []
    tab_knots = []
    tab_h_bounds = []
    for k in range(N_curves):
        # if res[k] is not None:
        tab_smooth_theta_coefs.append(res[k][0])
        tab_Q_smooth.append(res[k][1])
        tab_nb_iter.append(res[k][2])
        tab_h_opt.append(res[k][3])
        tab_lbda_opt.append(res[k][4])
        tab_knots.append(res[k][5])
        tab_nb_basis.append(res[k][6])
        tab_h_bounds.append(res[k][7])
        # else:
        #     tab_smooth_theta_coefs.append(None)
        #     tab_h_opt.append(None)
        #     tab_lbda_opt.append(None)
        #     tab_knots.append(None)
        #     tab_nb_basis.append(None)
        #     tab_h_bounds.append(None)
        #     tab_Q_smooth.append(None)
        #     tab_nb_iter.append(None)

    filename = filename_base

    dic = {"duration":duration, "tab_grid_arc_s":tab_grid_arc_s, "tab_smooth_theta_coefs":tab_smooth_theta_coefs,
           "tab_h_opt":tab_h_opt, "tab_lbda_opt":tab_lbda_opt, "tab_knots":tab_knots, "tab_nb_basis":tab_nb_basis, "tab_h_bounds":tab_h_bounds,
           "tab_Q_smooth":tab_Q_smooth, "tab_nb_iter":tab_nb_iter}
           
    if os.path.isfile(filename):
        print("Le fichier ", filename, " existe déjà.")
        filename = filename + '_bis'
    fil = open(filename,"xb")
    pickle.dump(dic,fil)
    fil.close()

    return 



def estimation_iteration_Tracking(filename_base, filename_simu, bounds_lbda, bounds_lbda_track, n_call_bayopt, tol, max_iter):

    filename = filename_simu 
    fil = open(filename,"rb")
    dic_init = pickle.load(fil)
    fil.close()

    tab_Y_scale = dic_init["tab_Y_scale"]
    tab_Z_hat_GS = dic_init["tab_Z_hat_GS"]
    tab_grid_arc_s = dic_init["tab_grid_arc_s"]
    N_curves = len(tab_Y_scale)

    time_init = time.time()


    res = Parallel(n_jobs=N_curves)(delayed(tracking_smoother)(tab_grid_arc_s[k], tab_Z_hat_GS[k][:,:3,:3], bounds_lbda, bounds_lbda_track, n_call_bayopt, tol, max_iter) for k in range(N_curves))


    time_end = time.time()
    duration = time_end - time_init
    
    tab_nb_iter = []
    tab_Q_smooth = []
    tab_smooth_theta_coefs = []
    tab_h_opt = []
    tab_lbda_opt = []
    tab_lbda_track_opt = []
    tab_nb_basis = []
    tab_knots = []
    tab_h_bounds = []
    for k in range(N_curves):
        # if res[k] is not None:
        tab_smooth_theta_coefs.append(res[k][0])
        tab_Q_smooth.append(res[k][1])
        tab_nb_iter.append(res[k][2])
        tab_h_opt.append(res[k][3])
        tab_lbda_opt.append(res[k][4])
        tab_lbda_track_opt.append(res[k][5])
        tab_knots.append(res[k][6])
        tab_nb_basis.append(res[k][7])
        tab_h_bounds.append(res[k][8])
        # else:
        #     tab_smooth_theta_coefs.append(None)
        #     tab_h_opt.append(None)
        #     tab_lbda_opt.append(None)
        #     tab_knots.append(None)
        #     tab_nb_basis.append(None)
        #     tab_h_bounds.append(None)
        #     tab_Q_smooth.append(None)
        #     tab_nb_iter.append(None)
        #     tab_lbda_track_opt.append(None)

    filename = filename_base

    dic = {"duration":duration, "tab_grid_arc_s":tab_grid_arc_s, "tab_smooth_theta_coefs":tab_smooth_theta_coefs, "tab_lbda_track_opt":tab_lbda_track_opt,
           "tab_h_opt":tab_h_opt, "tab_lbda_opt":tab_lbda_opt, "tab_knots":tab_knots, "tab_nb_basis":tab_nb_basis, "tab_h_bounds":tab_h_bounds,
           "tab_Q_smooth":tab_Q_smooth, "tab_nb_iter":tab_nb_iter}
           
    if os.path.isfile(filename):
        print("Le fichier ", filename, " existe déjà.")
        filename = filename + '_bis'
    fil = open(filename,"xb")
    pickle.dump(dic,fil)
    fil.close()

    return 

    



def compute_all_means(pop_x, lbda_bounds, n_call_bayopt=20, sigma=0.0):

    # pop_x est supposé avoir le meme nombre d'observations pour chaque courbe
    # pop_Q est calculer avec méthode GramScmidt
    # pop_theta est calculer avec méthode Least Squares

    n_samples = len(pop_x)
    dim = pop_x[0].shape[1]

    print('computation arc length...') 
    pop_arclgth = np.empty((n_samples), dtype=object)
    pop_L = np.zeros(n_samples)
    for k in range(n_samples):
        grid_time = np.linspace(0,1,len(pop_x[k]))
        h_deriv_bounds = np.array([np.max(grid_time[1:]-grid_time[:-1])*3, np.max(grid_time[1:]-grid_time[:-1])*5])
        derivatives, h_opt = compute_derivatives(pop_x[k], grid_time, deg=3, h=None, CV_optimization_h={"flag":True, "h_grid":h_deriv_bounds, "K":10, "method":'bayesian', "n_call":30, "verbose":False})
        grid_arc_s, L, arc_s, arc_s_dot = compute_arc_length(pop_x[k], grid_time, smooth=True, smoothing_param=h_opt)
        pop_arclgth[k] = grid_arc_s
        pop_L[k] = L 

    concat_grid_arc_s = np.unique(np.round(np.sort(np.concatenate(pop_arclgth)), decimals=3))
    N = len(concat_grid_arc_s)
    grid_time = np.linspace(0,1,N)
    pop_arclgth_reshape = np.zeros((n_samples,N))
    for k in range(n_samples):
        pop_arclgth_reshape[k] = interp1d(np.linspace(0,1,len(pop_x[k])), pop_arclgth[k])(grid_time)

    h_bounds = np.array([np.max((concat_grid_arc_s[1:]-concat_grid_arc_s[:-1])), np.min((np.max((concat_grid_arc_s[1:]-concat_grid_arc_s[:-1]))*8,0.08))])

    print('computation population parameters...') 

    pop_X = np.zeros((n_samples, N, dim))
    pop_x_scale = np.zeros((n_samples, N, dim))
    pop_x_scale_bis = np.empty((len(pop_x)), dtype=object)
    for k in range(n_samples):
        pop_x_scale_bis[k] = pop_x[k]/pop_L[k]
        pop_x_scale[k] = (interp1d(np.linspace(0,1,len(pop_x[k])), pop_x[k].T)(grid_time).T)/pop_L[k]
        pop_X[k] = (interp1d(pop_arclgth[k], pop_x[k].T)(concat_grid_arc_s).T)/pop_L[k] 

    pop_Q = np.zeros((n_samples, N, dim, dim))
    pop_Z = np.zeros((n_samples, N, dim+1, dim+1))
    pop_theta_coefs = np.empty((n_samples), dtype=object)
    pop_theta = np.zeros((n_samples, N, dim-1))

    knots = [concat_grid_arc_s[0]]
    grid_bis = concat_grid_arc_s[1:-1]
    for i in range(0,len(grid_bis),4):
        knots.append(grid_bis[i])
    knots.append(concat_grid_arc_s[-1])
    nb_basis = len(knots)+2

    Bspline_decom = VectorBSplineSmoothing(dim-1, nb_basis, domain_range=(0, 1), order=4, penalization=True, knots=knots)

    for k in range(n_samples):
        GS_orthog = GramSchmidtOrthogonalization(pop_X[k], concat_grid_arc_s, deg=3)
        h_deriv_bounds = np.array([np.max((0.01,np.max(concat_grid_arc_s[1:]-concat_grid_arc_s[:-1]))), np.max((0.05,np.max((concat_grid_arc_s[1:]-concat_grid_arc_s[:-1]))*5))])
        h_opt = GS_orthog.bayesian_optimization_hyperparameters(n_call_bayopt, h_bounds=h_deriv_bounds, verbose=False)
        Z_hat_GS, Q_hat_GS, X_hat_GS = GS_orthog.fit(h_opt) 
        pop_Z[k] = Z_hat_GS
        pop_Q[k] = Q_hat_GS
        
        local_approx_ode = LocalApproxFrenetODE(concat_grid_arc_s, Z=pop_Z[k])
        h_opt, lbda_opt, coefs_opt = local_approx_ode.bayesian_optimization_hyperparameters(n_call_bayopt=n_call_bayopt, lambda_bounds=lbda_bounds, h_bounds=h_bounds, n_splits=10, verbose=False, return_coefs=True, Bspline_repres=Bspline_decom)
        pop_theta_coefs[k] = coefs_opt
        pop_theta[k] = np.squeeze((Bspline_decom.basis_fct(concat_grid_arc_s).T @ coefs_opt).T)
    
    pop_theta_coefs = np.array(pop_theta_coefs)
    mu_Z0 = SE3.frechet_mean(pop_Z[:,0,:,:])

    res_pop = collections.namedtuple('res_pop', ['mu_Z0', 'pop_theta', 'pop_theta_coefs', 'pop_Z', 'pop_X', 'pop_x_scale', 'pop_x_scale_init', 'pop_arclgth', 'pop_arclgth_reshape', 'pop_L', 'concat_grid_arc_s'])
    out_pop = res_pop(mu_Z0, pop_theta, pop_theta_coefs, pop_Z, pop_X, pop_x_scale, pop_x_scale_bis, pop_arclgth, pop_arclgth_reshape, pop_L, concat_grid_arc_s)

    """ arithmetic mean """
    print('computation arithmetic mean...')

    mu_arithm = np.mean(pop_x_scale, axis=0)
    mu_s_arithm, mu_Z_arithm, coefs_opt_arithm, knots_arithm = mean_theta_from_mean_shape(mu_arithm, lbda_bounds, n_call_bayopt, nb_basis=None, knots_step=4)
    # mu_theta_arithm = VectorBSplineSmoothing(2, domain_range=(0, 1), order=4, penalization=False, knots=knots_arithm).evaluate_coefs(coefs_opt_arithm)

    # mu_arithm_arclgth = np.mean(pop_X, axis=0)
    # mu_s_arithm_arclgth, mu_Z_arithm_arclgth, coefs_opt_arithm_arclgth, knots_arithm_arclgth = mean_theta_from_mean_shape(mu_arithm_arclgth, h_deriv_bounds, h_bounds, lbda_bounds, n_call_bayopt, nb_basis=None, knots_step=3)
    # # mu_theta_arithm_arclgth = VectorBSplineSmoothing(2, domain_range=(0, 1), order=4, penalization=False, knots=knots_arithm_arclgth).evaluate_coefs(coefs_opt_arithm_arclgth)

    # res_mean_arithm = collections.namedtuple('res_mean_arithm', ['mu', 'mu_X_arclength', 'mu_s_arclgth', 'mu_s', 'mu_Z', 'mu_Z_arclgth', 'knots_arithm', 'coefs_opt_arithm', 'knots_arithm_arclgth', 'coefs_opt_arithm_arclgth'])
    # out_arithm = res_mean_arithm(mu_arithm, mu_arithm_arclgth, mu_s_arithm_arclgth, mu_s_arithm, mu_Z_arithm, mu_Z_arithm_arclgth, knots_arithm, coefs_opt_arithm, knots_arithm_arclgth, coefs_opt_arithm_arclgth)

    res_mean_arithm = collections.namedtuple('res_mean_arithm', ['mu', 'mu_s', 'mu_Z', 'knots_arithm', 'coefs_opt_arithm'])
    out_arithm = res_mean_arithm(mu_arithm, mu_s_arithm, mu_Z_arithm, knots_arithm, coefs_opt_arithm)

    """ SRVF mean """
    print('computation SRVF mean...')

    mu_srvf = SRVF(3).karcher_mean(pop_x_scale)
    mu_s_srvf, mu_Z_srvf, coefs_opt_srvf, knots_srvf = mean_theta_from_mean_shape(mu_srvf, lbda_bounds, n_call_bayopt, nb_basis=None, knots_step=4)
    # mu_theta_srvf = VectorBSplineSmoothing(2, domain_range=(0, 1), order=4, penalization=False, knots=knots_srvf).evaluate_coefs(coefs_opt_srvf)

    # mu_srvf_arclgth = SRVF(3).karcher_mean(pop_X)
    # mu_s_srvf_arclgth, mu_Z_srvf_arclgth, coefs_opt_srvf_arclgth, knots _srvf_arclgth = mean_theta_from_mean_shape(mu_srvf_arclgth, h_deriv_bounds, h_bounds, lbda_bounds, n_call_bayopt, nb_basis=None, knots_step=3)
    # # mu_theta_srvf_arclgth = VectorBSplineSmoothing(2, domain_range=(0, 1), order=4, penalization=False, knots=knots_srvf_arclgth).evaluate_coefs(coefs_opt_srvf_arclgth)

    # res_mean_SRVF = collections.namedtuple('res_mean_SRVF', ['mu', 'mu_X_arclength', 'mu_s_arclgth', 'mu_s', 'mu_Z', 'mu_Z_arclgth', 'knots_srvf', 'coefs_opt_srvf', 'knots_srvf_arclgth', 'coefs_opt_srvf_arclgth'])
    # out_SRVF = res_mean_SRVF(mu_srvf, mu_srvf_arclgth, mu_s_srvf_arclgth, mu_s_srvf, mu_Z_srvf, mu_Z_srvf_arclgth, knots_srvf, coefs_opt_srvf, knots_srvf_arclgth, coefs_opt_srvf_arclgth)

    res_mean_SRVF = collections.namedtuple('res_mean_SRVF', ['mu', 'mu_s', 'mu_Z', 'knots_srvf', 'coefs_opt_srvf'])
    out_SRVF = res_mean_SRVF(mu_srvf, mu_s_srvf, mu_Z_srvf, knots_srvf, coefs_opt_srvf)

    """ SRC mean """
    print('computation SRC mean...')
  
    mu_SRC, mu_theta_SRC, mu_s_SRC, mu_src_theta, gam_SRC = SRC(3).karcher_mean_bspline(pop_theta_coefs, pop_arclgth_reshape, 0.01, 20, nb_basis=None, lam=1, parallel=True, knots=knots)

    res_mean_SRC = collections.namedtuple('res_mean_SRC', ['mu', 'mu_theta', 'gam', 'mu_arclength', 'mu_src'])
    out_SRC = res_mean_SRC(mu_SRC, mu_theta_SRC, gam_SRC, mu_s_SRC, mu_src_theta)

    """ FC mean """
    print('computation FC mean...')

    mu_FC, mu_theta_FC, gam_mu_FC = Frenet_Curvatures(3).karcher_mean_bspline(pop_theta_coefs, pop_arclgth_reshape, nb_basis=None, knots=knots)

    res_mean_FC = collections.namedtuple('res_mean_FC', ['mu', 'mu_theta', 'gam'])
    out_FC = res_mean_FC(mu_FC, mu_theta_FC, gam_mu_FC)

    """ Stat Mean V1 """
    print('computation Stat Mean V1...')

    statmean_V1 = StatisticalMeanShapeV1(np.array([concat_grid_arc_s for k in range(n_samples)]),  pop_Q)
    h_opt, lbda_opt, coefs_opt = statmean_V1.bayesian_optimization_hyperparameters(n_call_bayopt, lbda_bounds, h_bounds, nb_basis, order=4, n_splits=10, verbose=False, return_coefs=True, knots=None, Bspline_repres=Bspline_decom)
    def mu_theta_V1_func(s):
        if isinstance(s, int) or isinstance(s, float):
            return np.squeeze(Bspline_decom.basis_fct(s).T @ coefs_opt)
        elif isinstance(s, np.ndarray):
            return np.squeeze((Bspline_decom.basis_fct(s).T @ coefs_opt).T)
    mu_theta_V1 = np.squeeze((Bspline_decom.basis_fct(concat_grid_arc_s).T @ coefs_opt).T)
    mu_Z_V1 = solve_FrenetSerret_ODE_SE(mu_theta_V1_func, concat_grid_arc_s, Z0=mu_Z0, timeout_seconds=60)
    mu_V1 = mu_Z_V1[:,:3,3]

    res_mean_V1 = collections.namedtuple('res_mean_V1', ['h_opt', 'lbda_opt', 'mu', 'mu_Z', 'mu_theta', 'coefs_opt'])
    out_V1 = res_mean_V1(h_opt, lbda_opt, mu_V1, mu_Z_V1, mu_theta_V1, coefs_opt)

    """ Stat Mean V2 """
    print('computation Stat Mean V2...')

    statmean_V2 = StatisticalMeanShapeV2(concat_grid_arc_s,  pop_Q)
    h_opt, lbda_opt, coefs_opt = statmean_V2.bayesian_optimization_hyperparameters(n_call_bayopt, lbda_bounds, h_bounds, nb_basis, order=4, n_splits=10, verbose=False, return_coefs=True, knots=None, sigma=sigma, Bspline_repres=Bspline_decom) 
    def mu_theta_V2_func(s):
        if isinstance(s, int) or isinstance(s, float):
            return np.squeeze(Bspline_decom.basis_fct(s).T @ coefs_opt)
        elif isinstance(s, np.ndarray):
            return np.squeeze((Bspline_decom.basis_fct(s).T @ coefs_opt).T)
    mu_theta_V2 =  np.squeeze((Bspline_decom.basis_fct(concat_grid_arc_s).T @ coefs_opt).T)
    mu_Z_V2 = solve_FrenetSerret_ODE_SE(mu_theta_V2_func, concat_grid_arc_s, Z0=mu_Z0, timeout_seconds=60)
    mu_V2 = mu_Z_V2[:,:3,3]

    res_mean_V2 = collections.namedtuple('res_mean_V2', ['h_opt', 'lbda_opt', 'mu', 'mu_Z', 'mu_theta', 'coefs_opt', 'gam', 'results_alignment'])
    out_V2 = res_mean_V2(h_opt, lbda_opt, mu_V2, mu_Z_V2, mu_theta_V2, coefs_opt, statmean_V2.gam, statmean_V2.res_align)

    """ Stat Mean V3 """
    print('computation Stat Mean V3...')

    statmean_V3 = StatisticalMeanShapeV3(concat_grid_arc_s,  pop_Q)
    h_opt, lbda_opt, coefs_opt = statmean_V3.bayesian_optimization_hyperparameters(n_call_bayopt, lbda_bounds, h_bounds, nb_basis, order=4, n_splits=10, verbose=False, return_coefs=True, knots=None, sigma=sigma, Bspline_repres=Bspline_decom) 
    def mu_theta_V3_func(s):
        if isinstance(s, int) or isinstance(s, float):
            return np.squeeze(Bspline_decom.basis_fct(s).T @ coefs_opt)
        elif isinstance(s, np.ndarray):
            return np.squeeze((Bspline_decom.basis_fct(s).T @ coefs_opt).T)
    mu_theta_V3 =  np.squeeze((Bspline_decom.basis_fct(concat_grid_arc_s).T @ coefs_opt).T)
    mu_Z_V3 = solve_FrenetSerret_ODE_SE(mu_theta_V3_func, concat_grid_arc_s, Z0=mu_Z0, timeout_seconds=60)
    mu_V3 = mu_Z_V3[:,:3,3]

    res_mean_V3 = collections.namedtuple('res_mean_V3', ['h_opt', 'lbda_opt', 'mu', 'mu_Z', 'mu_theta', 'coefs_opt', 'gam', 'results_alignment'])
    out_V3 = res_mean_V3(h_opt, lbda_opt, mu_V3, mu_Z_V3, mu_theta_V3, coefs_opt, statmean_V3.gam, statmean_V3.res_align)


    return out_pop, out_arithm, out_SRVF, out_SRC, out_FC, out_V1, out_V2, out_V3



def mean_theta_from_mean_shape(mu_x, lbda_bounds, n_call_bayopt, nb_basis=None, knots_step=None):
    """
        Compute theta et Z pour une courbe moyenne.
    """
    time = np.linspace(0,1,len(mu_x))
    h_deriv_bounds = np.array([np.max((time[1:]-time[:-1]))*3, np.max((time[1:]-time[:-1]))*5])
    derivatives, h_opt = compute_derivatives(mu_x, time, deg=3, h=None, CV_optimization_h={"flag":True, "h_grid":h_deriv_bounds, "K":10, "method":'bayesian', "n_call":30, "verbose":False})
    grid_arc_s, L, arc_s, arc_s_dot = compute_arc_length(mu_x, time, smooth=True, smoothing_param=h_opt)
    mu_x_scale = mu_x/L
    h_deriv_bounds = np.array([np.max((0.01,np.max((grid_arc_s[1:]-grid_arc_s[:-1])))), np.max((0.05,np.max((grid_arc_s[1:]-grid_arc_s[:-1]))*5))])
    GS_orthog = GramSchmidtOrthogonalization(mu_x_scale, grid_arc_s, deg=3)
    h_opt = GS_orthog.bayesian_optimization_hyperparameters(n_call_bayopt, h_bounds=h_deriv_bounds, verbose=False)
    Z_hat_GS, Q_hat_GS, X_hat_GS = GS_orthog.fit(h_opt) 
    local_approx_ode = LocalApproxFrenetODE(grid_arc_s, Z=Z_hat_GS)

    h_bounds = np.array([np.max((grid_arc_s[1:]-grid_arc_s[:-1])), np.min((np.max((grid_arc_s[1:]-grid_arc_s[:-1]))*8,0.08))])

    if knots_step is None:
        h_opt, lbda_opt, coefs_opt = local_approx_ode.bayesian_optimization_hyperparameters(n_call_bayopt=n_call_bayopt, lambda_bounds=lbda_bounds, h_bounds=h_bounds, nb_basis=nb_basis, n_splits=10, verbose=False, return_coefs=True)
        knots = None
    else:
        knots = []
        knots.append(grid_arc_s[0])
        for i in range(1,len(grid_arc_s)-1,knots_step):
            knots.append(grid_arc_s[i])
        knots.append(grid_arc_s[-1])
        h_opt, lbda_opt, coefs_opt = local_approx_ode.bayesian_optimization_hyperparameters(n_call_bayopt=n_call_bayopt, lambda_bounds=lbda_bounds, h_bounds=h_bounds, n_splits=10, verbose=False, return_coefs=True, knots=np.array(knots))
    
    return grid_arc_s, Z_hat_GS, coefs_opt, knots 





def compute_all_means_louper(res_pop, pop_x, lbda_bounds, n_call_bayopt=20, sigma=0.0):

    # pop_x est supposé avoir le meme nombre d'observations pour chaque courbe
    # pop_Q est calculer avec méthode GramScmidt
    # pop_theta est calculer avec méthode Least Squares

    n_samples = len(pop_x)
    dim = pop_x[0].shape[1]
    
    mu_Z0 = res_pop[0]
    pop_theta = res_pop[1]
    pop_theta_coefs = res_pop[2]
    pop_Z = res_pop[3]
    pop_X = res_pop[4]
    pop_x_scale = res_pop[5]
    pop_x_scale_bis = res_pop[6]
    pop_arclgth = res_pop[7]
    pop_arclgth_reshape = res_pop[8]
    pop_L = res_pop[9]
    concat_grid_arc_s = res_pop[10]

    pop_Q = np.zeros((n_samples, N, dim, dim))
    for k in range(n_samples):
        pop_x_scale[k] = centering(pop_x_scale[k])
        pop_Q[k] = pop_Z[k][:,:3,:3]

    N = len(concat_grid_arc_s)
    grid_time = np.linspace(0,1,N)
    h_bounds = np.array([np.max((concat_grid_arc_s[1:]-concat_grid_arc_s[:-1])), np.min((np.max((concat_grid_arc_s[1:]-concat_grid_arc_s[:-1]))*8,0.08))])

    knots = [concat_grid_arc_s[0]]
    grid_bis = concat_grid_arc_s[1:-1]
    for i in range(0,len(grid_bis),4):
        knots.append(grid_bis[i])
    knots.append(concat_grid_arc_s[-1])
    nb_basis = len(knots)+2

    Bspline_decom = VectorBSplineSmoothing(dim-1, nb_basis, domain_range=(0, 1), order=4, penalization=True, knots=knots)

    res_pop = collections.namedtuple('res_pop', ['mu_Z0', 'pop_theta', 'pop_theta_coefs', 'pop_Z', 'pop_X', 'pop_x_scale', 'pop_x_scale_init', 'pop_arclgth', 'pop_arclgth_reshape', 'pop_L', 'concat_grid_arc_s'])
    out_pop = res_pop(mu_Z0, pop_theta, pop_theta_coefs, pop_Z, pop_X, pop_x_scale, pop_x_scale_bis, pop_arclgth, pop_arclgth_reshape, pop_L, concat_grid_arc_s)

    """ arithmetic mean """
    print('computation arithmetic mean...')

    mu_arithm = np.mean(pop_x_scale, axis=0)
    mu_s_arithm, mu_Z_arithm, coefs_opt_arithm, knots_arithm = mean_theta_from_mean_shape(mu_arithm, lbda_bounds, n_call_bayopt, nb_basis=None, knots_step=4)

    res_mean_arithm = collections.namedtuple('res_mean_arithm', ['mu', 'mu_s', 'mu_Z', 'knots_arithm', 'coefs_opt_arithm'])
    out_arithm = res_mean_arithm(mu_arithm, mu_s_arithm, mu_Z_arithm, knots_arithm, coefs_opt_arithm)

    """ SRC mean """
    print('computation SRC mean...')
  
    mu_SRC, mu_theta_SRC, mu_s_SRC, mu_src_theta, gam_SRC = SRC(3).karcher_mean_bspline(pop_theta_coefs, pop_arclgth_reshape, 0.01, 20, nb_basis=None, lam=500, parallel=True, knots=knots)

    res_mean_SRC = collections.namedtuple('res_mean_SRC', ['mu', 'mu_theta', 'gam', 'mu_arclength', 'mu_src'])
    out_SRC = res_mean_SRC(mu_SRC, mu_theta_SRC, gam_SRC, mu_s_SRC, mu_src_theta)


    """ Stat Mean V2 """
    print('computation Stat Mean V2...')

    statmean_V2 = StatisticalMeanShapeV2(concat_grid_arc_s,  pop_Q)
    h_opt, lbda_opt, coefs_opt = statmean_V2.bayesian_optimization_hyperparameters(n_call_bayopt, lbda_bounds, h_bounds, nb_basis, order=4, n_splits=10, verbose=False, return_coefs=True, knots=None, sigma=sigma, Bspline_repres=Bspline_decom) 
    def mu_theta_V2_func(s):
        if isinstance(s, int) or isinstance(s, float):
            return np.squeeze(Bspline_decom.basis_fct(s).T @ coefs_opt)
        elif isinstance(s, np.ndarray):
            return np.squeeze((Bspline_decom.basis_fct(s).T @ coefs_opt).T)
    mu_theta_V2 =  np.squeeze((Bspline_decom.basis_fct(concat_grid_arc_s).T @ coefs_opt).T)
    mu_Z_V2 = solve_FrenetSerret_ODE_SE(mu_theta_V2_func, concat_grid_arc_s, Z0=mu_Z0, timeout_seconds=60)
    mu_V2 = mu_Z_V2[:,:3,3]

    res_mean_V2 = collections.namedtuple('res_mean_V2', ['h_opt', 'lbda_opt', 'mu', 'mu_Z', 'mu_theta', 'coefs_opt', 'gam', 'results_alignment'])
    out_V2 = res_mean_V2(h_opt, lbda_opt, mu_V2, mu_Z_V2, mu_theta_V2, coefs_opt, statmean_V2.gam, statmean_V2.res_align)

    """ Stat Mean V3 """
    print('computation Stat Mean V3...')

    statmean_V3 = StatisticalMeanShapeV3(concat_grid_arc_s,  pop_Q)
    h_opt, lbda_opt, coefs_opt = statmean_V3.bayesian_optimization_hyperparameters(n_call_bayopt, lbda_bounds, h_bounds, nb_basis, order=4, n_splits=10, verbose=False, return_coefs=True, knots=None, sigma=sigma, Bspline_repres=Bspline_decom) 
    def mu_theta_V3_func(s):
        if isinstance(s, int) or isinstance(s, float):
            return np.squeeze(Bspline_decom.basis_fct(s).T @ coefs_opt)
        elif isinstance(s, np.ndarray):
            return np.squeeze((Bspline_decom.basis_fct(s).T @ coefs_opt).T)
    mu_theta_V3 =  np.squeeze((Bspline_decom.basis_fct(concat_grid_arc_s).T @ coefs_opt).T)
    mu_Z_V3 = solve_FrenetSerret_ODE_SE(mu_theta_V3_func, concat_grid_arc_s, Z0=mu_Z0, timeout_seconds=60)
    mu_V3 = mu_Z_V3[:,:3,3]

    res_mean_V3 = collections.namedtuple('res_mean_V3', ['h_opt', 'lbda_opt', 'mu', 'mu_Z', 'mu_theta', 'coefs_opt', 'gam', 'results_alignment'])
    out_V3 = res_mean_V3(h_opt, lbda_opt, mu_V3, mu_Z_V3, mu_theta_V3, coefs_opt, statmean_V3.gam, statmean_V3.res_align)

    return out_pop, out_arithm, out_SRC, out_V2, out_V3
