import sys
sys.path.insert(1, '../')
from FrenetFDA.utils.Frenet_Serret_utils import *
from FrenetFDA.utils.smoothing_utils import *
from FrenetFDA.processing_Euclidean_curve.unified_estimate_Frenet_state_space.EM_Frenet_state_space_CV import FrenetStateSpaceCV_global, bayesian_CV_optimization_regularization_parameter
from FrenetFDA.processing_Euclidean_curve.unified_estimate_Frenet_state_space.iek_filter_smoother_Frenet_state import IEKFilterSmootherFrenetState
from FrenetFDA.processing_Euclidean_curve.preprocessing import *
from FrenetFDA.processing_Euclidean_curve.estimate_Frenet_path import GramSchmidtOrthogonalization, ConstrainedLocalPolynomialRegression
from FrenetFDA.processing_Euclidean_curve.estimate_Frenet_curvatures import ExtrinsicFormulas
from FrenetFDA.utils.Lie_group.SO3_utils import SO3
from FrenetFDA.processing_Frenet_path.smoothing import KarcherMeanSmoother, TrackingSmootherLinear
from FrenetFDA.processing_Frenet_path.estimate_Frenet_curvatures import TwoStepEstimatorKarcherMean, TwoStepEstimatorTracking
from pickle import *
import time 
import os.path
import os
import dill as pickle
from tqdm import tqdm



def compare_method_with_iteration(theta, arc_length_fct, N, mu0, K, nb_basis, bounds_h, bounds_lbda, bounds_lbda_track, n_call_bayopt, tol, max_iter):

    grid_time = np.linspace(0,1,N)
    arc_length = arc_length_fct(grid_time)
    mu_Z = solve_FrenetSerret_ODE_SE(theta, arc_length, mu0)
    mu_Q = mu_Z[:,:3,:3]
    Q_noisy = SO3.random_point_fisher(len(mu_Q), K, mean_directions=mu_Q)

    # Karcher Mean Smoother
    karcher_mean_smoother = TwoStepEstimatorKarcherMean(arc_length, Q_noisy)
    h_opt, lbda_opt = karcher_mean_smoother.bayesian_optimization_hyperparameters(n_call_bayopt=n_call_bayopt, lambda_bounds=bounds_lbda, h_bounds=bounds_h, nb_basis=nb_basis, epsilon=tol, max_iter=max_iter, n_splits=5, verbose=False)
    time_init = time.time()
    basis_theta_karch, Q_smooth_karch, nb_iter_karch = karcher_mean_smoother.fit(h_opt, lbda_opt, nb_basis=nb_basis, epsilon=tol, max_iter=max_iter)
    time_end = time.time()
    duration_karch = time_end - time_init
    print("fin Karcher mean smoother")

    # Tracking Smoother
    tracking_smoother = TwoStepEstimatorTracking(arc_length, Q_noisy)
    h_opt, lbda_opt, lbda_track_opt = tracking_smoother.bayesian_optimization_hyperparameters(n_call_bayopt=n_call_bayopt, lambda_track_bounds=bounds_lbda_track, lambda_bounds=bounds_lbda, h_bounds=bounds_h, nb_basis=nb_basis, epsilon=tol, max_iter=max_iter, n_splits=5, verbose=False)
    time_init = time.time()
    basis_theta_track, Q_smooth_track, nb_iter_track = tracking_smoother.fit(lbda_track_opt, h_opt, lbda_opt, nb_basis=nb_basis, epsilon=tol, max_iter=max_iter)
    time_end = time.time()
    duration_track = time_end - time_init
    print("fin tracking smoother")

    return Q_noisy, basis_theta_karch, Q_smooth_karch, nb_iter_karch, duration_karch, basis_theta_track, Q_smooth_track, nb_iter_track, duration_track



def init(theta, arc_length_fct, N, mu0, K):

    grid_time = np.linspace(0,1,N)
    arc_length = arc_length_fct(grid_time)
    mu_Z = solve_FrenetSerret_ODE_SE(theta, arc_length, mu0)
    mu_Q = mu_Z[:,:3,:3]
    Q_noisy = SO3.random_point_fisher(len(mu_Q), K, mean_directions=mu_Q)
    
    return Q_noisy


def karcher_mean_smoother(arc_length_fct, N, Q_noisy, nb_basis, bounds_h, bounds_lbda, n_call_bayopt, tol, max_iter):
    try:
        grid_time = np.linspace(0,1,N)
        arc_length = arc_length_fct(grid_time)
        karcher_mean_smoother = TwoStepEstimatorKarcherMean(arc_length, Q_noisy)
        h_opt, lbda_opt = karcher_mean_smoother.bayesian_optimization_hyperparameters(n_call_bayopt=n_call_bayopt, lambda_bounds=bounds_lbda, h_bounds=bounds_h, nb_basis=nb_basis, epsilon=tol, max_iter=max_iter, n_splits=5, verbose=True)
        basis_theta_karch, Q_smooth_karch, nb_iter_karch = karcher_mean_smoother.fit(h_opt, lbda_opt, nb_basis=nb_basis, epsilon=tol, max_iter=max_iter)
        return basis_theta_karch, Q_smooth_karch, nb_iter_karch
    except:
        return None


def tracking_smoother(arc_length_fct, N, Q_noisy, nb_basis, bounds_h, bounds_lbda, bounds_lbda_track, n_call_bayopt, tol, max_iter):
    try:
        grid_time = np.linspace(0,1,N)
        arc_length = arc_length_fct(grid_time)
        tracking_smoother = TwoStepEstimatorTracking(arc_length, Q_noisy)
        h_opt, lbda_opt, lbda_track_opt = tracking_smoother.bayesian_optimization_hyperparameters(n_call_bayopt=n_call_bayopt, lambda_track_bounds=bounds_lbda_track, lambda_bounds=bounds_lbda, h_bounds=bounds_h, nb_basis=nb_basis, epsilon=tol, max_iter=max_iter, n_splits=5, verbose=True)
        basis_theta_track, Q_smooth_track, nb_iter_track = tracking_smoother.fit(lbda_track_opt, h_opt, lbda_opt, nb_basis=nb_basis, epsilon=tol, max_iter=max_iter)
        return basis_theta_track, Q_smooth_track, nb_iter_track
    except:
        return None




def compare_method_with_iteration_parallel(filename_base, n_MC, theta, arc_length_fct, N, mu0, K, nb_basis, bounds_h, bounds_lbda, bounds_lbda_track, n_call_bayopt, tol, max_iter):

    time_init = time.time()

    with tqdm(total=n_MC) as pbar:
        res = Parallel(n_jobs=n_MC)(delayed(init)(theta, arc_length_fct, N, mu0, K) for k in range(n_MC))
    pbar.update()
    print(np.array(res).shape)

    time_end = time.time()
    duration = time_end - time_init

    Q_noisy_tab = np.zeros((n_MC, N, 3, 3))
    for i in range(n_MC):
        Q_noisy_tab[i] = res[i]

    filename = filename_base + "Q_noisy"

    dic = {"Q_noisy_tab":Q_noisy_tab, "duration":duration, "N":N, "alpha":np.sqrt(K), "nb_basis":nb_basis}

    if os.path.isfile(filename):
        print("Le fichier ", filename, " existe déjà.")
        filename = filename + '_bis'
    fil = open(filename,"xb")
    pickle.dump(dic,fil)
    fil.close()

    print('___________________________ End Init ___________________________')

    
    


    time_init = time.time()

    with tqdm(total=n_MC) as pbar:
        res = Parallel(n_jobs=n_MC)(delayed(tracking_smoother)(arc_length_fct, N, Q_noisy_tab[k], nb_basis, bounds_h, bounds_lbda, bounds_lbda_track, n_call_bayopt, tol, max_iter) for k in range(n_MC))
    pbar.update()

    time_end = time.time()
    duration = time_end - time_init

    basis_theta_tab = np.empty((n_MC), dtype=object)
    Q_smooth_tab = np.zeros((n_MC, N, 3, 3))
    nb_iter_tab = np.zeros(n_MC)
    for i in range(n_MC):
        if res[i] is not None:
            basis_theta_tab[i] = res[i][0]
            Q_smooth_tab[i] = res[i][1]
            nb_iter_tab[i] = res[i][2]

    filename = filename_base + "tracking_smoother"

    dic = {"duration":duration, "basis_theta_tab":basis_theta_tab, "Q_smooth_tab":Q_smooth_tab, "nb_iter_tab":nb_iter_tab}

    if os.path.isfile(filename):
        print("Le fichier ", filename, " existe déjà.")
        filename = filename + '_bis'
    fil = open(filename,"xb")
    pickle.dump(dic,fil)
    fil.close()

    print('___________________________ End Tracking ___________________________')
    
    
    

    
    time_init = time.time()

    with tqdm(total=n_MC) as pbar:
        res = Parallel(n_jobs=n_MC)(delayed(karcher_mean_smoother)(arc_length_fct, N, Q_noisy_tab[k], nb_basis, bounds_h, bounds_lbda, n_call_bayopt, tol, max_iter) for k in range(n_MC))
    pbar.update()

    time_end = time.time()
    duration = time_end - time_init

    basis_theta_tab = []
    Q_smooth_tab = np.zeros((n_MC, N, 3, 3))
    nb_iter_tab = np.zeros(n_MC)
    for i in range(n_MC):
        if res[i] is not None:
            basis_theta_tab.append(res[i][0])
            Q_smooth_tab[i] = res[i][1]
            nb_iter_tab[i] = res[i][2]

    filename = filename_base + "karcher_mean_smoother"

    dic = {"duration":duration, "basis_theta_tab":basis_theta_tab, "Q_smooth_tab":Q_smooth_tab, "nb_iter_tab":nb_iter_tab}

    if os.path.isfile(filename):
        print("Le fichier ", filename, " existe déjà.")
        filename = filename + '_bis'
    fil = open(filename,"xb")
    pickle.dump(dic,fil)
    fil.close()

    print('___________________________ End Karcher Mean ___________________________')