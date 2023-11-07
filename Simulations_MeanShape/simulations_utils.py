import numpy as np
import sys
sys.path.insert(1, '../')
from scipy.interpolate import interp1d
from FrenetFDA.utils.Frenet_Serret_utils import *
from FrenetFDA.shape_analysis.statistical_mean_shape import StatisticalMeanShapeV1, StatisticalMeanShapeV2, StatisticalMeanShapeV3
from FrenetFDA.shape_analysis.riemannian_geometries import SRVF, SRC, Frenet_Curvatures
from FrenetFDA.processing_Euclidean_curve.preprocessing import *
from FrenetFDA.utils.smoothing_utils import *
from FrenetFDA.processing_Euclidean_curve.estimate_Frenet_path import GramSchmidtOrthogonalization, GramSchmidtOrthogonalizationSphericalCurves
from FrenetFDA.processing_Frenet_path.estimate_Frenet_curvatures import LocalApproxFrenetODE
from FrenetFDA.utils.Lie_group.SE3_utils import SE3
from FrenetFDA.utils.Lie_group.SO3_utils import SO3
import collections


def mean_theta_from_mean_shape(mu_x, h_deriv_bounds, h_bounds, lbda_bounds, n_call_bayopt, nb_basis=None, knots_step=None):
    """
        Compute theta et Z pour une courbe moyenne.
    """
    time = np.linspace(0,1,len(mu_x))
    derivatives, h_opt = compute_derivatives(mu_x, time, deg=3, h=None, CV_optimization_h={"flag":True, "h_grid":h_deriv_bounds, "K":10, "method":'bayesian', "n_call":30, "verbose":False})
    grid_arc_s, L, arc_s, arc_s_dot = compute_arc_length(mu_x, time, smooth=True, smoothing_param=h_opt)
    mu_x_scale = mu_x/L
    GS_orthog = GramSchmidtOrthogonalization(mu_x_scale, grid_arc_s, deg=3)
    h_opt = GS_orthog.bayesian_optimization_hyperparameters(n_call_bayopt, h_bounds=h_deriv_bounds, verbose=False)
    Z_hat_GS, Q_hat_GS, X_hat_GS = GS_orthog.fit(h_opt) 
    local_approx_ode = LocalApproxFrenetODE(grid_arc_s, Z=Z_hat_GS)
    if knots_step is None:
        h_opt, lbda_opt, coefs_opt = local_approx_ode.bayesian_optimization_hyperparameters(n_call_bayopt=n_call_bayopt, lambda_bounds=lbda_bounds, h_bounds=h_bounds, nb_basis=nb_basis, n_splits=10, verbose=False, return_coefs=True)
        knots = None
    else:
        knots = []
        knots.append(grid_arc_s[0])
        for i in range(1,len(grid_arc_s)-1,3):
            knots.append(grid_arc_s[i])
        knots.append(grid_arc_s[-1])
        h_opt, lbda_opt, coefs_opt = local_approx_ode.bayesian_optimization_hyperparameters(n_call_bayopt=n_call_bayopt, lambda_bounds=lbda_bounds, h_bounds=h_bounds, n_splits=10, verbose=False, return_coefs=True, knots=np.array(knots))
    
    return grid_arc_s, Z_hat_GS, coefs_opt, knots


def compute_all_means_known_param(pop_x_scale, pop_Z, pop_theta_func, h_deriv_bounds, h_bounds, lbda_bounds, nb_basis, pop_arclgth, n_call_bayopt=20, sigma=0.0):

    n_samples = pop_Z.shape[0]
    pop_Q = pop_Z[:,:,:3,:3]
    pop_X = pop_Z[:,:,:3,3]
    arclgth = np.linspace(0,1,pop_Z.shape[1])

    Bspline_decom = VectorBSplineSmoothing(2, nb_basis, domain_range=(0, 1), order=4, penalization=False)
    mu_Z0 = SE3.frechet_mean(pop_Z[:,0,:,:])

    """ arithmetic mean """
    print('computation arithmetic mean...')

    time_init = time.time()
    mu_arithm = np.mean(pop_x_scale, axis=0)
    time_end = time.time()
    duration = time_end - time_init
    mu_s_arithm, mu_Z_arithm, coefs_opt, knots = mean_theta_from_mean_shape(mu_arithm, h_deriv_bounds, h_bounds, lbda_bounds, n_call_bayopt, nb_basis=None, knots_step=3)
    mu_theta_arithm = VectorBSplineSmoothing(2, domain_range=(0, 1), order=4, penalization=False, knots=knots).evaluate_coefs(coefs_opt)

    res_mean_arithm = collections.namedtuple('res_mean_arithm', ['mu', 'mu_s', 'mu_Z', 'mu_theta', 'duration'])
    out_arithm = res_mean_arithm(mu_arithm, mu_s_arithm, mu_Z_arithm, mu_theta_arithm, duration)

    """ SRVF mean """
    print('computation SRVF mean...')

    time_init = time.time()
    mu_srvf = SRVF(3).karcher_mean(pop_x_scale)
    time_end = time.time()
    duration = time_end - time_init
    mu_s_srvf, mu_Z_srvf, coefs_opt, knots = mean_theta_from_mean_shape(mu_srvf, h_deriv_bounds, h_bounds, lbda_bounds, n_call_bayopt, nb_basis=None, knots_step=3)
    mu_theta_srvf = VectorBSplineSmoothing(2, domain_range=(0, 1), order=4, penalization=False, knots=knots).evaluate_coefs(coefs_opt)

    res_mean_srvf = collections.namedtuple('res_mean_srvf', ['mu', 'mu_s', 'mu_Z', 'mu_theta', 'duration'])
    out_srvf = res_mean_srvf(mu_srvf, mu_s_srvf, mu_Z_srvf, mu_theta_srvf, duration)

    """ SRC mean """
    print('computation SRC mean...')

    time_init = time.time()
    mu_SRC, mu_theta_SRC, mu_s_SRC, mu_src_theta, gam_SRC = SRC(3).karcher_mean(pop_theta_func, pop_arclgth, 0.01, 20, lam=1, parallel=True)
    time_end = time.time()
    duration = time_end - time_init

    res_mean_SRC = collections.namedtuple('res_mean_SRC', ['mu', 'mu_theta', 'gam', 'mu_arclength', 'mu_src', 'duration'])
    out_SRC = res_mean_SRC(mu_SRC, mu_theta_SRC, gam_SRC, mu_s_SRC, mu_src_theta, duration)

    """ FC mean """
    print('computation FC mean...')

    time_init = time.time()
    mu_FC, mu_theta_FC, gam_mu_FC = Frenet_Curvatures(3).karcher_mean(pop_theta_func, pop_arclgth)
    time_end = time.time()
    duration = time_end - time_init

    res_mean_FC = collections.namedtuple('res_mean_FC', ['mu', 'mu_theta', 'gam', 'duration'])
    out_FC = res_mean_FC(mu_FC, mu_theta_FC, gam_mu_FC, duration)

    """ Stat Mean V1 """
    print('computation Stat Mean V1...')

    time_init = time.time()
    statmean_V1 = StatisticalMeanShapeV1(pop_arclgth,  pop_Q)
    h_opt, lbda_opt, coefs_opt = statmean_V1.bayesian_optimization_hyperparameters(n_call_bayopt, lbda_bounds, h_bounds, nb_basis, order=4, n_splits=10, verbose=False, return_coefs=True, knots=None) #, list_X=pop_X)
    mu_theta_V1_func = Bspline_decom.evaluate_coefs(coefs_opt)
    mu_theta_V1 = mu_theta_V1_func(arclgth)
    mu_Z_V1 = solve_FrenetSerret_ODE_SE(mu_theta_V1_func, arclgth, Z0=mu_Z0)
    mu_V1 = mu_Z_V1[:,:3,3]
    time_end = time.time()
    duration = time_end - time_init

    res_mean_V1 = collections.namedtuple('res_mean_V1', ['h_opt', 'lbda_opt', 'mu', 'mu_Z', 'mu_theta', 'mu_theta_func', 'coefs_opt', 'duration'])
    out_V1 = res_mean_V1(h_opt, lbda_opt, mu_V1, mu_Z_V1, mu_theta_V1, mu_theta_V1_func, coefs_opt, duration)

    """ Stat Mean V2 """
    print('computation Stat Mean V2...')

    time_init = time.time()
    statmean_V2 = StatisticalMeanShapeV2(arclgth,  pop_Q)
    h_opt, lbda_opt, coefs_opt = statmean_V2.bayesian_optimization_hyperparameters(n_call_bayopt, lbda_bounds, h_bounds, nb_basis, order=4, n_splits=10, verbose=False, return_coefs=True, knots=None, sigma=0.001) #, list_X=pop_X)
    mu_theta_V2_func = Bspline_decom.evaluate_coefs(coefs_opt)
    mu_theta_V2 = mu_theta_V2_func(arclgth)
    mu_Z_V2 = solve_FrenetSerret_ODE_SE(mu_theta_V2_func, arclgth, Z0=mu_Z0)
    mu_V2 = mu_Z_V2[:,:3,3]
    time_end = time.time()
    duration = time_end - time_init

    res_mean_V2 = collections.namedtuple('res_mean_V2', ['h_opt', 'lbda_opt', 'mu', 'mu_Z', 'mu_theta', 'mu_theta_func', 'coefs_opt', 'gam', 'results_alignment', 'duration'])
    out_V2 = res_mean_V2(h_opt, lbda_opt, mu_V2, mu_Z_V2, mu_theta_V2, mu_theta_V2_func, coefs_opt, statmean_V2.gam, statmean_V2.res_align, duration)

    """ Stat Mean V3 """
    print('computation Stat Mean V3...')

    time_init = time.time()
    statmean_V3 = StatisticalMeanShapeV3(arclgth,  pop_Q)
    h_opt, lbda_opt, coefs_opt = statmean_V3.bayesian_optimization_hyperparameters(n_call_bayopt, lbda_bounds, h_bounds, nb_basis, order=4, n_splits=10, verbose=False, return_coefs=True, knots=None, sigma=sigma) #, list_X=pop_X)
    mu_theta_V3_func = Bspline_decom.evaluate_coefs(coefs_opt)
    mu_theta_V3 = mu_theta_V3_func(arclgth)
    mu_Z_V3 = solve_FrenetSerret_ODE_SE(mu_theta_V3_func, arclgth, Z0=mu_Z0)
    mu_V3 = mu_Z_V3[:,:3,3]
    time_end = time.time()
    duration = time_end - time_init

    res_mean_V3 = collections.namedtuple('res_mean_V3', ['h_opt', 'lbda_opt', 'mu', 'mu_Z', 'mu_theta', 'mu_theta_func', 'coefs_opt', 'gam', 'results_alignment', 'duration'])
    out_V3 = res_mean_V3(h_opt, lbda_opt, mu_V3, mu_Z_V3, mu_theta_V3, mu_theta_V3_func, coefs_opt, statmean_V3.gam, statmean_V3.res_align, duration)

    return out_arithm, out_srvf, out_SRC, out_FC, out_V1, out_V2, out_V3



def compute_all_means(pop_x, h_deriv_bounds, h_bounds, lbda_bounds, nb_basis, pop_arclgth=None, pop_X=None, n_call_bayopt=20, sigma=0.0):

    # pop_x est supposé avoir le meme nombre d'observations pour chaque courbe
    # pop_Q est calculer avec méthode GramScmidt
    # pop_theta est calculer avec méthode Least Squares
    
    n_samples = len(pop_x)
    N = len(pop_x[0])
    grid_time = np.linspace(0,1,N)
    dim = pop_x[0].shape[1]
    
    if pop_arclgth is None:
        print('computation arc length...') 
        pop_arclgth = np.zeros((n_samples,N))
        pop_L = np.zeros(n_samples)
        for k in range(n_samples):
            derivatives, h_opt = compute_derivatives(pop_x[k], grid_time, deg=3, h=None, CV_optimization_h={"flag":True, "h_grid":h_deriv_bounds, "K":10, "method":'bayesian', "n_call":30, "verbose":False})
            grid_arc_s, L, arc_s, arc_s_dot = compute_arc_length(pop_x[k], grid_time, smooth=True, smoothing_param=h_opt)
            pop_arclgth[k] = grid_arc_s
            pop_L[k] = L 
    else:
        pop_L = pop_arclgth[:,-1]

    print('computation population parameters...') 

    pop_X = np.zeros(np.array(pop_x).shape)
    pop_x_scale = np.zeros(np.array(pop_x).shape)
    for k in range(n_samples):
        pop_x_scale[k] = pop_x[k]/pop_L[k]
        pop_X[k] = interp1d(pop_arclgth[k], pop_x_scale[k].T)(grid_time).T 

    pop_Q = np.zeros((n_samples, N, dim, dim))
    pop_Z = np.zeros((n_samples, N, dim+1, dim+1))
    # pop_theta_fct = np.empty((n_samples), dtype=object)
    pop_theta_coefs = np.empty((n_samples), dtype=object)
    pop_theta = np.zeros((n_samples, N, dim-1))

    Bspline_decom = VectorBSplineSmoothing(dim-1, nb_basis, domain_range=(0, 1), order=4, penalization=True)

    time_init = time.time()

    for k in range(n_samples):
        GS_orthog = GramSchmidtOrthogonalization(pop_X[k], grid_time, deg=3)
        h_opt = GS_orthog.bayesian_optimization_hyperparameters(n_call_bayopt, h_bounds=h_deriv_bounds, verbose=False)
        Z_hat_GS, Q_hat_GS, X_hat_GS = GS_orthog.fit(h_opt) 
        pop_Z[k] = Z_hat_GS
        pop_Q[k] = Q_hat_GS
        # try:
        local_approx_ode = LocalApproxFrenetODE(grid_time, Z=pop_Z[k])
        h_opt, lbda_opt, coefs_opt = local_approx_ode.bayesian_optimization_hyperparameters(n_call_bayopt=n_call_bayopt, lambda_bounds=lbda_bounds, h_bounds=h_bounds, n_splits=10, verbose=False, return_coefs=True, Bspline_repres=Bspline_decom)
        pop_theta_coefs[k] = coefs_opt
        # def func(s):
        #     if isinstance(s, int) or isinstance(s, float):
        #         return np.squeeze(Bspline_decom.basis_fct(s).T @ coefs_opt)
        #     elif isinstance(s, np.ndarray):
        #         return np.squeeze((Bspline_decom.basis_fct(s).T @ coefs_opt).T)
        # pop_theta_fct[k] = func
        pop_theta[k] = np.squeeze((Bspline_decom.basis_fct(grid_time).T @ coefs_opt).T)
        # except:
        #     pop_theta_fct[k] = lambda s: 0*s
        #     pop_theta[k] = np.zeros((N,2))
    
    time_end = time.time()
    duration = time_end - time_init

    pop_theta_coefs = np.array(pop_theta_coefs)
    mu_Z0 = SE3.frechet_mean(pop_Z[:,0,:,:])

    res_pop = collections.namedtuple('res_pop', ['mu_Z0', 'pop_theta', 'pop_theta_coefs', 'pop_Z', 'pop_X', 'pop_x_scale', 'pop_arclgth', 'pop_L', 'duration'])
    out_pop = res_pop(mu_Z0, pop_theta, pop_theta_coefs, pop_Z, pop_X, pop_x_scale, pop_arclgth, pop_L, duration)

    """ arithmetic mean """
    print('computation arithmetic mean...')

    time_init = time.time()
    mu_arithm = np.mean(pop_x_scale, axis=0)
    time_end = time.time()
    duration = time_end - time_init
    mu_s_arithm, mu_Z_arithm, coefs_opt_arithm, knots_arithm = mean_theta_from_mean_shape(mu_arithm, h_deriv_bounds, h_bounds, lbda_bounds, n_call_bayopt, nb_basis=None, knots_step=3)
    # mu_theta_arithm = VectorBSplineSmoothing(2, domain_range=(0, 1), order=4, penalization=False, knots=knots_arithm).evaluate_coefs(coefs_opt_arithm)

    # mu_arithm_arclgth = np.mean(pop_X, axis=0)
    # mu_s_arithm_arclgth, mu_Z_arithm_arclgth, coefs_opt_arithm_arclgth, knots_arithm_arclgth = mean_theta_from_mean_shape(mu_arithm_arclgth, h_deriv_bounds, h_bounds, lbda_bounds, n_call_bayopt, nb_basis=None, knots_step=3)
    # # mu_theta_arithm_arclgth = VectorBSplineSmoothing(2, domain_range=(0, 1), order=4, penalization=False, knots=knots_arithm_arclgth).evaluate_coefs(coefs_opt_arithm_arclgth)

    # res_mean_arithm = collections.namedtuple('res_mean_arithm', ['mu', 'mu_X_arclength', 'mu_s_arclgth', 'mu_s', 'mu_Z', 'mu_Z_arclgth', 'knots_arithm', 'coefs_opt_arithm', 'knots_arithm_arclgth', 'coefs_opt_arithm_arclgth'])
    # out_arithm = res_mean_arithm(mu_arithm, mu_arithm_arclgth, mu_s_arithm_arclgth, mu_s_arithm, mu_Z_arithm, mu_Z_arithm_arclgth, knots_arithm, coefs_opt_arithm, knots_arithm_arclgth, coefs_opt_arithm_arclgth)

    res_mean_arithm = collections.namedtuple('res_mean_arithm', ['mu', 'mu_s', 'mu_Z', 'knots_arithm', 'coefs_opt_arithm', 'duration'])
    out_arithm = res_mean_arithm(mu_arithm, mu_s_arithm, mu_Z_arithm, knots_arithm, coefs_opt_arithm, duration)

    """ SRVF mean """
    print('computation SRVF mean...')

    time_init = time.time()
    mu_srvf = SRVF(3).karcher_mean(pop_x_scale)
    time_end = time.time()
    duration = time_end - time_init
    mu_s_srvf, mu_Z_srvf, coefs_opt_srvf, knots_srvf = mean_theta_from_mean_shape(mu_srvf, h_deriv_bounds, h_bounds, lbda_bounds, n_call_bayopt, nb_basis=None, knots_step=3)
    # mu_theta_srvf = VectorBSplineSmoothing(2, domain_range=(0, 1), order=4, penalization=False, knots=knots_srvf).evaluate_coefs(coefs_opt_srvf)

    # mu_srvf_arclgth = SRVF(3).karcher_mean(pop_X)
    # mu_s_srvf_arclgth, mu_Z_srvf_arclgth, coefs_opt_srvf_arclgth, knots _srvf_arclgth = mean_theta_from_mean_shape(mu_srvf_arclgth, h_deriv_bounds, h_bounds, lbda_bounds, n_call_bayopt, nb_basis=None, knots_step=3)
    # # mu_theta_srvf_arclgth = VectorBSplineSmoothing(2, domain_range=(0, 1), order=4, penalization=False, knots=knots_srvf_arclgth).evaluate_coefs(coefs_opt_srvf_arclgth)

    # res_mean_SRVF = collections.namedtuple('res_mean_SRVF', ['mu', 'mu_X_arclength', 'mu_s_arclgth', 'mu_s', 'mu_Z', 'mu_Z_arclgth', 'knots_srvf', 'coefs_opt_srvf', 'knots_srvf_arclgth', 'coefs_opt_srvf_arclgth'])
    # out_SRVF = res_mean_SRVF(mu_srvf, mu_srvf_arclgth, mu_s_srvf_arclgth, mu_s_srvf, mu_Z_srvf, mu_Z_srvf_arclgth, knots_srvf, coefs_opt_srvf, knots_srvf_arclgth, coefs_opt_srvf_arclgth)

    res_mean_SRVF = collections.namedtuple('res_mean_SRVF', ['mu', 'mu_s', 'mu_Z', 'knots_srvf', 'coefs_opt_srvf', 'duration'])
    out_SRVF = res_mean_SRVF(mu_srvf, mu_s_srvf, mu_Z_srvf, knots_srvf, coefs_opt_srvf, duration)

    """ SRC mean """
    print('computation SRC mean...')
    
    time_init = time.time()
    mu_SRC, mu_theta_SRC, mu_s_SRC, mu_src_theta, gam_SRC = SRC(3).karcher_mean_bspline(pop_theta_coefs, pop_arclgth, 0.01, 20, nb_basis, lam=1, parallel=True)
    time_end = time.time()
    duration = time_end - time_init

    res_mean_SRC = collections.namedtuple('res_mean_SRC', ['mu', 'mu_theta', 'gam', 'mu_arclength', 'mu_src', 'duration'])
    out_SRC = res_mean_SRC(mu_SRC, mu_theta_SRC, gam_SRC, mu_s_SRC, mu_src_theta, duration)

    """ FC mean """
    print('computation FC mean...')

    time_init = time.time()
    mu_FC, mu_theta_FC, gam_mu_FC = Frenet_Curvatures(3).karcher_mean_bspline(pop_theta_coefs, pop_arclgth, nb_basis)
    time_end = time.time()
    duration = time_end - time_init

    res_mean_FC = collections.namedtuple('res_mean_FC', ['mu', 'mu_theta', 'gam', 'duration'])
    out_FC = res_mean_FC(mu_FC, mu_theta_FC, gam_mu_FC, duration)

    """ Stat Mean V1 """
    print('computation Stat Mean V1...')

    time_init = time.time()
    statmean_V1 = StatisticalMeanShapeV1(np.array([grid_time for k in range(n_samples)]),  pop_Q)
    h_opt, lbda_opt, coefs_opt = statmean_V1.bayesian_optimization_hyperparameters(n_call_bayopt, lbda_bounds, h_bounds, nb_basis, order=4, n_splits=10, verbose=False, return_coefs=True, knots=None) #, list_X=pop_X)
    def mu_theta_V1_func(s):
        if isinstance(s, int) or isinstance(s, float):
            return np.squeeze(Bspline_decom.basis_fct(s).T @ coefs_opt)
        elif isinstance(s, np.ndarray):
            return np.squeeze((Bspline_decom.basis_fct(s).T @ coefs_opt).T)
    mu_theta_V1 = np.squeeze((Bspline_decom.basis_fct(grid_time).T @ coefs_opt).T)
    mu_Z_V1 = solve_FrenetSerret_ODE_SE(mu_theta_V1_func, grid_time, Z0=mu_Z0, timeout_seconds=60)
    mu_V1 = mu_Z_V1[:,:3,3]
    time_end = time.time()
    duration = time_end - time_init

    res_mean_V1 = collections.namedtuple('res_mean_V1', ['h_opt', 'lbda_opt', 'mu', 'mu_Z', 'mu_theta', 'coefs_opt', 'duration'])
    out_V1 = res_mean_V1(h_opt, lbda_opt, mu_V1, mu_Z_V1, mu_theta_V1, coefs_opt, duration)

    """ Stat Mean V2 """
    print('computation Stat Mean V2...')

    time_init = time.time()
    statmean_V2 = StatisticalMeanShapeV2(grid_time,  pop_Q)
    h_opt, lbda_opt, coefs_opt = statmean_V2.bayesian_optimization_hyperparameters(n_call_bayopt, lbda_bounds, h_bounds, nb_basis, order=4, n_splits=10, verbose=False, return_coefs=True, knots=None, sigma=sigma) #, list_X=pop_X)
    def mu_theta_V2_func(s):
        if isinstance(s, int) or isinstance(s, float):
            return np.squeeze(Bspline_decom.basis_fct(s).T @ coefs_opt)
        elif isinstance(s, np.ndarray):
            return np.squeeze((Bspline_decom.basis_fct(s).T @ coefs_opt).T)
    mu_theta_V2 =  np.squeeze((Bspline_decom.basis_fct(grid_time).T @ coefs_opt).T)
    mu_Z_V2 = solve_FrenetSerret_ODE_SE(mu_theta_V2_func, grid_time, Z0=mu_Z0, timeout_seconds=60)
    mu_V2 = mu_Z_V2[:,:3,3]
    time_end = time.time()
    duration = time_end - time_init

    res_mean_V2 = collections.namedtuple('res_mean_V2', ['h_opt', 'lbda_opt', 'mu', 'mu_Z', 'mu_theta', 'coefs_opt', 'gam', 'results_alignment', 'duration'])
    out_V2 = res_mean_V2(h_opt, lbda_opt, mu_V2, mu_Z_V2, mu_theta_V2, coefs_opt, statmean_V2.gam, statmean_V2.res_align, duration)

    """ Stat Mean V3 """
    print('computation Stat Mean V3...')

    time_init = time.time()
    statmean_V3 = StatisticalMeanShapeV3(grid_time,  pop_Q)
    h_opt, lbda_opt, coefs_opt = statmean_V3.bayesian_optimization_hyperparameters(n_call_bayopt, lbda_bounds, h_bounds, nb_basis, order=4, n_splits=10, verbose=False, return_coefs=True, knots=None, sigma=sigma) #, list_X=pop_X)
    def mu_theta_V3_func(s):
        if isinstance(s, int) or isinstance(s, float):
            return np.squeeze(Bspline_decom.basis_fct(s).T @ coefs_opt)
        elif isinstance(s, np.ndarray):
            return np.squeeze((Bspline_decom.basis_fct(s).T @ coefs_opt).T)
    mu_theta_V3 =  np.squeeze((Bspline_decom.basis_fct(grid_time).T @ coefs_opt).T)
    mu_Z_V3 = solve_FrenetSerret_ODE_SE(mu_theta_V3_func, grid_time, Z0=mu_Z0, timeout_seconds=60)
    mu_V3 = mu_Z_V3[:,:3,3]
    time_end = time.time()
    duration = time_end - time_init

    res_mean_V3 = collections.namedtuple('res_mean_V3', ['h_opt', 'lbda_opt', 'mu', 'mu_Z', 'mu_theta', 'coefs_opt', 'gam', 'results_alignment', 'duration'])
    out_V3 = res_mean_V3(h_opt, lbda_opt, mu_V3, mu_Z_V3, mu_theta_V3, coefs_opt, statmean_V3.gam, statmean_V3.res_align, duration)


    return out_pop, out_arithm, out_SRVF, out_SRC, out_FC, out_V1, out_V2, out_V3







def compute_all_means_sphere(pop_x, h_deriv_bounds, h_bounds, lbda_bounds, nb_basis, n_call_bayopt=20, sigma=0.0):

    # pop_x est supposé avoir le meme nombre d'observations pour chaque courbe
    # pop_Q est calculer avec méthode GramScmidt
    # pop_theta est calculer avec méthode Least Squares
    
    n_samples = len(pop_x)
    N = len(pop_x[0])
    time = np.linspace(0,1,N)
    dim = pop_x[0].shape[1]
    
    print('computation arc length...') 
    pop_arclgth = np.zeros((n_samples,N))
    pop_L = np.zeros(n_samples)
    for k in range(n_samples):
        derivatives, h_opt = compute_derivatives(pop_x[k], time, deg=3, h=None, CV_optimization_h={"flag":True, "h_grid":h_deriv_bounds, "K":10, "method":'bayesian', "n_call":30, "verbose":False})
        grid_arc_s, L, arc_s, arc_s_dot = compute_arc_length(pop_x[k], time, smooth=True, smoothing_param=h_opt)
        pop_arclgth[k] = grid_arc_s
        pop_L[k] = L 

    print('computation population parameters...') 

    pop_X = np.zeros(np.array(pop_x).shape)
    pop_x_scale = np.zeros(np.array(pop_x).shape)
    for k in range(n_samples):
        pop_x_scale[k] = pop_x[k]/pop_L[k]
        pop_X[k] = interp1d(pop_arclgth[k], pop_x_scale[k].T)(time).T 

    pop_Q = np.zeros((n_samples, N, dim, dim))
    pop_Q_sphere = np.zeros((n_samples, N, dim, dim))
    pop_Z = np.zeros((n_samples, N, dim+1, dim+1))
    pop_theta_coefs = np.empty((n_samples), dtype=object)
    pop_theta = np.zeros((n_samples, N, dim-1))
    pop_kg_coefs = np.empty((n_samples), dtype=object)
    pop_kg = np.zeros((n_samples, N, dim-1))

    Bspline_decom = VectorBSplineSmoothing(dim-1, nb_basis, domain_range=(0, 1), order=4, penalization=True)
    Bspline_decom_sphere = VectorConstantBSplineSmoothing(nb_basis, domain_range=(0, 1), order=4, penalization=True)

    for k in range(n_samples):
        GS_orthog = GramSchmidtOrthogonalization(pop_X[k], time, deg=3)
        h_opt = GS_orthog.bayesian_optimization_hyperparameters(n_call_bayopt, h_bounds=h_deriv_bounds, verbose=False)
        Z_hat_GS, Q_hat_GS, X_hat_GS = GS_orthog.fit(h_opt) 
        pop_Z[k] = Z_hat_GS
        pop_Q[k] = Q_hat_GS
        local_approx_ode = LocalApproxFrenetODE(time, Z=pop_Z[k])
        h_opt, lbda_opt, coefs_opt = local_approx_ode.bayesian_optimization_hyperparameters(n_call_bayopt=n_call_bayopt, lambda_bounds=lbda_bounds, h_bounds=h_bounds, n_splits=10, verbose=False, return_coefs=True, Bspline_repres=Bspline_decom)
        pop_theta_coefs[k] = coefs_opt
        pop_theta[k] = np.squeeze((Bspline_decom.basis_fct(time).T @ coefs_opt).T)

        GS_sphere = GramSchmidtOrthogonalizationSphericalCurves(pop_X[k], time, deg=3)
        h_opt = GS_sphere.bayesian_optimization_hyperparameters(n_call_bayopt, h_bounds=h_deriv_bounds, verbose=False)
        Q_hat_GS, X_hat_GS = GS_sphere.fit(h_opt) 
        pop_Q_sphere[k] = Q_hat_GS
        local_approx_ode = LocalApproxFrenetODE(time, Q=pop_Q_sphere[k])
        h_opt, lbda_opt, coefs_opt = local_approx_ode.bayesian_optimization_hyperparameters(n_call_bayopt=n_call_bayopt, lambda_bounds=lbda_bounds, h_bounds=h_bounds, n_splits=10, verbose=False, return_coefs=True, Bspline_repres=Bspline_decom_sphere)
        pop_kg_coefs[k] = coefs_opt
        pop_kg[k] = np.squeeze((Bspline_decom_sphere.basis_fct(time).T @ coefs_opt).T)
    
    pop_theta_coefs = np.array(pop_theta_coefs)
    pop_kg_coefs = np.array(pop_kg_coefs)
    mu_Z0 = SE3.frechet_mean(pop_Z[:,0,:,:])
    mu_Q0_sphere = SO3.frechet_mean(pop_Q_sphere[:,0,:,:])

    res_pop = collections.namedtuple('res_pop', ['mu_Z0', 'pop_theta', 'pop_theta_coefs', 'pop_Z', 'pop_X', 'pop_x_scale', 'pop_arclgth', 'pop_L', 'pop_Q_sphere', 'pop_kg_coefs', 'pop_kg', 'mu_Q0_sphere'])
    out_pop = res_pop(mu_Z0, pop_theta, pop_theta_coefs, pop_Z, pop_X, pop_x_scale, pop_arclgth, pop_L, pop_Q_sphere, pop_kg_coefs, pop_kg, mu_Q0_sphere)

    """ arithmetic mean """
    print('computation arithmetic mean...')

    mu_arithm = np.mean(pop_x_scale, axis=0)
    mu_s_arithm, mu_Z_arithm, coefs_opt_arithm, knots_arithm = mean_theta_from_mean_shape(mu_arithm, h_deriv_bounds, h_bounds, lbda_bounds, n_call_bayopt, nb_basis=None, knots_step=3)
  
    mu_arithm_arclgth = np.mean(pop_X, axis=0)
    mu_s_arithm_arclgth, mu_Z_arithm_arclgth, coefs_opt_arithm_arclgth, knots_arithm_arclgth = mean_theta_from_mean_shape(mu_arithm_arclgth, h_deriv_bounds, h_bounds, lbda_bounds, n_call_bayopt, nb_basis=None, knots_step=3)
  
    res_mean_arithm = collections.namedtuple('res_mean_arithm', ['mu', 'mu_X_arclength', 'mu_s_arclgth', 'mu_s', 'mu_Z', 'mu_Z_arclgth', 'knots_arithm', 'coefs_opt_arithm', 'knots_arithm_arclgth', 'coefs_opt_arithm_arclgth'])
    out_arithm = res_mean_arithm(mu_arithm, mu_arithm_arclgth, mu_s_arithm_arclgth, mu_s_arithm, mu_Z_arithm, mu_Z_arithm_arclgth, knots_arithm, coefs_opt_arithm, knots_arithm_arclgth, coefs_opt_arithm_arclgth)


    """ SRVF mean """
    print('computation SRVF mean...')

    mu_srvf = SRVF(3).karcher_mean(pop_x_scale)
    mu_s_srvf, mu_Z_srvf, coefs_opt_srvf, knots_srvf = mean_theta_from_mean_shape(mu_srvf, h_deriv_bounds, h_bounds, lbda_bounds, n_call_bayopt, nb_basis=None, knots_step=3)

    mu_srvf_arclgth = SRVF(3).karcher_mean(pop_X)
    mu_s_srvf_arclgth, mu_Z_srvf_arclgth, coefs_opt_srvf_arclgth, knots_srvf_arclgth = mean_theta_from_mean_shape(mu_srvf_arclgth, h_deriv_bounds, h_bounds, lbda_bounds, n_call_bayopt, nb_basis=None, knots_step=3)

    res_mean_SRVF = collections.namedtuple('res_mean_SRVF', ['mu', 'mu_X_arclength', 'mu_s_arclgth', 'mu_s', 'mu_Z', 'mu_Z_arclgth', 'knots_srvf', 'coefs_opt_srvf', 'knots_srvf_arclgth', 'coefs_opt_srvf_arclgth'])
    out_SRVF = res_mean_SRVF(mu_srvf, mu_srvf_arclgth, mu_s_srvf_arclgth, mu_s_srvf, mu_Z_srvf, mu_Z_srvf_arclgth, knots_srvf, coefs_opt_srvf, knots_srvf_arclgth, coefs_opt_srvf_arclgth)

    """ SRC mean """
    print('computation SRC mean...')
  
    mu_SRC, mu_theta_SRC, mu_s_SRC, mu_src_theta, gam_SRC = SRC(3).karcher_mean_bspline(pop_theta_coefs, pop_arclgth, 0.01, 20, nb_basis, lam=1, parallel=True)

    res_mean_SRC = collections.namedtuple('res_mean_SRC', ['mu', 'mu_theta', 'gam', 'mu_arclength', 'mu_src'])
    out_SRC = res_mean_SRC(mu_SRC, mu_theta_SRC, gam_SRC, mu_s_SRC, mu_src_theta)

    """ FC mean """
    print('computation FC mean...')

    mu_FC, mu_theta_FC, gam_mu_FC = Frenet_Curvatures(3).karcher_mean_bspline(pop_theta_coefs, pop_arclgth, nb_basis)

    res_mean_FC = collections.namedtuple('res_mean_FC', ['mu', 'mu_theta', 'gam'])
    out_FC = res_mean_FC(mu_FC, mu_theta_FC, gam_mu_FC)

    """ Stat Mean V1 """
    print('computation Stat Mean V1...')

    statmean_V1 = StatisticalMeanShapeV1(np.array([time for k in range(n_samples)]),  pop_Q)
    h_opt, lbda_opt, coefs_opt = statmean_V1.bayesian_optimization_hyperparameters(n_call_bayopt, lbda_bounds, h_bounds, nb_basis, order=4, n_splits=10, verbose=False, return_coefs=True, knots=None, Bspline_repres=Bspline_decom) #, list_X=pop_X)
    def mu_theta_V1_func(s):
        if isinstance(s, int) or isinstance(s, float):
            return np.squeeze(Bspline_decom.basis_fct(s).T @ coefs_opt)
        elif isinstance(s, np.ndarray):
            return np.squeeze((Bspline_decom.basis_fct(s).T @ coefs_opt).T)
    mu_theta_V1 = np.squeeze((Bspline_decom.basis_fct(time).T @ coefs_opt).T)
    mu_Z_V1 = solve_FrenetSerret_ODE_SE(mu_theta_V1_func, time, Z0=mu_Z0, timeout_seconds=60)
    mu_V1 = mu_Z_V1[:,:3,3]

    res_mean_V1 = collections.namedtuple('res_mean_V1', ['h_opt', 'lbda_opt', 'mu', 'mu_Z', 'mu_theta', 'coefs_opt'])
    out_V1 = res_mean_V1(h_opt, lbda_opt, mu_V1, mu_Z_V1, mu_theta_V1, coefs_opt)

    """ Stat Mean V2 """
    print('computation Stat Mean V2...')

    statmean_V2 = StatisticalMeanShapeV2(time,  pop_Q)
    h_opt, lbda_opt, coefs_opt = statmean_V2.bayesian_optimization_hyperparameters(n_call_bayopt, lbda_bounds, h_bounds, nb_basis, order=4, n_splits=10, verbose=False, return_coefs=True, knots=None, sigma=sigma, Bspline_repres=Bspline_decom) #, list_X=pop_X)
    def mu_theta_V2_func(s):
        if isinstance(s, int) or isinstance(s, float):
            return np.squeeze(Bspline_decom.basis_fct(s).T @ coefs_opt)
        elif isinstance(s, np.ndarray):
            return np.squeeze((Bspline_decom.basis_fct(s).T @ coefs_opt).T)
    mu_theta_V2 =  np.squeeze((Bspline_decom.basis_fct(time).T @ coefs_opt).T)
    mu_Z_V2 = solve_FrenetSerret_ODE_SE(mu_theta_V2_func, time, Z0=mu_Z0, timeout_seconds=60)
    mu_V2 = mu_Z_V2[:,:3,3]

    res_mean_V2 = collections.namedtuple('res_mean_V2', ['h_opt', 'lbda_opt', 'mu', 'mu_Z', 'mu_theta', 'coefs_opt', 'gam', 'results_alignment'])
    out_V2 = res_mean_V2(h_opt, lbda_opt, mu_V2, mu_Z_V2, mu_theta_V2, coefs_opt, statmean_V2.gam, statmean_V2.res_align)

    """ Stat Mean V3 """
    print('computation Stat Mean V3...')

    statmean_V3 = StatisticalMeanShapeV3(time,  pop_Q)
    h_opt, lbda_opt, coefs_opt = statmean_V3.bayesian_optimization_hyperparameters(n_call_bayopt, lbda_bounds, h_bounds, nb_basis, order=4, n_splits=10, verbose=False, return_coefs=True, knots=None, sigma=sigma, Bspline_repres=Bspline_decom) #, list_X=pop_X)
    def mu_theta_V3_func(s):
        if isinstance(s, int) or isinstance(s, float):
            return np.squeeze(Bspline_decom.basis_fct(s).T @ coefs_opt)
        elif isinstance(s, np.ndarray):
            return np.squeeze((Bspline_decom.basis_fct(s).T @ coefs_opt).T)
    mu_theta_V3 =  np.squeeze((Bspline_decom.basis_fct(time).T @ coefs_opt).T)
    mu_Z_V3 = solve_FrenetSerret_ODE_SE(mu_theta_V3_func, time, Z0=mu_Z0, timeout_seconds=60)
    mu_V3 = mu_Z_V3[:,:3,3]

    res_mean_V3 = collections.namedtuple('res_mean_V3', ['h_opt', 'lbda_opt', 'mu', 'mu_Z', 'mu_theta', 'coefs_opt', 'gam', 'results_alignment'])
    out_V3 = res_mean_V3(h_opt, lbda_opt, mu_V3, mu_Z_V3, mu_theta_V3, coefs_opt, statmean_V3.gam, statmean_V3.res_align)


    """ Stat Mean V1 Adapted to sphere """
    print('computation Stat Mean V1 Adapted to sphere...')

    statmean_V1_sphere = StatisticalMeanShapeV1(np.array([time for k in range(n_samples)]),  pop_Q_sphere)
    h_opt, lbda_opt, coefs_opt = statmean_V1_sphere.bayesian_optimization_hyperparameters(n_call_bayopt, lbda_bounds, h_bounds, nb_basis, order=4, n_splits=10, verbose=False, return_coefs=True, knots=None, Bspline_repres=Bspline_decom_sphere) 
    def mu_theta_V1_func_sphere(s):
        if isinstance(s, int) or isinstance(s, float):
            return np.squeeze(Bspline_decom_sphere.basis_fct(s).T @ coefs_opt)
        elif isinstance(s, np.ndarray):
            return np.squeeze((Bspline_decom_sphere.basis_fct(s).T @ coefs_opt).T)
    mu_kg_V1 = np.squeeze((Bspline_decom_sphere.basis_fct(time).T @ coefs_opt).T)
    mu_Q_V1 = solve_FrenetSerret_ODE_SO(mu_theta_V1_func_sphere, time, Q0=mu_Q0_sphere, timeout_seconds=60)
    mu_V1 = mu_Q_V1[:,:3,0]

    res_mean_V1_sphere = collections.namedtuple('res_mean_V1_sphere', ['h_opt', 'lbda_opt', 'mu', 'mu_Q_sphere', 'mu_kg', 'coefs_opt'])
    out_V1_sphere = res_mean_V1_sphere(h_opt, lbda_opt, mu_V1, mu_Q_V1, mu_kg_V1, coefs_opt)

    """ Stat Mean V2 Adapted to sphere """
    print('computation Stat Mean V2 Adapted to sphere...')

    statmean_V2_sphere = StatisticalMeanShapeV2(time,  pop_Q_sphere)
    h_opt, lbda_opt, coefs_opt = statmean_V2_sphere.bayesian_optimization_hyperparameters(n_call_bayopt, lbda_bounds, h_bounds, nb_basis, order=4, n_splits=10, verbose=False, return_coefs=True, knots=None, sigma=sigma, Bspline_repres=Bspline_decom_sphere) 
    def mu_theta_V2_func_sphere(s):
        if isinstance(s, int) or isinstance(s, float):
            return np.squeeze(Bspline_decom_sphere.basis_fct(s).T @ coefs_opt)
        elif isinstance(s, np.ndarray):
            return np.squeeze((Bspline_decom_sphere.basis_fct(s).T @ coefs_opt).T)
    mu_kg_V2 =  np.squeeze((Bspline_decom_sphere.basis_fct(time).T @ coefs_opt).T)
    mu_Q_V2 = solve_FrenetSerret_ODE_SO(mu_theta_V2_func_sphere, time, Q0=mu_Q0_sphere, timeout_seconds=60)
    mu_V2 = mu_Q_V2[:,:3,0]

    res_mean_V2_sphere = collections.namedtuple('res_mean_V2_sphere', ['h_opt', 'lbda_opt', 'mu', 'mu_Q_sphere', 'mu_kg', 'coefs_opt', 'gam', 'results_alignment'])
    out_V2_sphere = res_mean_V2_sphere(h_opt, lbda_opt, mu_V2, mu_Q_V2, mu_kg_V2, coefs_opt, statmean_V2_sphere.gam, statmean_V2_sphere.res_align)

    """ Stat Mean V3 Adapted to sphere """
    print('computation Stat Mean V3 Adapted to sphere...')

    statmean_V3_sphere = StatisticalMeanShapeV3(time,  pop_Q_sphere)
    h_opt, lbda_opt, coefs_opt = statmean_V3_sphere.bayesian_optimization_hyperparameters(n_call_bayopt, lbda_bounds, h_bounds, nb_basis, order=4, n_splits=10, verbose=False, return_coefs=True, knots=None, sigma=sigma, Bspline_repres=Bspline_decom_sphere) 
    def mu_theta_V3_func_sphere(s):
        if isinstance(s, int) or isinstance(s, float):
            return np.squeeze(Bspline_decom_sphere.basis_fct(s).T @ coefs_opt)
        elif isinstance(s, np.ndarray):
            return np.squeeze((Bspline_decom_sphere.basis_fct(s).T @ coefs_opt).T)
    mu_kg_V3 =  np.squeeze((Bspline_decom_sphere.basis_fct(time).T @ coefs_opt).T)
    mu_Q_V3 = solve_FrenetSerret_ODE_SO(mu_theta_V3_func_sphere, time, Q0=mu_Q0_sphere, timeout_seconds=60)
    mu_V3 = mu_Q_V3[:,:3,0]

    res_mean_V3_sphere = collections.namedtuple('res_mean_V3_sphere', ['h_opt', 'lbda_opt', 'mu', 'mu_Q_sphere', 'mu_kg', 'coefs_opt', 'gam', 'results_alignment'])
    out_V3_sphere = res_mean_V3_sphere(h_opt, lbda_opt, mu_V3, mu_Q_V3, mu_kg_V3, coefs_opt, statmean_V3_sphere.gam, statmean_V3_sphere.res_align)

    return out_pop, out_arithm, out_SRVF, out_SRC, out_FC, out_V1, out_V2, out_V3, out_V1_sphere, out_V2_sphere, out_V3_sphere


def add_noise_pop(pop_X, sig_x):
    n_samples = len(pop_X)
    pop_X_noisy = np.zeros(pop_X.shape)
    if sig_x!=0:
        for k in range(n_samples):
            pop_X_noisy[k] = pop_X[k] + np.random.multivariate_normal(np.zeros(pop_X[k].shape[1]), sig_x**2*np.eye(3), size=pop_X[k].shape[0])
    return pop_X_noisy
    






def compute_pop_artihm_SRVF(pop_x, h_deriv_bounds, h_bounds, lbda_bounds, nb_basis, pop_arclgth=None, pop_X=None, n_call_bayopt=20, sigma=0.0):

    # pop_x est supposé avoir le meme nombre d'observations pour chaque courbe
    # pop_Q est calculer avec méthode GramScmidt
    # pop_theta est calculer avec méthode Least Squares
    
    n_samples = len(pop_x)
    N = len(pop_x[0])
    time = np.linspace(0,1,N)
    dim = pop_x[0].shape[1]
    
    if pop_arclgth is None:
        print('computation arc length...') 
        pop_arclgth = np.zeros((n_samples,N))
        pop_L = np.zeros(n_samples)
        for k in range(n_samples):
            derivatives, h_opt = compute_derivatives(pop_x[k], time, deg=3, h=None, CV_optimization_h={"flag":True, "h_grid":h_deriv_bounds, "K":10, "method":'bayesian', "n_call":30, "verbose":False})
            grid_arc_s, L, arc_s, arc_s_dot = compute_arc_length(pop_x[k], time, smooth=True, smoothing_param=h_opt)
            pop_arclgth[k] = grid_arc_s
            pop_L[k] = L 
    else:
        pop_L = pop_arclgth[:,-1]

    print('computation population parameters...') 

    pop_X = np.zeros(np.array(pop_x).shape)
    pop_x_scale = np.zeros(np.array(pop_x).shape)
    for k in range(n_samples):
        pop_x_scale[k] = pop_x[k]/pop_L[k]
        pop_X[k] = interp1d(pop_arclgth[k], pop_x_scale[k].T)(time).T 

    pop_Q = np.zeros((n_samples, N, dim, dim))
    pop_Z = np.zeros((n_samples, N, dim+1, dim+1))
    # pop_theta_fct = np.empty((n_samples), dtype=object)
    pop_theta_coefs = np.empty((n_samples), dtype=object)
    pop_theta = np.zeros((n_samples, N, dim-1))

    Bspline_decom = VectorBSplineSmoothing(dim-1, nb_basis, domain_range=(0, 1), order=4, penalization=True)

    for k in range(n_samples):
        GS_orthog = GramSchmidtOrthogonalization(pop_X[k], time, deg=3)
        h_opt = GS_orthog.bayesian_optimization_hyperparameters(n_call_bayopt, h_bounds=h_deriv_bounds, verbose=False)
        Z_hat_GS, Q_hat_GS, X_hat_GS = GS_orthog.fit(h_opt) 
        pop_Z[k] = Z_hat_GS
        pop_Q[k] = Q_hat_GS
        # try:
        local_approx_ode = LocalApproxFrenetODE(time, Z=pop_Z[k])
        h_opt, lbda_opt, coefs_opt = local_approx_ode.bayesian_optimization_hyperparameters(n_call_bayopt=n_call_bayopt, lambda_bounds=lbda_bounds, h_bounds=h_bounds, n_splits=10, verbose=False, return_coefs=True, Bspline_repres=Bspline_decom)
        pop_theta_coefs[k] = coefs_opt
        # def func(s):
        #     if isinstance(s, int) or isinstance(s, float):
        #         return np.squeeze(Bspline_decom.basis_fct(s).T @ coefs_opt)
        #     elif isinstance(s, np.ndarray):
        #         return np.squeeze((Bspline_decom.basis_fct(s).T @ coefs_opt).T)
        # pop_theta_fct[k] = func
        pop_theta[k] = np.squeeze((Bspline_decom.basis_fct(time).T @ coefs_opt).T)
        # except:
        #     pop_theta_fct[k] = lambda s: 0*s
        #     pop_theta[k] = np.zeros((N,2))

    pop_theta_coefs = np.array(pop_theta_coefs)
    
    mu_Z0 = SE3.frechet_mean(pop_Z[:,0,:,:])

    res_pop = collections.namedtuple('res_pop', ['mu_Z0', 'pop_theta', 'pop_theta_coefs', 'pop_Q', 'pop_Z', 'pop_X', 'pop_x_scale', 'pop_arclgth', 'pop_L'])
    out_pop = res_pop(mu_Z0, pop_theta, pop_theta_coefs, pop_Q, pop_Z, pop_X, pop_x_scale, pop_arclgth, pop_L)

    """ arithmetic mean """
    print('computation arithmetic mean...')

    mu_arithm = np.mean(pop_x_scale, axis=0)
    mu_s_arithm, mu_Z_arithm, coefs_opt_arithm, knots_arithm = mean_theta_from_mean_shape(mu_arithm, h_deriv_bounds, h_bounds, lbda_bounds, n_call_bayopt, nb_basis=None, knots_step=3)
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
    mu_s_srvf, mu_Z_srvf, coefs_opt_srvf, knots_srvf = mean_theta_from_mean_shape(mu_srvf, np.array([0.2,0.4]), np.array([0.05,0.15]), lbda_bounds, n_call_bayopt, nb_basis=None, knots_step=3)
    # mu_theta_srvf = VectorBSplineSmoothing(2, domain_range=(0, 1), order=4, penalization=False, knots=knots_srvf).evaluate_coefs(coefs_opt_srvf)

    # mu_srvf_arclgth = SRVF(3).karcher_mean(pop_X)
    # mu_s_srvf_arclgth, mu_Z_srvf_arclgth, coefs_opt_srvf_arclgth, knots _srvf_arclgth = mean_theta_from_mean_shape(mu_srvf_arclgth, h_deriv_bounds, h_bounds, lbda_bounds, n_call_bayopt, nb_basis=None, knots_step=3)
    # # mu_theta_srvf_arclgth = VectorBSplineSmoothing(2, domain_range=(0, 1), order=4, penalization=False, knots=knots_srvf_arclgth).evaluate_coefs(coefs_opt_srvf_arclgth)

    # res_mean_SRVF = collections.namedtuple('res_mean_SRVF', ['mu', 'mu_X_arclength', 'mu_s_arclgth', 'mu_s', 'mu_Z', 'mu_Z_arclgth', 'knots_srvf', 'coefs_opt_srvf', 'knots_srvf_arclgth', 'coefs_opt_srvf_arclgth'])
    # out_SRVF = res_mean_SRVF(mu_srvf, mu_srvf_arclgth, mu_s_srvf_arclgth, mu_s_srvf, mu_Z_srvf, mu_Z_srvf_arclgth, knots_srvf, coefs_opt_srvf, knots_srvf_arclgth, coefs_opt_srvf_arclgth)

    res_mean_SRVF = collections.namedtuple('res_mean_SRVF', ['mu', 'mu_s', 'mu_Z', 'knots_srvf', 'coefs_opt_srvf'])
    out_SRVF = res_mean_SRVF(mu_srvf, mu_s_srvf, mu_Z_srvf, knots_srvf, coefs_opt_srvf)


    return out_pop, out_arithm, out_SRVF




def compute_SRC_FC_StatMeans(pop_Q, pop_theta_coefs, pop_arclgth, mu_Z0, h_bounds, lbda_bounds, nb_basis, n_call_bayopt=20, sigma=0.0):

    n_samples = len(pop_arclgth)
    N = len(pop_arclgth[0])
    time = np.linspace(0,1,N)
    Bspline_decom = VectorBSplineSmoothing(2, nb_basis, domain_range=(0, 1), order=4, penalization=True)

    """ SRC mean """
    print('computation SRC mean...')
  
    mu_SRC, mu_theta_SRC, mu_s_SRC, mu_src_theta, gam_SRC = SRC(3).karcher_mean_bspline(pop_theta_coefs, pop_arclgth, 0.01, 20, nb_basis, lam=1, parallel=True)

    res_mean_SRC = collections.namedtuple('res_mean_SRC', ['mu', 'mu_theta', 'gam', 'mu_arclength', 'mu_src'])
    out_SRC = res_mean_SRC(mu_SRC, mu_theta_SRC, gam_SRC, mu_s_SRC, mu_src_theta)

    """ FC mean """
    print('computation FC mean...')

    mu_FC, mu_theta_FC, gam_mu_FC = Frenet_Curvatures(3).karcher_mean_bspline(pop_theta_coefs, pop_arclgth, nb_basis)

    res_mean_FC = collections.namedtuple('res_mean_FC', ['mu', 'mu_theta', 'gam'])
    out_FC = res_mean_FC(mu_FC, mu_theta_FC, gam_mu_FC)

    """ Stat Mean V1 """
    print('computation Stat Mean V1...')

    statmean_V1 = StatisticalMeanShapeV1(np.array([time for k in range(n_samples)]),  pop_Q)
    h_opt, lbda_opt, coefs_opt = statmean_V1.bayesian_optimization_hyperparameters(n_call_bayopt, lbda_bounds, h_bounds, nb_basis, order=4, n_splits=10, verbose=False, return_coefs=True, knots=None) #, list_X=pop_X)
    def mu_theta_V1_func(s):
        if isinstance(s, int) or isinstance(s, float):
            return np.squeeze(Bspline_decom.basis_fct(s).T @ coefs_opt)
        elif isinstance(s, np.ndarray):
            return np.squeeze((Bspline_decom.basis_fct(s).T @ coefs_opt).T)
    mu_theta_V1 =  np.squeeze((Bspline_decom.basis_fct(time).T @ coefs_opt).T)
    mu_Z_V1 = solve_FrenetSerret_ODE_SE(mu_theta_V1_func, time, Z0=mu_Z0, timeout_seconds=60)
    mu_V1 = mu_Z_V1[:,:3,3]

    res_mean_V1 = collections.namedtuple('res_mean_V1', ['h_opt', 'lbda_opt', 'mu', 'mu_Z', 'mu_theta', 'coefs_opt'])
    out_V1 = res_mean_V1(h_opt, lbda_opt, mu_V1, mu_Z_V1, mu_theta_V1, coefs_opt)

    """ Stat Mean V2 """
    print('computation Stat Mean V2...')

    statmean_V2 = StatisticalMeanShapeV2(time,  pop_Q)
    h_opt, lbda_opt, coefs_opt = statmean_V2.bayesian_optimization_hyperparameters(n_call_bayopt, lbda_bounds, h_bounds, nb_basis, order=4, n_splits=10, verbose=False, return_coefs=True, knots=None, sigma=sigma) #, list_X=pop_X)
    def mu_theta_V2_func(s):
        if isinstance(s, int) or isinstance(s, float):
            return np.squeeze(Bspline_decom.basis_fct(s).T @ coefs_opt)
        elif isinstance(s, np.ndarray):
            return np.squeeze((Bspline_decom.basis_fct(s).T @ coefs_opt).T)
    mu_theta_V2 =  np.squeeze((Bspline_decom.basis_fct(time).T @ coefs_opt).T)
    mu_Z_V2 = solve_FrenetSerret_ODE_SE(mu_theta_V2_func, time, Z0=mu_Z0, timeout_seconds=60)
    mu_V2 = mu_Z_V2[:,:3,3]

    res_mean_V2 = collections.namedtuple('res_mean_V2', ['h_opt', 'lbda_opt', 'mu', 'mu_Z', 'mu_theta', 'coefs_opt', 'gam', 'results_alignment'])
    out_V2 = res_mean_V2(h_opt, lbda_opt, mu_V2, mu_Z_V2, mu_theta_V2, coefs_opt, statmean_V2.gam, statmean_V2.res_align)

    """ Stat Mean V3 """
    print('computation Stat Mean V3...')

    statmean_V3 = StatisticalMeanShapeV3(time,  pop_Q)
    h_opt, lbda_opt, coefs_opt = statmean_V3.bayesian_optimization_hyperparameters(n_call_bayopt, lbda_bounds, h_bounds, nb_basis, order=4, n_splits=10, verbose=False, return_coefs=True, knots=None, sigma=sigma) #, list_X=pop_X)
    def mu_theta_V3_func(s):
        if isinstance(s, int) or isinstance(s, float):
            return np.squeeze(Bspline_decom.basis_fct(s).T @ coefs_opt)
        elif isinstance(s, np.ndarray):
            return np.squeeze((Bspline_decom.basis_fct(s).T @ coefs_opt).T)
    mu_theta_V3 =  np.squeeze((Bspline_decom.basis_fct(time).T @ coefs_opt).T)
    mu_Z_V3 = solve_FrenetSerret_ODE_SE(mu_theta_V3_func, time, Z0=mu_Z0, timeout_seconds=60)
    mu_V3 = mu_Z_V3[:,:3,3]

    res_mean_V3 = collections.namedtuple('res_mean_V3', ['h_opt', 'lbda_opt', 'mu', 'mu_Z', 'mu_theta', 'coefs_opt', 'gam', 'results_alignment'])
    out_V3 = res_mean_V3(h_opt, lbda_opt, mu_V3, mu_Z_V3, mu_theta_V3, coefs_opt, statmean_V3.gam, statmean_V3.res_align)


    return out_SRC, out_FC, out_V1, out_V2, out_V3
