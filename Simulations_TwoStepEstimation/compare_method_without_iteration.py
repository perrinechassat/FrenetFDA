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
from pickle import *
import time 
import os.path
import os
import dill as pickle
from tqdm import tqdm



def compare_method_without_iteration(theta, arc_length_fct, N, Gamma, mu0, P0, nb_basis, bounds_h, bounds_lbda, n_call_bayopt):

    grid_time = np.linspace(0,1,N)
    arc_length = arc_length_fct(grid_time)

    ## Definition of the states
    xi0 = np.random.multivariate_normal(np.zeros(6), P0)
    Z0 = mu0 @ SE3.exp(-xi0)
    Z = solve_FrenetSerret_ODE_SE(theta, arc_length, Z0=Z0)
    X = Z[:,:3,3]
    Y = X + np.random.multivariate_normal(np.zeros(3), Gamma, size=N)

    try:
        
        ## Init Gamma and s(t)
        derivatives, h_opt = compute_derivatives(Y, grid_time, deg=3, h=None, CV_optimization_h={"flag":True, "h_grid":np.array([bounds_h[0], bounds_h[-1]]), "K":10, "method":'bayesian', "n_call":30, "verbose":False})
        grid_arc_s, L, arc_s, arc_s_dot = compute_arc_length(Y, grid_time, smooth=True, smoothing_param=h_opt)
        print('fin arc length')

        ## Z GramSchmidt
        GS_orthog = GramSchmidtOrthogonalization(Y, grid_arc_s, deg=3)
        h_opt = GS_orthog.bayesian_optimization_hyperparameters(n_call_bayopt, h_bounds=bounds_h, verbose=False)
        Z_hat_GS, Q_hat_GS, X_hat_GS = GS_orthog.fit(h_opt) 
        print('fin Z GramSchmidt')

        ## Z CLP
        CLP_reg = ConstrainedLocalPolynomialRegression(Y, grid_arc_s, adaptative=False, deg_polynomial=3)
        h_opt = CLP_reg.bayesian_optimization_hyperparameters(n_call_bayopt, h_bounds=bounds_h, verbose=False)
        Z_hat_CLP, Q_hat_CLP, X_hat_CLP = CLP_reg.fit(h_opt)
        print('fin Z CLP')
    
        ## theta extrinsic
        extrins_model_theta = ExtrinsicFormulas(Y, grid_time, grid_arc_s, deg_polynomial=3)
        h_opt, lbda_opt = extrins_model_theta.bayesian_optimization_hyperparameters(n_call_bayopt=n_call_bayopt, lambda_bounds=bounds_lbda, h_bounds=bounds_h, nb_basis=nb_basis, n_splits=10, verbose=False)
        Basis_theta_hat_extrins = extrins_model_theta.Bspline_smooth_estimates(h_opt, nb_basis, regularization_parameter=lbda_opt)
        print('fin theta extrinsic')

        ## theta GS + LS
        local_approx_ode = LocalApproxFrenetODE(grid_arc_s, Z=Z_hat_GS)
        h_opt, lbda_opt = local_approx_ode.bayesian_optimization_hyperparameters(n_call_bayopt=n_call_bayopt, lambda_bounds=bounds_lbda, h_bounds=bounds_h, nb_basis=nb_basis, n_splits=10, verbose=False)
        Basis_theta_hat_GS_LS = local_approx_ode.Bspline_smooth_estimates(h_opt, nb_basis, regularization_parameter=lbda_opt)
        print('fin theta GS + LS')

        ## theta CLP + LS
        local_approx_ode = LocalApproxFrenetODE(grid_arc_s, Z=Z_hat_CLP)
        h_opt, lbda_opt = local_approx_ode.bayesian_optimization_hyperparameters(n_call_bayopt=n_call_bayopt, lambda_bounds=bounds_lbda, h_bounds=bounds_h, nb_basis=nb_basis, n_splits=10, verbose=False)
        Basis_theta_hat_CLP_LS = local_approx_ode.Bspline_smooth_estimates(h_opt, nb_basis, regularization_parameter=lbda_opt)
        print('fin theta CLP + LS')
    
        return Y, Z, Z_hat_GS, Z_hat_CLP, Basis_theta_hat_extrins, Basis_theta_hat_GS_LS, Basis_theta_hat_CLP_LS
    
    except:

        return None