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
from pickle import *
import time 
import os.path
import os
import dill as pickle
from tqdm import tqdm
 

""" ATTENTION PAS DE SENS SI ON CONNAIT LE PARAMETRE EXACT """


def monte_carlo_KarcherMean_smoother(theta, arc_length_fct, N, mu0, K, n_call_bayopt, h_bounds):

    grid_time = np.linspace(0,1,N)
    arc_length = arc_length_fct(grid_time, np.random.normal(0,1.5))
    mu_Z = solve_FrenetSerret_ODE_SE(theta, arc_length, mu0)
    mu_Q = mu_Z[:,:3,:3]
    Q_noisy = SO3.random_point_fisher(len(mu_Q), K, mean_directions=mu_Q)
    Smoother = KarcherMeanSmoother(grid=arc_length, Q=Q_noisy)
    h_opt = Smoother.bayesian_optimization_hyperparameters(theta, n_call_bayopt, h_bounds, n_splits=10, verbose=True)
    M_smooth = Smoother.fit(h_opt, theta)

    return M_smooth


def monte_carlo_Tracking_smoother(theta, arc_length_fct, N, mu0, K, n_call_bayopt, lbda_bounds):

    grid_time = np.linspace(0,1,N)
    arc_length = arc_length_fct(grid_time, np.random.normal(0,1.5))
    mu_Z = solve_FrenetSerret_ODE_SE(theta, arc_length, mu0)
    mu_Q = mu_Z[:,:3,:3]
    Q_noisy = SO3.random_point_fisher(len(mu_Q), K, mean_directions=mu_Q)
    Smoother = TrackingSmootherLinear(grid=arc_length, Q=Q_noisy)
    lbda_opt = Smoother.bayesian_optimization_hyperparameters(theta, n_call_bayopt, lbda_bounds, n_splits=10, verbose=True)
    M_smooth = Smoother.fit(lbda_opt, theta)

    return M_smooth

