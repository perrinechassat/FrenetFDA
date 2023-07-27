import sys
sys.path.insert(1, '../')
from FrenetFDA.utils.Frenet_Serret_utils import *
from FrenetFDA.utils.smoothing_utils import *
from FrenetFDA.processing_Euclidean_curve.unified_estimate_Frenet_state_space.EM_Frenet_state_space_CV import FrenetStateSpaceCV_global, bayesian_CV_optimization_regularization_parameter
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
from compare_smoother import compare_method_with_iteration_parallel, smoother_on_smooth_data
from compare_method_without_iteration import compare_method_without_iteration_parallel
import warnings
warnings.filterwarnings('ignore')

def theta(s):
    curv = lambda s : 2*np.cos(2*np.pi*s) + 5
    tors = lambda s : 2*np.sin(2*np.pi*s) + 1
    if isinstance(s, int) or isinstance(s, float):
        return np.array([curv(s), tors(s)])
    elif isinstance(s, np.ndarray):
        return np.vstack((curv(s), tors(s))).T
    else:
        raise ValueError('Variable is not a float, a int or a NumPy array.')
    

""" SCENARIO 2: observations y_i """

mu0 = np.eye(4) 
P0 = 0.01**2*np.eye(6)
n_MC = 90
bounds_lambda = np.array([[1e-09, 1e-03], [1e-09, 1e-03]])
bounds_h = np.array([0.05, 0.35])
n_call_bayopt = 50
def arc_length_fct(s):
   a = -0.7536625822195512
   return (np.exp(a*s) - 1)/(np.exp(a) - 1)

directory = r"results/scenario2/model_04/"
filename_base = "results/scenario2/model_04/"


print(" Scenario 2, simu 2: N=100, gamma=0.001 ")

N = 100
gamma = 0.001
Gamma = gamma**2*np.eye(3)
nb_basis = 10
filename = filename_base + "simu_1_"

compare_method_without_iteration_parallel(filename, n_MC, theta, arc_length_fct, N, Gamma, mu0, P0, nb_basis, bounds_h, bounds_lambda, n_call_bayopt)
