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
def arc_length_fct(s):
   a = -0.7536625822195512
   return (np.exp(a*s) - 1)/(np.exp(a) - 1)

bounds_lambda = np.array([[1e-09, 1e-03], [1e-09, 1e-03]])
bounds_lambda_track = np.array([1e-04, 1])
bounds_h = np.array([0.05, 0.35])
n_call_bayopt = 60
max_iter = 30
tol = 0.001

directory = r"results/scenario1/model_03/"
filename_base = "results/scenario1/model_03/"

current_directory = os.getcwd()
final_directory = os.path.join(current_directory, directory)
if not os.path.exists(final_directory):
   os.makedirs(final_directory)

filename = filename_base + "model"
dic = {"nb_iterations_simu": n_MC, "mu0": mu0, "theta":theta, "arc_length_fct": arc_length_fct, "tol":tol, "max_iter":max_iter, 
       "bounds_lambda": bounds_lambda, "bounds_h": bounds_h, "n_call_bayopt": n_call_bayopt, "bounds_lambda_track": bounds_lambda_track}
if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()


print(" Scenario 1, simu 1: N=100, alpha=10 ")

N = 100
nb_basis = 10
K = 10**2
filename = filename_base + "simu_1_"
compare_method_with_iteration_parallel(filename, n_MC, theta, arc_length_fct, N, mu0, K, nb_basis, bounds_h, bounds_lambda, bounds_lambda_track, n_call_bayopt, tol, max_iter)