import sys
sys.path.insert(1, '../')
from FrenetFDA.utils.Frenet_Serret_utils import *
from FrenetFDA.utils.smoothing_utils import *
from FrenetFDA.processing_Euclidean_curve.preprocessing import *
from pickle import *
from influence_init_theta import EM_from_init_theta
import warnings
warnings.filterwarnings('ignore')
import os.path
import os


directory = r"results/scenario3/influence_init_theta/"
filename_base = "results/scenario3/influence_init_theta/"

current_directory = os.getcwd()
final_directory = os.path.join(current_directory, directory)
if not os.path.exists(final_directory):
   os.makedirs(final_directory)

tol_EM = 0.1
max_iter_EM = 200
n_splits_CV = 5
n_call_bayopt = 25
# bounds_lambda = ((1e-09, 1e-05), (1e-09, 1e-05))
bounds_lambda = ((-9.0, -5.0), (-9.0, -5.0))
sigma_init = 0.1

print(" Influence init, simu 2")

filename = filename_base + "simu_2"

filename_simu = "/home/pchassat/FrenetFDA/Simulations_TwoStepEstimation/results/scenario2/model_04/simu_2"

EM_from_init_theta(filename, filename_simu, sigma_init, n_splits_CV, n_call_bayopt, bounds_lambda, tol_EM, max_iter_EM)
