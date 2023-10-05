import sys
sys.path.insert(1, '../')
from FrenetFDA.utils.Frenet_Serret_utils import *
from FrenetFDA.utils.smoothing_utils import *
from FrenetFDA.processing_Euclidean_curve.preprocessing import *
from pickle import *
from utils_functions import EM_from_init_theta
import warnings
warnings.filterwarnings('ignore')
import os.path
import os


directory = r"results/EM/trial_05/"
filename_base = "results/EM/trial_05/"

current_directory = os.getcwd()
final_directory = os.path.join(current_directory, directory)
if not os.path.exists(final_directory):
   os.makedirs(final_directory)

# tol_EM = 0.1
# max_iter_EM = 200
# n_splits_CV = 5
# n_call_bayopt = 25
# bounds_lambda = ((1e-09, 1e-05), (1e-09, 1e-05))
# sigma_init = 0.1

# print(" EM Group 1 ")

# group = "group_1"

# filename = filename_base + group + "_estimation_EM_0_1"

# filename_simu = "/home/pchassat/FrenetFDA/Sign_Language/results/trial_01/" + group + "_estimations_GS_leastsquares_theta"

# EM_from_init_theta(filename, filename_simu, sigma_init, n_splits_CV, n_call_bayopt, bounds_lambda, tol_EM, max_iter_EM)



tol_EM = 0.1
max_iter_EM = 200
n_splits_CV = 5
n_call_bayopt = 25
# bounds_lambda = ((1e-15, 1e-06), (1e-15, 1e-06))
bounds_lambda = np.array([[-15.0, -6.0], [-15.0, -6.0]]) 
sigma_init = 3

print(" EM Group 1 ")

group = "group_1"

filename = filename_base + group + "_estimation_EM"

filename_simu = "/home/pchassat/FrenetFDA/Sign_Language/results/trial_03/" + group + "_estimations_GS_leastsquares_theta"

EM_from_init_theta(filename, filename_simu, sigma_init, n_splits_CV, n_call_bayopt, bounds_lambda, tol_EM, max_iter_EM)