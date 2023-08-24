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


directory = r"results/EM/"
filename_base = "results/EM/"

current_directory = os.getcwd()
final_directory = os.path.join(current_directory, directory)
if not os.path.exists(final_directory):
   os.makedirs(final_directory)

tol_EM = 0.1
max_iter_EM = 200
n_splits_CV = 5
n_call_bayopt = 25
bounds_lambda = ((1e-09, 1e-05), (1e-09, 1e-05))
sigma_init = 0.15

print(" EM Group 2 ")

group = "group_2"

filename = filename_base + group + "_estimation_EM"

filename_simu = "/home/pchassat/FrenetFDA/Sign_Language/results/trial_01/" + group + "_estimation_GS_leastsquares_theta"

EM_from_init_theta(filename, filename_simu, sigma_init, n_splits_CV, n_call_bayopt, bounds_lambda, tol_EM, max_iter_EM)