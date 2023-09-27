import sys
sys.path.insert(1, '../')
from FrenetFDA.utils.Frenet_Serret_utils import *
from FrenetFDA.utils.smoothing_utils import *
from FrenetFDA.processing_Euclidean_curve.preprocessing import *
from pickle import *
from influence_init_theta import EM_from_init_theta_grid_search
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
# n_call_bayopt = 25
bounds_lambda = ((1e-09, 1e-05), (1e-09, 1e-05))
sigma_init = 0.1

lambda_grid = np.logspace(-9, -5, 10)
lambda_grids = [lambda_grid, lambda_grid]

print(" Influence init, simu 2")

filename = filename_base + "simu_2"

filename_simu = "/home/pchassat/FrenetFDA/Simulations_TwoStepEstimation/results/scenario2/model_04/simu_2"

EM_from_init_theta_grid_search(filename, filename_simu, sigma_init, n_splits_CV, lambda_grids, tol_EM, max_iter_EM)



# Ensuite, lancer l'EM pour chaque (lbda_1, lbda_2) de la grille pour max_iter fixé et sans seuil et voir la convergence dans chaque cas (sans opti). 
