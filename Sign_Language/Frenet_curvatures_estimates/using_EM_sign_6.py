import sys
sys.path.insert(1, '../../')
sys.path.insert(1, '../')
from FrenetFDA.utils.Frenet_Serret_utils import *
from FrenetFDA.utils.smoothing_utils import *
from FrenetFDA.processing_Euclidean_curve.unified_estimate_Frenet_state_space.EM_Frenet_state_space_CV import FrenetStateSpaceCV_global
from FrenetFDA.processing_Euclidean_curve.unified_estimate_Frenet_state_space.iek_filter_smoother_Frenet_state import IEKFilterSmootherFrenetState
from FrenetFDA.processing_Euclidean_curve.preprocessing import *
from FrenetFDA.processing_Euclidean_curve.estimate_Frenet_path import GramSchmidtOrthogonalization, ConstrainedLocalPolynomialRegression
from FrenetFDA.processing_Euclidean_curve.estimate_Frenet_curvatures import ExtrinsicFormulas
from FrenetFDA.processing_Frenet_path.estimate_Frenet_curvatures import ApproxFrenetODE, LocalApproxFrenetODE
import FrenetFDA.utils.visualization as visu
from pickle import *
import time 
import os.path
import os
import dill as pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
from utils_functions import EM_from_init_theta, EM_from_init_theta_single_group


def compute_list_Y_from_group(df, group):
    list_Y = []
    for gls in group:
        n_rep_Thomas = int(df[(df["glose"]==gls) & (df["name"]=='Thomas')]["n_rep"].values[0])
        for k in range(n_rep_Thomas):
            list_Y.append(df[(df["glose"]==gls) & (df["name"]=='Thomas')]["Rwrist"].values[0][k])
    return list_Y

# Load data frame of data
df = pd.read_pickle('../Process_Data/LSFtraj_5dis_cutrep.pkl')  

group = ["faire l'amour", 'Dimanche', 'Toujours', 'Autrefois', 'avril', 'train', "avoir l'air", 'femme']
# list_Y = compute_list_Y_from_group(df, group)

ind_group = []
ind_group.append(np.array([0,int(np.sum(df[(df["glose"]==group[0]) & (df["name"]=='Thomas')]["n_rep"].values))]))
for i in range(len(group)-1):
    ind_group.append(np.array([int(ind_group[i][1]),int(ind_group[i][1]+np.sum(df[(df["glose"]==group[i+1]) & (df["name"]=='Thomas')]["n_rep"].values))]))


directory = r"results/"
filename_base = "results/"
current_directory = os.getcwd()
final_directory = os.path.join(current_directory, directory)
if not os.path.exists(final_directory):
   os.makedirs(final_directory)


tol_EM = 0.1
max_iter_EM = 200
n_splits_CV = 5
n_call_bayopt = 30
bounds_lambda = np.array([[-30.0, -5.0], [-30.0, -5.0]])
sigma_init = 2.5

indice = 6

print('Start: Estimation curvatures with EM', '\n')

filename_simu = "/home/pchassat/FrenetFDA/Sign_Language/Frenet_curvatures_estimates/results/estimates_GS_least_squares_theta_bis"
filename = filename_base + "estimates_EM_" + str(indice)
EM_from_init_theta_single_group(filename, filename_simu, ind_group[indice], sigma_init, n_splits_CV, n_call_bayopt, bounds_lambda, tol_EM, max_iter_EM)

print('End: Estimation curvatures with EM.', '\n')



