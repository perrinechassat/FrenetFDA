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
from utils_functions import estimation_GS_group


def compute_list_Y_from_group(df, group):
    list_Y = []
    for gls in group:
        n_rep_Thomas = int(df[(df["glose"]==gls) & (df["name"]=='Thomas')]["n_rep"].values[0])
        for k in range(n_rep_Thomas):
            list_Y.append(df[(df["glose"]==gls) & (df["name"]=='Thomas')]["Rwrist"].values[0][k])
        n_rep_Aliza = int(df[(df["glose"]==gls) & (df["name"]=='Aliza')]["n_rep"].values[0])
        for k in range(n_rep_Aliza):
            list_Y.append(df[(df["glose"]==gls) & (df["name"]=='Aliza')]["Rwrist"].values[0][k])
    return list_Y

# Load data frame of data
df = pd.read_pickle('../Process_Data/LSFtraj_5dis_cutrep.pkl')  

group = ['aimer', 'autres', 'bleu', 'boire', 'bonbon', 'chagrin', 'chuchoter', 'cloche'] #93
list_Y = compute_list_Y_from_group(df, group)

directory = r"results/"
filename_base = "results/"
current_directory = os.getcwd()
final_directory = os.path.join(current_directory, directory)
if not os.path.exists(final_directory):
   os.makedirs(final_directory)


n_call_bayopt_der = 30
bounds_lbda = np.array([[-30, -5], [-30, -5]])
n_call_bayopt_theta = 60
bounds_h_der = np.array([0.05, 0.15])


print('Start: Estimation curvatures with GS and Least Squares', '\n')

filename = filename_base + "estimates_GS_least_squares_group_4"
estimation_GS_group(filename, list_Y, n_call_bayopt_der, bounds_h_der, bounds_lbda, n_call_bayopt_theta)

print('End: Estimation curvatures with GS and Least Squares.', '\n')