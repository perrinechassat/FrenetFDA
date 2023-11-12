import sys
sys.path.insert(1, '../')
sys.path.insert(1, '../../')
sys.path.insert(1, '../../../')
from FrenetFDA.utils.Frenet_Serret_utils import *
from FrenetFDA.utils.smoothing_utils import *
from FrenetFDA.processing_Euclidean_curve.unified_estimate_Frenet_state_space.EM_Frenet_state_space_CV import FrenetStateSpaceCV_global
from FrenetFDA.processing_Euclidean_curve.unified_estimate_Frenet_state_space.iek_filter_smoother_Frenet_state import IEKFilterSmootherFrenetState
from FrenetFDA.processing_Euclidean_curve.preprocessing import *
from FrenetFDA.processing_Euclidean_curve.estimate_Frenet_path import GramSchmidtOrthogonalization, ConstrainedLocalPolynomialRegression
from FrenetFDA.processing_Euclidean_curve.estimate_Frenet_curvatures import ExtrinsicFormulas
from FrenetFDA.processing_Frenet_path.estimate_Frenet_curvatures import ApproxFrenetODE, LocalApproxFrenetODE
import FrenetFDA.utils.visualization as visu
from utils_functions import *
import pickle 
import time 
import os.path
import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

directory = r"results/"
filename_base = "results/"

current_directory = os.getcwd()
final_directory = os.path.join(current_directory, directory)
if not os.path.exists(final_directory):
   os.makedirs(final_directory)

def compute_list_Y_from_group_single_signer(df, group, name):
    list_Y = np.empty((len(group)), dtype=object)
    j = 0
    for gls in group:
        list_traj = []
        n_rep = int(df[(df["glose"]==gls) & (df["name"]==name)]["n_rep"].values[0])
        for k in range(n_rep):
            list_traj.append(df[(df["glose"]==gls) & (df["name"]==name)]["Rwrist"].values[0][k])
        list_Y[j] = list_traj
        j+=1
    return list_Y

# Load data frame of data
df = pd.read_pickle('../Process_Data/LSFtraj_5dis_cutrep.pkl')  
group = ["avoir", 'Dimanche', 'Toujours', 'Autrefois', 'avril', 'train', "avoir l'air", 'femme', 'essayer', 'appeler']
list_Y = compute_list_Y_from_group_single_signer(df, group, 'Thomas')
N_sign = len(list_Y)

n_call_bayopt = 2
lbda_bounds = np.array([[-30,-5],[-30,-5]])
lam = 500

# time_init = time.time()
# res = Parallel(n_jobs=N_sign)(delayed(compute_all_means)(list_Y[k], lbda_bounds, n_call_bayopt=n_call_bayopt, sigma=lam) for k in range(N_sign))
# time_end = time.time()
# duration = time_end - time_init

# out_pop, out_arithm, out_srvf, out_SRC, out_FC, out_V1, out_V2, out_V3 = [], [], [], [], [], [], [], []
# for k in range(N_sign):
#     out_pop.append(res[k][0])
#     out_arithm.append(res[k][1])
#     out_srvf.append(res[k][2])
#     out_SRC.append(res[k][3])
#     out_FC.append(res[k][4])
#     out_V1.append(res[k][5])
#     out_V2.append(res[k][6])
#     out_V3.append(res[k][7])

# # SAVE
# filename = "means_Thomas"
# dic = {"duration":duration, "list_Y":list_Y, "res_pop":out_pop, "res_arithm":out_arithm, "res_SRVF":out_srvf, "res_SRC":out_SRC, "res_FC":out_FC, "res_V1":out_V1, "res_V2":out_V2, "res_V3":out_V3}

# if os.path.isfile(filename):
#     print("Le fichier ", filename, " existe déjà.")
#     filename = filename + '_bis'
# fil = open(filename,"xb")
# pickle.dump(dic,fil)
# fil.close()

# filename = "means_Thomas"
# with open(filename, 'rb') as fichier:
#     dic_init = pickle.load(fichier)
# # fil = open(filename,"rb")
# # dic_init = pickle.load(fil)
# # fil.close()
# res_pop = dic_init["res_pop"]

res_pop = np.load('res_pop_Thomas.npy', allow_pickle=True)

time_init = time.time()
res = Parallel(n_jobs=N_sign)(delayed(compute_all_means_louper)(res_pop[k], list_Y[k], lbda_bounds, n_call_bayopt=n_call_bayopt, sigma=lam) for k in range(N_sign))
time_end = time.time()
duration = time_end - time_init

out_arithm, out_SRC, out_V2, out_V3 = [], [], [], []
for k in range(N_sign):
    out_arithm.append(res[k][0])
    out_SRC.append(res[k][1])
    out_V2.append(res[k][2])
    out_V3.append(res[k][3])

# SAVE
filename = "means_Thomas_correct_test"
dic = {"duration":duration, "list_Y":list_Y, "res_arithm":out_arithm, "res_SRC":out_SRC, "res_V2":out_V2, "res_V3":out_V3}

if os.path.isfile(filename):
    print("Le fichier ", filename, " existe déjà.")
    filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()