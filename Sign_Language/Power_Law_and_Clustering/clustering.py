import sys
sys.path.insert(1, '../../')
from FrenetFDA.utils.Frenet_Serret_utils import *
from FrenetFDA.utils.smoothing_utils import *
from FrenetFDA.processing_Euclidean_curve.unified_estimate_Frenet_state_space.EM_Frenet_state_space_CV import FrenetStateSpaceCV_global
from FrenetFDA.processing_Euclidean_curve.unified_estimate_Frenet_state_space.iek_filter_smoother_Frenet_state import IEKFilterSmootherFrenetState
from FrenetFDA.processing_Euclidean_curve.preprocessing import *
from FrenetFDA.processing_Euclidean_curve.estimate_Frenet_path import GramSchmidtOrthogonalization, ConstrainedLocalPolynomialRegression
from FrenetFDA.processing_Euclidean_curve.estimate_Frenet_curvatures import ExtrinsicFormulas
from FrenetFDA.processing_Frenet_path.estimate_Frenet_curvatures import ApproxFrenetODE, LocalApproxFrenetODE
import FrenetFDA.utils.visualization as visu
from FrenetFDA.shape_analysis.riemannian_geometries import SRVF, SRC, Frenet_Curvatures
from pickle import *
import time 
import os.path
import os
import dill as pickle
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np

def compute_list_Y_from_group(df, group):
    list_Y = []
    for gls in group:
        n_rep_Thomas = int(df[(df["glose"]==gls) & (df["name"]=='Thomas')]["n_rep"].values[0])
        for k in range(n_rep_Thomas):
            list_Y.append(df[(df["glose"]==gls) & (df["name"]=='Thomas')]["Rwrist"].values[0][k])
        n_rep_Aliza = int(df[(df["glose"]==gls) & (df["name"]=='Aliza')]["n_rep"].values[0])
        for k in range(n_rep_Aliza):
            list_Y.append(df[(df["glose"]==gls) & (df["name"]=='Aliza')]["Rwrist"].values[0][k])

    ind_group = []
    ind_group.append(np.array([0,int(np.sum(df[df["glose"]==group[0]]["n_rep"].values))]))
    for i in range(len(group)-1):
        ind_group.append(np.array([int(ind_group[i][1]),int(ind_group[i][1]+np.sum(df[df["glose"]==group[i+1]]["n_rep"].values))]))
    return list_Y, ind_group

# Load data frame of data
df = pd.read_pickle('../Process_Data/LSFtraj_5dis_cutrep.pkl')  

group_1 = ["faire l'amour", 'coiffer', 'Dimanche', 'ne pas aimer', 'Toujours', 'Autrefois', 'avril', 'train', "avoir l'air"] #93
# group_2 = ['appeler', 'août', 'avant', 'animal', 'chercher', 'avion', 'besoin', 'asthme', 'bus'] #93
# group_3 = ['bouche bée', 'donner', 'avoir', 'bonjour', 'bouche cousue', 'brosse à dent', 'cerise', 'chaud', 'commander(resto)'] #93
# group_4 = ['aimer', 'autres', 'bleu', 'boire', 'bonbon', 'chagrin', 'chuchoter', 'cloche'] #93
# group_5 = ['eux trois', 'ils', 'commander(qqun)', 'croire', 'ne pas croire', 'curieux', 'jusqu’à', 'il faut'] #, 'elle'] #93
# group_6 = ['après-midi', 'débrouiller', 'éclair', 'enfant', 'désolé', 'décembre', 'jour', 'femme'] #90
# group_7 = ['jamais', 'dire', 'essayer', 'faim', 'je', 'janvier', 'jaune', 'Maman'] #93

list_Y_1, ind_group_1 = compute_list_Y_from_group(df, group_1)
# list_Y_2, ind_group_2 = compute_list_Y_from_group(df, group_2)
# list_Y_3, ind_group_3 = compute_list_Y_from_group(df, group_3)
# list_Y_4, ind_group_4 = compute_list_Y_from_group(df, group_4)
# list_Y_5, ind_group_5 = compute_list_Y_from_group(df, group_5)
# list_Y_6, ind_group_6 = compute_list_Y_from_group(df, group_6)
# list_Y_7, ind_group_7 = compute_list_Y_from_group(df, group_7)


max_n_samples = []
tab_list_Y = [list_Y_1] #, list_Y_2, list_Y_3, list_Y_4, list_Y_5, list_Y_6, list_Y_7]
for k in range(len(tab_list_Y)):
    max_n_samples.append(np.max([len(tab_list_Y[k][i]) for i in range(len(tab_list_Y[k]))]))
N = np.max(max_n_samples)
print(N)

grid_time = np.linspace(0,1,N)

full_grid_arc_s = np.load("results/full_grid_arc_s.npy", allow_pickle=True)
full_theta_scale = np.load("results/full_theta_scale.npy", allow_pickle=True)
full_grid_arc_s_reshape = np.load("results/full_grid_arc_s_reshape.npy", allow_pickle=True)
full_theta = np.load("results/full_theta.npy", allow_pickle=True)
full_x_reshape = np.load("results/full_x_reshape.npy", allow_pickle=True)

# matrices de pairwise distances
n_tot_curves = len(full_x_reshape)
print(n_tot_curves)

mat_SRC_scaled = np.zeros((n_tot_curves, n_tot_curves))
mat_SRC = np.zeros((n_tot_curves, n_tot_curves))
mat_SRVF = np.zeros((n_tot_curves, n_tot_curves))
mat_FC_scaled = np.zeros((n_tot_curves, n_tot_curves))
mat_FC = np.zeros((n_tot_curves, n_tot_curves))
for i in range(n_tot_curves):
    print(i)
    for j in range(i+1, n_tot_curves):

        mat_SRVF[i,j] = SRVF(3).dist(full_x_reshape[i], full_x_reshape[j])
        concact_arc_s = np.unique(np.round(np.sort(np.concatenate([full_grid_arc_s[i], full_grid_arc_s[j]])), decimals=8))
        
        fi = lambda s: interpolate.interp1d(full_grid_arc_s[i], full_theta_scale[i].T)(s).T
        fj = lambda s: interpolate.interp1d(full_grid_arc_s[j], full_theta_scale[j].T)(s).T
        mat_SRC_scaled[i,j] = SRC(3).dist_bis(fi, fj, full_grid_arc_s_reshape[i], full_grid_arc_s_reshape[j], grid_time, lam=100)
        mat_FC_scaled[i,j] = Frenet_Curvatures(3).dist(fi, fj, concact_arc_s)
        
        fi = lambda s: interpolate.interp1d(full_grid_arc_s[i], full_theta[i].T)(s).T
        fj = lambda s: interpolate.interp1d(full_grid_arc_s[j], full_theta[j].T)(s).T
        mat_SRC[i,j] = SRC(3).dist_bis(fi, fj, full_grid_arc_s_reshape[i], full_grid_arc_s_reshape[j], grid_time, lam=100)
        mat_FC[i,j] = Frenet_Curvatures(3).dist(fi, fj, concact_arc_s)

mat_SRC_scaled = mat_SRC_scaled + mat_SRC_scaled.T
mat_SRC = mat_SRC + mat_SRC.T
mat_SRVF = mat_SRVF + mat_SRVF.T 
mat_FC_scaled = mat_FC_scaled + mat_FC_scaled.T 
mat_FC = mat_FC + mat_FC.T


np.save("results/mat_SRC_scaled", mat_SRC_scaled, allow_pickle=True)
np.save("results/mat_SRC", mat_SRC, allow_pickle=True)
np.save("results/mat_SRVF", mat_SRVF, allow_pickle=True)
np.save("results/mat_FC_scaled", mat_FC_scaled, allow_pickle=True)
np.save("results/mat_FC", mat_FC, allow_pickle=True)