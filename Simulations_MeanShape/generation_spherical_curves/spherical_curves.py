import numpy as np
import sys
sys.path.insert(1, '../../')
sys.path.insert(1, '../')
from scipy.interpolate import interp1d
from FrenetFDA.utils.alignment_utils import align_vect_curvatures_fPCA, warp_curvatures, align_vect_SRC_fPCA, align_src
from FrenetFDA.utils.Frenet_Serret_utils import *
import FrenetFDA.utils.visualization as visu
from FrenetFDA.shape_analysis.statistical_mean_shape import StatisticalMeanShapeV1, StatisticalMeanShapeV2, StatisticalMeanShapeV3
from FrenetFDA.processing_Frenet_path.estimate_Frenet_curvatures import LocalApproxFrenetODE
from FrenetFDA.shape_analysis.riemannian_geometries import SRVF, SRC, Frenet_Curvatures
from simulations_utils import compute_all_means, mean_theta_from_mean_shape
from FrenetFDA.processing_Euclidean_curve.estimate_Frenet_path import GramSchmidtOrthogonalization, GramSchmidtOrthogonalizationSphericalCurves
from FrenetFDA.processing_Frenet_path.estimate_Frenet_curvatures import LocalApproxFrenetODE
from FrenetFDA.processing_Euclidean_curve.preprocessing import *
from FrenetFDA.utils.smoothing_utils import *
from spherical_curves_utils import *
from FrenetFDA.utils.Lie_group.SO3_utils import SO3
from simulations_utils import *
from pickle import *
import time 
import os.path
import os
import dill as pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

directory = r"results/"
filename_base = "results/pop_means_"

current_directory = os.getcwd()
final_directory = os.path.join(current_directory, directory)
if not os.path.exists(final_directory):
   os.makedirs(final_directory)

""" Generation des courbes """

n_samples = 20
K = 20
N = 100
grid = np.linspace(0,1,N)
dim = 3
Mu, pop_x, V_tab = generative_model_spherical_curves(n_samples, K, N, (0,1))

""" Ajout random parametrization """

def warping(t, b):
    if np.abs(b)<1e-15:
        return t
    else:
        return (np.exp(b*t) - 1)/(np.exp(b) - 1)   

rand_b = np.random.normal(0,2.5,n_samples)
pop_rand_param_x = np.zeros(pop_x.shape)
for i in range(n_samples):
    pop_rand_param_x[i] = interp1d(grid, pop_x[i].T)(warping(grid, rand_b[i])).T


n_call_bayopt = 30
lam = 1.0
nb_basis = 20
h_bounds = np.array([0.05,0.15])
h_deriv_bounds = np.array([0.1,0.3])
lbda_bounds = np.array([[-15.0,-8.0],[-15.0,-8.0]])


""" _________________ Amplitude and phase variability on theta and WITHOUT noise on x _________________ """

time_init = time.time()
out_arithm, out_srvf, out_SRC, out_FC, out_V1, out_V2, out_V3, out_V1_sphere, out_V2_sphere, out_V3_sphere = compute_all_means_sphere(pop_rand_param_x, h_deriv_bounds, h_bounds, lbda_bounds, nb_basis, n_call_bayopt=n_call_bayopt, sigma=lam)
time_end = time.time()
duration = time_end - time_init

# SAVE
filename = filename_base + "spherical_curves_without_noise" 
dic = {"duration":duration, "pop_x":pop_rand_param_x, "b":rand_b, "res_arithm":out_arithm, "res_SRVF":out_srvf, "res_SRC":out_SRC, "res_FC":out_FC,
        "res_V1":out_V1, "res_V2":out_V2, "res_V3":out_V3, "res_V1_sphere":out_V1_sphere, "res_V2_sphere":out_V2_sphere, "res_V3_sphere":out_V3_sphere}

if os.path.isfile(filename):
    print("Le fichier ", filename, " existe déjà.")
    filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()