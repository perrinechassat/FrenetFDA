import numpy as np
import sys
sys.path.insert(1, '../../')
sys.path.insert(1, '../')
from FrenetFDA.utils.Frenet_Serret_utils import *
import FrenetFDA.utils.visualization as visu
from FrenetFDA.shape_analysis.statistical_mean_shape import StatisticalMeanShapeV1, StatisticalMeanShapeV2, StatisticalMeanShapeV3
from FrenetFDA.processing_Frenet_path.estimate_Frenet_curvatures import LocalApproxFrenetODE
from FrenetFDA.shape_analysis.riemannian_geometries import SRVF, SRC, Frenet_Curvatures
from simulations_utils import *
from FrenetFDA.processing_Euclidean_curve.estimate_Frenet_path import GramSchmidtOrthogonalization
from FrenetFDA.processing_Frenet_path.estimate_Frenet_curvatures import LocalApproxFrenetODE
from FrenetFDA.processing_Euclidean_curve.preprocessing import *
from FrenetFDA.utils.smoothing_utils import *
from pickle import *
import time 
import os.path
import os
import dill as pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def compute_theta_srvf(mu_srvf, h_deriv_bounds, h_bounds, lbda_bounds, n_call_bayopt, nb_basis=None, knots_step=3):
    mu_s_srvf, mu_Z_srvf, coefs_opt_srvf, knots_srvf = mean_theta_from_mean_shape(mu_srvf, h_deriv_bounds, h_bounds, lbda_bounds, n_call_bayopt, nb_basis=None, knots_step=3)
    res_mean_SRVF = collections.namedtuple('res_mean_SRVF', ['mu', 'mu_s', 'mu_Z', 'knots_srvf', 'coefs_opt_srvf'])
    out_SRVF = res_mean_SRVF(mu_srvf, mu_s_srvf, mu_Z_srvf, knots_srvf, coefs_opt_srvf)
    return out_SRVF



directory = r"results/"
filename_base = "results/mean_gen_theta_varAmpPhase_"

current_directory = os.getcwd()
final_directory = os.path.join(current_directory, directory)
if not os.path.exists(final_directory):
   os.makedirs(final_directory)

n_samples = 20

n_MC = 80
n_call_bayopt = 30
lam = 1.0

nb_basis = 20
h_bounds = np.array([0.05,0.15])
h_deriv_bounds = np.array([0.3,0.4])
lbda_bounds = np.array([[-15.0,-5.0],[-15.0,-5.0]])

""" sig_x = 0.01 """

filename = "/home/pchassat/FrenetFDA/Simulations_MeanShape/generation_from_theta/results/mean_gen_theta_varAmpPhase_pop_Arithm_SRVF_with_noise_N_100_sig_01"
fil = open(filename,"rb")
dic_noisy_1 = pickle.load(fil)
fil.close()
res_SRVF_noisy = dic_noisy_1["res_SRVF"]
noisy_means_SRVF = np.array([res_SRVF_noisy[k].mu for k in range(n_MC)])

time_init = time.time()
res = Parallel(n_jobs=n_MC)(delayed(compute_theta_srvf)(noisy_means_SRVF[k], h_deriv_bounds, h_bounds, lbda_bounds, n_call_bayopt, nb_basis=None, knots_step=3) for k in range(n_MC))
time_end = time.time()
duration = time_end - time_init

# SAVE
filename = filename_base + "pop_SRVF_with_noise_N_100_sig_01_recomputed" 
dic = {"duration":duration, "res_SRVF":res}

if os.path.isfile(filename):
    print("Le fichier ", filename, " existe déjà.")
    filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()



# filename = "/home/pchassat/FrenetFDA/Simulations_MeanShape/generation_from_theta/results/mean_gen_theta_varAmpPhase_pop_Arithm_SRVF_with_noise_N_100_sig_005"
# fil = open(filename,"rb")
# dic_noisy_1 = pickle.load(fil)
# fil.close()
# res_SRVF_noisy = dic_noisy_1["res_SRVF"]
# noisy_means_SRVF = np.array([res_SRVF_noisy[k].mu for k in range(n_MC)])

# time_init = time.time()
# res = Parallel(n_jobs=n_MC)(delayed(compute_theta_srvf)(noisy_means_SRVF[k], h_deriv_bounds, h_bounds, lbda_bounds, n_call_bayopt, nb_basis=None, knots_step=3) for k in range(n_MC))
# time_end = time.time()
# duration = time_end - time_init

# # SAVE
# filename = filename_base + "pop_SRVF_with_noise_N_100_sig_005_recomputed" 
# dic = {"duration":duration, "res_SRVF":res}

# if os.path.isfile(filename):
#     print("Le fichier ", filename, " existe déjà.")
#     filename = filename + '_bis'
# fil = open(filename,"xb")
# pickle.dump(dic,fil)
# fil.close()




filename = "/home/pchassat/FrenetFDA/Simulations_MeanShape/generation_from_theta/results/mean_gen_theta_varAmpPhase_with_noise_N_200_sig_01"
fil = open(filename,"rb")
dic_noisy_1 = pickle.load(fil)
fil.close()
res_SRVF_noisy = dic_noisy_1["res_SRVF"]
noisy_means_SRVF = np.array([res_SRVF_noisy[k].mu for k in range(n_MC)])

time_init = time.time()
res = Parallel(n_jobs=n_MC)(delayed(compute_theta_srvf)(noisy_means_SRVF[k], h_deriv_bounds, h_bounds, lbda_bounds, n_call_bayopt, nb_basis=None, knots_step=3) for k in range(n_MC))
time_end = time.time()
duration = time_end - time_init

# SAVE
filename = filename_base + "pop_SRVF_with_noise_N_200_sig_01_recomputed" 
dic = {"duration":duration, "res_SRVF":res}

if os.path.isfile(filename):
    print("Le fichier ", filename, " existe déjà.")
    filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()