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

directory = r"results/"
filename_base = "results/mean_gen_theta_varAmpPhase_rand_tranfo"

current_directory = os.getcwd()
final_directory = os.path.join(current_directory, directory)
if not os.path.exists(final_directory):
   os.makedirs(final_directory)

# curv_ref = lambda s : 1.2*np.exp(3*np.sin(s*5))
# tors_ref = lambda s : 5*(0.3*s*5 - 0.5)
curv_ref = lambda s : np.exp(4*np.sin((s-0.15)*5)+0.25)/3
tors_ref = lambda s : 8*s - 3

n_samples = 20

n_MC = 80
n_call_bayopt = 30
lam = 1.0

filename = "/home/pchassat/FrenetFDA/Simulations_MeanShape/generation_from_theta/param_PhaseAmp_var_0"
fil = open(filename,"rb")
dic_pop = pickle.load(fil)
fil.close()
a_curv = dic_pop["a_curv"]
a_tors = dic_pop["a_tors"]
b = dic_pop["b"]


def theta(s, curv, tors, a_kappa=1, a_tau=1, b=0):
    def warping(t):
        if np.abs(b)<1e-15:
            return t
        else:
            return (np.exp(b*t) - 1)/(np.exp(b) - 1)           
    if isinstance(s, int) or isinstance(s, float):
        return np.array([a_kappa*curv(warping(s)), a_tau*tors(warping(s))])
    elif isinstance(s, np.ndarray):
        return np.vstack((a_kappa*curv(warping(s)), a_tau*tors(warping(s)))).T
    else:
        raise ValueError('Variable is not a float, a int or a NumPy array.')

pop_theta_func = np.array([lambda s, a1=a_curv[k], a2=a_tors[k], a3=b[k]: theta(s, curv_ref, tors_ref, a1, a2, a3) for k in range(n_samples)])


""" _______________________________________________________________ N = 100 _______________________________________________________________ """ 


N = 100

nb_basis = 20
h_bounds = np.array([0.03,0.1])
h_deriv_bounds = np.array([0.05,0.12])
lbda_bounds = np.array([[-15.0,-8.0],[-15.0,-8.0]])

grid = np.linspace(0,1,N)
arclgth = np.linspace(0,1,N)
pop_theta = np.array([theta(arclgth, curv_ref, tors_ref,  a_curv[k], a_tors[k], b[k]) for k in range(n_samples)])
pop_Z = []
for k in range(n_samples):
    Z = solve_FrenetSerret_ODE_SE(theta= lambda s: theta(s, curv_ref, tors_ref, a_curv[k], a_tors[k], b[k]), t_eval=arclgth, Z0=np.eye(4))
    pop_Z.append(Z)

pop_Z = np.array(pop_Z)
pop_Q = pop_Z[:,:,:3,:3]
pop_X = pop_Z[:,:,:3,3]
pop_L = np.ones(n_samples)
pop_x_scale = pop_X
pop_arclgth = np.array([arclgth for i in range(n_samples)])



def h(s,a):
    if np.abs(a)<1e-15:
        return s
    else:
        return (np.sin(2*np.pi*s) + s/a)*a


rand_rot = np.zeros((n_MC, n_samples, 3, 3))
rand_lgth = np.zeros((n_MC, n_samples))
rand_param = np.zeros((n_MC, n_samples, N))
rand_transf_pop_X_scaled = np.zeros((n_MC, n_samples, N, 3))
rand_transf_pop_X = np.zeros((n_MC, n_samples, N, 3))
for k in range(n_MC):
    rand_rot[k] = SO3.random_point_fisher(n_samples, 5, mean_directions=None)
    a_param = np.random.uniform(-0.12, 0.12, n_samples)
    for i in range(n_samples):
        rand_param[k,i] = h(grid, a_param[i])
    rand_lgth[k] = np.random.uniform(1, 10, n_samples)
    for i in range(n_samples):
        rand_transf_pop_X_scaled[k,i] = (rand_rot[k,i] @ interpolate.interp1d(grid, pop_X[i].T)(rand_param[k,i])).T 
        rand_transf_pop_X[k,i] = rand_transf_pop_X_scaled[k,i]*rand_lgth[k,i]



time_init = time.time()
res = Parallel(n_jobs=n_MC)(delayed(compute_all_means)(rand_transf_pop_X[k], h_deriv_bounds, h_bounds, lbda_bounds, nb_basis, n_call_bayopt=n_call_bayopt, sigma=lam) for k in range(n_MC))
time_end = time.time()
duration = time_end - time_init

out_pop, out_arithm, out_srvf, out_SRC, out_FC, out_V1, out_V2, out_V3 = [], [], [], [], [], [], [], []
for k in range(n_MC):
    out_pop.append(res[k][0])
    out_arithm.append(res[k][1])
    out_srvf.append(res[k][2])
    out_SRC.append(res[k][3])
    out_FC.append(res[k][4])
    out_V1.append(res[k][5])
    out_V2.append(res[k][6])
    out_V3.append(res[k][7])

# SAVE
filename = filename_base 
dic = {"duration":duration, "rand_rotations":rand_rot, "rand_lengths":rand_lgth, "rand_arclengths":rand_param , "arr_transf_x_scaled":rand_transf_pop_X_scaled, "arr_transf_x":rand_transf_pop_X, "res_pop":out_pop, "res_arithm":out_arithm, "res_SRVF":out_srvf, "res_SRC":out_SRC, "res_FC":out_FC, "res_V1":out_V1, "res_V2":out_V2, "res_V3":out_V3}

if os.path.isfile(filename):
    print("Le fichier ", filename, " existe déjà.")
    filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()




