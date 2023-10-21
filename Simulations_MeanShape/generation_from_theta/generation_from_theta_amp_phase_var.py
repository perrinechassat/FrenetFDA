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
filename_base = "results/mean_gen_theta_varAmpPhase_"

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

# sig_curv, sig_tors, sig_warp = 0.2, 1.5, 2
# a_curv = np.random.normal(1,sig_curv,n_samples)
# a_tors = np.random.normal(0,sig_tors,n_samples)
# b = np.random.normal(0,sig_warp,n_samples)

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


# """ _________________ Amplitude and phase variability on theta and WITHOUT noise on x _________________ """

# time_init = time.time()
# out_arithm, out_srvf, out_SRC, out_FC, out_V1, out_V2, out_V3 = compute_all_means_known_param(pop_x_scale, pop_Z, pop_theta_func, h_deriv_bounds, h_bounds, lbda_bounds, nb_basis, pop_arclgth, n_call_bayopt=n_call_bayopt, sigma=lam)
# time_end = time.time()
# duration = time_end - time_init

# # SAVE
# filename = filename_base + "without_noise_N_100" 
# dic = {"duration":duration, "res_arithm":out_arithm, "res_SRVF":out_srvf, "res_SRC":out_SRC, "res_FC":out_FC, "res_V1":out_V1, "res_V2":out_V2, "res_V3":out_V3}

# if os.path.isfile(filename):
#     print("Le fichier ", filename, " existe déjà.")
#     filename = filename + '_bis'
# fil = open(filename,"xb")
# pickle.dump(dic,fil)
# fil.close()



""" _________________ Amplitude and phase variability on theta and with noise on x _________________ """


nb_basis = 20
h_bounds = np.array([0.03,0.15])
h_deriv_bounds = np.array([0.1,0.3])
lbda_bounds = np.array([[-15.0,-5.0],[-15.0,-5.0]])

""" sig_x = 0.01 """
sig_x = 0.01

arr_noisy_x = np.zeros((n_MC, n_samples, N, 3))
for k in range(n_MC):
    arr_noisy_x[k] = add_noise_pop(pop_X, sig_x)


time_init = time.time()
res = Parallel(n_jobs=n_MC)(delayed(compute_pop_artihm_SRVF)(arr_noisy_x[k], h_deriv_bounds, h_bounds, lbda_bounds, nb_basis, n_call_bayopt=n_call_bayopt, sigma=lam) for k in range(n_MC))
time_end = time.time()
duration = time_end - time_init

# Bspline_decom = VectorBSplineSmoothing(2, nb_basis, domain_range=(0, 1), order=4, penalization=False)
out_pop, out_arithm, out_srvf = [], [], []
# pop_theta_fct = np.empty((n_MC, n_samples), dtype=object)
for k in range(n_MC):
    out_pop.append(res[k][0])
    out_arithm.append(res[k][1])
    out_srvf.append(res[k][2])
    # for i in range(n_samples):
    #     pop_theta_fct[k][i] = Bspline_decom.evaluate_coefs(out_pop[k].pop_theta_coefs[i])

# SAVE
filename = filename_base + "pop_Arithm_SRVF_with_noise_N_100_sig_01" 
dic = {"duration":duration, "arr_noisy_x":arr_noisy_x, "res_pop":out_pop, "res_arithm":out_arithm, "res_SRVF":out_srvf}

if os.path.isfile(filename):
    print("Le fichier ", filename, " existe déjà.")
    filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()

time_init = time.time()
res = Parallel(n_jobs=n_MC)(delayed(compute_SRC_FC_StatMeans)(out_pop[k].pop_Q, out_pop[k].pop_theta_coefs, out_pop[k].pop_arclgth, out_pop[k].mu_Z0, h_bounds, lbda_bounds, nb_basis, n_call_bayopt=n_call_bayopt, sigma=lam) for k in range(n_MC))
time_end = time.time()
duration = time_end - time_init

out_SRC, out_FC, out_V1, out_V2, out_V3 = [], [], [], [], []
for k in range(n_MC):
    out_SRC.append(res[k][0])
    out_FC.append(res[k][1])
    out_V1.append(res[k][2])
    out_V2.append(res[k][3])
    out_V3.append(res[k][4])

# SAVE
filename = filename_base + "pop_other_means_with_noise_N_100_sig_01" 
dic = {"duration":duration, "res_SRC":out_SRC, "res_FC":out_FC, "res_V1":out_V1, "res_V2":out_V2, "res_V3":out_V3}

if os.path.isfile(filename):
    print("Le fichier ", filename, " existe déjà.")
    filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()


# time_init = time.time()
# res = Parallel(n_jobs=n_MC)(delayed(compute_all_means)(arr_noisy_x[k], h_deriv_bounds, h_bounds, lbda_bounds, nb_basis, n_call_bayopt=n_call_bayopt, sigma=lam) for k in range(n_MC))
# time_end = time.time()
# duration = time_end - time_init

# out_pop, out_arithm, out_srvf, out_SRC, out_FC, out_V1, out_V2, out_V3 = [], [], [], [], [], [], [], []
# for k in range(n_MC):
#     out_pop.append(res[k][0])
#     out_arithm.append(res[k][1])
#     out_srvf.append(res[k][2])
#     out_SRC.append(res[k][3])
#     out_FC.append(res[k][4])
#     out_V1.append(res[k][5])
#     out_V2.append(res[k][6])
#     out_V3.append(res[k][7])

# # SAVE
# filename = filename_base + "with_noise_N_100_sig_01" 
# dic = {"duration":duration, "arr_noisy_x":arr_noisy_x, "res_pop":out_pop, "res_arithm":out_arithm, "res_SRVF":out_srvf, "res_SRC":out_SRC, "res_FC":out_FC, "res_V1":out_V1, "res_V2":out_V2, "res_V3":out_V3}

# if os.path.isfile(filename):
#     print("Le fichier ", filename, " existe déjà.")
#     filename = filename + '_bis'
# fil = open(filename,"xb")
# pickle.dump(dic,fil)
# fil.close()




""" sig_x = 0.005 """
sig_x = 0.005

arr_noisy_x = np.zeros((n_MC, n_samples, N, 3))
for k in range(n_MC):
    arr_noisy_x[k] = add_noise_pop(pop_X, sig_x)


time_init = time.time()
res = Parallel(n_jobs=n_MC)(delayed(compute_pop_artihm_SRVF)(arr_noisy_x[k], h_deriv_bounds, h_bounds, lbda_bounds, nb_basis, n_call_bayopt=n_call_bayopt, sigma=lam) for k in range(n_MC))
time_end = time.time()
duration = time_end - time_init

# Bspline_decom = VectorBSplineSmoothing(2, nb_basis, domain_range=(0, 1), order=4, penalization=False)
out_pop, out_arithm, out_srvf = [], [], []
# pop_theta_fct = np.empty((n_MC, n_samples), dtype=object)
for k in range(n_MC):
    out_pop.append(res[k][0])
    out_arithm.append(res[k][1])
    out_srvf.append(res[k][2])
    # for i in range(n_samples):
    #     pop_theta_fct[k][i] = Bspline_decom.evaluate_coefs(out_pop[k].pop_theta_coefs[i])

# SAVE
filename = filename_base + "pop_Arithm_SRVF_with_noise_N_100_sig_005" 
dic = {"duration":duration, "arr_noisy_x":arr_noisy_x, "res_pop":out_pop, "res_arithm":out_arithm, "res_SRVF":out_srvf}

if os.path.isfile(filename):
    print("Le fichier ", filename, " existe déjà.")
    filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()


time_init = time.time()
res = Parallel(n_jobs=n_MC)(delayed(compute_SRC_FC_StatMeans)(out_pop[k].pop_Q, out_pop[k].pop_theta_coefs, out_pop[k].pop_arclgth, out_pop[k].mu_Z0, h_bounds, lbda_bounds, nb_basis, n_call_bayopt=n_call_bayopt, sigma=lam) for k in range(n_MC))
time_end = time.time()
duration = time_end - time_init

out_SRC, out_FC, out_V1, out_V2, out_V3 = [], [], [], [], []
for k in range(n_MC):
    out_SRC.append(res[k][0])
    out_FC.append(res[k][1])
    out_V1.append(res[k][2])
    out_V2.append(res[k][3])
    out_V3.append(res[k][4])

# SAVE
filename = filename_base + "pop_other_means_with_noise_N_100_sig_005" 
dic = {"duration":duration, "res_SRC":out_SRC, "res_FC":out_FC, "res_V1":out_V1, "res_V2":out_V2, "res_V3":out_V3}

if os.path.isfile(filename):
    print("Le fichier ", filename, " existe déjà.")
    filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()


# time_init = time.time()
# res = Parallel(n_jobs=n_MC)(delayed(compute_all_means)(arr_noisy_x[k], h_deriv_bounds, h_bounds, lbda_bounds, nb_basis, n_call_bayopt=n_call_bayopt, sigma=lam) for k in range(n_MC))
# time_end = time.time()
# duration = time_end - time_init

# out_pop, out_arithm, out_srvf, out_SRC, out_FC, out_V1, out_V2, out_V3 = [], [], [], [], [], [], [], []
# for k in range(n_MC):
#     out_pop.append(res[k][0])
#     out_arithm.append(res[k][1])
#     out_srvf.append(res[k][2])
#     out_SRC.append(res[k][3])
#     out_FC.append(res[k][4])
#     out_V1.append(res[k][5])
#     out_V2.append(res[k][6])
#     out_V3.append(res[k][7])

# # SAVE
# filename = filename_base + "with_noise_N_100_sig_005" 
# dic = {"duration":duration, "arr_noisy_x":arr_noisy_x, "res_pop":out_pop, "res_arithm":out_arithm, "res_SRVF":out_srvf, "res_SRC":out_SRC, "res_FC":out_FC, "res_V1":out_V1, "res_V2":out_V2, "res_V3":out_V3}

# if os.path.isfile(filename):
#     print("Le fichier ", filename, " existe déjà.")
#     filename = filename + '_bis'
# fil = open(filename,"xb")
# pickle.dump(dic,fil)
# fil.close()











""" _______________________________________________________________ N = 200 _______________________________________________________________ """ 


N = 200

nb_basis = 25

h_bounds = np.array([0.015,0.1])
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


""" _________________ Amplitude and phase variability on theta and WITHOUT noise on x _________________ """

time_init = time.time()
out_arithm, out_srvf, out_SRC, out_FC, out_V1, out_V2, out_V3 = compute_all_means_known_param(pop_x_scale, pop_Z, pop_theta_func, h_deriv_bounds, h_bounds, lbda_bounds, nb_basis, pop_arclgth, n_call_bayopt=n_call_bayopt, sigma=lam)
time_end = time.time()
duration = time_end - time_init

# SAVE
filename = filename_base + "without_noise_N_200" 
dic = {"duration":duration, "res_arithm":out_arithm, "res_SRVF":out_srvf, "res_SRC":out_SRC, "res_FC":out_FC, "res_V1":out_V1, "res_V2":out_V2, "res_V3":out_V3}

if os.path.isfile(filename):
    print("Le fichier ", filename, " existe déjà.")
    filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()



""" _________________ Amplitude and phase variability on theta and with noise on x _________________ """


h_bounds = np.array([0.02,0.15])
h_deriv_bounds = np.array([0.05,0.3])
lbda_bounds = np.array([[-15.0,-5.0],[-15.0,-5.0]])


""" sig_x = 0.01 """
sig_x = 0.01

arr_noisy_x = np.zeros((n_MC, n_samples, N, 3))
for k in range(n_MC):
    arr_noisy_x[k] = add_noise_pop(pop_X, sig_x)

time_init = time.time()
res = Parallel(n_jobs=n_MC)(delayed(compute_all_means)(arr_noisy_x[k], h_deriv_bounds, h_bounds, lbda_bounds, nb_basis, n_call_bayopt=n_call_bayopt, sigma=lam) for k in range(n_MC))
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
filename = filename_base + "with_noise_N_200_sig_01"
dic = {"duration":duration, "arr_noisy_x":arr_noisy_x, "res_pop":out_pop, "res_arithm":out_arithm, "res_SRVF":out_srvf, "res_SRC":out_SRC, "res_FC":out_FC, "res_V1":out_V1, "res_V2":out_V2, "res_V3":out_V3}

if os.path.isfile(filename):
    print("Le fichier ", filename, " existe déjà.")
    filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()


""" sig_x = 0.005 """
sig_x = 0.005

arr_noisy_x = np.zeros((n_MC, n_samples, N, 3))
for k in range(n_MC):
    arr_noisy_x[k] = add_noise_pop(pop_X, sig_x)

time_init = time.time()
res = Parallel(n_jobs=n_MC)(delayed(compute_all_means)(arr_noisy_x[k], h_deriv_bounds, h_bounds, lbda_bounds, nb_basis, n_call_bayopt=n_call_bayopt, sigma=lam) for k in range(n_MC))
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
filename = filename_base + "with_noise_N_200_sig_005"
dic = {"duration":duration, "arr_noisy_x":arr_noisy_x, "res_pop":out_pop, "res_arithm":out_arithm, "res_SRVF":out_srvf, "res_SRC":out_SRC, "res_FC":out_FC, "res_V1":out_V1, "res_V2":out_V2, "res_V3":out_V3}

if os.path.isfile(filename):
    print("Le fichier ", filename, " existe déjà.")
    filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()