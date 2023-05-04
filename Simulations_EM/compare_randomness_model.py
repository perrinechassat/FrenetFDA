import sys
sys.path.insert(1, '../')
from FrenetFDA.utils.Frenet_Serret_utils import *
from FrenetFDA.utils.smoothing_utils import *
from FrenetFDA.processing_Euclidean_curve.unified_estimate_Frenet_state_space.EM_Frenet_state_space import FrenetStateSpace
from FrenetFDA.processing_Euclidean_curve.unified_estimate_Frenet_state_space.iek_filter_smoother_Frenet_state import IEKFilterSmootherFrenetState
from FrenetFDA.processing_Euclidean_curve.preprocessing import *
from FrenetFDA.processing_Euclidean_curve.estimate_Frenet_path import GramSchmidtOrthogonalization, ConstrainedLocalPolynomialRegression
from FrenetFDA.processing_Euclidean_curve.estimate_Frenet_curvatures import ExtrinsicFormulas
from pickle import *
import time
import os.path
import os
import dill as pickle
from simulations_tools import * 
from tqdm import tqdm

""" 

Scenario 2: Simulation code to compare the results in function of the randomness level in the model on Z. 
    
    Scenario:
        1. $\Sigma = \sigma^2 I$ with $\sigma = 0$
        2. $\Sigma = \sigma^2 I$ with $\sigma = 0.01$
        3. $\Sigma = \sigma^2 I$ with $\sigma = 0.1$
    and all the others parameters are the same for all scenarios.

"""

directory = r"results/scenario_2/model_01"
filename_base = "results/scenario_2/model_01/"

current_directory = os.getcwd()
final_directory = os.path.join(current_directory, directory)
if not os.path.exists(final_directory):
   os.makedirs(final_directory)


""" Definition of the true parameters """

## Theta 
def theta(s):
   curv = lambda s : 2*np.cos(2*np.pi*s) + 5
   tors = lambda s : 2*np.sin(2*np.pi*s) + 1
   if isinstance(s, int) or isinstance(s, float):
      return np.array([curv(s), tors(s)])
   elif isinstance(s, np.ndarray):
      return np.vstack((curv(s), tors(s))).T
   else:
      raise ValueError('Variable is not a float, a int or a NumPy array.')
   
arc_length_fct = lambda s: s
# def warping(s,a):
#     if np.abs(a)<1e-15:
#         return s
#     else:
#         return (np.exp(a*s) - 1)/(np.exp(a) - 1)    

## Gamma
gamma = 0.001
Gamma = gamma**2*np.eye(3)

## mu_0 and P_0
mu0 = np.eye(4)
P0 = 0.001**2*np.eye(6)

## number of samples and basis fct
N = 200
nb_basis = 15

## grid of parameters
bandwidth_grid_init = np.array([0.05, 0.1, 0.12, 0.15, 0.17, 0.2, 0.25])
reg_param_grid_init = np.array([1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01])

## Param EM
max_iter = 200
tol = 1e-3
reg_param_grid_EM = np.array([[1e-06,1e-06], [1e-05,1e-05], [1e-04,1e-04], [1e-03,1e-03], [1e-02,1e-02], [1e-01,1e-01]])
reg_param_grid_EM = np.array(np.meshgrid(*reg_param_grid_EM.T)).reshape((2,-1))
reg_param_grid_EM = np.moveaxis(reg_param_grid_EM, 0,1)


""" Number of simulations """
N_simu = 100


filename = filename_base + "model"
dic = {"nb_iterations_simu": N_simu, "P0": P0, "mu0": mu0, "theta":theta, "Gamma":Gamma, "reg_param_grid_EM":reg_param_grid_EM, "max_iter":max_iter, "tol":tol, "N":N, 
       "arc_length_fct": arc_length_fct, "bandwidth_grid_init" : bandwidth_grid_init, "nb_basis":nb_basis, "reg_param_grid_init": reg_param_grid_init}
if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()



""" S2.1: sigma = 0 """

# print('--------------------- Start scenario 2.1 ---------------------')

# time_init = time.time()

# Sigma = None 

# with tqdm(total=N_simu) as pbar:
#    res_S2_1 = Parallel(n_jobs=50)(delayed(scenario_2)(theta, Sigma, mu0, P0, Gamma, N, arc_length_fct, nb_basis, bandwidth_grid_init, reg_param_grid_init, reg_param_grid_EM, max_iter, tol) for k in range(N_simu))
#    pbar.update()

# time_end = time.time()
# duration = time_end - time_init

# filename = filename_base + "scenario_2_1"

# dic = {"results_S2_1":res_S2_1, "Sigma": Sigma}

# if os.path.isfile(filename):
#    print("Le fichier ", filename, " existe déjà.")
#    filename = filename + '_bis'
# fil = open(filename,"xb")
# pickle.dump(dic,fil)
# fil.close()

# print('End of scenario 2.1: time spent', duration, 'seconds. \n')



""" S2.2: sigma = 0.01 """

print('--------------------- Start scenario 2.2 ---------------------')

time_init = time.time()

sigma = 0.01
Sigma = lambda s: np.array([[sigma**2 + 0*s, 0*s],[0*s, sigma**2 + 0*s]])

with tqdm(total=N_simu) as pbar:
   res_S2_2 = Parallel(n_jobs=50)(delayed(scenario_2)(theta, Sigma, mu0, P0, Gamma, N, arc_length_fct, nb_basis, bandwidth_grid_init, reg_param_grid_init, reg_param_grid_EM, max_iter, tol) for k in range(N_simu))
   pbar.update()

time_end = time.time()
duration = time_end - time_init

filename = filename_base + "scenario_2_2"

dic = {"results_S2_2":res_S2_2, "Sigma": Sigma}

if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()

print('End of scenario 2.2: time spent', duration, 'seconds. \n')





""" S2.3: sigma = 0.1 """

print('--------------------- Start scenario 2.3 ---------------------')

time_init = time.time()

sigma = 0.1
Sigma = lambda s: np.array([[sigma**2 + 0*s, 0*s],[0*s, sigma**2 + 0*s]])

with tqdm(total=N_simu) as pbar:
   res_S2_3 = Parallel(n_jobs=50)(delayed(scenario_2)(theta, Sigma, mu0, P0, Gamma, N, arc_length_fct, nb_basis, bandwidth_grid_init, reg_param_grid_init, reg_param_grid_EM, max_iter, tol) for k in range(N_simu))
   pbar.update()

time_end = time.time()
duration = time_end - time_init

filename = filename_base + "scenario_2_3"

dic = {"results_S2_3":res_S2_3, "Sigma": Sigma}

if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()

print('End of scenario 2.3: time spent', duration, 'seconds. \n')
