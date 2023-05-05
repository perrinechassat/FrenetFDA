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

Scenario 3: Simulation code to compare the results in function of the observation noise. 
    
    Scenario:
        1. $\Gamma = \gamma^2 I$ with $\gamma = 0.001$
        2. $\Gamma = \gamma^2 I$ with $\gamma = 0.005$
    and all the others parameters are the same for all scenarios.

"""

directory = r"results/scenario_3/model_01"
filename_base = "results/scenario_3/model_01/"

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

## Sigma
Sigma = None 
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
max_iter = 300
tol = 1e-3
# reg_param_grid_EM = np.array([[1e-06,1e-06], [1e-05,1e-05], [1e-04,1e-04], [1e-03,1e-03], [1e-02,1e-02], [1e-01,1e-01]])
reg_param_grid_EM = np.array([[1e-04,1e-04], [1e-03,1e-03], [1e-02,1e-02], [1e-01,1e-01]])
reg_param_grid_EM = np.array(np.meshgrid(*reg_param_grid_EM.T)).reshape((2,-1))
reg_param_grid_EM = np.moveaxis(reg_param_grid_EM, 0,1)
## Number of simulations 
N_simu = 100


filename = filename_base + "model"
dic = {"nb_iterations_simu": N_simu, "P0": P0, "mu0": mu0, "theta":theta, "Sigma":Sigma, "reg_param_grid_EM":reg_param_grid_EM, "max_iter":max_iter, "tol":tol, "N":N, 
       "arc_length_fct": arc_length_fct, "bandwidth_grid_init" : bandwidth_grid_init, "nb_basis":nb_basis, "reg_param_grid_init": reg_param_grid_init}
if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()


""" S3.1: gamma = 0.001 """

print('--------------------- Start scenario 3.1 ---------------------')

time_init = time.time()

## Gamma
gamma = 0.001
Gamma = gamma**2*np.eye(3)


with tqdm(total=N_simu) as pbar:
   res_S3_1 = Parallel(n_jobs=50)(delayed(scenario_1_1_bis)(theta, Sigma, mu0, P0, Gamma, N, arc_length_fct, nb_basis, bandwidth_grid_init, reg_param_grid_init, reg_param_grid_EM, max_iter, tol) for k in range(N_simu))
   pbar.update()

time_end = time.time()
duration = time_end - time_init

filename = filename_base + "scenario_3_1"

dic = {"results_S3_1":res_S3_1, "gamma": gamma}

if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()

print('End of scenario 3.1: time spent', duration, 'seconds. \n')



""" S3.2: gamma = 0.005 """

print('--------------------- Start scenario 3.2 ---------------------')

time_init = time.time()

## Gamma
gamma = 0.005
Gamma = gamma**2*np.eye(3)


with tqdm(total=N_simu) as pbar:
   res_S3_2 = Parallel(n_jobs=50)(delayed(scenario_1_1_bis)(theta, Sigma, mu0, P0, Gamma, N, arc_length_fct, nb_basis, bandwidth_grid_init, reg_param_grid_init, reg_param_grid_EM, max_iter, tol) for k in range(N_simu))
   pbar.update()

time_end = time.time()
duration = time_end - time_init

filename = filename_base + "scenario_3_2"

dic = {"results_S3_2":res_S3_2, "gamma": gamma}

if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()

print('End of scenario 3.2: time spent', duration, 'seconds. \n')







""" 

Scenario 4: Simulation code to compare the results in function of the number of sample points. 
    
    Scenario:
        1. N = 200   (same as Scenario 3.1)
        2. N = 100
    and all the others parameters are the same for all scenarios.

"""


directory = r"results/scenario_4/model_01"
filename_base = "results/scenario_4/model_01/"

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

## Sigma
Sigma = None 
## mu_0 and P_0
mu0 = np.eye(4)
P0 = 0.001**2*np.eye(6)
## Gamma
gamma = 0.001
Gamma = gamma**2*np.eye(3)
## grid of parameters
bandwidth_grid_init = np.array([0.05, 0.1, 0.12, 0.15, 0.17, 0.2, 0.25])
reg_param_grid_init = np.array([1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01])
## Param EM
max_iter = 300
tol = 1e-3
# reg_param_grid_EM = np.array([[1e-06,1e-06], [1e-05,1e-05], [1e-04,1e-04], [1e-03,1e-03], [1e-02,1e-02], [1e-01,1e-01]])
reg_param_grid_EM = np.array([[1e-04,1e-04], [1e-03,1e-03], [1e-02,1e-02], [1e-01,1e-01]])
reg_param_grid_EM = np.array(np.meshgrid(*reg_param_grid_EM.T)).reshape((2,-1))
reg_param_grid_EM = np.moveaxis(reg_param_grid_EM, 0,1)
## Number of simulations 
N_simu = 100


filename = filename_base + "model"
dic = {"nb_iterations_simu": N_simu, "P0": P0, "mu0": mu0, "theta":theta, "Gamma":Gamma, "Sigma":Sigma, "reg_param_grid_EM":reg_param_grid_EM, "max_iter":max_iter, "tol":tol, 
       "arc_length_fct": arc_length_fct, "bandwidth_grid_init" : bandwidth_grid_init, "reg_param_grid_init": reg_param_grid_init}
if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()



""" S4.2: N = 100 """

print('--------------------- Start scenario 4.2 ---------------------')

time_init = time.time()

## number of samples and basis fct
N = 100
nb_basis = 10

with tqdm(total=N_simu) as pbar:
   res_S4_2 = Parallel(n_jobs=50)(delayed(scenario_1_1_bis)(theta, Sigma, mu0, P0, Gamma, N, arc_length_fct, nb_basis, bandwidth_grid_init, reg_param_grid_init, reg_param_grid_EM, max_iter, tol) for k in range(N_simu))
   pbar.update()

time_end = time.time()
duration = time_end - time_init

filename = filename_base + "scenario_4_2"

dic = {"results_S4_2":res_S4_2, "N": N, "nb_basis": nb_basis}

if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()

print('End of scenario 4.2: time spent', duration, 'seconds. \n')





""" 

Scenario 5: Simulation code to compare the results in function of the model parameter theta(s) and s(t). 
    
    Scenario:
        1. theta_1(s) = (curv_1(s), tors_1(s)) + s(t) = t                                       (same as Scenario 3.1)
        2. theta_1(s) = (curv_1(s), tors_1(s)) + s(t) = (np.sin(2*np.pi*t) + t/0.08)*0.08 
        3. theta_2(s) = (curv_2(s), tors_2(s)) + s(t) = t 
        4. theta_2(s) = (curv_2(s), tors_2(s)) + s(t) = (np.sin(2*np.pi*t) + t/0.08)*0.08 

        with:
            curv_1(s) = 2*cos(2*pi*s) + 5
            tors_1(s) = 2*sin(2*pi*s) + 1

            curv_2(s) = exp(3*sin(2*pi*(s-0.2)))
            tors_2(s) = 5*s - 2

    and all the others parameters are the same for all scenarios.

"""

directory = r"results/scenario_5/model_01"
filename_base = "results/scenario_5/model_01/"

current_directory = os.getcwd()
final_directory = os.path.join(current_directory, directory)
if not os.path.exists(final_directory):
   os.makedirs(final_directory)


""" Definition of the true parameters """


## Sigma
Sigma = None 
## mu_0 and P_0
mu0 = np.eye(4)
P0 = 0.001**2*np.eye(6)
## number of samples and basis fct
N = 200
nb_basis = 15
## Gamma
gamma = 0.001
Gamma = gamma**2*np.eye(3)
## grid of parameters
bandwidth_grid_init = np.array([0.05, 0.1, 0.12, 0.15, 0.17, 0.2, 0.25])
reg_param_grid_init = np.array([1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01])
## Param EM
max_iter = 300
tol = 1e-3
# reg_param_grid_EM = np.array([[1e-06,1e-06], [1e-05,1e-05], [1e-04,1e-04], [1e-03,1e-03], [1e-02,1e-02], [1e-01,1e-01]])
reg_param_grid_EM = np.array([[1e-04,1e-04], [1e-03,1e-03], [1e-02,1e-02], [1e-01,1e-01]])
reg_param_grid_EM = np.array(np.meshgrid(*reg_param_grid_EM.T)).reshape((2,-1))
reg_param_grid_EM = np.moveaxis(reg_param_grid_EM, 0,1)
## Number of simulations 
N_simu = 100


filename = filename_base + "model"
dic = {"nb_iterations_simu": N_simu, "P0": P0, "mu0": mu0, "Gamma":Gamma, "Sigma":Sigma, "reg_param_grid_EM":reg_param_grid_EM, "max_iter":max_iter, "tol":tol, "N":N, 
     "bandwidth_grid_init" : bandwidth_grid_init, "nb_basis":nb_basis, "reg_param_grid_init": reg_param_grid_init}
if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()



def theta_1(s):
    curv = lambda s : 2*np.cos(2*np.pi*s) + 5
    tors = lambda s : 2*np.sin(2*np.pi*s) + 1
    if isinstance(s, int) or isinstance(s, float):
        return np.array([curv(s), tors(s)])
    elif isinstance(s, np.ndarray):
        return np.vstack((curv(s), tors(s))).T
    else:
        raise ValueError('Variable is not a float, a int or a NumPy array.')
    
def theta_2(s):
    curv = lambda s : np.exp(3*np.sin(2*np.pi*(s-0.2)))
    tors = lambda s : 5*s - 2
    if isinstance(s, int) or isinstance(s, float):
        return np.array([curv(s), tors(s)])
    elif isinstance(s, np.ndarray):
        return np.vstack((curv(s), tors(s))).T
    else:
        raise ValueError('Variable is not a float, a int or a NumPy array.')
    

arc_length_id = lambda s: s
arc_length_tilde = lambda s: (np.sin(2*np.pi*s) + s/0.07)*0.07



""" S5.1: theta_1 + s_id """

# idem scenario 3.1


""" S5.2: theta_1 + s_tilde """

print('--------------------- Start scenario 5.2 ---------------------')

time_init = time.time()

with tqdm(total=N_simu) as pbar:
   res_S5_2 = Parallel(n_jobs=50)(delayed(scenario_1_1_bis)(theta_1, Sigma, mu0, P0, Gamma, N, arc_length_tilde, nb_basis, bandwidth_grid_init, reg_param_grid_init, reg_param_grid_EM, max_iter, tol) for k in range(N_simu))
   pbar.update()

time_end = time.time()
duration = time_end - time_init

filename = filename_base + "scenario_5_2"

dic = {"results_S5_2":res_S5_2, "arc_length": arc_length_tilde, "theta": theta_1}

if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()


print('End of scenario 5.2: time spent', duration, 'seconds. \n')


""" S5.3: theta_2 + s_id """

print('--------------------- Start scenario 5.3 ---------------------')

time_init = time.time()

with tqdm(total=N_simu) as pbar:
   res_S5_3 = Parallel(n_jobs=50)(delayed(scenario_1_1_bis)(theta_2, Sigma, mu0, P0, Gamma, N, arc_length_id, nb_basis, bandwidth_grid_init, reg_param_grid_init, reg_param_grid_EM, max_iter, tol) for k in range(N_simu))
   pbar.update()

time_end = time.time()
duration = time_end - time_init

filename = filename_base + "scenario_5_3"

dic = {"results_S5_3":res_S5_3, "arc_length": arc_length_id, "theta": theta_2}

if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()

print('End of scenario 5.3: time spent', duration, 'seconds. \n')



""" S5.4: theta_2 + s_tilde """

print('--------------------- Start scenario 5.4 ---------------------')

time_init = time.time()

with tqdm(total=N_simu) as pbar:
   res_S5_4 = Parallel(n_jobs=50)(delayed(scenario_1_1_bis)(theta_2, Sigma, mu0, P0, Gamma, N, arc_length_tilde, nb_basis, bandwidth_grid_init, reg_param_grid_init, reg_param_grid_EM, max_iter, tol) for k in range(N_simu))
   pbar.update()

time_end = time.time()
duration = time_end - time_init

filename = filename_base + "scenario_5_4"

dic = {"results_S5_4":res_S5_4, "arc_length": arc_length_tilde, "theta": theta_2}

if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()

print('End of scenario 5.4: time spent', duration, 'seconds. \n')