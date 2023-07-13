import sys
sys.path.insert(1, '../')
from FrenetFDA.utils.Frenet_Serret_utils import *
from FrenetFDA.utils.smoothing_utils import *
from FrenetFDA.processing_Euclidean_curve.unified_estimate_Frenet_state_space.EM_Frenet_state_space_CV import FrenetStateSpaceCV_global, bayesian_CV_optimization_regularization_parameter
from FrenetFDA.processing_Euclidean_curve.unified_estimate_Frenet_state_space.iek_filter_smoother_Frenet_state import IEKFilterSmootherFrenetState
from FrenetFDA.processing_Euclidean_curve.preprocessing import *
from FrenetFDA.processing_Euclidean_curve.estimate_Frenet_path import GramSchmidtOrthogonalization, ConstrainedLocalPolynomialRegression
from FrenetFDA.processing_Euclidean_curve.estimate_Frenet_curvatures import ExtrinsicFormulas
from pickle import *
import time 
import os.path
import os
import dill as pickle
from tqdm import tqdm
from compare_smoother import compare_method_with_iteration
from compare_method_without_iteration import compare_method_without_iteration


def theta(s):
    curv = lambda s : 2*np.cos(2*np.pi*s) + 5
    tors = lambda s : 2*np.sin(2*np.pi*s) + 1
    if isinstance(s, int) or isinstance(s, float):
        return np.array([curv(s), tors(s)])
    elif isinstance(s, np.ndarray):
        return np.vstack((curv(s), tors(s))).T
    else:
        raise ValueError('Variable is not a float, a int or a NumPy array.')
    

""" SCENARIO 2: observations y_i """

mu0 = np.eye(4) 
P0 = 0.01**2*np.eye(6)
n_MC = 90
bounds_lambda = np.array([[1e-09, 1e-03], [1e-09, 1e-03]])
bounds_h = np.array([0.05, 0.35])
n_call_bayopt = 100
a = np.random.normal(0,1)
def arc_length_fct(s):
    if abs(a) < 1e-04:
        return s
    else:
        return (np.exp(a*s) - 1)/(np.exp(a) - 1)

directory = r"results/scenario2/model_01/"
filename_base = "results/scenario2/model_01/"

current_directory = os.getcwd()
final_directory = os.path.join(current_directory, directory)
if not os.path.exists(final_directory):
   os.makedirs(final_directory)

filename = filename_base + "model"
dic = {"nb_iterations_simu": n_MC, "P0": P0, "mu0": mu0, "theta":theta, "arc_length_fct": arc_length_fct, "a": a,
       "bounds_lambda": bounds_lambda, "bounds_h": bounds_h, "n_call_bayopt": n_call_bayopt}
if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()



print(" Scenario 2, simu 1: N=100, gamma=0.001 ")

N = 100
gamma = 0.001
Gamma = gamma**2*np.eye(3)
nb_basis = 10

time_init = time.time()

with tqdm(total=n_MC) as pbar:
   res = Parallel(n_jobs=-1)(delayed(compare_method_without_iteration)(theta, arc_length_fct, N, Gamma, mu0, P0, nb_basis, bounds_h, bounds_lambda, n_call_bayopt) for k in range(n_MC))
   pbar.update()

time_end = time.time()
duration = time_end - time_init

filename = filename_base + "simu_1"

dic = {"results":res, "duration":duration, "N":N, "gamma":gamma, "nb_basis":nb_basis}

if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()



print(" Scenario 2, simu 2: N=100, gamma=0.005 ")

N = 100
gamma = 0.005
Gamma = gamma**2*np.eye(3)
nb_basis = 10

time_init = time.time()

with tqdm(total=n_MC) as pbar:
   res = Parallel(n_jobs=-1)(delayed(compare_method_without_iteration)(theta, arc_length_fct, N, Gamma, mu0, P0, nb_basis, bounds_h, bounds_lambda, n_call_bayopt) for k in range(n_MC))
   pbar.update()

time_end = time.time()
duration = time_end - time_init

filename = filename_base + "simu_2"

dic = {"results":res, "duration":duration, "N":N, "gamma":gamma, "nb_basis":nb_basis}

if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()



print(" Scenario 2, simu 3: N=200, gamma=0.001 ")

N = 200
gamma = 0.001
Gamma = gamma**2*np.eye(3)
nb_basis = 15

time_init = time.time()

with tqdm(total=n_MC) as pbar:
   res = Parallel(n_jobs=-1)(delayed(compare_method_without_iteration)(theta, arc_length_fct, N, Gamma, mu0, P0, nb_basis, bounds_h, bounds_lambda, n_call_bayopt) for k in range(n_MC))
   pbar.update()

time_end = time.time()
duration = time_end - time_init

filename = filename_base + "simu_3"

dic = {"results":res, "duration":duration, "N":N, "gamma":gamma, "nb_basis":nb_basis}

if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()



print(" Scenario 2, simu 4: N=200, gamma=0.005 ")

N = 200
gamma = 0.005
Gamma = gamma**2*np.eye(3)
nb_basis = 15

time_init = time.time()

with tqdm(total=n_MC) as pbar:
   res = Parallel(n_jobs=-1)(delayed(compare_method_without_iteration)(theta, arc_length_fct, N, Gamma, mu0, P0, nb_basis, bounds_h, bounds_lambda, n_call_bayopt) for k in range(n_MC))
   pbar.update()

time_end = time.time()
duration = time_end - time_init

filename = filename_base + "simu_4"

dic = {"results":res, "duration":duration, "N":N, "gamma":gamma, "nb_basis":nb_basis}

if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()




""" SCENARIO 1: observations U_i """


bounds_lambda = np.array([[1e-09, 1e-03], [1e-09, 1e-03]])
bounds_lambda_track = np.array([1e-04, 1])
bounds_h = np.array([0.05, 1])
n_call_bayopt = 20
max_iter = 25
tol = 0.001


directory = r"results/scenario1/model_01/"
filename_base = "results/scenario1/model_01/"

current_directory = os.getcwd()
final_directory = os.path.join(current_directory, directory)
if not os.path.exists(final_directory):
   os.makedirs(final_directory)

filename = filename_base + "model"
dic = {"nb_iterations_simu": n_MC, "mu0": mu0, "theta":theta, "arc_length_fct": arc_length_fct, "a": a, "tol":tol, "max_iter":max_iter, 
       "bounds_lambda": bounds_lambda, "bounds_h": bounds_h, "n_call_bayopt": n_call_bayopt, "bounds_lambda_track": bounds_lambda_track}
if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()


print(" Scenario 1, simu 1: N=100, alpha=10 ")

N = 100
nb_basis = 10
K = 10**2

time_init = time.time()

with tqdm(total=n_MC) as pbar:
   res = Parallel(n_jobs=-1)(delayed(compare_method_with_iteration)(theta, arc_length_fct, N, mu0, K, nb_basis, bounds_h, bounds_lambda, bounds_lambda_track, n_call_bayopt, tol, max_iter) for k in range(n_MC))
   pbar.update()

time_end = time.time()
duration = time_end - time_init

filename = filename_base + "simu_1"

dic = {"results":res, "duration":duration, "N":N, "alpha":np.sqrt(K), "nb_basis":nb_basis}

if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()



print(" Scenario 1, simu 2: N=100, alpha=20 ")

N = 100
nb_basis = 10
K = 20**2

time_init = time.time()

with tqdm(total=n_MC) as pbar:
   res = Parallel(n_jobs=-1)(delayed(compare_method_with_iteration)(theta, arc_length_fct, N, mu0, K, nb_basis, bounds_h, bounds_lambda, bounds_lambda_track, n_call_bayopt, tol, max_iter) for k in range(n_MC))
   pbar.update()

time_end = time.time()
duration = time_end - time_init

filename = filename_base + "simu_2"

dic = {"results":res, "duration":duration, "N":N, "alpha":np.sqrt(K), "nb_basis":nb_basis}

if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()


print(" Scenario 1, simu 3: N=200, alpha=10 ")

N = 200
nb_basis = 15
K = 10**2

time_init = time.time()

with tqdm(total=n_MC) as pbar:
   res = Parallel(n_jobs=-1)(delayed(compare_method_with_iteration)(theta, arc_length_fct, N, mu0, K, nb_basis, bounds_h, bounds_lambda, bounds_lambda_track, n_call_bayopt, tol, max_iter) for k in range(n_MC))
   pbar.update()

time_end = time.time()
duration = time_end - time_init

filename = filename_base + "simu_3"

dic = {"results":res, "duration":duration, "N":N, "alpha":np.sqrt(K), "nb_basis":nb_basis}

if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()



print(" Scenario 1, simu 4: N=200, alpha=20 ")

N = 200
nb_basis = 10
K = 20**2

time_init = time.time()

with tqdm(total=n_MC) as pbar:
   res = Parallel(n_jobs=-1)(delayed(compare_method_with_iteration)(theta, arc_length_fct, N, mu0, K, nb_basis, bounds_h, bounds_lambda, bounds_lambda_track, n_call_bayopt, tol, max_iter) for k in range(n_MC))
   pbar.update()

time_end = time.time()
duration = time_end - time_init

filename = filename_base + "simu_4"

dic = {"results":res, "duration":duration, "N":N, "alpha":np.sqrt(K), "nb_basis":nb_basis}

if os.path.isfile(filename):
   print("Le fichier ", filename, " existe déjà.")
   filename = filename + '_bis'
fil = open(filename,"xb")
pickle.dump(dic,fil)
fil.close()