import sys
sys.path.insert(1, '../')
from FrenetFDA.utils.Frenet_Serret_utils import *
from FrenetFDA.utils.smoothing_utils import *
from FrenetFDA.processing_Euclidean_curve.unified_estimate_Frenet_state_space.EM_Frenet_state_space import FrenetStateSpace
from FrenetFDA.processing_Euclidean_curve.unified_estimate_Frenet_state_space.iek_filter_smoother_Frenet_state import IEKFilterSmootherFrenetState
from FrenetFDA.processing_Euclidean_curve.preprocessing import *
from FrenetFDA.processing_Euclidean_curve.estimate_Frenet_path import GramSchmidtOrthogonalization, ConstrainedLocalPolynomialRegression
from FrenetFDA.processing_Euclidean_curve.estimate_Frenet_curvatures import ExtrinsicFormulas
from FrenetFDA.processing_Frenet_path.estimate_Frenet_curvatures import ApproxFrenetODE, LocalApproxFrenetODE
import time


def model_scenario(theta, Sigma, mu0, P0, Gamma, N, arc_length_fct):
   """ 
      If Sigma==None the model is deterministic. 

   """
   time_grid = np.linspace(0,1,N)
   arc_length = arc_length_fct(time_grid)
   L = np.zeros((6,2))
   L[0,1], L[2,0] = 1, 1
   xi0 = np.random.multivariate_normal(np.zeros(6), P0)
   Z0 = mu0 @ SE3.exp(-xi0)
   if Sigma is None:
      Z = solve_FrenetSerret_ODE_SE(theta, arc_length, Z0=Z0, method='Linearized')
   else:
      Z = solve_FrenetSerret_SDE_SE3(theta, Sigma, L, arc_length, Z0=Z0)
   Q = Z[:,:3,:3]
   X = Z[:,:3,3]
   Y = X + np.random.multivariate_normal(np.zeros(3), Gamma, size=(len(X)))

   return Z, Q, X, Y



def scenario_1_1(theta, Sigma, mu0, P0, Gamma, N, arc_length_fct, nb_basis, bandwidth_grid_init, reg_param_grid_init, reg_param_grid_EM, max_iter_EM, tol_EM):
   """ Scenario 1.1: Initialization method LP + GS + Extrinsic formulas """

   
   ####     Generation of one sample of the model    ####

   Z, Q, X, Y = model_scenario(theta, Sigma, mu0, P0, Gamma, N, arc_length_fct)

   ####     Initialization of parameters    ####

   grid_time = np.linspace(0,1,N)
   derivatives, h_opt = compute_derivatives(Y, grid_time, deg=3, h=None, CV_optimization_h={"flag":True, "h_grid":bandwidth_grid_init, "K":10})
   grid_arc_s, L, arc_s, arc_s_dot = compute_arc_length(Y, grid_time, smooth=True, smoothing_param=h_opt)

   # Gamma
   Gamma_hat = ((Y - derivatives[0]).T @ (Y - derivatives[0]))/N

   # Z_hat
   GS_orthog = GramSchmidtOrthogonalization(Y, grid_arc_s, deg=3)
   h_opt, err_h = GS_orthog.grid_search_CV_optimization_bandwidth(bandwidth_grid=bandwidth_grid_init, K_split=10)
   Z_GS, Q_GS, X_GS = GS_orthog.fit(h_opt) 

   # mu0_hat
   mu0_hat = Z_GS[0]

   # theta_hat
   extrins_model_theta = ExtrinsicFormulas(Y, grid_time, grid_arc_s, deg_polynomial=3)
   h_opt, nb_basis_opt, regularization_parameter_opt, CV_error_tab = extrins_model_theta.grid_search_optimization_hyperparameters(bandwidth_grid_init, np.array([nb_basis]), reg_param_grid_init, method='2', n_splits=10)
   Basis_extrins = extrins_model_theta.Bspline_smooth_estimates(h_opt, nb_basis_opt, regularization_parameter=regularization_parameter_opt)

   # Sigma_hat
   mu_Z_theta_extrins = solve_FrenetSerret_ODE_SE(Basis_extrins.evaluate, grid_arc_s, Z0=mu0_hat, method='Linearized')
   Sigma_hat = np.zeros((N, 2, 2))
   for i in range(N):
      L = np.zeros((6,2))
      L[0,1], L[2,0] = 1, 1
      xi = -SE3.log(np.linalg.inv(mu_Z_theta_extrins[i])@Z_GS[i])
      Sigma_hat[i] = L.T @ xi[:,np.newaxis] @ xi[np.newaxis,:] @ L
   sig = np.sqrt((np.mean(Sigma_hat[:,0,0]) + np.mean(Sigma_hat[:,1,1]))/2)
   # print('sigma:', sig)
   Sigma_hat = lambda s: sig**2*np.array([[1+0*s, 0*s], [0*s, 1+0*s]])
   
   # P0_hat
   P0_hat = sig**2*np.eye(6)

   try:
      ####     Run the EM    ####
      FS_statespace = FrenetStateSpace(grid_arc_s, Y)
      FS_statespace.expectation_maximization(tol_EM, max_iter_EM, nb_basis=nb_basis, regularization_parameter_list=reg_param_grid_EM, init_params = {"W":Gamma_hat, "coefs":Basis_extrins.coefs, "mu0":mu0_hat, "Sigma":Sigma_hat, "P0":P0_hat}, init_states = None, method='autre', model_Sigma='scalar')
   except:
      FS_statespace = [sig, Basis_extrins, Z_GS]

   return FS_statespace, Z_GS, Z




def scenario_1_2(theta, Sigma, mu0, P0, Gamma, N, arc_length_fct, nb_basis, bandwidth_grid_init, reg_param_grid_init, reg_param_grid_EM, max_iter_EM, tol_EM):
   """ Scenario 1.2: Initialization method LP + GS + Approx Ode """


   ####     Generation of one sample of the model    ####

   Z, Q, X, Y = model_scenario(theta, Sigma, mu0, P0, Gamma, N, arc_length_fct)

   ####     Initialization of parameters    ####

   time = np.linspace(0,1,N)
   derivatives, h_opt = compute_derivatives(Y, time, deg=3, h=None, CV_optimization_h={"flag":True, "h_grid":bandwidth_grid_init, "K":10})
   grid_arc_s, L, arc_s, arc_s_dot = compute_arc_length(Y, time, smooth=True, smoothing_param=h_opt)

   # Gamma
   Gamma_hat = ((Y - derivatives[0]).T @ (Y - derivatives[0]))/N

   # Z_hat
   GS_orthog = GramSchmidtOrthogonalization(Y, grid_arc_s, deg=3)
   h_opt, err_h = GS_orthog.grid_search_CV_optimization_bandwidth(bandwidth_grid=bandwidth_grid_init, K_split=10)
   Z_GS, Q_GS, X_GS = GS_orthog.fit(h_opt) 

   # mu0_hat
   mu0_hat = Z_GS[0]

   # theta_hat
   approx_ode = ApproxFrenetODE(grid_arc_s, Z=Z_GS)
   nb_basis_opt, regularization_parameter_opt, tab_GCV_scores_GS = approx_ode.grid_search_optimization_hyperparameters(nb_basis_list=np.array([nb_basis]), regularization_parameter_list=reg_param_grid_init)
   Bspline_approxODE = approx_ode.Bspline_smooth_estimates(nb_basis, regularization_parameter=regularization_parameter_opt)

   # Sigma_hat
   mu_Z_theta_extrins = solve_FrenetSerret_ODE_SE(Bspline_approxODE.evaluate, grid_arc_s, Z0=mu0_hat, method='Linearized')
   Sigma_hat = np.zeros((N, 2, 2))
   for i in range(N):
      L = np.zeros((6,2))
      L[0,1], L[2,0] = 1, 1
      xi = -SE3.log(np.linalg.inv(mu_Z_theta_extrins[i])@Z_GS[i])
      Sigma_hat[i] = L.T @ xi[:,np.newaxis] @ xi[np.newaxis,:] @ L
   sig = np.sqrt((np.mean(Sigma_hat[:,0,0]) + np.mean(Sigma_hat[:,1,1]))/2)
   Sigma_hat = lambda s: sig**2*np.array([[1+0*s, 0*s], [0*s, 1+0*s]])
   
   # P0_hat
   P0_hat = sig**2*np.eye(6)

   ####     Run the EM    ####
   try:
      FS_statespace = FrenetStateSpace(grid_arc_s, Y)
      FS_statespace.expectation_maximization(tol_EM, max_iter_EM, nb_basis=nb_basis, regularization_parameter_list=reg_param_grid_EM, init_params = {"W":Gamma_hat, "coefs":Bspline_approxODE.coefs, "mu0":mu0_hat, "Sigma":Sigma_hat, "P0":P0_hat}, init_states = None, method='autre', model_Sigma='scalar')
   except:
      FS_statespace = [sig, Bspline_approxODE, Z_GS]

   return FS_statespace, Z_GS, Z



def scenario_1_3(theta, Sigma, mu0, P0, Gamma, N, arc_length_fct, nb_basis, bandwidth_grid_init, reg_param_grid_init, reg_param_grid_EM, max_iter_EM, tol_EM):
   """ Scenario 1.3: Initialization method LP + GS + Local Approx Ode """
   
    ####     Generation of one sample of the model    ####

   Z, Q, X, Y = model_scenario(theta, Sigma, mu0, P0, Gamma, N, arc_length_fct)

   ####     Initialization of parameters    ####

   time = np.linspace(0,1,N)
   derivatives, h_opt = compute_derivatives(Y, time, deg=3, h=None, CV_optimization_h={"flag":True, "h_grid":bandwidth_grid_init, "K":10})
   grid_arc_s, L, arc_s, arc_s_dot = compute_arc_length(Y, time, smooth=True, smoothing_param=h_opt)

   # Gamma
   Gamma_hat = ((Y - derivatives[0]).T @ (Y - derivatives[0]))/N

   # Z_hat
   GS_orthog = GramSchmidtOrthogonalization(Y, grid_arc_s, deg=3)
   h_opt, err_h = GS_orthog.grid_search_CV_optimization_bandwidth(bandwidth_grid=bandwidth_grid_init, K_split=10)
   Z_GS, Q_GS, X_GS = GS_orthog.fit(h_opt) 

   # mu0_hat
   mu0_hat = Z_GS[0]

   # theta_hat
   local_approx_ode = LocalApproxFrenetODE(grid_arc_s, Z=Z_GS)
   h_opt, nb_basis_opt, regularization_parameter_opt, CV_error_tab = local_approx_ode.grid_search_optimization_hyperparameters(bandwidth_list=bandwidth_grid_init, nb_basis_list=np.array([nb_basis]), regularization_parameter_list=reg_param_grid_init, method='2', parallel=False)
   Bspline_localapproxODE = local_approx_ode.Bspline_smooth_estimates(h_opt, nb_basis, regularization_parameter=regularization_parameter_opt)


   # Sigma_hat
   mu_Z_theta = solve_FrenetSerret_ODE_SE(Bspline_localapproxODE.evaluate, grid_arc_s, Z0=mu0_hat, method='Linearized')
   Sigma_hat = np.zeros((N, 2, 2))
   for i in range(N):
      L = np.zeros((6,2))
      L[0,1], L[2,0] = 1, 1
      xi = -SE3.log(np.linalg.inv(mu_Z_theta[i])@Z_GS[i])
      Sigma_hat[i] = L.T @ xi[:,np.newaxis] @ xi[np.newaxis,:] @ L
   sig = np.sqrt((np.mean(Sigma_hat[:,0,0]) + np.mean(Sigma_hat[:,1,1]))/2)
   Sigma_hat = lambda s: sig**2*np.array([[1+0*s, 0*s], [0*s, 1+0*s]])
   
   # P0_hat
   P0_hat = sig**2*np.eye(6)

   ####     Run the EM    ####
   try:
      FS_statespace = FrenetStateSpace(grid_arc_s, Y)
      FS_statespace.expectation_maximization(tol_EM, max_iter_EM, nb_basis=nb_basis, regularization_parameter_list=reg_param_grid_EM, init_params = {"W":Gamma_hat, "coefs":Bspline_localapproxODE.coefs, "mu0":mu0_hat, "Sigma":Sigma_hat, "P0":P0_hat}, init_states = None, method='autre', model_Sigma='scalar')
   except:
      FS_statespace = [sig, Bspline_localapproxODE, Z_GS]

   return FS_statespace, Z_GS, Z



def scenario_1_4(theta, Sigma, mu0, P0, Gamma, N, arc_length_fct, nb_basis, bandwidth_grid_init, reg_param_grid_init, reg_param_grid_EM, max_iter_EM, tol_EM):
   """ Scenario 1.4: Initialization method CLP + Approx Ode """
   
   ####     Generation of one sample of the model    ####

   Z, Q, X, Y = model_scenario(theta, Sigma, mu0, P0, Gamma, N, arc_length_fct)

   ####     Initialization of parameters    ####

   time = np.linspace(0,1,N)
   derivatives, h_opt = compute_derivatives(Y, time, deg=3, h=None, CV_optimization_h={"flag":True, "h_grid":bandwidth_grid_init, "K":10})
   grid_arc_s, L, arc_s, arc_s_dot = compute_arc_length(Y, time, smooth=True, smoothing_param=h_opt)

   # Gamma
   Gamma_hat = ((Y - derivatives[0]).T @ (Y - derivatives[0]))/N

   # Z_hat
   CLP_reg = ConstrainedLocalPolynomialRegression(Y, grid_arc_s, adaptative=False, deg_polynomial=3)
   h_opt, err_h = CLP_reg.grid_search_CV_optimization_bandwidth(bandwidth_grid=bandwidth_grid_init, K_split=10)
   Z_CLP, Q_CLP, X_CLP = CLP_reg.fit(h_opt)

   # mu0_hat
   mu0_hat = Z_CLP[0]

   # theta_hat
   approx_ode = ApproxFrenetODE(grid_arc_s, Z=Z_CLP)
   nb_basis_opt, regularization_parameter_opt, tab_GCV_scores = approx_ode.grid_search_optimization_hyperparameters(nb_basis_list=np.array([nb_basis]), regularization_parameter_list=reg_param_grid_init)
   Bspline_approxODE = approx_ode.Bspline_smooth_estimates(nb_basis, regularization_parameter=regularization_parameter_opt)

   # Sigma_hat
   mu_Z_theta_extrins = solve_FrenetSerret_ODE_SE(Bspline_approxODE.evaluate, grid_arc_s, Z0=mu0_hat, method='Linearized')
   Sigma_hat = np.zeros((N, 2, 2))
   for i in range(N):
      L = np.zeros((6,2))
      L[0,1], L[2,0] = 1, 1
      xi = -SE3.log(np.linalg.inv(mu_Z_theta_extrins[i])@Z_CLP[i])
      Sigma_hat[i] = L.T @ xi[:,np.newaxis] @ xi[np.newaxis,:] @ L
   sig = np.sqrt((np.mean(Sigma_hat[:,0,0]) + np.mean(Sigma_hat[:,1,1]))/2)
   Sigma_hat = lambda s: sig**2*np.array([[1+0*s, 0*s], [0*s, 1+0*s]])
   
   # P0_hat
   P0_hat = sig**2*np.eye(6)

   ####     Run the EM    ####
   try:
      FS_statespace = FrenetStateSpace(grid_arc_s, Y)
      FS_statespace.expectation_maximization(tol_EM, max_iter_EM, nb_basis=nb_basis, regularization_parameter_list=reg_param_grid_EM, init_params = {"W":Gamma_hat, "coefs":Bspline_approxODE.coefs, "mu0":mu0_hat, "Sigma":Sigma_hat, "P0":P0_hat}, init_states = None, method='autre', model_Sigma='scalar')
   except:
      FS_statespace = [sig, Bspline_approxODE, Z_CLP]
   
   return FS_statespace, Z_CLP, Z



def scenario_1_5(theta, Sigma, mu0, P0, Gamma, N, arc_length_fct, nb_basis, bandwidth_grid_init, reg_param_grid_init, reg_param_grid_EM, max_iter_EM, tol_EM):
   """ Scenario 1.5: Initialization method CLP + Local Approx Ode """
   
   ####     Generation of one sample of the model    ####

   Z, Q, X, Y = model_scenario(theta, Sigma, mu0, P0, Gamma, N, arc_length_fct)

   ####     Initialization of parameters    ####

   time = np.linspace(0,1,N)
   derivatives, h_opt = compute_derivatives(Y, time, deg=3, h=None, CV_optimization_h={"flag":True, "h_grid":bandwidth_grid_init, "K":10})
   grid_arc_s, L, arc_s, arc_s_dot = compute_arc_length(Y, time, smooth=True, smoothing_param=h_opt)

   # Gamma
   Gamma_hat = ((Y - derivatives[0]).T @ (Y - derivatives[0]))/N

   # Z_hat
   CLP_reg = ConstrainedLocalPolynomialRegression(Y, grid_arc_s, adaptative=False, deg_polynomial=3)
   h_opt, err_h = CLP_reg.grid_search_CV_optimization_bandwidth(bandwidth_grid=bandwidth_grid_init, K_split=10)
   Z_CLP, Q_CLP, X_CLP = CLP_reg.fit(h_opt)

   # mu0_hat
   mu0_hat = Z_CLP[0]

   # theta_hat
   local_approx_ode = LocalApproxFrenetODE(grid_arc_s, Z=Z_CLP)
   h_opt, nb_basis_opt, regularization_parameter_opt, CV_error_tab = local_approx_ode.grid_search_optimization_hyperparameters(bandwidth_list=bandwidth_grid_init, nb_basis_list=np.array([nb_basis]), regularization_parameter_list=reg_param_grid_init, method='2', parallel=False)
   Bspline_localapproxODE = local_approx_ode.Bspline_smooth_estimates(h_opt, nb_basis, regularization_parameter=regularization_parameter_opt)


   # Sigma_hat
   mu_Z_theta_extrins = solve_FrenetSerret_ODE_SE(Bspline_localapproxODE.evaluate, grid_arc_s, Z0=mu0_hat, method='Linearized')
   Sigma_hat = np.zeros((N, 2, 2))
   for i in range(N):
      L = np.zeros((6,2))
      L[0,1], L[2,0] = 1, 1
      xi = -SE3.log(np.linalg.inv(mu_Z_theta_extrins[i])@Z_CLP[i])
      Sigma_hat[i] = L.T @ xi[:,np.newaxis] @ xi[np.newaxis,:] @ L
   sig = np.sqrt((np.mean(Sigma_hat[:,0,0]) + np.mean(Sigma_hat[:,1,1]))/2)
   Sigma_hat = lambda s: sig**2*np.array([[1+0*s, 0*s], [0*s, 1+0*s]])
   
   # P0_hat
   P0_hat = sig**2*np.eye(6)

   ####     Run the EM    ####
   try:
      FS_statespace = FrenetStateSpace(grid_arc_s, Y)
      FS_statespace.expectation_maximization(tol_EM, max_iter_EM, nb_basis=nb_basis, regularization_parameter_list=reg_param_grid_EM, init_params = {"W":Gamma_hat, "coefs":Bspline_localapproxODE.coefs, "mu0":mu0_hat, "Sigma":Sigma_hat, "P0":P0_hat}, init_states = None, method='autre', model_Sigma='scalar')
   except:
      FS_statespace = [sig, Bspline_localapproxODE, Z_CLP]

   return FS_statespace, Z_CLP, Z



def scenario_2(theta, Sigma, mu0, P0, Gamma, N, arc_length_fct, nb_basis, bandwidth_grid_init, reg_param_grid_init, reg_param_grid_EM, max_iter_EM, tol_EM):
   """ Scenario 2: Initialization fixed to ... and only sigma will be modified. """

   FS_statespace, Z_init, Z = scenario_1_1(theta, Sigma, mu0, P0, Gamma, N, arc_length_fct, nb_basis, bandwidth_grid_init, reg_param_grid_init, reg_param_grid_EM, max_iter_EM, tol_EM)

   return FS_statespace, Z_init, Z 



def scenario_1_1_bis(theta, Sigma, mu0, P0, Gamma, N, arc_length_fct, nb_basis, bandwidth_grid_init, reg_param_grid_init, reg_param_grid_EM, max_iter_EM, tol_EM):
   """ Scenario 1.1: Initialization method LP + GS + Extrinsic formulas """

   # try:
   
   ####     Generation of one sample of the model    ####

   Z, Q, X, Y = model_scenario(theta, Sigma, mu0, P0, Gamma, N, arc_length_fct)

   ####     Initialization of parameters    ####

   grid_time = np.linspace(0,1,N)
   derivatives, h_opt = compute_derivatives(Y, grid_time, deg=3, h=None, CV_optimization_h={"flag":True, "h_grid":bandwidth_grid_init, "K":10})
   grid_arc_s, L, arc_s, arc_s_dot = compute_arc_length(Y, grid_time, smooth=True, smoothing_param=h_opt)

   # Gamma
   Gamma_hat = ((Y - derivatives[0]).T @ (Y - derivatives[0]))/N

   # Z_hat  
   # print('GS')
   GS_orthog = GramSchmidtOrthogonalization(Y, grid_arc_s, deg=3)
   h_opt, err_h = GS_orthog.grid_search_CV_optimization_bandwidth(bandwidth_grid=bandwidth_grid_init, K_split=10)
   Z_GS, Q_GS, X_GS = GS_orthog.fit(h_opt) 

   # mu0_hat
   mu0_hat = Z_GS[0]

   # theta_hat
   # print('Extrins')
   extrins_model_theta = ExtrinsicFormulas(Y, grid_time, grid_arc_s, deg_polynomial=3)
   h_opt, nb_basis_opt, regularization_parameter_opt, CV_error_tab = extrins_model_theta.grid_search_optimization_hyperparameters(bandwidth_grid_init, np.array([nb_basis]), reg_param_grid_init, method='2', n_splits=10)
   Basis_extrins = extrins_model_theta.Bspline_smooth_estimates(h_opt, nb_basis_opt, regularization_parameter=regularization_parameter_opt)
   # Basis_extrins.plot()

   # Sigma_hat
   # mu_Z_theta_extrins = solve_FrenetSerret_ODE_SE(Basis_extrins.evaluate, grid_arc_s, Z0=mu0_hat, method='Linearized')
   # Sigma_hat = np.zeros((N, 2, 2))
   # for i in range(N):
   #    L = np.zeros((6,2))
   #    L[0,1], L[2,0] = 1, 1
   #    xi = -SE3.log(np.linalg.inv(mu_Z_theta_extrins[i])@Z_GS[i])
   #    Sigma_hat[i] = L.T @ xi[:,np.newaxis] @ xi[np.newaxis,:] @ L
   # sig = np.sqrt((np.mean(Sigma_hat[:,0,0]) + np.mean(Sigma_hat[:,1,1]))/2)
   # print('sigma:', sig)
   sig = 0.03
   Sigma_hat = lambda s: sig**2*np.array([[1+0*s, 0*s], [0*s, 1+0*s]])
   
   # P0_hat
   P0_hat = sig**2*np.eye(6)

   # print('EM')
   ####     Run the EM    ####
   FS_statespace = FrenetStateSpace(grid_arc_s, Y)
   FS_statespace.expectation_maximization(tol_EM, max_iter_EM, nb_basis=nb_basis, regularization_parameter_list=reg_param_grid_EM, init_params = {"W":Gamma_hat, "coefs":Basis_extrins.coefs, "mu0":mu0_hat, "Sigma":Sigma_hat, "P0":P0_hat}, init_states = None, method='autre', model_Sigma='scalar')

   return FS_statespace, Z_GS, Z
   
   # except:

   #    return None

   