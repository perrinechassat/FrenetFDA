from FrenetFDA.processing_Euclidean_curve.unified_estimate_Frenet_state_space.iek_filter_smoother_Frenet_state import IEKFilterSmootherFrenetState
import FrenetFDA.utils.visualization as visu
from FrenetFDA.utils.Lie_group.SE3_utils import SE3
from FrenetFDA.utils.Lie_group.SO3_utils import SO3
from FrenetFDA.utils.smoothing_utils import VectorBSplineSmoothing
from FrenetFDA.utils.Frenet_Serret_utils import *
from pickle import *
import os.path
import dill as pickle
import numpy as np
from scipy.linalg import block_diag
from scipy.interpolate import interp1d
from skfda.representation.basis import VectorValued, BSpline
from skfda.misc.regularization import TikhonovRegularization, compute_penalty_matrix
from skfda.misc.operators import LinearDifferentialOperator
import time as ttime
from sklearn.model_selection import KFold
from skopt import gp_minimize
from skopt.plots import plot_convergence


class FrenetStateSpaceCV_byit:


    def __init__(self, grid_obs, Y_obs, dim=3):
        self.n = dim
        self.dim_g = int((1/2)*dim*(dim+1)) # dimension of the Lie Algebra se(n)
        self.Y = Y_obs[1:] #array of shape (N,n,)
        self.grid = grid_obs
        self.domain_range = (grid_obs[0], grid_obs[-1])
        self.rho = np.array([1,0,0])
        self.H = np.hstack((np.zeros((self.n, self.n)), np.eye(self.n)))
        # Def for case dim=3
        self.L = np.array([[0,1],[0,0],[1,0],[0,0],[0,0],[0,0]])

    def expectation_maximization(self, tol, max_iter, nb_basis, bounds_lambda, n_calls, init_params = None, order=4, method='approx', model_Sigma='scalar', score_lambda='MSE_YmuX', verbose=False, n_splits_CV=5):

        self.verbose = verbose
        self.v = (1/2)*(self.grid[1:]+self.grid[:-1])
        self.init_tab()

        if init_params is None:
            raise Exception("Must pass in argument an initial guess estimate for the parameters")
        else:
            self.Gamma = init_params["Gamma"]
            self.coefs = init_params["coefs"]
            self.mu0 = init_params["mu0"]
            self.Sigma = init_params["Sigma"]
            self.P0 = init_params["P0"]

        # N_param_smoothing = len(regularization_parameter_list)    
        # if N_param_smoothing==0:
        #     penalization = False
        #     regularization_parameter_list = np.array([0])
        # else:
        #     penalization = True
        self.Bspline_decomp = VectorBSplineSmoothing(self.n-1, nb_basis, domain_range=self.domain_range, order=order, penalization=True)

        k = 0
        rel_error = 2*tol
        val_expected_loglikelihood = 0
        self.tab_increment()

        time_init = ttime.time()

        while k < max_iter and rel_error > tol:
            if self.verbose:
                print('-------------- Iteration',k+1,'/',max_iter,' --------------')

            self.iteration_EM(n_splits_CV, bounds_lambda, n_calls, method, model_Sigma, score_lambda)

            new_val_expected_loglikelihood = self.expected_loglikelihood()
            if self.verbose:
                print('value of expected_loglikelihood: ', new_val_expected_loglikelihood)
            rel_error = np.abs(val_expected_loglikelihood - new_val_expected_loglikelihood)
            if self.verbose:
                print('relative error: ', rel_error)
            self.tab_increment(rel_error, new_val_expected_loglikelihood)
            val_expected_loglikelihood = new_val_expected_loglikelihood
            k+=1
            
        time_end = ttime.time()
        self.nb_iterations = k
        self.duration = time_end - time_init

        if self.verbose:
            print('End expectation maximization algo. \n Number of iterations:', k, ', total duration:', self.duration, ' seconds.')



    def iteration_EM(self, n_CV, bounds_lambda, n_calls, method='approx', model_Sigma='scalar', score='MSE_YX'):

        ## CV optimization of lambda
        def func(x):
            score_lambda = np.zeros(n_CV)
            kf = KFold(n_splits=n_CV, shuffle=True)
            ind_CV = 0

            for train_index, test_index in kf.split(self.grid[1:]):
                # print('     --> Start CV step n°', ind_CV+1)
                Y_train = self.Y[train_index]
                Y_test = self.Y[test_index]
                grid_train = np.concatenate((np.array([self.grid[0]]), self.grid[1:][train_index]))
                grid_test = np.concatenate((np.array([self.grid[0]]), self.grid[1:][test_index]))

                self.N = len(Y_train)
                self.u = grid_train[1:] - grid_train[:-1]
                self.v = (grid_train[1:] + grid_train[:-1])/2
                self.grid_obs = grid_train
                self.basis_matrix = self.Bspline_decomp.basis(np.expand_dims(self.v, 1),).reshape((self.Bspline_decomp.basis.n_basis, -1)).T
                self.E_step(grid_train, Y_train)
                self.__approx_distribution_r(method)
                Gamma, P0, mu0, coefs, L_tilde, Sigma, sigma_square, regularization_parameter = self.M_step(0.001, 5, x, model_Sigma)

                if score=='MSE_YX':
                    Z_reconst = solve_FrenetSerret_SDE_SE3(lambda s: self.theta_from_coefs(coefs, s), Sigma, self.L, grid_test, Z0=mu0)
                    X_reconst_test = Z_reconst[1:,:3,3]
                    score_lambda[ind_CV] = np.linalg.norm(X_reconst_test - Y_test)**2

                elif score=='MSE_YmuX':
                    Z_reconst = solve_FrenetSerret_ODE_SE(lambda s: self.theta_from_coefs(coefs, s), grid_test, Z0=mu0)
                    X_reconst_test = Z_reconst[1:,:3,3]
                    score_lambda[ind_CV] = np.linalg.norm(X_reconst_test - Y_test)**2

                ind_CV += 1

            # print(score_lambda)
            print('val x:', x, 'score:', np.mean(score_lambda))
            return np.mean(score_lambda)

        # Do a bayesian optimisation and return the optimal parameter (lambda_kappa, lambda_tau)
        
        res = gp_minimize(func,               # the function to minimize
                        bounds_lambda,        # the bounds on each dimension of x
                        acq_func="EI",        # the acquisition function
                        n_calls=n_calls,       # the number of evaluations of f
                        n_random_starts=2,    # the number of random initialization points
                        random_state=1,       # the random seed
                        n_jobs=1,            # use all the cores for parallel calculation
                        verbose=True)
        lbda_opt = res.x
        print(res.x_iters)
        print(res.func_vals)
        lbda_opt = np.array([lbda_opt[0], lbda_opt[1]])
        print('the optimal hyperparameters selected are: ', lbda_opt)

        self.N = len(self.Y)
        self.grid_obs = self.grid
        self.u = self.grid[1:] - self.grid[:-1]
        self.v = (1/2)*(self.grid[1:]+self.grid[:-1])
        self.basis_matrix = self.Bspline_decomp.basis(np.expand_dims(self.v, 1),).reshape((self.Bspline_decomp.basis.n_basis, -1)).T
        self.E_step(self.grid, self.Y)
        self.__approx_distribution_r(method)
        self.Gamma, self.P0, self.mu0, self.coefs, self.L_tilde, self.Sigma, self.sigma_square, self.regularization_parameter = self.M_step(0.001, 5, lbda_opt, model_Sigma)


    def E_step(self, grid, Y):
        """
            Expectation step: Suppose that self.sigma, self.Gamma, self.a_theta, self.mu0, self.P0 are known. Call the tracking and smoothing method.

        """
        if self.verbose:
            print('___ E step ___')
        kalman_filter = IEKFilterSmootherFrenetState(self.n, self.Gamma, self.Sigma, self.theta, Z0=self.mu0, P0=self.P0)
        kalman_filter.smoothing(grid, Y)
        self.Z = kalman_filter.smooth_Z
        self.Q = kalman_filter.smooth_Q
        self.X = kalman_filter.smooth_X
        self.P = kalman_filter.smooth_P
        self.P_dble = kalman_filter.smooth_P_dble


    def M_step(self, tol, max_iter, lbda, model_Sigma):
        """
            Maximization step:

        """
        if self.verbose:
            print('___ M step ___')

        # Optimization of W
        Gamma = self.opti_Gamma()
        P0 = self.P[0]
        mu0 = self.Z[0]

        # Optimization of theta given lambda
        regularization_parameter, regularization_parameter_matrix = self.Bspline_decomp.check_regularization_parameter(lbda)
        coefs, mat_weights, weights, L_tilde = self.opti_coefs(tol, max_iter, regularization_parameter_matrix)
        sigma_square, Sigma, expect_MSE = self.opti_Sigma(coefs, weights, regularization_parameter_matrix)
        # self.coefs, self.Sigma, self.mat_weights, self.weights, self.expect_MSE, err_obs, self.L_tilde = self.opti_coefs_and_Sigma(tol, max_iter, self.regularization_parameter_matrix, model_Sigma)
        # self.plot_theta()
        return Gamma, P0, mu0, coefs, L_tilde, Sigma, sigma_square, regularization_parameter
        

    def theta_from_coefs(self, coefs, s):
        if isinstance(s, int) or isinstance(s, float):
            return np.squeeze(self.Bspline_decomp.basis_fct(s).T @ coefs)
        elif isinstance(s, np.ndarray):
            return np.squeeze((self.Bspline_decomp.basis_fct(s).T @ coefs).T)
        else:
            raise ValueError('Variable is not a float, a int or a NumPy array.')
        

    def theta(self, s):
        if isinstance(s, int) or isinstance(s, float):
            return np.squeeze(self.Bspline_decomp.basis_fct(s).T @ self.coefs)
        elif isinstance(s, np.ndarray):
            return np.squeeze((self.Bspline_decomp.basis_fct(s).T @ self.coefs).T)
        else:
            raise ValueError('Variable is not a float, a int or a NumPy array.')
    

    def plot_theta(self):
        visu.plot_array_2D(self.v, [self.r_tilde[:,0], self.theta(self.v)[:,0]], 'curv ')
        visu.plot_array_2D(self.v, [self.r_tilde[:,1], self.theta(self.v)[:,1]], 'tors ')



    def __approx_distribution_r(self, method='monte_carlo', N_rand=500):
        self.r_tilde = np.zeros((self.N, self.n-1))
        self.cov_r_tilde = np.zeros((self.N, self.n-1, self.n-1))

        if method=='monte_carlo':
            r = np.zeros((self.N, self.dim_g))
            cov_r = np.zeros((self.N, self.dim_g, self.dim_g))
            for i in range(self.N):
                mat_cov = np.hstack((np.vstack((self.P[i], self.P_dble[i].T)), np.vstack((self.P_dble[i], self.P[i+1]))))
                set_rand_xi = np.random.multivariate_normal(np.zeros(2*self.dim_g), mat_cov, size=N_rand)
                rand_obs = np.zeros((N_rand, self.dim_g))
                for j in range(N_rand):
                    rand_obs[j] = -(1/self.u[i])*SE3.log(SE3.exp(set_rand_xi[j][6:])@np.linalg.inv(self.Z[i+1])@self.Z[i]@SE3.exp(-set_rand_xi[j][:6]))
                    r[i] += rand_obs[j]
                r[i] = r[i]/N_rand
                for j in range(N_rand):
                    cov_r[i] += (rand_obs[j] - r[i])[:,np.newaxis] @ (rand_obs[j] - r[i])[:,np.newaxis].T
                cov_r[i] = cov_r[i]/N_rand
                self.cov_r_tilde[i] = self.L.T @ cov_r[i] @ self.L
                self.r_tilde[i] = self.L.T @ r[i]

        else: 
            for i in range(self.N):
                inv_Zi1_Zi = np.linalg.inv(self.Z[i+1])@self.Z[i]
                self.r_tilde[i] = -(1/self.u[i])*(self.L.T @ SE3.log(inv_Zi1_Zi))
                Ad = SE3.Ad_group(inv_Zi1_Zi)
                self.cov_r_tilde[i] = (1/self.u[i]**2)*self.L.T @ (self.P[i+1] - self.P_dble[i].T @ Ad.T - Ad @ self.P_dble[i] + Ad @ self.P[i] @ Ad.T) @ self.L


    def __compute_weights(self, coefs):
        L_tilde = np.zeros((self.N,self.n-1,self.n-1))
        L_tilde_inv = np.zeros((self.N,self.n-1,self.n-1))
        theta_v = np.reshape(self.basis_matrix @ coefs, (-1,self.n-1))
        weights = np.zeros((self.N,self.n-1,self.n-1))
        for i in range(self.N):
            phi_vi = SE3.Ad_group(SE3.exp(-(self.grid_obs[i+1]-self.v[i])*np.array([theta_v[i,1],0,theta_v[i,0],1,0,0])))
            L_tilde[i] = self.L.T @ phi_vi @ self.L
            L_tilde_inv[i] = np.linalg.inv(L_tilde[i])
            weights[i] = self.u[i]*L_tilde_inv[i].T@L_tilde_inv[i]
        mat_weights = block_diag(*weights)
        return mat_weights, weights, L_tilde
        
    
    def __opti_coefs(self, mat_weights, reg_param_mat):
        left = self.basis_matrix.T @ mat_weights @ self.basis_matrix + reg_param_mat @ self.Bspline_decomp.penalty_matrix    
        right = self.basis_matrix.T @ mat_weights @ np.reshape(self.r_tilde, (self.N*(self.n-1),))
        coefs_tem = np.linalg.solve(left, right)
        coefs_tem = np.reshape(coefs_tem,(-1,2))
        new_coefs = np.reshape(coefs_tem, (-1,))
        return new_coefs
    

    def opti_Sigma(self, coefs, weights, reg_param_mat):
        res = np.zeros((self.N, self.n-1, self.n-1))
        err = np.reshape(self.basis_matrix @ coefs, (-1,self.n-1)) - self.r_tilde
        for i in range(self.N):
            res[i] = weights[i] @ (err[i][:,np.newaxis] @ err[i][np.newaxis,:] + self.cov_r_tilde[i]) 
        sigma_square = (1/(2*self.N))*(np.trace(np.sum(res, axis=0))) # + coefs.T @ reg_param_mat @ self.Bspline_decomp.penalty_matrix @ coefs)
        print('sigma_square :', sigma_square, (1/(2*self.N))*np.trace(np.sum(res, axis=0)), (1/(2*self.N))*(coefs.T @ reg_param_mat @ self.Bspline_decomp.penalty_matrix @ coefs))
        Sigma = lambda s: sigma_square*np.eye(2)
        return sigma_square, Sigma, res
    
    
    def opti_coefs(self, tol, max_iter, reg_param_mat):
        mat_weights, weights, L_tilde = self.__compute_weights(self.coefs)
        old_theta = np.reshape(self.basis_matrix @ self.coefs, (-1,self.n-1))
        rel_error = 2*tol 
        k = 0 
        while rel_error > tol and k < max_iter:

            coefs_opt = self.__opti_coefs(mat_weights, reg_param_mat)
            mat_weights, weights, L_tilde = self.__compute_weights(coefs_opt)
            new_theta = np.reshape(self.basis_matrix @ coefs_opt, (-1,self.n-1))
            rel_error = np.linalg.norm(old_theta - new_theta)/np.linalg.norm(old_theta)
            if self.verbose:
                print('iteration for optimization of coefs and Sigma:', k, ', relative error:', rel_error)
            old_theta = new_theta
            k += 1

        return coefs_opt, mat_weights, weights, L_tilde
    

    def opti_Gamma(self):
        Gamma = np.zeros((self.n,self.n))
        for i in range(1,self.N+1):
            Gamma += (self.Y[i-1]-self.X[i])[:,np.newaxis]@(self.Y[i-1]-self.X[i])[np.newaxis,:] + self.Q[i]@self.H@self.P[i]@self.H.T@self.Q[i].T
        Gamma = Gamma/self.N
        return Gamma


    def expected_loglikelihood(self):
        # P0 / mu0
        val = np.log(np.linalg.det(self.P0))
        print('v P0:', -np.log(np.linalg.det(self.P0))) 
        # Gamma
        val += self.N*np.log(np.linalg.det(self.Gamma)) 
        print('v gamma:', -self.N*np.log(np.linalg.det(self.Gamma)))
        # Theta / Sigma
        val += 2*self.N*np.log(self.sigma_square)
        print('v sigma:', -2*self.N*np.log(self.sigma_square))
        # print('val l1:', val)
        v_bis = 0
        for i in range(self.N):
            v_bis += np.log(np.linalg.det( self.u[i]*self.L_tilde[i] @ self.L_tilde[i].T ))
            val += np.log(np.linalg.det( self.u[i]*self.L_tilde[i] @ self.L_tilde[i].T )) # Sigma deja dans W_tilde
            # val += (1/self.sigma_square)*np.trace(self.expect_MSE[i]) 

            # val += (1/2)*np.log(np.linalg.det(self.W_tilde[i])) # Sigma deja dans W_tilde
            # val += (1/2)*np.trace(np.linalg.inv(self.Sigma(self.v[i]))@self.residuals[i]) 
        # Penalization
        # print('val l2:', val)
        # val += (1/self.sigma_square)*self.coefs.T @ self.regularization_parameter_matrix @ self.Bspline_decomp.penalty_matrix @ self.coefs     
        # print('val l3:', val)   
        print('v weights:', -v_bis)
        return -val

    def init_tab(self):
        self.tab_rel_error = []
        self.tab_expected_loglikelihood = []
        self.tab_Z = []
        self.tab_X = []
        self.tab_Q = []
        self.tab_sigma = []
        self.tab_theta = []
        self.tab_coefs = []
        self.tab_Gamma = []
        self.tab_P0 = []
        self.tab_mu0 = []
        self.tab_regularization_parameter = []


    def tab_increment(self, rel_error=None, val_expected_loglikelihood=None):
        if rel_error is not None:
            self.tab_rel_error.append(rel_error)
        if val_expected_loglikelihood is not None:
            self.tab_expected_loglikelihood.append(val_expected_loglikelihood)
        if hasattr(self, "Z"):
            self.tab_Z.append(self.Z)
        if hasattr(self, "X"):
            self.tab_X.append(self.X)
        if hasattr(self, "Q"):
            self.tab_Q.append(self.Q)
        if hasattr(self, "Sigma"):
            # self.tab_sigma.append(np.array([self.Sigma(vi) for vi in self.v]))
            self.tab_sigma.append(np.sqrt(np.trace(self.Sigma(self.v[1]))/2))
        if hasattr(self, "theta"):
            self.tab_theta.append(self.theta(self.v))
        if hasattr(self, "coefs"):
            self.tab_coefs.append(self.coefs)
        if hasattr(self, "Gamma"):
            self.tab_Gamma.append(self.Gamma)
        if hasattr(self, "P0"):
            self.tab_P0.append(self.P0)
        if hasattr(self, "mu0"):
            self.tab_mu0.append(self.mu0)
        if hasattr(self, "regularization_parameter"):
            self.tab_regularization_parameter.append(self.regularization_parameter)
        

    def save_tab_results(self, filename):
        dic = {"tab_rel_error": self.tab_rel_error, "tab_expected_loglikelihood": self.tab_expected_loglikelihood, "tab_Z": self.tab_Z, "tab_X": self.tab_X, 
               "tab_Q": self.tab_Q, "tab_sigma": self.tab_sigma, "tab_theta": self.tab_theta, "tab_Gamma": self.tab_Gamma, "duration":self.duration, "nb_iterations":self.nb_iterations}
        if os.path.isfile(filename):
            print("Le fichier ", filename, " existe déjà.")
            filename = filename + '_bis'
        fil = open(filename,"xb")
        pickle.dump(dic,fil)
        fil.close()

   
    # def CV_split_Y(self, n_splits):
    #     Y_train = []
    #     grid_train = []
    #     Y_test = []
    #     grid_test = []
    #     kf = KFold(n_splits=n_splits, shuffle=True)
    #     for train_index, test_index in kf.split(self.grid[1:]):
    #         Y_train.append(self.Y[train_index])
    #         Y_test.append(self.Y[test_index])
    #         grid_train.append(np.concatenate((np.array([self.grid[0]]), self.grid[1:][train_index])))
    #         grid_test.append(np.concatenate((np.array([self.grid[0]]), self.grid[1:][test_index])))
    #     return Y_train, grid_train, Y_test, grid_test

    # def iteration_EM(self, Y_train, grid_train, Y_test, grid_test, bounds_lambda, n_calls, method='approx', model_Sigma='scalar', score_lambda='MSE_YX'):
    #     n_split = len(Y_train)
    #     # K = len(regularization_parameter_list)
    #     # score_lambda_matrix = np.zeros((K,K))
    #     for k in range(n_split):
    #         if self.verbose:
    #             print('Iteration CV:', k)
    #         self.N = len(Y_train[k])
    #         self.u = grid_train[k][1:] - grid_train[k][:-1]
    #         self.v = (grid_train[k][1:] + grid_train[k][:-1])/2
    #         self.grid_obs = grid_train[k]
    #         self.basis_matrix = self.Bspline_decomp.basis(np.expand_dims(self.v, 1),).reshape((self.Bspline_decomp.basis.n_basis, -1)).T
    #         self.E_step(grid_train[k], Y_train[k])
    #         self.__approx_distribution_r(method)
    #         # M step
    #         Gamma = self.opti_Gamma()
    #         P0 = self.P[0]
    #         mu0 = self.Z[0]
    #         score_lambda_matrix = score_lambda_matrix + self.opti_lambda(0.001, 5, regularization_parameter_list, model_Sigma, score_lambda, Y_test[k], grid_test[k], mu0)
    #     score_lambda_matrix = score_lambda_matrix/n_split
    #     self.score_lambda_matrix = score_lambda_matrix
    #     ind = np.squeeze(np.array(np.where(score_lambda_matrix==np.min(score_lambda_matrix))))
    #     lbda_opt = np.array([regularization_parameter_list[ind[0]], regularization_parameter_list[ind[1]]]) 
    #     self.regularization_parameter, self.regularization_parameter_matrix = self.Bspline_decomp.check_regularization_parameter(lbda_opt)
    #     if self.verbose:
    #         print('Optimal chosen lambda:', lbda_opt)
        
    #     self.N = len(self.Y)
    #     self.grid_obs = self.grid
    #     self.u = self.grid[1:] - self.grid[:-1]
    #     self.v = (1/2)*(self.grid[1:]+self.grid[:-1])
    #     self.basis_matrix = self.Bspline_decomp.basis(np.expand_dims(self.v, 1),).reshape((self.Bspline_decomp.basis.n_basis, -1)).T
    #     self.E_step(self.grid, self.Y)
    #     self.__approx_distribution_r(method)
    #     self.coefs, self.Sigma, self.mat_weights, self.weights, self.expect_MSE, err_obs, self.L_tilde = self.opti_coefs_and_Sigma(0.001, 5, self.regularization_parameter_matrix, model_Sigma)
        


    # def M_step(self, tol, max_iter, reg_param_list, model_Sigma, score_lbda, true_theta):
    #     """
    #         Maximization step:

    #     """
    #     if self.verbose:
    #         print('___ M step ___')

    #     # Optimization of W
    #     self.Gamma = self.opti_Gamma()
    #     self.P0 = self.P[0]
    #     self.mu0 = self.Z[0]

    #     # Optimization of lambda
    #     if len(reg_param_list)==1:
    #         self.regularization_parameter, self.regularization_parameter_matrix = self.Bspline_decomp.check_regularization_parameter(np.array([reg_param_list[0], reg_param_list[0]]))
    #     else:
    #         self.regularization_parameter, self.score_lambda_matrix, _ = self.opti_lambda(tol, max_iter, reg_param_list, model_Sigma, score_lbda, true_theta)
    #         self.regularization_parameter, self.regularization_parameter_matrix = self.Bspline_decomp.check_regularization_parameter(self.regularization_parameter)
        
    #     # Optimization of theta given lambda
    #     self.coefs, self.Sigma, self.mat_weights, self.weights, self.expect_MSE, err_obs, self.L_tilde = self.opti_coefs_and_Sigma(tol, max_iter, self.regularization_parameter_matrix, model_Sigma)
    #     # self.plot_theta()

    
    # def __compute_weights(self, Sigma, L_tilde_inv):
    #     weights = np.zeros((self.N,self.n-1,self.n-1))
    #     for i in range(self.N):
    #         weights[i] = self.u[i]*L_tilde_inv[i].T@np.linalg.inv(Sigma(self.v[i]))@L_tilde_inv[i]
    #     mat_weights = block_diag(*weights)
    #     return mat_weights, weights
        
    
    # def __opti_coefs(self, mat_weights, reg_param_mat):
    #     left = self.basis_matrix.T @ mat_weights @ self.basis_matrix + reg_param_mat @ self.Bspline_decomp.penalty_matrix    
    #     right = self.basis_matrix.T @ mat_weights @ np.reshape(self.r_tilde, (self.N*(self.n-1),))
    #     coefs_tem = np.linalg.solve(left, right)
    #     coefs_tem = np.reshape(coefs_tem,(-1,2))
    #     new_coefs = np.reshape(coefs_tem, (-1,))
    #     return new_coefs
    

    # def __compute_L_tilde(self, coefs):
    #     L_tilde = np.zeros((self.N,self.n-1,self.n-1))
    #     L_tilde_inv = np.zeros((self.N,self.n-1,self.n-1))
    #     theta_v = np.reshape(self.basis_matrix @ coefs, (-1,self.n-1))

    #     for i in range(self.N):
    #         phi_vi = SE3.Ad_group(SE3.exp(-(self.grid_obs[i+1]-self.v[i])*np.array([theta_v[i,1],0,theta_v[i,0],1,0,0])))
    #         L_tilde[i] = self.L.T @ phi_vi @ self.L
    #         L_tilde_inv[i] = np.linalg.inv(L_tilde[i])
    #     return L_tilde, L_tilde_inv


    # def __opti_Sigma(self, coefs, model_Sigma, L_tilde_inv):
    #     res = np.zeros((self.N, self.n-1, self.n-1))
    #     err = np.reshape(self.basis_matrix @ coefs, (-1,self.n-1)) - self.r_tilde
    #     for i in range(self.N):
    #         res[i] = self.u[i]*L_tilde_inv[i] @ (err[i][:,np.newaxis] @ err[i][np.newaxis,:] + self.cov_r_tilde[i]) @ L_tilde_inv[i]
    #     if model_Sigma=='vector':
    #         sigma_square_1 = np.mean(res[:,0,0])
    #         sigma_square_2 = np.mean(res[:,1,1])
    #         Sigma = lambda s: np.array([[sigma_square_1,0], [0, sigma_square_2]])
    #     else:                                                               # Case 'single_constant' by default
    #         sigma_square = np.mean((res[:,0,0]+res[:,1,1])/2)
    #         Sigma = lambda s: sigma_square*np.eye(2)
    #     return Sigma, res, err
    
    
    # def opti_coefs_and_Sigma(self, tol, max_iter, reg_param_mat, model_Sigma):

    #     coefs_opt = self.__opti_coefs(np.eye(self.basis_matrix.shape[0]), reg_param_mat)
    #     L_tilde, L_tilde_inv = self.__compute_L_tilde(coefs_opt)
    #     Sigma_opt, expect_MSE, err_obs = self.__opti_Sigma(coefs_opt, model_Sigma, L_tilde_inv)
    #     mat_weights, weights = self.__compute_weights(Sigma_opt, L_tilde_inv)

    #     old_theta = np.reshape(self.basis_matrix @ coefs_opt, (-1,self.n-1))
    #     rel_error = 2*tol 
    #     k = 0 
    #     while rel_error > tol and k < max_iter:
            
    #         coefs_opt = self.__opti_coefs(mat_weights, reg_param_mat)
    #         L_tilde, L_tilde_inv = self.__compute_L_tilde(coefs_opt)
    #         Sigma_opt, expect_MSE, err_obs = self.__opti_Sigma(coefs_opt, model_Sigma, L_tilde_inv)
    #         mat_weights, weights = self.__compute_weights(Sigma_opt, L_tilde_inv)
            
    #         new_theta = np.reshape(self.basis_matrix @ coefs_opt, (-1,self.n-1))
    #         rel_error = np.linalg.norm(old_theta - new_theta)/np.linalg.norm(old_theta)
    #         if self.verbose:
    #             print('iteration for optimization of coefs and Sigma:', k, ', relative error:', rel_error)
    #         old_theta = new_theta
    #         k += 1

    #     return coefs_opt, Sigma_opt, mat_weights, weights, expect_MSE, err_obs, L_tilde
    

    # def opti_lambda(self, tol, max_iter, reg_param_list, model_Sigma, score, Y_test, grid_test, mu0):
    #     K = len(reg_param_list)
    #     score_lambda_matrix = np.zeros((K,K))
    #     for i in range(K):
    #         for j in range(K):
    #             reg_param = np.array([reg_param_list[i], reg_param_list[j]])
    #             lbda, reg_param_mat = self.Bspline_decomp.check_regularization_parameter(reg_param)
    #             coefs, Sigma, mat_weights, weights, expect_MSE, err_obs, L_tilde = self.opti_coefs_and_Sigma(tol, max_iter, reg_param_mat, model_Sigma)
    #             hat_matrix = self.basis_matrix @ np.linalg.inv(self.basis_matrix.T @ mat_weights @ self.basis_matrix + reg_param_mat @ self.Bspline_decomp.penalty_matrix) @ self.basis_matrix.T @ mat_weights
                
    #             if score=='MSE_YX':
    #                 Z_reconst = solve_FrenetSerret_SDE_SE3(lambda s: self.theta_from_coefs(coefs, s), Sigma, self.L, grid_test, Z0=mu0)
    #                 X_reconst_test = Z_reconst[1:,:3,3]
    #                 score_lambda_matrix[i,j] = np.linalg.norm(X_reconst_test - Y_test)**2

    #             elif score=='MSE_YmuX':
    #                 Z_reconst = solve_FrenetSerret_ODE_SE(lambda s: self.theta_from_coefs(coefs, s), grid_test, Z0=mu0)
    #                 X_reconst_test = Z_reconst[1:,:3,3]
    #                 score_lambda_matrix[i,j] = np.linalg.norm(X_reconst_test - Y_test)**2
    #             else: 
    #                 raise Exception("Invalid term for optimization of lamnda score.")
    
    #     return score_lambda_matrix
 

    # def opti_Gamma(self):
    #     Gamma = np.zeros((self.n,self.n))
    #     for i in range(1,self.N+1):
    #         Gamma += (self.Y[i-1]-self.X[i])[:,np.newaxis]@(self.Y[i-1]-self.X[i])[np.newaxis,:] + self.Q[i]@self.H@self.P[i]@self.H.T@self.Q[i].T
    #     Gamma = Gamma/self.N
    #     return Gamma


    # def expected_loglikelihood(self):
    #     val = (6/2)*self.N*np.log(2*np.pi)
    #     # P0 / mu0
    #     val += (1/2)*np.log(np.linalg.det(self.P0)) 
    #     # Gamma
    #     val += (1/2)*self.N*np.log(np.linalg.det(self.Gamma)) 
    #     # Theta / Sigma
    #     for i in range(self.N):
    #         val += (1/2)*np.log(np.linalg.det( self.u[i]*self.L_tilde[i] @ self.Sigma(self.v[i]) @ self.L_tilde[i].T )) # Sigma deja dans W_tilde
    #         val += (1/2)*np.trace(np.linalg.inv(self.Sigma(self.v[i]))@self.expect_MSE[i]) 
    #         # val += (1/2)*np.log(np.linalg.det(self.W_tilde[i])) # Sigma deja dans W_tilde
    #         # val += (1/2)*np.trace(np.linalg.inv(self.Sigma(self.v[i]))@self.residuals[i]) 
    #     # Penalization
    #     val += self.coefs.T @ self.regularization_parameter_matrix @ self.Bspline_decomp.penalty_matrix @ self.coefs        
    #     return -val














class FrenetStateSpaceCV_global:


    def __init__(self, grid_obs, Y_obs, bornes_theta, dim=3):
        self.n = dim
        self.dim_g = int((1/2)*dim*(dim+1)) # dimension of the Lie Algebra se(n)
        self.Y = Y_obs #array of shape (N,n,)
        self.grid = grid_obs
        self.domain_range = (bornes_theta[0], bornes_theta[1])
        self.N = len(self.Y)
        self.rho = np.array([1,0,0])
        self.H = np.hstack((np.zeros((self.n, self.n)), np.eye(self.n)))
        # Def for case dim=3
        self.L = np.array([[0,1],[0,0],[1,0],[0,0],[0,0],[0,0]])
        self.u = self.grid[1:] - self.grid[:-1]
        self.v = (1/2)*(self.grid[1:]+self.grid[:-1])


    def expectation_maximization(self, tol, max_iter, nb_basis, regularization_parameter, init_params = None, order=4, method='approx', model_Sigma='scalar', verbose=False):

        self.verbose = verbose
        self.init_tab()

        if init_params is None:
            raise Exception("Must pass in argument an initial guess estimate for the parameters")
        else:
            self.Gamma = init_params["Gamma"]
            self.coefs = init_params["coefs"]
            self.mu0 = init_params["mu0"]
            self.Sigma = init_params["Sigma"]
            self.P0 = init_params["P0"]
        
        self.regularization_parameter = regularization_parameter
        self.Bspline_decomp = VectorBSplineSmoothing(self.n-1, nb_basis, domain_range=self.domain_range, order=order, penalization=True)
        V = np.expand_dims(self.v, 1)
        self.basis_matrix = self.Bspline_decomp.basis(V,).reshape((self.Bspline_decomp.basis.n_basis, -1)).T

        k = 0
        rel_error = 2*tol
        val_expected_loglikelihood = 0
        self.tab_increment()

        time_init = ttime.time()

        while k < max_iter and rel_error > tol:
            if self.verbose:
                print('-------------- Iteration',k+1,'/',max_iter,' --------------')

            # st = ttime.time()
            self.E_step()
            # ft = ttime.time()
            # print('time E step:', ft-st)
            # st = ttime.time()
            self.__approx_distribution_r(method)
            # ft = ttime.time()
            # print('time approx step:', ft-st)
            # st = ttime.time()
            self.M_step(0.001, 5, model_Sigma)
            # ft = ttime.time()
            # print('time M step:', ft-st)

            new_val_expected_loglikelihood = self.expected_loglikelihood()
            if self.verbose:
                print('value of expected_loglikelihood: ', new_val_expected_loglikelihood)
            rel_error = np.abs(val_expected_loglikelihood - new_val_expected_loglikelihood)
            if self.verbose:
                print('relative error: ', rel_error)
            self.tab_increment(rel_error, new_val_expected_loglikelihood)
            val_expected_loglikelihood = new_val_expected_loglikelihood
            k+=1
            
        time_end = ttime.time()
        self.nb_iterations = k
        self.duration = time_end - time_init

        if self.verbose:
            print('End expectation maximization algo. \n Number of iterations:', k, ', total duration:', self.duration, ' seconds.')


    def E_step(self):
        """
            Expectation step: Suppose that self.sigma, self.Gamma, self.a_theta, self.mu0, self.P0 are known. Call the tracking and smoothing method.

        """
        if self.verbose:
            print('___ E step ___')
        kalman_filter = IEKFilterSmootherFrenetState(self.n, self.Gamma, self.Sigma, self.theta, Z0=self.mu0, P0=self.P0)
        kalman_filter.smoothing(self.grid, self.Y)
        self.Z = kalman_filter.smooth_Z
        self.Q = kalman_filter.smooth_Q
        self.X = kalman_filter.smooth_X
        self.P = kalman_filter.smooth_P
        self.P_dble = kalman_filter.smooth_P_dble


    def M_step(self, tol, max_iter, model_Sigma):
        """
            Maximization step:

        """
        if self.verbose:
            print('___ M step ___')

        # Optimization of W
        self.opti_Gamma()
        self.P0 = self.P[0]
        self.mu0 = self.Z[0]

        # Optimization of theta given lambda
        self.regularization_parameter, self.regularization_parameter_matrix = self.Bspline_decomp.check_regularization_parameter(self.regularization_parameter)
        self.coefs, self.mat_weights, self.weights, self.L_tilde = self.opti_coefs(tol, max_iter, self.regularization_parameter_matrix)
        self.sigma_square, self.Sigma, self.expect_MSE = self.opti_Sigma(self.coefs, self.weights, self.regularization_parameter_matrix)
        # self.plot_theta()


    def theta_from_coefs(self, coefs, s):
        if isinstance(s, int) or isinstance(s, float):
            return np.squeeze(self.Bspline_decomp.basis_fct(s).T @ coefs)
        elif isinstance(s, np.ndarray):
            return np.squeeze((self.Bspline_decomp.basis_fct(s).T @ coefs).T)
        else:
            raise ValueError('Variable is not a float, a int or a NumPy array.')
        

    def theta(self, s):
        if isinstance(s, int) or isinstance(s, float):
            return np.squeeze(self.Bspline_decomp.basis_fct(s).T @ self.coefs)
        elif isinstance(s, np.ndarray):
            return np.squeeze((self.Bspline_decomp.basis_fct(s).T @ self.coefs).T)
        else:
            raise ValueError('Variable is not a float, a int or a NumPy array.')
    

    def plot_theta(self):
        visu.plot_array_2D(self.v, [self.r_tilde[:,0], self.theta(self.v)[:,0]], 'curv ')
        visu.plot_array_2D(self.v, [self.r_tilde[:,1], self.theta(self.v)[:,1]], 'tors ')



    def __approx_distribution_r(self, method='monte_carlo', N_rand=500):
        self.r_tilde = np.zeros((self.N, self.n-1))
        self.cov_r_tilde = np.zeros((self.N, self.n-1, self.n-1))

        if method=='monte_carlo':
            r = np.zeros((self.N, self.dim_g))
            cov_r = np.zeros((self.N, self.dim_g, self.dim_g))
            for i in range(self.N):
                mat_cov = np.hstack((np.vstack((self.P[i], self.P_dble[i].T)), np.vstack((self.P_dble[i], self.P[i+1]))))
                set_rand_xi = np.random.multivariate_normal(np.zeros(2*self.dim_g), mat_cov, size=N_rand)
                rand_obs = np.zeros((N_rand, self.dim_g))
                for j in range(N_rand):
                    rand_obs[j] = -(1/self.u[i])*SE3.log(SE3.exp(set_rand_xi[j][6:])@np.linalg.inv(self.Z[i+1])@self.Z[i]@SE3.exp(-set_rand_xi[j][:6]))
                    r[i] += rand_obs[j]
                r[i] = r[i]/N_rand
                for j in range(N_rand):
                    cov_r[i] += (rand_obs[j] - r[i])[:,np.newaxis] @ (rand_obs[j] - r[i])[:,np.newaxis].T
                cov_r[i] = cov_r[i]/N_rand
                self.cov_r_tilde[i] = self.L.T @ cov_r[i] @ self.L
                self.r_tilde[i] = self.L.T @ r[i]

        else: 
            for i in range(self.N):
                inv_Zi1_Zi = np.linalg.inv(self.Z[i+1])@self.Z[i]
                self.r_tilde[i] = -(1/self.u[i])*(self.L.T @ SE3.log(inv_Zi1_Zi))
                Ad = SE3.Ad_group(inv_Zi1_Zi)
                self.cov_r_tilde[i] = (1/self.u[i]**2)*self.L.T @ (self.P[i+1] - self.P_dble[i].T @ Ad.T - Ad @ self.P_dble[i] + Ad @ self.P[i] @ Ad.T) @ self.L


    def __compute_weights(self, coefs):
        L_tilde = np.zeros((self.N,self.n-1,self.n-1))
        L_tilde_inv = np.zeros((self.N,self.n-1,self.n-1))
        theta_v = np.reshape(self.basis_matrix @ coefs, (-1,self.n-1))
        weights = np.zeros((self.N,self.n-1,self.n-1))
        for i in range(self.N):
            phi_vi = SE3.Ad_group(SE3.exp(-(self.grid[i+1]-self.v[i])*np.array([theta_v[i,1],0,theta_v[i,0],1,0,0])))
            L_tilde[i] = self.L.T @ phi_vi @ self.L
            L_tilde_inv[i] = np.linalg.inv(L_tilde[i])
            weights[i] = self.u[i]*L_tilde_inv[i].T@L_tilde_inv[i]
        mat_weights = block_diag(*weights)
        return mat_weights, weights, L_tilde
        
    
    def __opti_coefs(self, mat_weights, reg_param_mat):
        left = self.basis_matrix.T @ mat_weights @ self.basis_matrix + reg_param_mat @ self.Bspline_decomp.penalty_matrix    
        right = self.basis_matrix.T @ mat_weights @ np.reshape(self.r_tilde, (self.N*(self.n-1),))
        coefs_tem = np.linalg.solve(left, right)
        coefs_tem = np.reshape(coefs_tem,(-1,2))
        new_coefs = np.reshape(coefs_tem, (-1,))
        return new_coefs


    def opti_Sigma(self, coefs, weights, reg_param_mat):
        res = np.zeros((self.N, self.n-1, self.n-1))
        err = np.reshape(self.basis_matrix @ coefs, (-1,self.n-1)) - self.r_tilde
        for i in range(self.N):
            res[i] = weights[i] @ (err[i][:,np.newaxis] @ err[i][np.newaxis,:] + self.cov_r_tilde[i]) 
        sigma_square = (1/(2*self.N))*(np.trace(np.sum(res, axis=0)) + coefs.T @ reg_param_mat @ self.Bspline_decomp.penalty_matrix @ coefs)
        # print('sigma_square :', sigma_square, (1/(2*self.N))*np.trace(np.sum(res, axis=0)), (1/(2*self.N))*(coefs.T @ reg_param_mat @ self.Bspline_decomp.penalty_matrix @ coefs))
        Sigma = lambda s: sigma_square*np.eye(2)
        return sigma_square, Sigma, res
    
    
    def opti_coefs(self, tol, max_iter, reg_param_mat):
        mat_weights, weights, L_tilde = self.__compute_weights(self.coefs)
        old_theta = np.reshape(self.basis_matrix @ self.coefs, (-1,self.n-1))
        rel_error = 2*tol 
        k = 0 
        while rel_error > tol and k < max_iter:

            coefs_opt = self.__opti_coefs(mat_weights, reg_param_mat)
            mat_weights, weights, L_tilde = self.__compute_weights(coefs_opt)
            new_theta = np.reshape(self.basis_matrix @ coefs_opt, (-1,self.n-1))
            rel_error = np.linalg.norm(old_theta - new_theta)/np.linalg.norm(old_theta)
            if self.verbose:
                print('iteration for optimization of coefs and Sigma:', k, ', relative error:', rel_error)
            old_theta = new_theta
            k += 1

        return coefs_opt, mat_weights, weights, L_tilde
    

    def opti_Gamma(self):
        Gamma = np.zeros((self.n,self.n))
        for i in range(1,self.N+1):
            Gamma += (self.Y[i-1]-self.X[i])[:,np.newaxis]@(self.Y[i-1]-self.X[i])[np.newaxis,:] + self.Q[i]@self.H@self.P[i]@self.H.T@self.Q[i].T
        self.Gamma = Gamma/self.N


    def expected_loglikelihood(self):
        # P0 / mu0
        val = np.log(np.linalg.det(self.P0))
        # print('v P0:', -np.log(np.linalg.det(self.P0))) 
        # Gamma
        val += self.N*np.log(np.linalg.det(self.Gamma)) 
        # print('v gamma:', -self.N*np.log(np.linalg.det(self.Gamma)))
        # Theta / Sigma
        val += 2*self.N*np.log(self.sigma_square)
        # print('v sigma:', -2*self.N*np.log(self.sigma_square))
        # print('val l1:', val)
        v_bis = 0
        for i in range(self.N):
            v_bis += np.log(np.linalg.det( self.u[i]*self.L_tilde[i] @ self.L_tilde[i].T ))
            val += np.log(np.linalg.det( self.u[i]*self.L_tilde[i] @ self.L_tilde[i].T )) # Sigma deja dans W_tilde
            # val += (1/self.sigma_square)*np.trace(self.expect_MSE[i]) 

        # print('v weights:', -v_bis)
        return -val


    def init_tab(self):
        self.tab_rel_error = []
        self.tab_expected_loglikelihood = []
        self.tab_Z = []
        self.tab_X = []
        self.tab_Q = []
        self.tab_sigma = []
        self.tab_theta = []
        self.tab_coefs = []
        self.tab_Gamma = []
        self.tab_P0 = []
        self.tab_mu0 = []


    def tab_increment(self, rel_error=None, val_expected_loglikelihood=None):
        if rel_error is not None:
            self.tab_rel_error.append(rel_error)
        if val_expected_loglikelihood is not None:
            self.tab_expected_loglikelihood.append(val_expected_loglikelihood)
        if hasattr(self, "Z"):
            self.tab_Z.append(self.Z)
        if hasattr(self, "X"):
            self.tab_X.append(self.X)
        if hasattr(self, "Q"):
            self.tab_Q.append(self.Q)
        if hasattr(self, "Sigma"):
            # self.tab_sigma.append(np.array([self.Sigma(vi) for vi in self.v]))
            self.tab_sigma.append(np.sqrt(np.trace(self.Sigma(self.v[1]))/2))
        if hasattr(self, "theta"):
            self.tab_theta.append(self.theta(self.v))
        if hasattr(self, "coefs"):
            self.tab_coefs.append(self.coefs)
        if hasattr(self, "Gamma"):
            self.tab_Gamma.append(self.Gamma)
        if hasattr(self, "P0"):
            self.tab_P0.append(self.P0)
        if hasattr(self, "mu0"):
            self.tab_mu0.append(self.mu0)
        

    def save_tab_results(self, filename):
        dic = {"tab_rel_error": self.tab_rel_error, "tab_expected_loglikelihood": self.tab_expected_loglikelihood, "tab_Z": self.tab_Z, "tab_X": self.tab_X, 
               "tab_Q": self.tab_Q, "tab_sigma": self.tab_sigma, "tab_theta": self.tab_theta, "tab_Gamma": self.tab_Gamma, "duration":self.duration, "nb_iterations":self.nb_iterations}
        if os.path.isfile(filename):
            print("Le fichier ", filename, " existe déjà.")
            filename = filename + '_bis'
        fil = open(filename,"xb")
        pickle.dump(dic,fil)
        fil.close()





def bayesian_CV_optimization_regularization_parameter(n_CV, n_call_bayopt, lambda_bounds, grid_obs, Y_obs, tol, max_iter, nb_basis, init_params = None, order=4, method='approx', verbose=False):

    ## CV optimization of lambda
    
    def func(x):
        score_lambda = np.zeros(n_CV)
        kf = KFold(n_splits=n_CV, shuffle=True)
        ind_CV = 0

        for train_index, test_index in kf.split(grid_obs[1:]):
            # print('     --> Start CV step n°', ind_CV+1)
            Y_train = Y_obs[train_index]
            Y_test = Y_obs[test_index]
            grid_train = np.concatenate((np.array([grid_obs[0]]), grid_obs[1:][train_index]))
            grid_test = np.concatenate((np.array([grid_obs[0]]), grid_obs[1:][test_index]))
            
            FS_statespace = FrenetStateSpaceCV_global(grid_train, Y_train, bornes_theta=np.array([0,1]))
            FS_statespace.expectation_maximization(tol, max_iter, nb_basis=nb_basis, regularization_parameter=x, init_params=init_params, method=method, order=order, verbose=verbose)
            
            Z_reconst = solve_FrenetSerret_ODE_SE(FS_statespace.theta, grid_test, Z0=FS_statespace.mu0)
            X_reconst_test = Z_reconst[1:,:3,3]
            score_lambda[ind_CV] = np.linalg.norm(X_reconst_test - Y_test)**2
            
            ind_CV += 1

        # print(score_lambda)
        # print('val x:', x, 'score:', np.mean(score_lambda))
        return np.mean(score_lambda)

    # Do a bayesian optimisation and return the optimal parameter (lambda_kappa, lambda_tau)
    
    res_bayopt = gp_minimize(func,               # the function to minimize
                    lambda_bounds,        # the bounds on each dimension of x
                    acq_func="EI",        # the acquisition function
                    n_calls=n_call_bayopt,       # the number of evaluations of f
                    n_random_starts=2,    # the number of random initialization points
                    random_state=1,       # the random seed
                    n_jobs=1,            # use all the cores for parallel calculation
                    verbose=True)
    lbda_opt = res_bayopt.x
    # print(res_bayopt.x_iters)
    # print(res_bayopt.func_vals)
    lbda_opt = np.array([lbda_opt[0], lbda_opt[1]])
    print('the optimal hyperparameters selected are: ', lbda_opt)

    FS_statespace = FrenetStateSpaceCV_global(grid_obs, Y_obs[1:], bornes_theta=np.array([0,1]))
    FS_statespace.expectation_maximization(tol, max_iter, nb_basis=nb_basis, regularization_parameter=lbda_opt, init_params=init_params, method=method, order=order, verbose=verbose)

    return FS_statespace, res_bayopt

        








class FrenetStateSpaceCV_global_bis:


    def __init__(self, grid_obs, Y_obs, bornes_theta, dim=3):
        self.n = dim
        self.dim_g = int((1/2)*dim*(dim+1)) # dimension of the Lie Algebra se(n)
        self.Y = Y_obs #array of shape (N,n,)
        self.grid = grid_obs
        self.domain_range = (bornes_theta[0], bornes_theta[1])
        self.N = len(self.Y)
        self.rho = np.array([1,0,0])
        self.H = np.hstack((np.zeros((self.n, self.n)), np.eye(self.n)))
        # Def for case dim=3
        self.L = np.array([[0,1],[0,0],[1,0],[0,0],[0,0],[0,0]])
        self.u = self.grid[1:] - self.grid[:-1]
        self.v = (1/2)*(self.grid[1:]+self.grid[:-1])


    def expectation_maximization(self, tol, max_iter, nb_basis, regularization_parameter, init_params = None, order=4, method='approx', model_Sigma='scalar', verbose=False):

        self.verbose = verbose
        self.init_tab()

        if init_params is None:
            raise Exception("Must pass in argument an initial guess estimate for the parameters")
        else:
            self.Gamma = init_params["Gamma"]
            self.coefs = init_params["coefs"]
            self.mu0 = init_params["mu0"]
            self.Sigma = init_params["Sigma"]
            self.P0 = init_params["P0"]
        
        self.regularization_parameter = regularization_parameter
        self.Bspline_decomp = VectorBSplineSmoothing(self.n-1, nb_basis, domain_range=self.domain_range, order=order, penalization=True)
        V = np.expand_dims(self.v, 1)
        self.basis_matrix = self.Bspline_decomp.basis(V,).reshape((self.Bspline_decomp.basis.n_basis, -1)).T

        k = 0
        rel_error = 2*tol
        val_expected_loglikelihood = 0
        self.tab_increment()

        time_init = ttime.time()

        while k < max_iter and rel_error > tol:
            if self.verbose:
                print('-------------- Iteration',k+1,'/',max_iter,' --------------')

            # st = ttime.time()
            self.E_step()
            # ft = ttime.time()
            # print('time E step:', ft-st)
            # st = ttime.time()
            self.__approx_distribution_r(method)
            # ft = ttime.time()
            # print('time approx step:', ft-st)
            # st = ttime.time()
            self.M_step(0.001, 5, model_Sigma)
            # ft = ttime.time()
            # print('time M step:', ft-st)

            new_val_expected_loglikelihood = self.expected_loglikelihood()
            if self.verbose:
                print('value of expected_loglikelihood: ', new_val_expected_loglikelihood)
            rel_error = np.abs(val_expected_loglikelihood - new_val_expected_loglikelihood)
            if self.verbose:
                print('relative error: ', rel_error)
            self.tab_increment(rel_error, new_val_expected_loglikelihood)
            val_expected_loglikelihood = new_val_expected_loglikelihood
            k+=1
            
        time_end = ttime.time()
        self.nb_iterations = k
        self.duration = time_end - time_init

        if self.verbose:
            print('End expectation maximization algo. \n Number of iterations:', k, ', total duration:', self.duration, ' seconds.')



    def E_step(self):
        """
            Expectation step: Suppose that self.sigma, self.Gamma, self.a_theta, self.mu0, self.P0 are known. Call the tracking and smoothing method.

        """
        if self.verbose:
            print('___ E step ___')
        kalman_filter = IEKFilterSmootherFrenetState(self.n, self.Gamma, self.Sigma, self.theta, Z0=self.mu0, P0=self.P0)
        kalman_filter.smoothing(self.grid, self.Y)
        self.Z = kalman_filter.smooth_Z
        self.Q = kalman_filter.smooth_Q
        self.X = kalman_filter.smooth_X
        self.P = kalman_filter.smooth_P
        self.P_dble = kalman_filter.smooth_P_dble



    def M_step(self, tol, max_iter, model_Sigma):
        """
            Maximization step:

        """
        if self.verbose:
            print('___ M step ___')

        # Optimization of W
        self.opti_Gamma()
        self.P0 = self.P[0]
        self.mu0 = self.Z[0]

        # Optimization of theta given lambda
        self.regularization_parameter, self.regularization_parameter_matrix = self.Bspline_decomp.check_regularization_parameter(self.regularization_parameter)
        # self.coefs, self.mat_weights, self.weights, self.L_tilde = self.opti_coefs(tol, max_iter, self.regularization_parameter_matrix)
        # self.sigma_square, self.Sigma, self.expect_MSE = self.opti_Sigma(self.coefs, self.weights, self.regularization_parameter_matrix)
        self.coefs, self.sigma_square, self.Sigma, self.mat_weights, self.weights, self.expect_MSE, self.L_tilde = self.opti_coefs_and_Sigma(tol, max_iter, self.regularization_parameter_matrix, model_Sigma)
        # self.plot_theta()
        

    def theta_from_coefs(self, coefs, s):
        if isinstance(s, int) or isinstance(s, float):
            return np.squeeze(self.Bspline_decomp.basis_fct(s).T @ coefs)
        elif isinstance(s, np.ndarray):
            return np.squeeze((self.Bspline_decomp.basis_fct(s).T @ coefs).T)
        else:
            raise ValueError('Variable is not a float, a int or a NumPy array.')
        

    def theta(self, s):
        if isinstance(s, int) or isinstance(s, float):
            return np.squeeze(self.Bspline_decomp.basis_fct(s).T @ self.coefs)
        elif isinstance(s, np.ndarray):
            return np.squeeze((self.Bspline_decomp.basis_fct(s).T @ self.coefs).T)
        else:
            raise ValueError('Variable is not a float, a int or a NumPy array.')
    

    def plot_theta(self):
        visu.plot_array_2D(self.v, [self.r_tilde[:,0], self.theta(self.v)[:,0]], 'curv ')
        visu.plot_array_2D(self.v, [self.r_tilde[:,1], self.theta(self.v)[:,1]], 'tors ')



    def __approx_distribution_r(self, method='monte_carlo', N_rand=500):
        self.r_tilde = np.zeros((self.N, self.n-1))
        self.cov_r_tilde = np.zeros((self.N, self.n-1, self.n-1))

        if method=='monte_carlo':
            r = np.zeros((self.N, self.dim_g))
            cov_r = np.zeros((self.N, self.dim_g, self.dim_g))
            for i in range(self.N):
                mat_cov = np.hstack((np.vstack((self.P[i], self.P_dble[i].T)), np.vstack((self.P_dble[i], self.P[i+1]))))
                set_rand_xi = np.random.multivariate_normal(np.zeros(2*self.dim_g), mat_cov, size=N_rand)
                rand_obs = np.zeros((N_rand, self.dim_g))
                for j in range(N_rand):
                    rand_obs[j] = -(1/self.u[i])*SE3.log(SE3.exp(set_rand_xi[j][6:])@np.linalg.inv(self.Z[i+1])@self.Z[i]@SE3.exp(-set_rand_xi[j][:6]))
                    r[i] += rand_obs[j]
                r[i] = r[i]/N_rand
                for j in range(N_rand):
                    cov_r[i] += (rand_obs[j] - r[i])[:,np.newaxis] @ (rand_obs[j] - r[i])[:,np.newaxis].T
                cov_r[i] = cov_r[i]/N_rand
                self.cov_r_tilde[i] = self.L.T @ cov_r[i] @ self.L
                self.r_tilde[i] = self.L.T @ r[i]

        else: 
            for i in range(self.N):
                inv_Zi1_Zi = np.linalg.inv(self.Z[i+1])@self.Z[i]
                self.r_tilde[i] = -(1/self.u[i])*(self.L.T @ SE3.log(inv_Zi1_Zi))
                Ad = SE3.Ad_group(inv_Zi1_Zi)
                self.cov_r_tilde[i] = (1/self.u[i]**2)*self.L.T @ (self.P[i+1] - self.P_dble[i].T @ Ad.T - Ad @ self.P_dble[i] + Ad @ self.P[i] @ Ad.T) @ self.L
    

    def __compute_weights(self, Sigma, L_tilde_inv):
        weights = np.zeros((self.N,self.n-1,self.n-1))
        for i in range(self.N):
            weights[i] = self.u[i]*L_tilde_inv[i].T@np.linalg.inv(Sigma(self.v[i]))@L_tilde_inv[i]
        mat_weights = block_diag(*weights)
        return mat_weights, weights
        
    
    def __opti_coefs(self, mat_weights, reg_param_mat):
        left = self.basis_matrix.T @ mat_weights @ self.basis_matrix + reg_param_mat @ self.Bspline_decomp.penalty_matrix    
        right = self.basis_matrix.T @ mat_weights @ np.reshape(self.r_tilde, (self.N*(self.n-1),))
        coefs_tem = np.linalg.solve(left, right)
        coefs_tem = np.reshape(coefs_tem,(-1,2))
        new_coefs = np.reshape(coefs_tem, (-1,))
        return new_coefs
    

    def __compute_L_tilde(self, coefs):
        L_tilde = np.zeros((self.N,self.n-1,self.n-1))
        L_tilde_inv = np.zeros((self.N,self.n-1,self.n-1))
        theta_v = np.reshape(self.basis_matrix @ coefs, (-1,self.n-1))
        for i in range(self.N):
            phi_vi = SE3.Ad_group(SE3.exp(-(self.grid[i+1]-self.v[i])*np.array([theta_v[i,1],0,theta_v[i,0],1,0,0])))
            L_tilde[i] = self.L.T @ phi_vi @ self.L
            L_tilde_inv[i] = np.linalg.inv(L_tilde[i])
        return L_tilde, L_tilde_inv


    def __opti_Sigma(self, coefs, model_Sigma, L_tilde_inv):
        res = np.zeros((self.N, self.n-1, self.n-1))
        err = np.reshape(self.basis_matrix @ coefs, (-1,self.n-1)) - self.r_tilde
        for i in range(self.N):
            res[i] = self.u[i]*L_tilde_inv[i] @ (err[i][:,np.newaxis] @ err[i][np.newaxis,:] + self.cov_r_tilde[i]) @ L_tilde_inv[i].T
        sigma_square = (1/(2*self.N))*(np.trace(np.sum(res, axis=0))) # + coefs.T @ reg_param_mat @ self.Bspline_decomp.penalty_matrix @ coefs)
        print('sigma_square :', sigma_square)
        Sigma = lambda s: sigma_square*np.eye(2)
        return sigma_square, Sigma, res
    
    
    def opti_coefs_and_Sigma(self, tol, max_iter, reg_param_mat, model_Sigma):
        L_tilde, L_tilde_inv = self.__compute_L_tilde(self.coefs)
        mat_weights, weights = self.__compute_weights(self.Sigma, L_tilde_inv)
        old_theta = np.reshape(self.basis_matrix @ self.coefs, (-1,self.n-1))
        rel_error = 2*tol 
        k = 0 
        while rel_error > tol and k < max_iter:

            coefs_opt = self.__opti_coefs(mat_weights, reg_param_mat)
            L_tilde, L_tilde_inv = self.__compute_L_tilde(coefs_opt)
            sigma_square, Sigma_opt, expect_MSE = self.__opti_Sigma(coefs_opt, model_Sigma, L_tilde_inv)
            mat_weights, weights = self.__compute_weights(Sigma_opt, L_tilde_inv)

            new_theta = np.reshape(self.basis_matrix @ coefs_opt, (-1,self.n-1))
            rel_error = np.linalg.norm(old_theta - new_theta)/np.linalg.norm(old_theta)
            if self.verbose:
                print('iteration for optimization of coefs and Sigma:', k, ', relative error:', rel_error)
            old_theta = new_theta
            k += 1

        return coefs_opt, sigma_square, Sigma_opt, mat_weights, weights, expect_MSE, L_tilde


    
    def __opti_coefs(self, mat_weights, reg_param_mat):
        left = self.basis_matrix.T @ mat_weights @ self.basis_matrix + reg_param_mat @ self.Bspline_decomp.penalty_matrix    
        right = self.basis_matrix.T @ mat_weights @ np.reshape(self.r_tilde, (self.N*(self.n-1),))
        coefs_tem = np.linalg.solve(left, right)
        coefs_tem = np.reshape(coefs_tem,(-1,2))
        new_coefs = np.reshape(coefs_tem, (-1,))
        return new_coefs
    

    def __compute_L_tilde(self, coefs):
        L_tilde = np.zeros((self.N,self.n-1,self.n-1))
        L_tilde_inv = np.zeros((self.N,self.n-1,self.n-1))
        theta_v = np.reshape(self.basis_matrix @ coefs, (-1,self.n-1))
        for i in range(self.N):
            phi_vi = SE3.Ad_group(SE3.exp(-(self.grid[i+1]-self.v[i])*np.array([theta_v[i,1],0,theta_v[i,0],1,0,0])))
            L_tilde[i] = self.L.T @ phi_vi @ self.L
            L_tilde_inv[i] = np.linalg.inv(L_tilde[i])
        return L_tilde, L_tilde_inv
    

    def opti_Gamma(self):
        Gamma = np.zeros((self.n,self.n))
        for i in range(1,self.N+1):
            Gamma += (self.Y[i-1]-self.X[i])[:,np.newaxis]@(self.Y[i-1]-self.X[i])[np.newaxis,:] + self.Q[i]@self.H@self.P[i]@self.H.T@self.Q[i].T
        self.Gamma = Gamma/self.N


    def expected_loglikelihood(self):
        # P0 / mu0
        val = np.log(np.linalg.det(self.P0))
        print('v P0:', -np.log(np.linalg.det(self.P0))) 
        # Gamma
        val += self.N*np.log(np.linalg.det(self.Gamma)) 
        print('v gamma:', -self.N*np.log(np.linalg.det(self.Gamma)))
        # Theta / Sigma
        val += 2*self.N*np.log(self.sigma_square)
        print('v sigma:', -2*self.N*np.log(self.sigma_square))
        # print('val l1:', val)
        v_bis = 0
        for i in range(self.N):
            v_bis += np.log(np.linalg.det( self.u[i]*self.L_tilde[i] @ self.L_tilde[i].T ))
            val += np.log(np.linalg.det( self.u[i]*self.L_tilde[i] @ self.L_tilde[i].T )) # Sigma deja dans W_tilde
            # val += (1/self.sigma_square)*np.trace(self.expect_MSE[i]) 

            # val += (1/2)*np.log(np.linalg.det(self.W_tilde[i])) # Sigma deja dans W_tilde
            # val += (1/2)*np.trace(np.linalg.inv(self.Sigma(self.v[i]))@self.residuals[i]) 
        # Penalization
        # print('val l2:', val)
        print('v weights:', -v_bis)
        val += self.coefs.T @ self.regularization_parameter_matrix @ self.Bspline_decomp.penalty_matrix @ self.coefs     
        # print('val l3:', val)   
        print('v pen:', self.coefs.T @ self.regularization_parameter_matrix @ self.Bspline_decomp.penalty_matrix @ self.coefs)
        return -val


    def init_tab(self):
        self.tab_rel_error = []
        self.tab_expected_loglikelihood = []
        self.tab_Z = []
        self.tab_X = []
        self.tab_Q = []
        self.tab_sigma = []
        self.tab_theta = []
        self.tab_coefs = []
        self.tab_Gamma = []
        self.tab_P0 = []
        self.tab_mu0 = []


    def tab_increment(self, rel_error=None, val_expected_loglikelihood=None):
        if rel_error is not None:
            self.tab_rel_error.append(rel_error)
        if val_expected_loglikelihood is not None:
            self.tab_expected_loglikelihood.append(val_expected_loglikelihood)
        if hasattr(self, "Z"):
            self.tab_Z.append(self.Z)
        if hasattr(self, "X"):
            self.tab_X.append(self.X)
        if hasattr(self, "Q"):
            self.tab_Q.append(self.Q)
        if hasattr(self, "Sigma"):
            # self.tab_sigma.append(np.array([self.Sigma(vi) for vi in self.v]))
            self.tab_sigma.append(np.sqrt(np.trace(self.Sigma(self.v[1]))/2))
        if hasattr(self, "theta"):
            self.tab_theta.append(self.theta(self.v))
        if hasattr(self, "coefs"):
            self.tab_coefs.append(self.coefs)
        if hasattr(self, "Gamma"):
            self.tab_Gamma.append(self.Gamma)
        if hasattr(self, "P0"):
            self.tab_P0.append(self.P0)
        if hasattr(self, "mu0"):
            self.tab_mu0.append(self.mu0)
        

    def save_tab_results(self, filename):
        dic = {"tab_rel_error": self.tab_rel_error, "tab_expected_loglikelihood": self.tab_expected_loglikelihood, "tab_Z": self.tab_Z, "tab_X": self.tab_X, 
               "tab_Q": self.tab_Q, "tab_sigma": self.tab_sigma, "tab_theta": self.tab_theta, "tab_Gamma": self.tab_Gamma, "duration":self.duration, "nb_iterations":self.nb_iterations}
        if os.path.isfile(filename):
            print("Le fichier ", filename, " existe déjà.")
            filename = filename + '_bis'
        fil = open(filename,"xb")
        pickle.dump(dic,fil)
        fil.close()