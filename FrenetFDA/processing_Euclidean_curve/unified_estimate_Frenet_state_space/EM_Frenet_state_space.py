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


class MLE:

    def __init__(self, grid_obs, Y_obs, Z_obs):
        self.grid = grid_obs
        self.Y = Y_obs[1:]
        self.Z = Z_obs
        self.X = Z_obs[:,:3,3]
        self.Q = Z_obs[:,:3,:3]
        self.N = len(self.Y)
        self.u = self.grid[1:] - self.grid[:-1]
        self.v = (self.grid[1:] + self.grid[:-1])/2
        # self.v = self.grid[:-1]
        L = np.zeros((6,2))
        L[0,1], L[2,0] = 1, 1
        self.L = L
    
    def opti_other_param(self):
        self.mu0 = self.Z[0]
        self.P0 = np.zeros((6,6))
        Gamma = np.zeros((3,3))
        for i in range(1,self.N+1):
            Gamma += (self.Y[i-1]-self.X[i])[:,np.newaxis]@(self.Y[i-1]-self.X[i])[np.newaxis,:] 
        self.Gamma = Gamma/self.N
        return self.mu0, self.P0, self.Gamma
    
    def def_model_theta(self, nb_basis):
        self.nb_basis = nb_basis
        self.Bspline_decomp = VectorBSplineSmoothing(2, nb_basis, domain_range=(self.grid[0], self.grid[-1]), order=4, penalization=True)
        V = np.expand_dims(self.v, 1)
        self.basis_matrix = self.Bspline_decomp.basis(V,).reshape((self.Bspline_decomp.basis.n_basis, -1)).T
        r = np.zeros((self.N,6))
        r_tilde = np.zeros((self.N,2))
        for i in range(self.N):
            r[i] = -(1/self.u[i])*SE3.log(np.linalg.inv(self.Z[i+1])@self.Z[i])
            r_tilde[i] = self.L.T@r[i] 
        self.r = r
        self.r_tilde = r_tilde

    
    def compute_L_tilde(self, coefs):
        L_tilde = np.zeros((self.N,2,2))
        L_tilde_inv = np.zeros((self.N,2,2))
        theta_v = np.reshape(self.basis_matrix @ coefs, (-1,2))
        for i in range(self.N):
            phi_vi = SE3.Ad_group(SE3.exp(-(self.grid[i+1]-self.v[i])*np.array([theta_v[i,1],0,theta_v[i,0],1,0,0])))
            L_tilde[i] = self.L.T @ phi_vi @ self.L
            L_tilde_inv[i] = np.linalg.inv(L_tilde[i])
        return L_tilde, L_tilde_inv
    

    def compute_weights(self, coefs):
        L_tilde, L_tilde_inv = self.compute_L_tilde(coefs)
        weights = np.zeros((self.N,2,2))
        for i in range(self.N):
            weights[i] = self.u[i]*L_tilde_inv[i].T@L_tilde_inv[i]
        mat_weights = block_diag(*weights)
        return mat_weights, weights
    

    def step_opti_coefs(self, mat_weights, reg_param_mat):
        left = self.basis_matrix.T @ mat_weights @ self.basis_matrix + reg_param_mat @ self.Bspline_decomp.penalty_matrix    
        right = self.basis_matrix.T @ mat_weights @ np.reshape(self.r_tilde, (self.N*(2),))
        coefs_tem = np.linalg.solve(left, right)
        coefs_tem = np.reshape(coefs_tem,(-1,2))
        new_coefs = np.reshape(coefs_tem, (-1,))
        return new_coefs
        

    def opti_coefs(self, reg_param):
        lbda, reg_param_mat = self.Bspline_decomp.check_regularization_parameter(reg_param)
        coefs_init = np.zeros((self.nb_basis*2))
        mat_weights, weights = self.compute_weights(coefs_init)
    
        tol = 0.001
        max_iter = 100
        old_theta = np.reshape(self.basis_matrix @ coefs_init, (-1,2))
        rel_error = 2*tol
        k = 0 
        # print('Coefs optimization:')
        while rel_error > tol and k < max_iter:
            coefs_opt = self.step_opti_coefs(mat_weights, reg_param_mat)
            mat_weights, weights = self.compute_weights(coefs_opt)
            new_theta = np.reshape(self.basis_matrix @ coefs_opt, (-1,2))
            rel_error = np.linalg.norm(old_theta - new_theta)/np.linalg.norm(new_theta)
            # print('     iteration:', k, ', relative error:', rel_error)
            old_theta = new_theta
            k += 1
        self.coefs = coefs_opt
        self.mat_weights = mat_weights
        self.weigths = weights
        self.hat_matrix = self.basis_matrix @ np.linalg.inv(self.basis_matrix.T @ mat_weights @ self.basis_matrix + reg_param_mat @ self.Bspline_decomp.penalty_matrix) @ self.basis_matrix.T @ mat_weights


    def opti_sigma(self):
        error = (np.eye(self.hat_matrix.shape[0])-self.hat_matrix)@np.reshape(self.r_tilde, (self.N*2,))
        sigma_square = (1/(2*self.N))*np.trace(self.mat_weights@(error[:,np.newaxis]@error[np.newaxis,:]))
        self.sigma_square = sigma_square
        self.sigma = np.sqrt(sigma_square)
    

    def compute_GCV_criteria(self):
        error = (np.eye(self.hat_matrix.shape[0])-self.hat_matrix)@np.reshape(self.r_tilde, (self.N*2,))
        GCV = self.N*(np.linalg.norm(error)**2)/(np.trace((np.eye(self.hat_matrix.shape[0])-self.hat_matrix))**2) 
        V0 = self.N*(np.linalg.norm(error)**2)/(np.trace(np.linalg.inv(self.mat_weights)@(np.eye(self.hat_matrix.shape[0])-self.hat_matrix))**2) 
        V1 = self.N*np.squeeze((error[np.newaxis,:]@self.mat_weights@error))/(np.trace((np.eye(self.hat_matrix.shape[0])-self.hat_matrix))**2)
        V2 = self.N*(np.linalg.norm(self.mat_weights@error)**2)/(np.trace(self.mat_weights@(np.eye(self.hat_matrix.shape[0])-self.hat_matrix))**2)
        U0 = (1/self.N)*(np.linalg.norm(error)**2) - (self.sigma_square/self.N)*np.trace(np.linalg.inv(self.mat_weights)) + 2*(self.sigma_square/self.N)*np.trace(np.linalg.inv(self.mat_weights)@self.hat_matrix)
        U1 = (1/self.N)*np.squeeze((error[np.newaxis,:]@self.mat_weights@error)) - (self.sigma_square/self.N)*np.trace(np.eye(self.mat_weights.shape[0])) + 2*(self.sigma_square/self.N)*np.trace(self.hat_matrix)
        U2 = (1/self.N)*(np.linalg.norm(self.mat_weights@error)**2) - (self.sigma_square/self.N)*np.trace(self.mat_weights) + 2*(self.sigma_square/self.N)*np.trace(self.mat_weights@self.hat_matrix)
        L = np.squeeze(self.N*np.log(np.squeeze((error[np.newaxis,:]@self.mat_weights@error))) + np.sum(np.log(np.linalg.det(np.linalg.inv(self.weigths)))) + np.log(self.N)*np.trace(self.hat_matrix))
        
        return GCV, V0, V1, V2, U0, U1, U2, L
    

    def theta(self, s):
        if isinstance(s, int) or isinstance(s, float):
            return np.squeeze(self.Bspline_decomp.basis_fct(s).T @ self.coefs)
        elif isinstance(s, np.ndarray):
            return np.squeeze((self.Bspline_decomp.basis_fct(s).T @ self.coefs).T)
        else:
            raise ValueError('Variable is not a float, a int or a NumPy array.')
    
    def compute_reconst_criterion(self):
        Sigma_opt = lambda s: self.sigma_square*np.array([[1 +0*s, 0*s], [0*s, 1+0*s]])
        Z_reconst = solve_FrenetSerret_SDE_SE3(self.theta, Sigma_opt, self.L, self.grid, Z0=self.mu0)
        mu_Z_reconst = solve_FrenetSerret_ODE_SE(self.theta, self.grid, Z0=self.mu0)
        X_reconst = Z_reconst[:,:3,3]

        error = (np.eye(self.hat_matrix.shape[0])-self.hat_matrix)@np.reshape(self.r_tilde, (self.N*2,))
        error_Y = X_reconst[1:] - self.Y

        mse_theta = np.linalg.norm(error)**2
        mse_Y = np.linalg.norm(error_Y)**2
        wmse_theta = np.squeeze((error[np.newaxis,:]@self.mat_weights@error))
        wmse_Y = 0
        for i in range(self.N):
            wmse_Y += np.squeeze((error_Y[i][np.newaxis,:]@np.linalg.inv(self.Gamma)@error_Y[i]))
        loglik_hat = wmse_Y + (1/self.sigma_square)*wmse_theta + np.sum(np.log(np.linalg.det(np.linalg.inv(self.weigths))))
        return mse_theta, wmse_theta, mse_Y, wmse_Y, loglik_hat, Z_reconst, mu_Z_reconst
    
        # Ksplit = KFold(n_splits=K_split, shuffle=True)
        # err = []
        # for train_index, test_index in Ksplit.split(self.grid[1:]):
        #     t_train, t_test = self.grid[1:][train_index], self.grid[1:][test_index]
        #     data_train, data_test = [train_index,:], data[test_index,:]
        #     derivatives = self.fit(data_train, t_train, t_test, bandwidth_grid[j])
        #     diff = derivatives[0] - data_test
        #     err.append(np.linalg.norm(diff)**2)
        # CV_2 = np.mean(err)



    def compute_true_MSE(self, theta, sigma):
        # true_theta = theta(self.grid[:-1])
        true_theta = theta(self.v)
        error = true_theta - np.reshape(self.basis_matrix @ self.coefs, (-1,2))
        MSE_theta = np.linalg.norm(error, axis=0)**2
        MSE = np.linalg.norm(error)**2
        error_sigma = np.linalg.norm(sigma-self.sigma)**2
        return MSE_theta, MSE, error_sigma


class FrenetStateSpace:


    def __init__(self, grid_obs, Y_obs, dim=3, bornes_theta=None):
        self.n = dim
        self.dim_g = int((1/2)*dim*(dim+1)) # dimension of the Lie Algebra se(n)
        self.Y = Y_obs[1:] #array of shape (N,n,)
        self.grid = grid_obs 
        self.domain_range = (grid_obs[0], grid_obs[-1])
        self.N = len(self.Y)
        self.rho = np.array([1,0,0])
        self.H = np.hstack((np.zeros((self.n, self.n)), np.eye(self.n)))
        # Def for case dim=3
        self.L = np.array([[0,1],[0,0],[1,0],[0,0],[0,0],[0,0]])
        self.u = self.grid[1:] - self.grid[:-1]
        self.v = (1/2)*(self.grid[1:]+self.grid[:-1])


    def maximum_complete_likelihood_estimation(self, Z_obs, nb_basis, regularization_parameter_list = [], order=4,  model_Sigma='scalar', score_lambda='GCV', true_theta=None):
        self.Z = Z_obs
        self.X = Z_obs[:,:3,3]
        self.Q = Z_obs[:,:3,:3]
        self.mu0 = self.Z[0]
        self.P0 = np.zeros((6,6))
        Gamma = np.zeros((3,3))
        for i in range(1,self.N+1):
            Gamma += (self.Y[i-1]-self.X[i])[:,np.newaxis]@(self.Y[i-1]-self.X[i])[np.newaxis,:] 
        self.Gamma = Gamma/self.N

        self.nb_basis = nb_basis
        N_param_smoothing = len(regularization_parameter_list)    
        if N_param_smoothing==0:
            penalization = False
            regularization_parameter_list = np.array([0])
        else:
            penalization = True
        self.Bspline_decomp = VectorBSplineSmoothing(2, nb_basis, domain_range=(self.grid[0], self.grid[-1]), order=4, penalization=penalization)
        V = np.expand_dims(self.v, 1)
        self.basis_matrix = self.Bspline_decomp.basis(V,).reshape((self.Bspline_decomp.basis.n_basis, -1)).T
        r = np.zeros((self.N,self.dim_g))
        r_tilde = np.zeros((self.N,self.n-1))
        for i in range(self.N):
            r[i] = -(1/self.u[i])*SE3.log(np.linalg.inv(self.Z[i+1])@self.Z[i])
            r_tilde[i] = self.L.T@r[i] 
        self.r = r
        self.r_tilde = r_tilde
        self.cov_r_tilde = np.zeros((self.N, self.n-1, self.n-1))

        self.regularization_parameter, self.score_lambda_matrix, self.true_MSE = self.opti_lambda(0.001, 10, regularization_parameter_list, model_Sigma, score_lambda, true_theta=true_theta)
        self.regularization_parameter, self.regularization_parameter_matrix = self.Bspline_decomp.check_regularization_parameter(self.regularization_parameter)
        # Optimization of theta given lambda
        self.coefs, self.Sigma, self.mat_weights, self.weights, self.expect_MSE, err_obs, self.L_tilde = self.opti_coefs_and_Sigma(0.001, 10, self.regularization_parameter_matrix, model_Sigma)
        


    def expectation_maximization(self, tol, max_iter, nb_basis, regularization_parameter_list = [], init_params = None, order=4, method='approx', model_Sigma='scalar', score_lambda='GCV', true_theta=None, verbose=False):

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

        N_param_smoothing = len(regularization_parameter_list)    
        if N_param_smoothing==0:
            penalization = False
            regularization_parameter_list = np.array([0])
        else:
            penalization = True
        self.Bspline_decomp = VectorBSplineSmoothing(self.n-1, nb_basis, domain_range=(self.grid[0], self.grid[-1]), order=order, penalization=penalization)
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

            self.E_step()
            self.__approx_distribution_r(method)
            self.M_step(0.001, 5, regularization_parameter_list, model_Sigma, score_lambda, true_theta)

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



    def M_step(self, tol, max_iter, reg_param_list, model_Sigma, score_lbda, true_theta):
        """
            Maximization step:

        """
        if self.verbose:
            print('___ M step ___')

        # Optimization of W
        self.opti_Gamma()
        self.P0 = self.P[0]
        self.mu0 = self.Z[0]

        # Optimization of lambda
        self.regularization_parameter, self.score_lambda_matrix, _ = self.opti_lambda(tol, max_iter, reg_param_list, model_Sigma, score_lbda, true_theta)
        self.regularization_parameter, self.regularization_parameter_matrix = self.Bspline_decomp.check_regularization_parameter(self.regularization_parameter)
        
        # Optimization of theta given lambda
        self.coefs, self.Sigma, self.mat_weights, self.weights, self.expect_MSE, err_obs, self.L_tilde = self.opti_coefs_and_Sigma(tol, max_iter, self.regularization_parameter_matrix, model_Sigma)
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
            res[i] = self.u[i]*L_tilde_inv[i] @ (err[i][:,np.newaxis] @ err[i][np.newaxis,:] + self.cov_r_tilde[i]) @ L_tilde_inv[i]
        if model_Sigma=='vector':
            sigma_square_1 = np.mean(res[:,0,0])
            sigma_square_2 = np.mean(res[:,1,1])
            Sigma = lambda s: np.array([[sigma_square_1,0], [0, sigma_square_2]])
        else:                                                               # Case 'single_constant' by default
            sigma_square = np.mean((res[:,0,0]+res[:,1,1])/2)
            Sigma = lambda s: sigma_square*np.eye(2)
        return Sigma, res, err
    
    
    def opti_coefs_and_Sigma(self, tol, max_iter, reg_param_mat, model_Sigma):
        L_tilde, L_tilde_inv = self.__compute_L_tilde(self.coefs)
        mat_weights, weights = self.__compute_weights(self.Sigma, L_tilde_inv)
        old_theta = np.reshape(self.basis_matrix @ self.coefs, (-1,self.n-1))
        rel_error = 2*tol 
        k = 0 
        while rel_error > tol and k < max_iter:

            coefs_opt = self.__opti_coefs(mat_weights, reg_param_mat)
            L_tilde, L_tilde_inv = self.__compute_L_tilde(coefs_opt)
            Sigma_opt, expect_MSE, err_obs = self.__opti_Sigma(coefs_opt, model_Sigma, L_tilde_inv)
            mat_weights, weights = self.__compute_weights(Sigma_opt, L_tilde_inv)

            new_theta = np.reshape(self.basis_matrix @ coefs_opt, (-1,self.n-1))
            rel_error = np.linalg.norm(old_theta - new_theta)/np.linalg.norm(old_theta)
            if self.verbose:
                print('iteration for optimization of coefs and Sigma:', k, ', relative error:', rel_error)
            old_theta = new_theta
            k += 1

        return coefs_opt, Sigma_opt, mat_weights, weights, expect_MSE, err_obs, L_tilde
    

    # def opti_lambda(self, tol, max_iter, reg_param_list, model_Sigma):
    #     K = len(reg_param_list)

    #     GCV_scores = np.zeros(K)
    #     for k in range(K):

    #         lbda, reg_param_mat = self.Bspline_decomp.check_regularization_parameter(reg_param_list[k])
    #         coefs, Sigma, mat_weights, expect_MSE, err_obs, L_tilde = self.opti_coefs_and_Sigma(tol, max_iter, reg_param_mat, model_Sigma)
    #         err = np.reshape(err_obs, (self.N*(self.n-1),))
    #         hat_matrix = self.basis_matrix @ np.linalg.inv(self.basis_matrix.T @ mat_weights @ self.basis_matrix + reg_param_mat @ self.Bspline_decomp.penalty_matrix) @ self.basis_matrix.T @ mat_weights
    #         numerator = (1/self.N)*np.squeeze(err[np.newaxis,:] @ mat_weights.T @ mat_weights @ err[:,np.newaxis]) 
    #         denominator = ((1/self.N)*np.trace(mat_weights @ (np.eye(mat_weights.shape[0]) - hat_matrix)))**2
    #         GCV_scores[k] = numerator/denominator
    #         if self.verbose:
    #             print('lambda value:', lbda, ' GCV score:', GCV_scores[k]) 

    #     ind = np.argmin(GCV_scores)
    #     lbda_opt = reg_param_list[ind] 
    #     if self.verbose:
    #         print('Optimal chosen lambda:', lbda_opt)
    #     return lbda_opt

    def opti_lambda(self, tol, max_iter, reg_param_list, model_Sigma, score, true_theta=None):
        K = len(reg_param_list)
        score_lambda_matrix = np.zeros((K,K))
        if true_theta is not None:
            true_MSE = np.zeros((K,K))
            true_theta_arr = true_theta(self.v)
        else:
            true_MSE = None
        for i in range(K):
            for j in range(K):
                reg_param = np.array([reg_param_list[i], reg_param_list[j]])
                lbda, reg_param_mat = self.Bspline_decomp.check_regularization_parameter(reg_param)
                coefs, Sigma, mat_weights, weights, expect_MSE, err_obs, L_tilde = self.opti_coefs_and_Sigma(tol, max_iter, reg_param_mat, model_Sigma)
                hat_matrix = self.basis_matrix @ np.linalg.inv(self.basis_matrix.T @ mat_weights @ self.basis_matrix + reg_param_mat @ self.Bspline_decomp.penalty_matrix) @ self.basis_matrix.T @ mat_weights
                if score=='GCV':
                    score_lambda_matrix[i,j] = self.__GCV_score(hat_matrix)
                elif score=='V0':
                    score_lambda_matrix[i,j] = self.__V_score(hat_matrix, mat_weights, k=0)
                elif score=='V1':
                    score_lambda_matrix[i,j] = self.__V_score(hat_matrix, mat_weights, k=1)
                elif score=='V2':
                    score_lambda_matrix[i,j] = self.__V_score(hat_matrix, mat_weights, k=2)
                elif score=='U0' and model_Sigma!='vector':
                    score_lambda_matrix[i,j] = self.__U_score(hat_matrix, mat_weights, np.trace(Sigma(0))/2, k=0)
                elif score=='U1' and model_Sigma!='vector':
                    score_lambda_matrix[i,j] = self.__U_score(hat_matrix, mat_weights, np.trace(Sigma(0))/2, k=1)
                elif score=='U2' and model_Sigma!='vector':
                    score_lambda_matrix[i,j] = self.__U_score(hat_matrix, mat_weights, np.trace(Sigma(0))/2, k=2)
                elif score=='L':
                    score_lambda_matrix[i,j] = self.__L_score(hat_matrix, mat_weights, weights)
                elif score=='MSE_YX':
                    score_lambda_matrix[i,j] = self.__MSE_YX_score(Sigma, coefs)
                elif score=='MSE_YmuX':
                    score_lambda_matrix[i,j] = self.__MSE_YmuX_score(coefs)
                elif score=='true_MSE' and true_theta is not None:
                    score_lambda_matrix[i,j] = self.__true_MSE(true_theta, coefs)
                else: 
                    raise Exception("Invalid term for optimization of lamnda score.")
                if true_theta is not None:
                    error = true_theta_arr - np.reshape(self.basis_matrix @ coefs, (-1,2))
                    MSE_theta = np.linalg.norm(error, axis=0)**2
                    true_MSE[i,j] = np.linalg.norm(error)**2

        ind = np.squeeze(np.array(np.where(score_lambda_matrix==np.min(score_lambda_matrix))))
        lbda_opt = np.array([reg_param_list[ind[0]], reg_param_list[ind[1]]]) 
        if self.verbose:
            print('Optimal chosen lambda:', lbda_opt)
        return lbda_opt, score_lambda_matrix, true_MSE
    

    def __GCV_score(self, hat_matrix):
        error = (np.eye(hat_matrix.shape[0])-hat_matrix)@np.reshape(self.r_tilde, (self.N*2,))
        return self.N*(np.linalg.norm(error)**2)/(np.trace((np.eye(hat_matrix.shape[0])-hat_matrix))**2) 
        
    def __V_score(self, hat_matrix, mat_weights, k=0):
        error = (np.eye(hat_matrix.shape[0])-hat_matrix)@np.reshape(self.r_tilde, (self.N*2,))
        if k==0:
            return self.N*(np.linalg.norm(error)**2)/(np.trace(np.linalg.inv(mat_weights)@(np.eye(hat_matrix.shape[0])-hat_matrix))**2) 
        elif k==1:
            return self.N*np.squeeze((error[np.newaxis,:]@mat_weights@error))/(np.trace((np.eye(hat_matrix.shape[0])-hat_matrix))**2)
        elif k==2:
            return self.N*(np.linalg.norm(mat_weights@error)**2)/(np.trace(mat_weights@(np.eye(hat_matrix.shape[0])-hat_matrix))**2)
        
    def __U_score(self, hat_matrix, mat_weights, sigma_square, k=0):
        error = (np.eye(hat_matrix.shape[0])-hat_matrix)@np.reshape(self.r_tilde, (self.N*2,))
        if k==0:
            return (1/self.N)*(np.linalg.norm(error)**2) - (sigma_square/self.N)*np.trace(np.linalg.inv(mat_weights)) + 2*(sigma_square/self.N)*np.trace(np.linalg.inv(mat_weights)@hat_matrix)
        elif k==1:
            return (1/self.N)*np.squeeze((error[np.newaxis,:]@mat_weights@error)) - (sigma_square/self.N)*np.trace(np.eye(mat_weights.shape[0])) + 2*(sigma_square/self.N)*np.trace(hat_matrix)
        elif k==2:
            return (1/self.N)*(np.linalg.norm(mat_weights@error)**2) - (sigma_square/self.N)*np.trace(mat_weights) + 2*(sigma_square/self.N)*np.trace(mat_weights@hat_matrix)

    def __L_score(self, hat_matrix, mat_weights, weigths):
        error = (np.eye(hat_matrix.shape[0])-hat_matrix)@np.reshape(self.r_tilde, (self.N*2,))
        return np.squeeze(self.N*np.log(np.squeeze((error[np.newaxis,:]@mat_weights@error))) + np.sum(np.log(np.linalg.det(np.linalg.inv(weigths)))) + np.log(self.N)*np.trace(hat_matrix))

    def __MSE_YX_score(self, Sigma, coefs):
        Z_reconst = solve_FrenetSerret_SDE_SE3(lambda s: self.theta_from_coefs(coefs, s), Sigma, self.L, self.grid, Z0=self.mu0)
        X_reconst = Z_reconst[:,:3,3]
        error_Y = X_reconst[1:] - self.Y
        score = np.linalg.norm(error_Y)**2
        return score
    
    def __MSE_YmuX_score(self, coefs):
        Z_reconst = solve_FrenetSerret_ODE_SE(lambda s: self.theta_from_coefs(coefs, s), self.grid, Z0=self.mu0)
        X_reconst = Z_reconst[:,:3,3]
        error_Y = X_reconst[1:] - self.Y
        score = np.linalg.norm(error_Y)**2
        return score

    def __true_MSE(self, true_theta, coefs):
        error = true_theta(self.v) - np.reshape(self.basis_matrix @ coefs, (-1,2))
        score = np.linalg.norm(error)**2
        return score

    def opti_Gamma(self):
        Gamma = np.zeros((self.n,self.n))
        for i in range(1,self.N+1):
            Gamma += (self.Y[i-1]-self.X[i])[:,np.newaxis]@(self.Y[i-1]-self.X[i])[np.newaxis,:] + self.Q[i]@self.H@self.P[i]@self.H.T@self.Q[i].T
        self.Gamma = Gamma/self.N


    def expected_loglikelihood(self):
        val = (6/2)*self.N*np.log(2*np.pi)
        # P0 / mu0
        val += (1/2)*np.log(np.linalg.det(self.P0)) 
        # Gamma
        val += (1/2)*self.N*np.log(np.linalg.det(self.Gamma)) 
        # Theta / Sigma
        for i in range(self.N):
            val += (1/2)*np.log(np.linalg.det( self.u[i]*self.L_tilde[i] @ self.Sigma(self.v[i]) @ self.L_tilde[i].T )) # Sigma deja dans W_tilde
            val += (1/2)*np.trace(np.linalg.inv(self.Sigma(self.v[i]))@self.expect_MSE[i]) 
            # val += (1/2)*np.log(np.linalg.det(self.W_tilde[i])) # Sigma deja dans W_tilde
            # val += (1/2)*np.trace(np.linalg.inv(self.Sigma(self.v[i]))@self.residuals[i]) 
        # Penalization
        val += self.coefs.T @ self.regularization_parameter_matrix @ self.Bspline_decomp.penalty_matrix @ self.coefs        
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

   # def __compute_SSE(self, coefs, mat_weights):
    #     err = np.reshape((np.reshape(self.basis_matrix @ coefs, (-1,self.n-1)) - self.r_tilde), (self.N*(self.n-1),))
    #     SSE = np.diag(np.reshape(err @ mat_weights, (-1,self.n-1)).T @ (np.reshape(self.basis_matrix @ coefs, (-1,self.n-1)) - self.r_tilde))
    #     # print('part 1 SSE', SSE)
    #     # SSE = SSE + np.sum(np.reshape(np.diag(mat_weights @ block_diag(*self.cov_r_tilde)), (-1,self.n-1)), axis=0)
    #     # print('part 2 SSE', SSE)
    #     return SSE


    # def compute_sampling_variance(self):
    #     SSE = self.__compute_SSE(self.coefs, self.mat_weights)
    #     self.residuals_error = np.array([SSE[i]/(self.N-self.Bspline_decomp.nb_basis[i]) for i in range(self.n-1)])
    #     res_mat = block_diag(*np.apply_along_axis(np.diag, 1, np.array([self.residuals_error for i in range(self.N)])))
    #     S = np.linalg.inv(self.basis_matrix.T @ self.mat_weights @ self.basis_matrix + self.regularization_parameter_matrix @ self.Bspline_decomp.penalty_matrix) @ self.basis_matrix.T @ self.mat_weights
    #     self.sampling_variance_coeffs = S @ res_mat @ S.T 
    #     self.sampling_variance_yhat = self.basis_matrix @ self.sampling_variance_coeffs @ self.basis_matrix.T

    #     def confidence_limits(s):
    #         val = self.theta(s)
    #         basis_fct_arr = self.Bspline_decomp.basis_fct(s).T 
    #         if (self.n-1)==1:
    #             if isinstance(s, int) or isinstance(s, float):
    #                 error = 0
    #             elif isinstance(s, np.ndarray):
    #                 error = np.zeros((len(s)))
    #             else:
    #                 raise ValueError('Variable is not a float, a int or a NumPy array.')
    #             error = 0.95*np.sqrt(np.diag(basis_fct_arr @ self.sampling_variance_coeffs @ basis_fct_arr.T))
    #         else:
    #             if isinstance(s, int) or isinstance(s, float):
    #                 error = np.zeros((self.n-1))
    #             elif isinstance(s, np.ndarray):
    #                 error = np.zeros((self.n-1, len(s)))
    #             else:
    #                 raise ValueError('Variable is not a float, a int or a NumPy array.')
    #             for i in range(self.n-1):
    #                 error[i] = 0.95*np.sqrt(np.diag(basis_fct_arr[i] @ self.sampling_variance_coeffs @ basis_fct_arr[i].T))
    #         upper_limit = val + error.T
    #         lower_limit = val - error.T
    #         return lower_limit, upper_limit
    #     self.confidence_limits = confidence_limits


        # def __compute_coefs_CV(self, r_tilde_train, mat_weights_train, basis_matrix_train, reg_param_mat):
    #     left = basis_matrix_train.T @ mat_weights_train @ basis_matrix_train + reg_param_mat @ self.Bspline_decomp.penalty_matrix    
    #     right = basis_matrix_train.T @ mat_weights_train @ np.reshape(r_tilde_train, (self.N*(self.n-1),))
    #     coefs_tem = np.linalg.solve(left, right)
    #     coefs_tem = np.reshape(coefs_tem,(-1,2))
    #     new_coefs = np.reshape(coefs_tem, (-1,))
    #     return new_coefs
    

    # def opti_lambda_CV_MSE(self, tol, max_iter, reg_param_list, model_Sigma, score, n_splits):
    #     K = len(reg_param_list)
    #     score_lambda_matrix = np.zeros((K,K))
    #     for i in range(K):
    #         for j in range(K):
    #             reg_param = np.array([reg_param_list[i], reg_param_list[j]])
    #             lbda, reg_param_mat = self.Bspline_decomp.check_regularization_parameter(reg_param)
    #             n_splits = 5
    #             kf = KFold(n_splits=n_splits, shuffle=True)
    #             for train_index, test_index in kf.split(self.v):
    #                 r_train = self.r_tilde[train_index]
    #                 r_test = self.r_tilde[test_index]
    #                 grid_train = self.v[train_index]
    #                 grid_test = self.v[test_index]
    #                 basis_matrix_train = self.Bspline_decomp.basis(np.expand_dims(grid_train, 1),).reshape((self.Bspline_decomp.basis.n_basis, -1)).T
    #                 mat_weights_train = np.eye(basis_matrix_train.shape[0])
    #                 coefs_train = self.__compute_coefs_CV(r_train, mat_weights_train, basis_matrix_train, reg_param_mat)
                    
    #                 L_tilde, L_tilde_inv = self.__compute_L_tilde(coefs_train)
    #                 mat_weights_train, weights = self.__compute_weights(self.Sigma, L_tilde_inv)
    #                 old_theta = np.reshape(self.basis_matrix @ coefs_train, (-1,self.n-1))
    #                 rel_error = 2*tol 
    #                 k = 0 
    #                 while rel_error > tol and k < max_iter:
    #                     coefs_opt = self.__compute_coefs_CV(r_train, mat_weights_train, basis_matrix_train, reg_param_mat)
    #                     L_tilde, L_tilde_inv = self.__compute_L_tilde(coefs_opt)
    #                     Sigma_opt, expect_MSE, err_obs = self.__opti_Sigma(coefs_opt, model_Sigma, L_tilde_inv)
    #                     mat_weights_train, weights = self.__compute_weights(Sigma_opt, L_tilde_inv)

    #                     new_theta = np.reshape(self.basis_matrix @ coefs_opt, (-1,self.n-1))
    #                     rel_error = np.linalg.norm(old_theta - new_theta)/np.linalg.norm(old_theta)
    #                     if self.verbose:
    #                         print('iteration for optimization of coefs and Sigma:', k, ', relative error:', rel_error)
    #                     old_theta = new_theta
    #                     k += 1




    # def __compute_weights(self, coefs, Sigma):
    #     self.W_tilde = np.zeros((self.N,self.n-1,self.n-1))
    #     weights = np.zeros((self.N,self.n-1,self.n-1))
    #     for i in range(self.N):
    #         theta_vi = self.theta_from_coefs(coefs,self.v[i])
    #         phi_vi = SE3.Ad_group(SE3.exp(-(self.grid[i+1]-self.v[i])*np.array([theta_vi[1],0,theta_vi[0],1,0,0])))
    #         L_tilde = self.L.T @ phi_vi @ self.L
    #         self.W_tilde[i] = self.u[i]*L_tilde @ Sigma(self.v[i]) @ L_tilde.T
    #         if np.allclose(self.coefs, np.zeros(np.sum(self.Bspline_decomp.nb_basis)), rtol=1e-10):
    #             W_tilde_inv = (1/self.u[i])*np.eye(2)
    #         else:
    #             try:
    #                 W_tilde_inv = np.linalg.inv(self.W_tilde[i])
    #             except:
    #                 print("At point ", i, "tilde_Wi singular matrix. Values of theta at that point: ", theta_vi)
    #                 W_tilde_inv = (1/self.u[i])*np.eye(2)
    #         weights[i] = (self.u[i]**2)*W_tilde_inv
    #     mat_weights = block_diag(*weights)
    #     return mat_weights


    # def __opti_Sigma(self, coefs, model_Sigma):
    #     res = np.zeros((self.N, self.n-1, self.n-1))
    #     for i in range(self.N):
    #         theta_vi = self.theta_from_coefs(coefs,self.v[i])
    #         phi_vi = SE3.Ad_group(SE3.exp(-(self.grid[i+1]-self.v[i])*np.array([theta_vi[1],0,theta_vi[0],1,0,0])))
    #         Li = self.L.T @ phi_vi @ self.L
    #         inv_Li = np.linalg.inv(Li)
    #         res[i] = self.u[i]* inv_Li @ ((theta_vi - self.r_tilde[i])[:,np.newaxis] @ (theta_vi - self.r_tilde[i])[np.newaxis,:] + self.cov_r_tilde[i]) @ inv_Li

    #     self.residuals = res
    #     # new_grid_v = np.concatenate(([0], self.v, [1]))
    #     # if model_Sigma=='single_continuous':
    #     #     sigma_square_arr = (res[:,0,0]+res[:,1,1])/2
    #     #     sigma_square_arr = np.concatenate(([sigma_square_arr[0]], sigma_square_arr, [sigma_square_arr[-1]]))
    #     #     sigma_square = interp1d(new_grid_v, sigma_square_arr)
    #     #     Sigma = lambda s: sigma_square(s)*np.eye(2)
    #     if model_Sigma=='vector':
    #         sigma_square_1 = np.mean(res[:,0,0])
    #         sigma_square_2 = np.mean(res[:,1,1])
    #         Sigma = lambda s: np.array([[sigma_square_1,0], [0, sigma_square_2]])
    #         if self.verbose:
    #             print("sigma_1:", np.sqrt(sigma_square_1), "sigma_2:", np.sqrt(sigma_square_2))
    #     # elif model_Sigma=='diagonal_continuous':
    #     #     sigma_square_1_arr = np.concatenate(([res[0,0,0]], res[:,0,0], [res[-1,0,0]]))
    #     #     sigma_square_2_arr = np.concatenate(([res[0,1,1]], res[:,1,1], [res[-1,1,1]]))
    #     #     sigma_square_1 = interp1d(new_grid_v, sigma_square_1_arr)
    #     #     sigma_square_2 = interp1d(new_grid_v, sigma_square_2_arr)
    #     #     Sigma = lambda s: np.array([[sigma_square_1(s),0], [0, sigma_square_2(s)]])
    #     else: # Case 'single_constant' by default
    #         sigma_square = np.mean((res[:,0,0]+res[:,1,1])/2)
    #         Sigma = lambda s: sigma_square*np.eye(2)
    #         if self.verbose:
    #             print("sigma:", np.sqrt(sigma_square))
         
    #     return Sigma


    # def opti_coefs_and_Sigma(self, tol, max_iter, reg_param_mat, model_Sigma):
    #     mat_weights = self.__compute_weights(self.coefs, self.Sigma)
    #     old_theta = self.theta_from_coefs(self.coefs,self.v) 
    #     rel_error = 2*tol 
    #     k = 0 
    #     while rel_error > tol and k < max_iter:

    #         coefs_opt = self.__opti_coefs(mat_weights, reg_param_mat)
    #         Sigma_opt = self.__opti_Sigma(coefs_opt, model_Sigma)
    #         new_theta = self.theta_from_coefs(coefs_opt,self.v)
    #         rel_error = np.linalg.norm(old_theta - new_theta)/np.linalg.norm(old_theta)
    #         if self.verbose:
    #             print('iteration for optimization of coefs and Sigma:', k, ', relative error:', rel_error)

    #         mat_weights = self.__compute_weights(coefs_opt, Sigma_opt)
    #         old_theta = new_theta
    #         k += 1

    #     return coefs_opt, Sigma_opt, mat_weights
    


    # def opti_lambda(self, tol, max_iter, reg_param_list, model_Sigma):
    #     K = len(reg_param_list)
    #     # self.tab_GCV_0 = np.zeros(K)
    #     # self.tab_GCV_1 = np.zeros(K)
    #     # self.tab_GCV_2 = np.zeros(K)
    #     # # self.tab_U_0 = np.zeros(K)
    #     # # self.tab_U_1 = np.zeros(K)
    #     # # self.tab_U_2 = np.zeros(K)
    #     # self.tab_L = np.zeros(K)

    #     GCV_scores = np.zeros(K)
    #     for k in range(K):
    #         stt = ttime.time()

    #         lbda, reg_param_mat = self.Bspline_decomp.check_regularization_parameter(reg_param_list[k])
    #         coefs, Sigma, mat_weights = self.opti_coefs_and_Sigma(tol, max_iter, reg_param_mat, model_Sigma)

    #         err = np.reshape((np.reshape(self.basis_matrix @ coefs, (-1,self.n-1)) - self.r_tilde), (self.N*(self.n-1),))
    #         hat_matrix = self.basis_matrix @ np.linalg.inv(self.basis_matrix.T @ mat_weights @ self.basis_matrix + reg_param_mat @ self.Bspline_decomp.penalty_matrix) @ self.basis_matrix.T @ mat_weights
            
    #         # V k = 2
    #         numerator = (1/self.N)*np.squeeze(err[np.newaxis,:] @ mat_weights.T @ mat_weights @ err[:,np.newaxis]) 
    #         denominator = ((1/self.N)*np.trace(mat_weights @ (np.eye(mat_weights.shape[0]) - hat_matrix)))**2
    #         GCV_scores_2 = numerator/denominator

    #         # # V k = 1
    #         # numerator = (1/self.N)*np.squeeze(err[np.newaxis,:] @ mat_weights @ err[:,np.newaxis]) 
    #         # denominator = ((1/self.N)*np.trace((np.eye(mat_weights.shape[0]) - hat_matrix)))**2
    #         # GCV_scores_1 = numerator/denominator

    #         # # V k = 0
    #         # numerator = (1/self.N)*np.squeeze(err[np.newaxis,:] @ err[:,np.newaxis]) 
    #         # denominator = ((1/self.N)*np.trace(np.linalg.inv(mat_weights) @ (np.eye(mat_weights.shape[0]) - hat_matrix)))**2
    #         # GCV_scores_0 = numerator/denominator

    #         # # U k = 2
    #         # U_2 = (1/self.N)*np.squeeze(err[np.newaxis,:] @ mat_weights.T @ mat_weights @ err[:,np.newaxis]) - (sig/self.N)*np.trace(mat_weights) + (2*sig/self.N)*np.trace(mat_weights @ hat_matrix)
    #         # self.tab_U_2[k] = U_2

    #         # # U k = 1
    #         # U_1 = (1/self.N)*np.squeeze(err[np.newaxis,:] @ mat_weights @ err[:,np.newaxis]) - (sig/self.N)*np.trace(np.eye(mat_weights.shape[0])) + (2*sig/self.N)*np.trace(hat_matrix)
    #         # self.tab_U_1[k] = U_1

    #         # # U k = 0
    #         # U_0 = (1/self.N)*np.squeeze(err[np.newaxis,:] @ err[:,np.newaxis]) - (sig/self.N)*np.trace(np.linalg.inv(mat_weights)) + (2*sig/self.N)*np.trace(np.linalg.inv(mat_weights) @ hat_matrix)
    #         # self.tab_U_0[k] = U_0

    #         # L_score
    #         # L = self.N*np.log(np.squeeze(err[np.newaxis,:] @ mat_weights @ err[:,np.newaxis])) + np.log(self.N)*np.trace(hat_matrix)
    #         # self.tab_L[k] = L

    #         GCV_scores[k] = GCV_scores_2
    #         if self.verbose:
    #             print('lambda value:', lbda, ' GCV score:', GCV_scores[k]) #, ' GCV score 1:', GCV_scores_1, ' GCV score 0:', GCV_scores_0, 'L score:', L) #, 'SSE:', SSE, 'df:', df_lbda) #, 'error_L:', Error_L, 'GCV_score_L:', GCV_scores_L)

    #         edd = ttime.time()
    #         print('time one step opti lbda', edd-stt)

    #     ind = np.argmin(GCV_scores)
    #     lbda_opt = reg_param_list[ind] 
    #     if self.verbose:
    #         print('Lambda OPTI:', lbda_opt)
    #     return lbda_opt



    # def MLE_fulldata(self, init_Z, kernel, coefs_0=None, regularization_parameter=None):
    #     self.kernel = kernel
    #     if regularization_parameter is not None:
    #         self.regularization_parameter = regularization_parameter
    #     else:
    #         self.regularization_parameter = 0
    #     self.Z = init_Z
    #     self.Q = self.Z[:,:self.n,:self.n]
    #     self.X = self.Z[:,:self.n,self.n]
    #     self.P = np.zeros((self.N+1,self.dim_g,self.dim_g))
    #     self.P_dble = np.zeros((self.N,self.dim_g,self.dim_g))
    #     # il nous faut quand même un a_theta_init pour le calcul des Wi
    #     """ M_step without conditional expectation """
    #     if coefs_0 is not None:
    #         self.coefs = coefs_0
    #     else:
    #         self.coefs = np.zeros((self.n*self.N))

    #     # Optimization of W
    #     # self.opti_Gamma()
    #     # Optimization of a_theta
    #     tol = 0.01
    #     rel_error = 2*tol
    #     k = 0
    #     self.__approx_distribution_r()
    #     self.__compute_F_G_tilde_W()
    #     # self.plot_theta()
    #     # Iterative least squares
    #     while rel_error > tol and k < 5:
    #         print('iteration of M_step:', k)
    #         old_coefs = self.coefs
    #         new_coefs = self.opti_theta()
    #         self.__compute_F_G_tilde_W()
    #         # self.plot_theta()
    #         rel_error = np.linalg.norm(old_coefs - new_coefs)/np.linalg.norm(old_coefs)
    #         # print(rel_error)
    #         k += 1
    #     # self.plot_theta()

    #     print('Number of iterations in M step: ', k)

    #     val_expected_loglikelihood = 0
    #     for i in range(self.N):
    #         val_expected_loglikelihood += (1/2)*np.log(np.linalg.det(self.u[i]*self.tilde_W[i]))
    #     val_expected_loglikelihood = val_expected_loglikelihood + self.regularization_parameter*self.coefs.T @ self.kernel(self.v[:,np.newaxis],self.v[:,np.newaxis]) @ self.coefs
    #     print("value of expected_loglikelihood:", -val_expected_loglikelihood)


    # def initialized_from_state(self, init_Z, coefs_0, nb_basis, reg_param_list=[], P=None, P_dble=None, order=4, method='approx', model_Sigma='single_constant'):

    #     N_param_smoothing = len(reg_param_list)    
    #     if N_param_smoothing==0:
    #         penalization = False
    #         reg_param_list = np.array([0])
    #     else:
    #         penalization = True
    #     self.Bspline_decomp = VectorBSplineSmoothing(self.n-1, nb_basis, domain_range=(self.grid[0], self.grid[-1]), order=order, penalization=penalization)
    #     V = np.expand_dims(self.v, 1)
    #     self.basis_matrix = self.Bspline_decomp.basis(V,).reshape((self.Bspline_decomp.basis.n_basis, -1)).T

    #     self.Z = init_Z
    #     self.Q = self.Z[:,:self.n,:self.n]
    #     self.X = self.Z[:,:self.n,self.n]
    #     if P is not None:
    #         self.P = P
    #     else:
    #         self.P = np.zeros((self.N+1,self.dim_g,self.dim_g))
    #     if P_dble is not None:
    #         self.P_dble = P_dble
    #     else:
    #         self.P_dble = np.zeros((self.N,self.dim_g,self.dim_g))
    #     # il nous faut quand même un a_theta_init pour le calcul des Wi
    #     """ M_step without conditional expectation """
    #     self.coefs = coefs_0
    #     self.__approx_distribution_r(method)
    #     self.Sigma = self.__opti_Sigma(self.coefs, model_Sigma)
    #     self.M_step(0.001, 5, reg_param_list, model_Sigma)
    #     # self.P0 = np.vstack((np.hstack((self.sigma_square*np.eye(self.n), np.zeros((self.n,self.n)))),np.zeros((self.n,2*self.n))))
    #     print("value of expected_loglikelihood:", self.expected_loglikelihood())  



    # def initialized_from_state(self, init_Z, nb_basis, reg_param_list=[], coefs_0=None, order=4, model_Sigma='single_constant'):

    #     N_param_smoothing = len(reg_param_list)    
    #     if N_param_smoothing==0:
    #         penalization = False
    #         reg_param_list = np.array([0])
    #     else:
    #         penalization = True
    #     self.Bspline_decomp = VectorBSplineSmoothing(self.n-1, nb_basis, domain_range=(self.grid[0], self.grid[-1]), order=order, penalization=penalization)
    #     V = np.expand_dims(self.v, 1)
    #     self.basis_matrix = self.Bspline_decomp.basis(V,).reshape((self.Bspline_decomp.basis.n_basis, -1)).T

    #     self.Z = init_Z
    #     self.Q = self.Z[:,:self.n,:self.n]
    #     self.X = self.Z[:,:self.n,self.n]
    #     self.P = np.zeros((self.N+1,self.dim_g,self.dim_g))
    #     self.P_dble = np.zeros((self.N,self.dim_g,self.dim_g))
    #     # il nous faut quand même un a_theta_init pour le calcul des Wi
    #     """ M_step without conditional expectation """
    #     if coefs_0 is None:
    #         coefs_0 = np.zeros((np.sum(self.Bspline_decomp.nb_basis)))
    #     self.coefs = coefs_0
    #     self.__approx_distribution_r(method="approx")
    #     self.Sigma, self.residuals = self.__opti_Sigma(self.coefs, model_Sigma)
    #     self.M_step(0.001, 5, reg_param_list, model_Sigma)
        
    #     return self.Bspline_decomp, self.Sigma, self.Gamma, self.mu0, self.P0
