from FrenetFDA.processing_Euclidean_curve_bis.unified_estimate_Frenet_state_space.iek_filter_smoother_Frenet_state import IEKFilterSmootherFrenetState
import FrenetFDA.utils.visualization as visu
from FrenetFDA.utils.Lie_group.SE3_utils import SE3
from FrenetFDA.utils.Lie_group.SO3_utils import SO3
from FrenetFDA.utils.smoothing_utils import VectorBSplineSmoothing
from pickle import *
import time
import os.path
import dill as pickle
import numpy as np
from scipy.linalg import block_diag
from scipy.interpolate import interp1d
from skfda.representation.basis import VectorValued, BSpline
from skfda.misc.regularization import TikhonovRegularization, compute_penalty_matrix
from skfda.misc.operators import LinearDifferentialOperator


class FrenetStateSpace:

    def __init__(self, grid_obs, Y_obs, dim=3):
        self.n = dim
        self.dim_g = int((1/2)*dim*(dim+1)) # dimension of the Lie Algebra se(n)
        self.Y = Y_obs #array of shape (N,n,)
        self.grid = grid_obs
        self.domain_range = (grid_obs[0], grid_obs[-1])
        self.N = len(Y_obs)
        self.rho = np.array([1,0,0])
        self.H = np.hstack((np.zeros((self.n, self.n)), np.eye(self.n)))
        # Def for case dim=3
        self.L = np.array([[0,1],[0,0],[1,0],[0,0],[0,0],[0,0]])
        self.u = self.grid[1:] - self.grid[:-1]
        self.v = (1/2)*(self.grid[1:]+self.grid[:-1])


    # def BSpline_basis_representation(self, nb_basis, order=4, penalization=True):
    #     self.nb_basis = nb_basis
    #     basis_kappa = BSpline(domain_range=(0,1), n_basis=nb_basis, order=order)
    #     basis_tau = BSpline(domain_range=(0,1), n_basis=nb_basis, order=order)
    #     self.basis = VectorValued([basis_kappa, basis_tau])
    #     def basis_fct(s):
    #         return np.squeeze(self.basis.evaluate(s))
    #     self.basis_fct = basis_fct
    #     V = np.expand_dims(self.v, 1)
    #     self.basis_matrix = self.basis(V,).reshape((self.basis.n_basis, -1)).T
    #     if penalization:
    #         self.penalty_matrix = compute_penalty_matrix(basis_iterable=(self.basis,),regularization_parameter=1,regularization=TikhonovRegularization(LinearDifferentialOperator(2)))
    #     else:
    #         self.penalty_matrix = np.zeros((nb_basis*2, nb_basis*2))
        

    # def Kernel_basis_representation(self, kernel, penalization=True):
    #     self.kernel = kernel
    #     V = np.expand_dims(self.v, 1)
    #     def basis_fct(s):
    #         if isinstance(s, int) or isinstance(s, float):
    #             return np.kron(self.kernel(V, np.array([s])[:,np.newaxis]), np.eye(2))
    #         elif isinstance(s, np.ndarray):
    #             return np.reshape(np.kron(self.kernel(V, s[:,np.newaxis]), np.eye(2)), (-1,len(s),2))
    #         else:
    #             raise ValueError('Variable is not a float, a int or a NumPy array.')
    #     self.nb_basis = self.N
    #     self.basis_fct = basis_fct
    #     self.basis_matrix = np.kron(self.kernel(V,V),np.eye(2))
    #     if penalization:
    #         self.penalty_matrix = np.kron(self.kernel(V,V),np.eye(2))
    #     else:
    #         self.penalty_matrix = np.zeros((self.basis_matrix.shape))


    def expectation_maximization(self, tol, max_iter, nb_basis, regularization_parameter_list = [], init_params = None, init_states = None, order=4, method='approx', model_Sigma='single_constant'):

        self.init_tab()
        if init_params is None and init_states is None:
            raise Exception("Must pass in argument an initial guess estimate for the state or for the parameters")
        elif init_params is not None and init_states is not None:
            raise Exception("Must choose between an initial guess estimate for the state or for the parameters")
        elif init_states is not None and init_params is None:
            self.initialized_from_state(init_states["Z"], init_states["init_coefs"], P=init_states["P"], P_dble=init_states["P_dble"])
        else:
            self.W = init_params["W"]
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
        val_loglikelihood = 0
        self.tab_increment()
        time_init = time.time()

        while k < max_iter and rel_error > tol:
            print('-------------- Iteration',k+1,'/',max_iter,' --------------')
            self.E_step()
            self.__approx_distribution_r(method)
            self.M_step(0.001, 5, regularization_parameter_list, model_Sigma)

            new_val_loglikelihood = self.loglikelihood()
            print('value of loglikelihood: ', new_val_loglikelihood)
            rel_error = np.abs(val_loglikelihood - new_val_loglikelihood)
            print('relative error: ', rel_error)
            self.tab_increment(rel_error, new_val_loglikelihood)
            val_loglikelihood = new_val_loglikelihood
            k+=1
        time_end = time.time()
        self.nb_iterations = k
        self.duration = time_end - time_init
        print('End. \n Number of iterations:', k, ', total duration:', self.duration, ' seconds.')


    def E_step(self):
        """
            Expectation step: Suppose that self.sigma, self.W, self.a_theta, self.mu0, self.P0 are known. Call the tracking and smoothing method.

        """
        print('___ E step ___')
        kalman_filter = IEKFilterSmootherFrenetState(self.n, self.W, self.Sigma, self.theta, Z0=self.mu0, P0=self.P0)
        kalman_filter.smoothing(self.grid, self.Y)
        self.Z = kalman_filter.smooth_Z
        self.Q = kalman_filter.smooth_Q
        self.X = kalman_filter.smooth_X
        self.P = kalman_filter.smooth_P
        self.P_dble = kalman_filter.smooth_P_dble


    def M_step(self, tol, max_iter, reg_param_list, model_Sigma):
        """
        
        """
        print('___ M step ___')
        # Optimization of W
        self.opti_W()
        # Optimization of lambda
        self.regularization_parameter = self.opti_lambda(tol, max_iter, reg_param_list, model_Sigma)
        self.regularization_parameter, self.regularization_parameter_matrix = self.Bspline_decomp.check_regularization_parameter(self.regularization_parameter)
        # Optimization of theta given lambda
        self.coefs, self.Sigma, self.mat_weights = self.opti_coefs_and_Sigma(tol, max_iter, self.regularization_parameter_matrix, model_Sigma)
        self.P0 = self.P[0]
        self.mu0 = self.Z[0]

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
        r = np.zeros((self.N, self.dim_g))
        self.r_tilde = np.zeros((self.N, self.n-1))
        cov_r = np.zeros((self.N, self.dim_g, self.dim_g))
        self.cov_r_tilde = np.zeros((self.N, self.n-1, self.n-1))
        if method=='monte_carlo':
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
                r[i] = -(1/self.u[i])*SE3.log(np.linalg.inv(self.Z[i+1])@self.Z[i])
                self.r_tilde[i] = self.L.T @ r[i]
                Ad = SE3.Ad_group(np.linalg.inv(self.Z[i+1])@self.Z[i])
                cov_r[i] = (1/self.u[i]**2)*(self.P[i+1] - self.P_dble[i].T @ Ad.T - Ad @ self.P_dble[i] + Ad @ self.P[i] @ Ad.T)
                self.cov_r_tilde[i] = self.L.T @ cov_r[i] @ self.L


    def __compute_weights(self, coefs, Sigma):
        self.W_tilde = np.zeros((self.N,self.n-1,self.n-1))
        weights = np.zeros((self.N,self.n-1,self.n-1))
        for i in range(self.N):
            theta_vi = self.theta_from_coefs(coefs,self.v[i])
            phi_vi = SE3.Ad_group(SE3.exp(-(self.grid[i+1]-self.v[i])*np.array([theta_vi[1],0,theta_vi[0],1,0,0])))
            L_tilde = self.L.T @ phi_vi @ self.L
            self.W_tilde[i] = self.u[i]*L_tilde @ Sigma(self.v[i]) @ L_tilde.T
            if np.allclose(self.coefs, np.zeros(np.sum(self.Bspline_decomp.nb_basis)), rtol=1e-10):
                W_tilde_inv = np.hstack((np.vstack((np.eye(self.n),np.zeros((self.n,self.n)))),np.zeros((2*self.n, self.n))))
            else:
                try:
                    W_tilde_inv = np.linalg.inv(self.W_tilde[i])
                except:
                    print("At point ", i, "tilde_Wi singular matrix. Values of theta at that point: ", theta_vi)
                    W_tilde_inv = np.hstack((np.vstack((np.eye(self.n),np.zeros((self.n,self.n)))),np.zeros((2*self.n, self.n))))
            weights[i] = (self.u[i]**2)*W_tilde_inv
        mat_weights = block_diag(*weights)
        return mat_weights
    
    
    def __opti_coefs(self, mat_weights, reg_param_mat):
        left = self.basis_matrix.T @ mat_weights @ self.basis_matrix + reg_param_mat @ self.Bspline_decomp.penalty_matrix    
        right = self.basis_matrix.T @ mat_weights @ np.reshape(self.r_tilde, (self.N*(self.n-1),))
        coefs_tem = np.linalg.solve(left, right)
        coefs_tem = np.reshape(coefs_tem,(-1,2))
        new_coefs = np.reshape(coefs_tem, (-1,))
        return new_coefs
    

    def __opti_Sigma(self, coefs, model_Sigma):
        res = np.zeros((self.N, self.n-1, self.n-1))
        for i in range(self.N):
            theta_vi = self.theta_from_coefs(coefs,self.v[i])
            phi_vi = SE3.Ad_group(SE3.exp(-(self.grid[i+1]-self.v[i])*np.array([theta_vi[1],0,theta_vi[0],1,0,0])))
            Li = self.L.T @ phi_vi @ self.L
            inv_Li = np.linalg.inv(Li)
            res[i] = self.u[i]* inv_Li @ ((theta_vi - self.r_tilde[i])[:,np.newaxis] @ (theta_vi - self.r_tilde[i])[np.newaxis,:] + self.cov_r_tilde[i]) @ inv_Li
        self.residuals = res
        new_grid_v = np.concatenate(([0], self.v, [1]))
        if model_Sigma=='single_continuous':
            sigma_square_arr = (res[:,0,0]+res[:,1,1])/2
            sigma_square_arr = np.concatenate(([sigma_square_arr[0]], sigma_square_arr, [sigma_square_arr[-1]]))
            sigma_square = interp1d(new_grid_v, sigma_square_arr)
            Sigma = lambda s: sigma_square(s)*np.eye(2)
        elif model_Sigma=='diagonal_constant':
            sigma_square_1 = np.mean(res[:,0,0])
            sigma_square_2 = np.mean(res[:,1,1])
            Sigma = lambda s: np.array([[sigma_square_1,0], [0, sigma_square_2]])
            print("sigma_1:", np.sqrt(sigma_square_1), "sigma_2:", np.sqrt(sigma_square_2))
        elif model_Sigma=='diagonal_continuous':
            sigma_square_1_arr = np.concatenate(([res[0,0,0]], res[:,0,0], [res[-1,0,0]]))
            sigma_square_2_arr = np.concatenate(([res[0,1,1]], res[:,1,1], [res[-1,1,1]]))
            sigma_square_1 = interp1d(new_grid_v, sigma_square_1_arr)
            sigma_square_2 = interp1d(new_grid_v, sigma_square_2_arr)
            Sigma = lambda s: np.array([[sigma_square_1(s),0], [0, sigma_square_2(s)]])
        else: # Case 'single_constant' by default
            sigma_square = np.mean((res[:,0,0]+res[:,1,1])/2)
            Sigma = lambda s: sigma_square*np.eye(2)
            print("sigma:", np.sqrt(sigma_square))
        return Sigma


    def opti_coefs_and_Sigma(self, tol, max_iter, reg_param_mat, model_Sigma):
        mat_weights = self.__compute_weights(self.coefs, self.Sigma)
        old_theta = self.theta_from_coefs(self.coefs,self.v) 
        rel_error = 2*tol 
        k = 0 
        while rel_error > tol and k < max_iter:

            coefs_opt = self.__opti_coefs(mat_weights, reg_param_mat)
            Sigma_opt = self.__opti_Sigma(coefs_opt, model_Sigma)
            new_theta = self.theta_from_coefs(coefs_opt,self.v)
            rel_error = np.linalg.norm(old_theta - new_theta)/np.linalg.norm(old_theta)
            print('iteration for optimization of coefs and Sigma:', k, ', relative error:', rel_error)

            mat_weights = self.__compute_weights(coefs_opt, Sigma_opt)
            old_theta = new_theta
            k += 1

        return coefs_opt, Sigma_opt, mat_weights
    


    def opti_lambda(self, tol, max_iter, reg_param_list, model_Sigma):
        K = len(reg_param_list)
        GCV_scores = np.zeros(K)
        for k in range(K):
            lbda, reg_param_mat = self.Bspline_decomp.check_regularization_parameter(reg_param_list[k])
            coefs, Sigma, mat_weights = self.opti_coefs_and_Sigma(tol, max_iter, reg_param_mat, model_Sigma)
            # err = np.reshape((np.reshape(self.basis_matrix @ coefs, (-1,self.n-1)) - self.r_tilde), (self.N*(self.n-1),))

            SSE = self.__compute_SSE(coefs, mat_weights)
            df_lbda = self.basis_matrix @ np.linalg.inv(self.basis_matrix.T @ mat_weights @ self.basis_matrix + reg_param_mat @ self.Bspline_decomp.penalty_matrix) @ self.basis_matrix.T @ mat_weights
            df_lbda = np.sum(np.reshape(np.diag(df_lbda), (-1,self.n-1)), axis=0)
            GCV_scores[k] = np.sum(np.array([(self.N*SSE[i])/((self.N - df_lbda[i])**2) for i in range(self.n-1)]))

            # SSE = np.squeeze(err[np.newaxis,:] @ mat_weights @ err[:,np.newaxis]) + np.trace(mat_weights @ block_diag(*self.cov_r_tilde))
            # df_lbda = np.trace(self.basis_matrix @ np.linalg.inv(self.basis_matrix.T @ mat_weights @ self.basis_matrix + reg_param_mat @ self.Bspline_decomp.penalty_matrix) @ self.basis_matrix.T @ mat_weights)  
            # GCV_scores[k] = (self.N*SSE)/((self.N - df_lbda)**2)
            print('lambda value:', lbda, ' GCV score:', GCV_scores[k], 'SSE:', SSE, 'df:', df_lbda)
        ind = np.argmin(GCV_scores)
        lbda_opt = reg_param_list[ind] 
        return lbda_opt


        
    def __compute_SSE(self, coefs, mat_weights):
        err = np.reshape((np.reshape(self.basis_matrix @ coefs, (-1,self.n-1)) - self.r_tilde), (self.N*(self.n-1),))
        SSE = np.diag(np.reshape(err @ mat_weights, (-1,self.n-1)).T @ (np.reshape(self.basis_matrix @ coefs, (-1,self.n-1)) - self.r_tilde))
        # print('part 1 SSE', SSE)
        # SSE = SSE + np.sum(np.reshape(np.diag(mat_weights @ block_diag(*self.cov_r_tilde)), (-1,self.n-1)), axis=0)
        # print('part 2 SSE', SSE)
        return SSE


    def compute_sampling_variance(self):
        SSE = self.__compute_SSE(self.coefs, self.mat_weights)
        self.residuals_error = np.array([SSE[i]/(self.N-self.Bspline_decomp.nb_basis[i]) for i in range(self.n-1)])
        res_mat = block_diag(*np.apply_along_axis(np.diag, 1, np.array([self.residuals_error for i in range(self.N)])))
        S = np.linalg.inv(self.basis_matrix.T @ self.mat_weights @ self.basis_matrix + self.regularization_parameter_matrix @ self.Bspline_decomp.penalty_matrix) @ self.basis_matrix.T @ self.mat_weights
        self.sampling_variance_coeffs = S @ res_mat @ S.T 
        self.sampling_variance_yhat = self.basis_matrix @ self.sampling_variance_coeffs @ self.basis_matrix.T

        def confidence_limits(s):
            val = self.theta(s)
            basis_fct_arr = self.Bspline_decomp.basis_fct(s).T 
            if (self.n-1)==1:
                if isinstance(s, int) or isinstance(s, float):
                    error = 0
                elif isinstance(s, np.ndarray):
                    error = np.zeros((len(s)))
                else:
                    raise ValueError('Variable is not a float, a int or a NumPy array.')
                error = 0.95*np.sqrt(np.diag(basis_fct_arr @ self.sampling_variance_coeffs @ basis_fct_arr.T))
            else:
                if isinstance(s, int) or isinstance(s, float):
                    error = np.zeros((self.n-1))
                elif isinstance(s, np.ndarray):
                    error = np.zeros((self.n-1, len(s)))
                else:
                    raise ValueError('Variable is not a float, a int or a NumPy array.')
                for i in range(self.n-1):
                    error[i] = 0.95*np.sqrt(np.diag(basis_fct_arr[i] @ self.sampling_variance_coeffs @ basis_fct_arr[i].T))
            upper_limit = val + error.T
            lower_limit = val - error.T
            return lower_limit, upper_limit
        self.confidence_limits = confidence_limits
        


    def opti_W(self):
        W = np.zeros((self.n,self.n))
        for i in range(1,self.N+1):
            W += (self.Y[i-1]-self.X[i])[:,np.newaxis]@(self.Y[i-1]-self.X[i])[np.newaxis,:] + self.Q[i]@self.H@self.P[i]@self.H.T@self.Q[i].T
        self.W = W/self.N


    def loglikelihood(self):
        val = (1/2)*self.N*np.log(np.linalg.det(self.W)) + (1/2)*np.log(np.linalg.det(self.P0))
        for i in range(self.N):
            val += (1/2)*np.log(np.linalg.det(self.Sigma(self.v[i])))
            val += (1/2)*np.log(np.linalg.det(self.W_tilde[i])) 
        regularization_parameter_matrix = np.diag(np.concatenate([np.repeat(self.regularization_parameter[i], self.Bspline_decomp.nb_basis[i]) for i in range(self.n-1)]))
        val = val + self.coefs.T @ regularization_parameter_matrix @ self.Bspline_decomp.penalty_matrix @ self.coefs        
        return -val


    def init_tab(self):
        self.tab_rel_error = []
        self.tab_loglikelihood = []
        self.tab_Z = []
        self.tab_X = []
        self.tab_Q = []
        self.tab_sigma = []
        self.tab_theta = []
        self.tab_W = []


    def tab_increment(self, rel_error=None, val_loglikelihood=None):
        if rel_error is not None:
            self.tab_rel_error.append(rel_error)
        if val_loglikelihood is not None:
            self.tab_loglikelihood.append(val_loglikelihood)
        if hasattr(self, "Z"):
            self.tab_Z.append(self.Z)
        if hasattr(self, "X"):
            self.tab_X.append(self.X)
        if hasattr(self, "Q"):
            self.tab_Q.append(self.Q)
        if hasattr(self, "Sigma"):
            self.tab_sigma.append(np.array([self.Sigma(vi) for vi in self.v]))
        if hasattr(self, "theta"):
            self.tab_theta.append(self.theta(self.v))
        if hasattr(self, "W"):
            self.tab_W.append(self.W)


    def save_tab_results(self, filename):
        dic = {"tab_rel_error": self.tab_rel_error, "tab_loglikelihood": self.tab_loglikelihood, "tab_Z": self.tab_Z, "tab_X": self.tab_X, 
               "tab_Q": self.tab_Q, "tab_sigma": self.tab_sigma, "tab_theta": self.tab_theta, "tab_W": self.tab_W, "duration":self.duration, "nb_iterations":self.nb_iterations}
        if os.path.isfile(filename):
            print("Le fichier ", filename, " existe déjà.")
            filename = filename + '_bis'
        fil = open(filename,"xb")
        pickle.dump(dic,fil)
        fil.close()


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
    #     # self.opti_W()
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

    #     val_loglikelihood = 0
    #     for i in range(self.N):
    #         val_loglikelihood += (1/2)*np.log(np.linalg.det(self.u[i]*self.tilde_W[i]))
    #     val_loglikelihood = val_loglikelihood + self.regularization_parameter*self.coefs.T @ self.kernel(self.v[:,np.newaxis],self.v[:,np.newaxis]) @ self.coefs
    #     print("value of loglikelihood:", -val_loglikelihood)


    def initialized_from_state(self, init_Z, coefs_0, nb_basis, reg_param_list=[], P=None, P_dble=None, order=4, method='approx', model_Sigma='single_constant'):

        N_param_smoothing = len(reg_param_list)    
        if N_param_smoothing==0:
            penalization = False
            reg_param_list = np.array([0])
        else:
            penalization = True
        self.Bspline_decomp = VectorBSplineSmoothing(self.n-1, nb_basis, domain_range=(self.grid[0], self.grid[-1]), order=order, penalization=penalization)
        V = np.expand_dims(self.v, 1)
        self.basis_matrix = self.Bspline_decomp.basis(V,).reshape((self.Bspline_decomp.basis.n_basis, -1)).T

        self.Z = init_Z
        self.Q = self.Z[:,:self.n,:self.n]
        self.X = self.Z[:,:self.n,self.n]
        if P is not None:
            self.P = P
        else:
            self.P = np.zeros((self.N+1,self.dim_g,self.dim_g))
        if P_dble is not None:
            self.P_dble = P_dble
        else:
            self.P_dble = np.zeros((self.N,self.dim_g,self.dim_g))
        # il nous faut quand même un a_theta_init pour le calcul des Wi
        """ M_step without conditional expectation """
        self.coefs = coefs_0
        self.__approx_distribution_r(method)
        self.Sigma = self.__opti_Sigma(self.coefs, model_Sigma)
        self.M_step(0.001, 5, reg_param_list, model_Sigma)
        # self.P0 = np.vstack((np.hstack((self.sigma_square*np.eye(self.n), np.zeros((self.n,self.n)))),np.zeros((self.n,2*self.n))))
        print("value of loglikelihood:", self.loglikelihood())  
