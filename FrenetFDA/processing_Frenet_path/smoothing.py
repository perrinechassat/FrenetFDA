"""
    TO COMPLETE: 

    Class to make smoothing of Q if we have noisy observations of the Frenet Path as input.
    Include:
        - Smoothing by tracking ?
        - Smoothing by karcher_mean ? 

"""

import numpy as np
from FrenetFDA.utils.smoothing_utils import compute_weight_neighbors_local_smoothing, VectorBSplineSmoothing, grid_search_GCV_optimization_Bspline_hyperparameters
from FrenetFDA.utils.Frenet_Serret_utils import solve_FrenetSerret_ODE_SE, solve_FrenetSerret_ODE_SO
from FrenetFDA.utils.Lie_group.SO3_utils import SO3
from FrenetFDA.utils.Lie_group.SE3_utils import SE3
from scipy.linalg import expm
from sklearn.model_selection import KFold
from skopt import gp_minimize
from geomstats.geometry.special_orthogonal import SpecialOrthogonal

class KarcherMeanSmoother:
    
    def __init__(self, grid, Q=None, Z=None, adaptive=False):  
        if Q is None and Z is None:
            raise Exception("Either Q or Z must be passed as input.") 
        elif Z is None:
            self.N, self.dim, _ = Q.shape
            if len(grid)!=self.N:
                raise Exception("Invalide dimension of Q and grid.")
            self.grid = grid
            self.Q = Q 
            self.Z = None
        else:
            self.N, dim, _ = Z.shape
            if len(grid)!=self.N:
                raise Exception("Invalide dimension of Q and grid.")
            self.grid = grid
            self.dim = dim-1
            self.Q = Z[:,:self.dim,:self.dim]
            self.Z = Z
        self.dim_theta = len(np.diag(np.eye(self.dim), k=1))
        self.adaptive_ind = adaptive


    def __fit_SE(self, h, theta, grid_obs, obs_Z, grid_eval=None):
        if grid_eval is None:
            grid_eval = grid_obs
        N_grid_eval = len(grid_eval)
        neighbor_obs, weight, grid_double, delta = compute_weight_neighbors_local_smoothing(grid_obs, grid_eval, h, self.adaptive_ind)

        M = np.zeros((N_grid_eval, self.dim+1, self.dim+1))
        for q in range(N_grid_eval):
            Obs_q = obs_Z[neighbor_obs[q]]
            theta_q = np.multiply(delta[q][:,np.newaxis],theta(grid_double[q]))
            a_theta_q = np.stack((theta_q[:,1], np.zeros(len(theta_q)), theta_q[:,0], np.ones(len(theta_q)), np.zeros(len(theta_q)), np.zeros(len(theta_q))), axis=-1)
            exp_a_theta_q = np.apply_along_axis(SE3.exp, 1, a_theta_q) # n_q,4,4
            mat_product = np.matmul(Obs_q, exp_a_theta_q)
            M[q] = SE3.frechet_mean(mat_product, weights=weight[q])
        return M


    def __fit_SO(self, h, theta, grid_obs, obs_Q, grid_eval=None):
        if grid_eval is None:
            grid_eval = grid_obs
        N_grid_eval = len(grid_eval)
        neighbor_obs, weight, grid_double, delta = compute_weight_neighbors_local_smoothing(grid_obs, grid_eval, h, self.adaptive_ind)

        M = np.zeros((N_grid_eval, self.dim, self.dim))
        for q in range(N_grid_eval):
            Obs_q = obs_Q[neighbor_obs[q]]
            theta_q = np.multiply(delta[q][:,np.newaxis],theta(grid_double[q]))
            a_theta_q = np.stack((theta_q[:,1], np.zeros(len(theta_q)), theta_q[:,0]), axis=-1)
            exp_a_theta_q = np.apply_along_axis(SO3.exp, 1, a_theta_q) # n_q,3,3
            # mat_product = Obs_q
            # ind_norm = np.where(np.linalg.matrix_norm(A_q) > 0.)[0]
            # mat_product[ind_norm] = np.matmul(Obs_q[ind_norm].double(), exp_A_q[ind_norm].double())
            mat_product = np.matmul(Obs_q, exp_a_theta_q)
            for i in range(len(mat_product)):
                if np.linalg.det(mat_product[i])<0:
                    mat_product[i] = mat_product[i]@np.diag((1,1,-1))
            M[q] = SO3.frechet_mean(mat_product, weights=weight[q])
            
        return M
        

    def fit(self, h, theta, grid_eval=None):
        if self.Z is None:
            M = self.__fit_SO(h, theta, self.grid, self.Q, grid_eval=grid_eval)
        else:
            M = self.__fit_SE(h, theta, self.grid, self.Z, grid_eval=grid_eval)
        return M
    

    def bayesian_optimization_hyperparameters(self, theta, n_call_bayopt, h_bounds, n_splits=10, verbose=True):

        # ## CV optimization of h

        def func(x):
            print(x)
            score = np.zeros(n_splits)
            kf = KFold(n_splits=n_splits, shuffle=True)
            grid_split = self.grid[1:-1]
            ind_CV = 0

            for train_index, test_index in kf.split(grid_split):
                train_index = train_index+1
                test_index = test_index+1
                train_index = np.concatenate((np.array([0]), train_index, np.array([len(self.grid[1:-1])+1])))
                grid_train = self.grid[train_index]
                # grid_test = self.grid[test_index]
                Q_train = self.Q[train_index]
                # Q_smooth = self.__fit_SO(x[0], theta, grid_train, Q_train, grid_eval=grid_train)
                # dist = np.mean(SO3.geodesic_distance(Q_train, Q_smooth))
                Q_test = self.Q[test_index]
                Q_smooth = self.__fit_SO(x[0], theta, grid_train, Q_train, grid_eval=self.grid)
                dist = np.mean(SO3.geodesic_distance(Q_test, Q_smooth[test_index]))
                score[ind_CV] = dist
                ind_CV += 1 

            return np.mean(score)

        # Do a bayesian optimisation and return the optimal parameter (lambda_kappa, lambda_tau)
        
        res_bayopt = gp_minimize(func,               # the function to minimize
                        [h_bounds],        # the bounds on each dimension of x
                        acq_func="EI",        # the acquisition function
                        n_calls=n_call_bayopt,       # the number of evaluations of f
                        n_random_starts=2,    # the number of random initialization points
                        random_state=1,       # the random seed
                        # n_jobs=1,            # use all the cores for parallel calculation
                        verbose=verbose)
        h_opt = res_bayopt.x[0]

        return h_opt

    


class TrackingSmootherLinear:

    def __init__(self, grid, Q=None, Z=None):  
        if Q is None and Z is None:
            raise Exception("Either Q or Z must be passed as input.") 
        elif Z is None:
            self.N, self.dim, _ = Q.shape
            if len(grid)!=self.N:
                raise Exception("Invalide dimension of Q and grid.")
            self.grid = grid
            self.Q = Q 
            self.Q_lin = np.concatenate((self.Q[:,:,0], self.Q[:,:,1], self.Q[:,:,2]), axis=1)
            self.Z = None
        else:
            self.N, dim, _ = Z.shape
            if len(grid)!=self.N:
                raise Exception("Invalide dimension of Q and grid.")
            self.grid = grid
            self.dim = dim-1
            self.Q = Z[:,:self.dim,:self.dim]
            self.Q_lin = np.concatenate((self.Q[:,:,0], self.Q[:,:,1], self.Q[:,:,2]), axis=1)
            self.Z = Z
        self.dim_theta = len(np.diag(np.eye(self.dim), k=1))
        self.u = self.grid[1:] - self.grid[:-1]
        self.v = (self.grid[1:] + self.grid[:-1])/2


    def __fit(self, lbda, theta, grid, Qlin):

        p = self.dim*self.dim
        N_obs = len(grid)
        Y = np.zeros((N_obs, p+1, p+1))
        A_tilde = np.zeros((N_obs, p+1, p+1))
        B_tilde = np.zeros((N_obs, p+1, p))
        grid_u = grid[1:] - grid[:-1]
        grid_v = (grid[1:] + grid[:-1])/2
        R = np.zeros((N_obs, p, p))

        for q in range(N_obs):
            Obs_q = Qlin[q]
            Y[q] = np.vstack((np.hstack((np.eye(p), -Obs_q[:,np.newaxis])), np.hstack((-Obs_q, np.array([np.linalg.norm(Obs_q)**2])))[np.newaxis,:]))
            # Y[q] = np.concatenate((np.concatenate((np.eye(p),-Obs_q[:,np.newaxis]),axis=1), np.concatenate((-Obs_q[np.newaxis,:], np.linalg.norm(Obs_q)**2),axis=1)),axis=0)

            if q < N_obs-1:

                R[q] = lbda*grid_u[q]*np.eye(p)
                theta_q = theta(grid_v[q])
                Atheta_q = SO3.wedge(np.array([theta_q[1], 0, theta_q[0]]))
                big_Atheta_q = np.kron(-Atheta_q, np.eye(self.dim))
                extend_mat = np.concatenate((np.concatenate((grid_u[q]*big_Atheta_q, np.eye(p)), axis=1), np.zeros((p,2*p))), axis=0)
                exp_extend_mat = expm(extend_mat)
                A = exp_extend_mat[:p,:p]
                B = grid_u[q] * exp_extend_mat[:p,p:]
                A_tilde[q] = np.concatenate((np.concatenate((A,np.zeros((p,1))),axis=1), np.concatenate((np.zeros((1,p)), np.eye(1)),axis=1)),axis=0)
                B_tilde[q] = np.concatenate((B, np.zeros((1,p))), axis=0)

        M0 = Qlin[0]
        U, Z, K, P = self.tracking(M0, Y, R, A_tilde, B_tilde, N_obs-1)
        M_hat = np.reshape(Z[:,:p], (-1,self.dim,self.dim), order='F')

        so3 = SpecialOrthogonal(3)
        M_proj = so3.projection(M_hat)
        return M_proj


    def fit(self, lbda, theta):

        M = self.__fit(lbda, theta, self.grid, self.Q_lin)
        return M
    

    # def get_X0(self, P0):
    #     K = P0[:self.dim,:self.dim]
    #     a = P0[self.dim,:self.dim]
    #     c = P0[:self.dim,self.dim]
    #     b = P0[self.dim,self.dim]
    #     return -np.linalg.inv(K+K.T) @ (a+c.T)


    def state_space_model(self, A, z_t_minus_1, B, u_t_minus_1):
        """
        Calculates the state at time t given the state at time t-1 and
        the control inputs applied at time t-1
        """
        state_estimate_t = (A @ z_t_minus_1) + (B @ u_t_minus_1)
        return state_estimate_t


    def tracking(self, X0, Q, R, A, B, N):
        P = [None] * (N + 1)
        Qf = Q[N]
        P[N] = Qf
        for i in range(N, 0, -1):
            # Discrete-time Algebraic Riccati equation to calculate the optimal state cost matrix
            P[i-1] = Q[i-1] + A[i-1].T @ P[i] @ A[i-1] - (A[i-1].T @ P[i] @ B[i-1]) @ np.linalg.inv(R[i-1] + B[i-1].T @ P[i] @ B[i-1]) @ (B[i-1].T @ P[i] @ A[i-1])

        # Create a list of N elements
        K = [None] * N
        u = [None] * N
        z = np.zeros((N+1,self.dim*self.dim+1))
        # if X0 is None:
        #     X0 = self.get_X0(P[0])
        #     z[0] = np.concatenate((X0,np.squeeze(np.eye(self.dim),axis=0)),axis=0)
        #     # print(X0)
        # else:
        z[0] = np.hstack((X0, 1))
        # z[0] = np.concatenate((X0,np.eye(1)),axis=0)
        # z[0] = np.concatenate((X0,np.array([1])))
        for i in range(0,N):
            # Calculate the optimal feedback gain K
            K[i] = np.linalg.inv(R[i] + B[i].T @ P[i+1] @ B[i]) @ B[i].T @ P[i+1] @ A[i]
            u[i] = -K[i] @ z[i]
            # print(u[i])
            z[i+1] = self.state_space_model(A[i], z[i], B[i], u[i])

        Z = np.array(z)
        return np.array(u), Z, np.array(K), np.array(P)
    

    def bayesian_optimization_hyperparameters(self, theta, n_call_bayopt, lbda_bounds, n_splits=10, verbose=True):

        # ## CV optimization of h

        def func(x):
            score = np.zeros(n_splits)
            kf = KFold(n_splits=n_splits, shuffle=True)
            grid_split = self.grid[1:-1]
            ind_CV = 0

            for train_index, test_index in kf.split(grid_split):
                train_index = train_index+1
                test_index = test_index+1
                train_index = np.concatenate((np.array([0]), train_index, np.array([len(self.grid[1:-1])+1])))
                grid_train = self.grid[train_index]
                Q_train = self.Q[train_index]
                Qlin_train= np.concatenate((Q_train[:,:,0], Q_train[:,:,1], Q_train[:,:,2]), axis=1)
                Q_smooth = self.__fit(x[0], theta, grid_train, Qlin_train)
                dist = np.mean(SO3.geodesic_distance(Q_train, Q_smooth))
                score[ind_CV] = dist
                ind_CV += 1 

            return np.mean(score)

        # Do a bayesian optimisation and return the optimal parameter (lambda_kappa, lambda_tau)
        
        res_bayopt = gp_minimize(func,               # the function to minimize
                        [lbda_bounds],        # the bounds on each dimension of x
                        acq_func="EI",        # the acquisition function
                        n_calls=n_call_bayopt,       # the number of evaluations of f
                        n_random_starts=2,    # the number of random initialization points
                        random_state=1,       # the random seed
                        # n_jobs=1,            # use all the cores for parallel calculation
                        verbose=verbose)
        lbda_opt = res_bayopt.x[0]

        return lbda_opt
    




# class TrackingSmoother:

#     def __init__(self, grid, Q=None, Z=None):  
#         if Q is None and Z is None:
#             raise Exception("Either Q or Z must be passed as input.") 
#         elif Z is None:
#             self.N, self.dim, _ = Q.shape
#             if len(grid)!=self.N:
#                 raise Exception("Invalide dimension of Q and grid.")
#             self.grid = grid
#             self.Q = Q 
#             self.Z = None
#         else:
#             self.N, dim, _ = Z.shape
#             if len(grid)!=self.N:
#                 raise Exception("Invalide dimension of Q and grid.")
#             self.grid = grid
#             self.dim = dim-1
#             self.Q = Z[:,:self.dim,:self.dim]
#             self.Z = Z
#         self.dim_theta = len(np.diag(np.eye(self.dim), k=1))
#         self.u = self.grid[1:] - self.grid[:-1]
#         self.v = (self.grid[1:] + self.grid[:-1])/2


#     def fit(self, lbda, theta):

#         M = np.zeros((self.N, 2*self.dim, 2*self.dim))
#         A_tilde = np.zeros((self.N, 2*self.dim, 2*self.dim))
#         B_tilde = np.zeros((self.N, 2*self.dim, self.dim))
#         R = np.zeros((self.N, self.dim, self.dim))

#         for q in range(self.N):
#             Obs_q = self.Q[q]
#             M[q] = np.concatenate((np.concatenate((np.eye(self.dim),-Obs_q),axis=1), np.concatenate((-Obs_q.T, Obs_q.T @ Obs_q),axis=1)),axis=0)

#             if q < self.N-1:
#                 R[q] = lbda*self.u[q]*np.eye(self.dim)
#                 theta_q = theta(self.grid[q])
#                 exp_Aq = SO3.exp(self.u[q]*np.array([theta_q[1], 0, theta_q[0]]))
#                 extend_A = np.concatenate((np.concatenate((self.u[q]*SO3.wedge(np.array([theta_q[1], 0, theta_q[0]])), np.eye(3)), axis=1), np.zeros((3,6))), axis=0)
#                 phi_A = expm(extend_A)[:self.dim,self.dim:]
#                 B = self.u[q] * phi_A
#                 A_tilde[q] = np.concatenate((np.concatenate((exp_Aq,np.zeros((self.dim,self.dim))),axis=1), np.concatenate((np.zeros((self.dim,self.dim)), np.eye(self.dim)),axis=1)),axis=0)
#                 B_tilde[q] = np.concatenate((B, np.zeros((self.dim,self.dim))), axis=0)

#         M0 = self.Q[0]
#         U, Z, K, P = self.tracking(M0, M, R, A_tilde, B_tilde, self.N-1)
#         M_hat = Z[:,:self.dim,:]
#         return M_hat



#     def get_X0(self, P0):
#         K = P0[:self.dim,:self.dim]
#         a = P0[self.dim,:self.dim]
#         c = P0[:self.dim,self.dim]
#         b = P0[self.dim,self.dim]
#         return -np.linalg.inv(K+K.T) @ (a+c.T)


#     def state_space_model(self, A, z_t_minus_1, B, u_t_minus_1):
#         """
#         Calculates the state at time t given the state at time t-1 and
#         the control inputs applied at time t-1
#         """
#         state_estimate_t = (A @ z_t_minus_1) + (B @ u_t_minus_1)
#         return state_estimate_t


#     def tracking(self, X0, Q, R, A, B, N):
#         P = [None] * (N + 1)
#         Qf = Q[N]
#         P[N] = Qf
#         for i in range(N, 0, -1):
#             # Discrete-time Algebraic Riccati equation to calculate the optimal state cost matrix
#             P[i-1] = Q[i-1] + A[i-1].T @ P[i] @ A[i-1] - (A[i-1].T @ P[i] @ B[i-1]) @ np.linalg.inv(R[i-1] + B[i-1].T @ P[i] @ B[i-1]) @ (B[i-1].T @ P[i] @ A[i-1])

#         # Create a list of N elements
#         K = [None] * N
#         u = [None] * N
#         z = [None] * (N+1)
#         if X0 is None:
#             X0 = self.get_X0(P[0])
#             z[0] = np.concatenate((X0,np.squeeze(np.eye(self.dim),axis=0)),axis=0)
#             # print(X0)
#         else:
#             z[0] = np.concatenate((X0,np.eye(self.dim)),axis=0)
#         # z[0] = np.concatenate((X0,np.array([1])))
#         for i in range(0,N):
#             # Calculate the optimal feedback gain K
#             K[i] = np.linalg.inv(R[i] + B[i].T @ P[i+1] @ B[i]) @ B[i].T @ P[i+1] @ A[i]
#             u[i] = -K[i] @ z[i]
#             # print(u[i])
#             z[i+1] = self.state_space_model(A[i], z[i], B[i], u[i])

#         Z = np.array(z)
#         return np.array(u), Z, np.array(K), np.array(P)