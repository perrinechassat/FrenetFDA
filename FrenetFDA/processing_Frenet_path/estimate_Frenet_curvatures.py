import numpy as np
from FrenetFDA.utils.smoothing_utils import compute_weight_neighbors_local_smoothing, VectorBSplineSmoothing, grid_search_GCV_optimization_Bspline_hyperparameters
from FrenetFDA.utils.Frenet_Serret_utils import solve_FrenetSerret_ODE_SE, solve_FrenetSerret_ODE_SO
from FrenetFDA.utils.Lie_group.SO3_utils import SO3
from FrenetFDA.utils.Lie_group.SE3_utils import SE3
from FrenetFDA.processing_Frenet_path.smoothing import KarcherMeanSmoother, TrackingSmootherLinear
from skopt import gp_minimize
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
# from memory_profiler import profile

class ApproxFrenetODE:

    def __init__(self, grid, Q=None, Z=None):  
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
        self.__raw_estimates()


    def raw_estimates(self):
        return self.raw_grid, self.raw_data
    

    def __raw_estimates(self):
        grid_v = (self.grid[1:] + self.grid[:-1])/2
        grid_u = self.grid[1:] - self.grid[:-1]
        raw_theta = np.zeros((self.N-1, self.dim_theta))
        for i in range(self.N-1):
            obs_vi = -(1/grid_u[i])*SO3.log(self.Q[i+1].T @ self.Q[i])
            raw_theta[i,0] = obs_vi[2]
            raw_theta[i,1] = obs_vi[0]
        self.raw_data = raw_theta
        self.raw_grid = grid_v


    def Bspline_smooth_estimates(self, nb_basis, order=4, regularization_parameter=None):
        if regularization_parameter is None:
            penalization = False
        else:
            penalization = True
        Bspline_repres = VectorBSplineSmoothing(self.dim_theta, nb_basis, domain_range=(self.grid[0], self.grid[-1]), order=order, penalization=penalization)
        Bspline_repres.fit(self.raw_grid, self.raw_data, regularization_parameter=regularization_parameter)
        cf_limits = Bspline_repres.compute_confidence_limits(0.95)
        return Bspline_repres
    

    def bayesian_optimization_hyperparameters(self):
        """
            Not implemented yet.
        """
        pass


    def grid_search_optimization_hyperparameters(self, nb_basis_list, regularization_parameter_list, order=4, parallel=False):
        """
            Optimization of smoothing parameters: number of basis and regularization parameter.

        """
        nb_basis_opt, regularization_parameter_opt, tab_GCV_scores = grid_search_GCV_optimization_Bspline_hyperparameters(self.dim_theta, self.raw_grid, self.raw_data, nb_basis_list, regularization_parameter_list, order=order, parallel=parallel)
        
        return nb_basis_opt, regularization_parameter_opt, tab_GCV_scores



class LocalApproxFrenetODE:

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


    def raw_estimates(self, h):
        grid_theta, raw_theta, weight_theta = self.__raw_estimates(h, self.grid, self.Q)
        return grid_theta, raw_theta, weight_theta


    def __raw_estimates(self, h, grid, Q):
        
        N_grid = len(grid)
        neighbor_obs, weight, grid_double, delta = compute_weight_neighbors_local_smoothing(grid, grid, h, self.adaptive_ind)

        Omega, S, Kappa, Tau = [], [], [], []
        for q in range(N_grid):
            if q==0:
                s = grid[0]*np.ones(len(neighbor_obs[q]))
            elif q==N_grid-1:
                s = grid[-1]*np.ones(len(neighbor_obs[q]))
            else:
                s = grid_double[q]
            S += list(s)

            N_q = len(neighbor_obs[q])
            Obs_q = Q[neighbor_obs[q]]
            w_q = weight[q]
            u_q = np.copy(delta[q])
            omega_q = np.multiply(w_q,np.power(u_q,2))
            if q!=0 and q!=N_grid-1:
                v_q = np.where(u_q==0)[0]
                u_q[u_q==0] = 1
            R_q = np.zeros((N_q, self.dim))
            for j in range(N_q):
                if (q!=0 or j!=0) and (q!=N_grid-1 or j!=N_q-1):
                    R_q[j] = -SO3.log(np.transpose(np.ascontiguousarray(Q[q]))@np.ascontiguousarray(Obs_q[j]))/u_q[j]
            if q!=0 and q!=N_grid-1:
                R_q[v_q] = np.abs(0*R_q[v_q])
            kappa = np.squeeze(R_q[:,2])
            tau = np.squeeze(R_q[:,0])

            Omega = np.append(Omega, omega_q.tolist())
            Kappa = np.append(Kappa, kappa.tolist())
            Tau = np.append(Tau, tau.tolist())        

        Ms, Momega, Mkappa, Mtau = self.__compute_sort_unique_val(np.around(S, 8), Omega, Kappa, Tau)

        # Test pour enlever les valeurs à zeros.
        Momega = np.asarray(Momega)
        ind_nozero = np.where(Momega!=0.)
        weight_theta = np.squeeze(Momega[ind_nozero])
        grid_theta = Ms[ind_nozero]
        raw_theta = np.stack((np.squeeze(np.asarray(Mkappa)[ind_nozero]),np.squeeze(np.asarray(Mtau)[ind_nozero])), axis=1)

        return grid_theta, raw_theta, weight_theta

    
    def __compute_sort_unique_val(self, S, Omega, Kappa, Tau):
        """
        Step of function Compute Raw Curvature, compute the re-ordering of the data.
        ...
        """
        uniqueS = np.unique(S)
        nb_unique_val = len(uniqueS)
        mOmega = np.zeros(nb_unique_val)
        mKappa = np.zeros(nb_unique_val)
        mTau   = np.zeros(nb_unique_val)
        for ijq in range(nb_unique_val):
            id_ijq      = np.where(S==uniqueS[ijq])[0]
            Omega_ijq   = Omega[id_ijq]
            Kappa_ijq   = Kappa[id_ijq]
            Tau_ijq     = Tau[id_ijq]
            mOmega[ijq] = np.sum(Omega_ijq)
            if mOmega[ijq]>0:
                mKappa[ijq] = (np.ascontiguousarray(Omega_ijq[np.where(Omega_ijq>0)]) @ np.ascontiguousarray(np.transpose(Kappa_ijq[np.where(Omega_ijq>0)])))/mOmega[ijq]
                mTau[ijq]   = (np.ascontiguousarray(Omega_ijq[np.where(Omega_ijq>0)]) @ np.ascontiguousarray(np.transpose(Tau_ijq[np.where(Omega_ijq>0)])))/mOmega[ijq]
            else:
                mKappa[ijq] = 0
                mTau[ijq]   = 0
        return uniqueS, mOmega, mKappa, mTau
    

    def Bspline_smooth_estimates(self, h, nb_basis, order=4, regularization_parameter=None):

        grid_theta, raw_theta, weight_theta = self.raw_estimates(h)
        if regularization_parameter is None:
            penalization = False
        else:
            penalization = True
        Bspline_repres = VectorBSplineSmoothing(self.dim_theta, nb_basis, domain_range=(self.grid[0], self.grid[-1]), order=order, penalization=penalization)
        Bspline_repres.fit(grid_theta, raw_theta, weights=weight_theta, regularization_parameter=regularization_parameter)
        cf_limits = Bspline_repres.compute_confidence_limits(0.95)
        return Bspline_repres



    def __step_cross_val(self, train_index, test_index, h, lbda, Bspline_repres):
        train_index = train_index+1
        test_index = test_index+1
        train_index = np.concatenate((np.array([0]), train_index, np.array([len(self.grid[1:-1])+1])))
        grid_train = self.grid[train_index]
        Q_train = self.Q[train_index]
        Q_test = self.Q[test_index]
        grid_theta_train, raw_theta_train, weight_theta_train = self.__raw_estimates(h, grid_train, Q_train)
        Bspline_repres.fit(grid_theta_train, raw_theta_train, weights=weight_theta_train, regularization_parameter=lbda)
        if self.Z is None:
            Q_test_pred = solve_FrenetSerret_ODE_SO(Bspline_repres.evaluate, self.grid, self.Q[0])
            dist = np.mean(SO3.geodesic_distance(Q_test, Q_test_pred[test_index]))
        else:
            Z_test_pred = solve_FrenetSerret_ODE_SE(Bspline_repres.evaluate, self.grid, self.Z[0])
            dist = np.mean(SE3.geodesic_distance(self.Z[test_index], Z_test_pred[test_index]))
        return dist


    def grid_search_optimization_hyperparameters(self, bandwidth_list, nb_basis_list, regularization_parameter_list, order=4, parallel=False, method='1', n_splits=10):
        
        N_param_basis = len(nb_basis_list)
        N_param_smoothing = len(regularization_parameter_list)
        N_param_bandwidth = len(bandwidth_list)

        if N_param_basis==0 or N_param_bandwidth==0:
            raise Exception("nb_basis_list and bandwidth_list cannot be empty.")
    
        if N_param_smoothing==0:
            penalization = False
            N_param_smoothing = 1
            regularization_parameter_list = np.array([0])
        else:
            penalization = True

        if regularization_parameter_list.ndim == 1:
            regularization_parameter_list = np.stack([regularization_parameter_list for i in range(self.dim_theta)], axis=-1)
        if nb_basis_list.ndim == 1:
            nb_basis_list = np.stack([nb_basis_list for i in range(self.dim_theta)], axis=-1)


        if method=='1':
            
            # print('Begin grid search optimisation with', N_param_basis*N_param_smoothing*N_param_bandwidth, 'combinations of parameters...')

            error_bandwidth = np.zeros(N_param_bandwidth)
            tab_GCV_scores = np.zeros((N_param_basis, N_param_bandwidth, N_param_smoothing, self.dim_theta))
            for i in range(N_param_basis):
                nb_basis = nb_basis_list[i]
                Bspline_repres = VectorBSplineSmoothing(self.dim_theta, nb_basis, domain_range=(self.grid[0], self.grid[-1]), order=order, penalization=penalization)
                for j in range(N_param_bandwidth):
                    grid_theta, raw_theta, weight_theta = self.raw_estimates(bandwidth_list[j])
                    V = np.expand_dims(grid_theta, 1)
                    basis_matrix = Bspline_repres.basis(V,).reshape((Bspline_repres.basis.n_basis, -1)).T # shape (self.dim*N, np.sum(self.nb_basis))
                    data, weights_matrix = Bspline_repres.check_data(grid_theta, raw_theta, weight_theta)
                    for k in range(N_param_smoothing):
                        tab_GCV_scores[i,j,k] = Bspline_repres.GCV_score(basis_matrix, data, weights_matrix, regularization_parameter_list[k])
            for j in range(N_param_bandwidth):
                grid_theta, raw_theta, weight_theta = self.raw_estimates(bandwidth_list[j])
                nb_basis_opt = np.zeros((self.dim_theta))
                regularization_parameter_opt = np.zeros((self.dim_theta))
                for i in range(self.dim_theta):
                    ind = np.unravel_index(np.argmin(tab_GCV_scores[:,j,:,i], axis=None), tab_GCV_scores[:,j,:,i].shape)
                    nb_basis_opt[i] = nb_basis_list[ind[0],i]
                    regularization_parameter_opt[i] = regularization_parameter_list[ind[1],i]
                Bspline_repres = VectorBSplineSmoothing(self.dim_theta, nb_basis_opt, domain_range=(self.grid[0], self.grid[-1]), order=order, penalization=penalization)
                Bspline_repres.fit(grid_theta, raw_theta, weights=weight_theta, regularization_parameter=regularization_parameter_opt)
                if self.Z is None:
                    Q_pred = solve_FrenetSerret_ODE_SO(Bspline_repres.evaluate, self.grid, self.Q[0])
                    error_bandwidth[j] = np.mean(SO3.geodesic_distance(self.Q, Q_pred))
                else:
                    Z_pred = solve_FrenetSerret_ODE_SE(Bspline_repres.evaluate, self.grid, self.Z[0])
                    error_bandwidth[j] = np.mean(SE3.geodesic_distance(self.Z, Z_pred))

            # ind = np.argmin(error_bandwidth)
            # h_opt = bandwidth_list[ind]
            # ind = np.unravel_index(np.argmin(tab_GCV_scores[:,ind,:], axis=None), tab_GCV_scores[:,ind,:].shape)
            # nb_basis_opt = nb_basis_list[ind[0]]
            # regularization_parameter_opt = regularization_parameter_list[ind[1]]

            ind_h = np.argmin(error_bandwidth)
            h_opt = bandwidth_list[ind_h]
            nb_basis_opt = np.zeros((self.dim_theta))
            regularization_parameter_opt = np.zeros((self.dim_theta))
            for i in range(self.dim_theta):
                ind = np.unravel_index(np.argmin(tab_GCV_scores[:,ind_h,:,i], axis=None), tab_GCV_scores[:,ind_h,:,i].shape)
                nb_basis_opt[i] = nb_basis_list[ind[0],i]
                regularization_parameter_opt[i] = regularization_parameter_list[ind[1],i]

            # print('Optimal parameters selected by grid search optimisation: ', 'bandwidth =', h_opt, 'nb_basis =', nb_basis_opt, 'regularization_parameter =', regularization_parameter_opt)
            return h_opt, nb_basis_opt, regularization_parameter_opt, tab_GCV_scores, error_bandwidth


        elif method=='2':
            
            regularization_parameter_list = np.array(np.meshgrid(*regularization_parameter_list.T)).reshape((2,-1))
            regularization_parameter_list = np.moveaxis(regularization_parameter_list, 0,1)
            nb_basis_list = np.array(np.meshgrid(*nb_basis_list.T)).reshape((2,-1))
            nb_basis_list = np.moveaxis(nb_basis_list, 0,1)

            N_param_basis = len(nb_basis_list)
            N_param_smoothing = len(regularization_parameter_list)
            N_param_bandwidth = len(bandwidth_list)

            # print('Begin grid search optimisation with', N_param_basis*N_param_smoothing*N_param_bandwidth, 'combinations of parameters...')

            kf = KFold(n_splits=n_splits, shuffle=True)
            grid_split = self.grid[1:-1]

            CV_error_tab = np.zeros((N_param_basis, N_param_bandwidth, N_param_smoothing))
            if parallel:
                for i in range(N_param_basis):
                    nb_basis = nb_basis_list[i]
                    Bspline_repres = VectorBSplineSmoothing(self.dim_theta, nb_basis, domain_range=(self.grid[0], self.grid[-1]), order=order, penalization=penalization)
                    for j in range(N_param_bandwidth):
                        h = bandwidth_list[j]
                        for k in range(N_param_smoothing):
                            lbda = regularization_parameter_list[k]
                            func = lambda train_ind, test_ind : self.__step_cross_val(train_ind, test_ind, h, lbda, Bspline_repres)
                            CV_err = Parallel(n_jobs=10)(delayed(func)(train_index, test_index) for train_index, test_index in kf.split(grid_split))
                            CV_error_tab[i,j,k] = np.mean(CV_err)


            else:
                for i in range(N_param_basis):
                    nb_basis = nb_basis_list[i]
                    Bspline_repres = VectorBSplineSmoothing(self.dim_theta, nb_basis, domain_range=(self.grid[0], self.grid[-1]), order=order, penalization=penalization)
                    for j in range(N_param_bandwidth):
                        h = bandwidth_list[j]
                        CV_err_lbda = np.zeros((N_param_smoothing,n_splits))
                        k_split = 0
                        for train_index, test_index in kf.split(grid_split):
                            train_index = train_index+1
                            test_index = test_index+1
                            train_index = np.concatenate((np.array([0]), train_index, np.array([len(self.grid[1:-1])+1])))
                            grid_train = self.grid[train_index]
                            Q_train = self.Q[train_index]
                            Q_test = self.Q[test_index]
                            grid_theta_train, raw_theta_train, weight_theta_train = self.__raw_estimates(h, grid_train, Q_train)
                            for k in range(N_param_smoothing):
                                lbda = regularization_parameter_list[k]
                                Bspline_repres.fit(grid_theta_train, raw_theta_train, weights=weight_theta_train, regularization_parameter=lbda)
                                if self.Z is None:
                                    Q_test_pred = solve_FrenetSerret_ODE_SO(Bspline_repres.evaluate, self.grid, self.Q[0])
                                    dist = np.mean(SO3.geodesic_distance(Q_test, Q_test_pred[test_index]))
                                else:
                                    Z_test_pred = solve_FrenetSerret_ODE_SE(Bspline_repres.evaluate, self.grid, self.Z[0])
                                    dist = np.mean(SE3.geodesic_distance(self.Z[test_index], Z_test_pred[test_index]))
                                CV_err_lbda[k,k_split] = dist
                            k_split += 1 
                        CV_error_tab[i,j,:] = np.mean(CV_err_lbda, axis=1) 

            ind = np.unravel_index(np.argmin(CV_error_tab, axis=None), CV_error_tab.shape)
            nb_basis_opt = nb_basis_list[ind[0]]
            h_opt = bandwidth_list[ind[1]]
            regularization_parameter_opt = regularization_parameter_list[ind[2]]
            
            # print('Optimal parameters selected by grid search optimisation: ', 'bandwidth =', h_opt, 'nb_basis =', nb_basis_opt, 'regularization_parameter =', regularization_parameter_opt)
            return h_opt, nb_basis_opt, regularization_parameter_opt, CV_error_tab

    
    def bayesian_optimization_hyperparameters(self, n_call_bayopt, lambda_bounds, h_bounds, nb_basis, order=4, n_splits=10, verbose=True, return_basis=False):

        # ## CV optimization of lambda
        Bspline_repres = VectorBSplineSmoothing(self.dim_theta, nb_basis, domain_range=(self.grid[0], self.grid[-1]), order=order, penalization=True)

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
                Q_test = self.Q[test_index]
                grid_theta_train, raw_theta_train, weight_theta_train = self.__raw_estimates(x[0], grid_train, Q_train)
                lbda = np.array([x[1],x[2]])
                Bspline_repres.fit(grid_theta_train, raw_theta_train, weights=weight_theta_train, regularization_parameter=lbda)
                if self.Z is None:
                    Q_test_pred = solve_FrenetSerret_ODE_SO(Bspline_repres.evaluate, self.grid, self.Q[0])
                    dist = np.mean(SO3.geodesic_distance(Q_test, Q_test_pred[test_index]))
                else:
                    # try:
                    #     Z_test_pred = solve_FrenetSerret_ODE_SE(Bspline_repres.evaluate, self.grid, self.Z[0])
                    # except:
                    #     Z_test_pred = solve_FrenetSerret_ODE_SE(Bspline_repres.evaluate, self.grid, self.Z[0], method='Radau')
                    Z_test_pred = solve_FrenetSerret_ODE_SE(Bspline_repres.evaluate, self.grid, self.Z[0]) #, method='Radau')
                    dist = np.mean(SE3.geodesic_distance(self.Z[test_index], Z_test_pred[test_index]))

                score[ind_CV] = dist
                ind_CV += 1 

            return np.mean(score)

        # Do a bayesian optimisation and return the optimal parameter (lambda_kappa, lambda_tau)
        
        bounds = np.array([[h_bounds[0], h_bounds[1]], [lambda_bounds[0,0], lambda_bounds[0,1]], [lambda_bounds[1,0], lambda_bounds[1,1]]])
        res_bayopt = gp_minimize(func,               # the function to minimize
                        bounds,        # the bounds on each dimension of x
                        acq_func="EI",        # the acquisition function
                        n_calls=n_call_bayopt,       # the number of evaluations of f
                        n_random_starts=2,    # the number of random initialization points
                        random_state=1,       # the random seed
                        n_jobs=1,            # use all the cores for parallel calculation
                        verbose=verbose)
        param_opt = res_bayopt.x
        h_opt = param_opt[0]
        lbda_opt = np.array([param_opt[1], param_opt[2]])

        if return_basis:
            grid_theta, raw_theta, weight_theta = self.raw_estimates(h_opt)
            Bspline_repres.fit(grid_theta, raw_theta, weights=weight_theta, regularization_parameter=lbda_opt)
            return h_opt, lbda_opt, Bspline_repres

        else:
            return h_opt, lbda_opt
    


"""___________________________________________________________________  Karcher Mean  ___________________________________________________________________"""    


class TwoStepEstimatorKarcherMean:

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


    def fit(self, h, lbda, nb_basis, epsilon=1e-03, max_iter=30):

        # LS_theta = LocalApproxFrenetODE(self.grid, Q=self.Q)
        # basis_theta = LS_theta.Bspline_smooth_estimates(h, nb_basis, order=4, regularization_parameter=lbda)

        basis_theta = VectorBSplineSmoothing(self.dim_theta, nb_basis, domain_range=(self.grid[0], self.grid[-1]), order=4, penalization=True)
        Q_smooth = self.Q
        grid_theta, raw_theta, weight_theta = LocalApproxFrenetODE(self.grid, Q=self.Q).raw_estimates(h)
        basis_theta.fit(grid_theta, raw_theta, weights=weight_theta, regularization_parameter=lbda)
        theta  = basis_theta.evaluate(self.grid)
        Dtheta = theta
        k=0
        while np.linalg.norm(Dtheta)>=epsilon*np.linalg.norm(theta) and k<max_iter:
            # print('iteration n°', k)
            theta_old = theta
            Smoother = KarcherMeanSmoother(self.grid, Q=Q_smooth)
            Q_smooth = Smoother.fit(h, basis_theta.evaluate)
            # LS_theta = LocalApproxFrenetODE(self.grid, Q=Q_smooth)
            # basis_theta = LS_theta.Bspline_smooth_estimates(h, nb_basis, order=4, regularization_parameter=lbda)
            grid_theta, raw_theta, weight_theta = LocalApproxFrenetODE(self.grid, Q=Q_smooth).raw_estimates(h)
            basis_theta.fit(grid_theta, raw_theta, weights=weight_theta, regularization_parameter=lbda)
            theta  = basis_theta.evaluate(self.grid)
            Dtheta = theta - theta_old
            k += 1  
            # print('relative error:', np.linalg.norm(Dtheta)/np.linalg.norm(theta))

        return basis_theta, Q_smooth, k 
    

    def __fit(self, grid, Q, h, lbda, nb_basis, epsilon=1e-03, max_iter=30):

        # LS_theta = LocalApproxFrenetODE(grid, Q=Q)
        basis_theta = VectorBSplineSmoothing(self.dim_theta, nb_basis, domain_range=(self.grid[0], self.grid[-1]), order=4, penalization=True)
        Q_smooth = Q
        # basis_theta = LS_theta.Bspline_smooth_estimates(h, nb_basis, order=4, regularization_parameter=lbda)
        grid_theta, raw_theta, weight_theta = LocalApproxFrenetODE(grid, Q=Q).raw_estimates(h)
        basis_theta.fit(grid_theta, raw_theta, weights=weight_theta, regularization_parameter=lbda)
        theta  = basis_theta.evaluate(grid)
        Dtheta = theta
        k=0
        while np.linalg.norm(Dtheta)>=epsilon*np.linalg.norm(theta) and k<max_iter:
            # print('iteration smoother n°', k)
            theta_old = theta
            Smoother = KarcherMeanSmoother(grid, Q=Q_smooth)
            Q_smooth = Smoother.fit(h, basis_theta.evaluate)
            grid_theta, raw_theta, weight_theta = LocalApproxFrenetODE(grid, Q=Q_smooth).raw_estimates(h)
            basis_theta.fit(grid_theta, raw_theta, weights=weight_theta, regularization_parameter=lbda)
            # basis_theta = LS_theta.Bspline_smooth_estimates(h, nb_basis, order=4, regularization_parameter=lbda)
            theta  = basis_theta.evaluate(grid)
            Dtheta = theta - theta_old
            del Smoother
            k += 1  

        return basis_theta.evaluate


    def func_CV(self, x, nb_basis, order=4, epsilon=1e-03, max_iter=30, n_splits=5):
        score = np.zeros(n_splits)
        kf = KFold(n_splits=n_splits, shuffle=True)
        grid_split = self.grid[1:-1]
        ind_CV = 0

        for train_index, test_index in kf.split(grid_split):
            # print('-------------- ind_CV:', ind_CV+1)
            train_index = train_index+1
            test_index = test_index+1
            train_index = np.concatenate((np.array([0]), train_index, np.array([len(self.grid[1:-1])+1])))
            grid_train = self.grid[train_index]
            Q_train = self.Q[train_index]
            Q_test = self.Q[test_index]
            lbda = np.array([x[1],x[2]])
            # print(lbda)
            func_basis_theta = self.__fit(grid_train, Q_train, x[0], lbda, nb_basis, epsilon=epsilon, max_iter=max_iter)
            # print('end fit')
            Q_test_pred = solve_FrenetSerret_ODE_SO(func_basis_theta, self.grid, self.Q[0])
            dist = np.mean(SO3.geodesic_distance(Q_test, Q_test_pred[test_index]))
            # print('end distance')
            score[ind_CV] = dist
            ind_CV += 1 

        return np.mean(score)


    def bayesian_optimization_hyperparameters(self, n_call_bayopt, lambda_bounds, h_bounds, nb_basis, order=4, epsilon=1e-03, max_iter=30, n_splits=5, verbose=True):

        def func(x):
            # print('hyperparam:', x)
            score = np.zeros(n_splits)
            kf = KFold(n_splits=n_splits, shuffle=True)
            grid_split = self.grid[1:-1]
            ind_CV = 0

            for train_index, test_index in kf.split(grid_split):
                # print('-------------- ind_CV:', ind_CV+1)
                train_index = train_index+1
                test_index = test_index+1
                train_index = np.concatenate((np.array([0]), train_index, np.array([len(self.grid[1:-1])+1])))
                grid_train = self.grid[train_index]
                Q_train = self.Q[train_index]
                Q_test = self.Q[test_index]
                lbda = np.array([x[1],x[2]])
                # print(lbda)
                func_basis_theta = self.__fit(grid_train, Q_train, x[0], lbda, nb_basis, epsilon=epsilon, max_iter=max_iter)
                # print('end fit')
                Q_test_pred = solve_FrenetSerret_ODE_SO(func_basis_theta, self.grid, self.Q[0])
                dist = np.mean(SO3.geodesic_distance(Q_test, Q_test_pred[test_index]))
                # print('end distance')
                score[ind_CV] = dist
                ind_CV += 1 

            return np.mean(score)

        # Do a bayesian optimisation and return the optimal parameter (lambda_kappa, lambda_tau)
        
        bounds = np.array([[h_bounds[0], h_bounds[1]], [lambda_bounds[0,0], lambda_bounds[0,1]], [lambda_bounds[1,0], lambda_bounds[1,1]]])
        res_bayopt = gp_minimize(func,               # the function to minimize
                        bounds,        # the bounds on each dimension of x
                        acq_func="EI",        # the acquisition function
                        n_calls=n_call_bayopt,       # the number of evaluations of f
                        n_initial_points=2,    # the number of random initialization points
                        random_state=2,       # the random seed
                        n_jobs=1,            # use all the cores for parallel calculation
                        verbose=verbose)
        param_opt = res_bayopt.x
        h_opt = param_opt[0]
        lbda_opt = np.array([param_opt[1], param_opt[2]])

        return h_opt, lbda_opt
    


"""___________________________________________________________________  Tracking  ___________________________________________________________________"""    


class TwoStepEstimatorTracking:

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


    # @profile
    def fit(self, lbda_track, h, lbda, nb_basis, epsilon=1e-03, max_iter=30):

        # LS_theta = LocalApproxFrenetODE(self.grid, Q=self.Q)
        # basis_theta = LS_theta.Bspline_smooth_estimates(h, nb_basis, order=4, regularization_parameter=lbda)
        basis_theta = VectorBSplineSmoothing(self.dim_theta, nb_basis, domain_range=(self.grid[0], self.grid[-1]), order=4, penalization=True)
        grid_theta, raw_theta, weight_theta = LocalApproxFrenetODE(self.grid, Q=self.Q).raw_estimates(h)
        basis_theta.fit(grid_theta, raw_theta, weights=weight_theta, regularization_parameter=lbda)
        Q_smooth = self.Q
        theta  = basis_theta.evaluate(self.grid)
        Dtheta = theta
        k=0
        while np.linalg.norm(Dtheta)>=epsilon*np.linalg.norm(theta) and k<max_iter:
            # print('iteration n°', k)
            theta_old = theta
            Smoother = TrackingSmootherLinear(self.grid, Q=Q_smooth)
            Q_smooth = Smoother.fit(lbda_track, basis_theta.evaluate)
            grid_theta, raw_theta, weight_theta = LocalApproxFrenetODE(self.grid, Q=Q_smooth).raw_estimates(h)
            basis_theta.fit(grid_theta, raw_theta, weights=weight_theta, regularization_parameter=lbda)
            # LS_theta = LocalApproxFrenetODE(self.grid, Q=Q_smooth)
            # basis_theta = LS_theta.Bspline_smooth_estimates(h, nb_basis, order=4, regularization_parameter=lbda)
            theta  = basis_theta.evaluate(self.grid)
            Dtheta = theta - theta_old
            k += 1  
            # print('relative error:', np.linalg.norm(Dtheta)/np.linalg.norm(theta))

        return basis_theta, Q_smooth, k 
    
    # @profile
    def __fit(self, grid, Q, lbda_track, h, lbda, nb_basis, epsilon=1e-03, max_iter=30):
        
        # LS_theta = LocalApproxFrenetODE(grid, Q=Q)
        # basis_theta = LS_theta.Bspline_smooth_estimates(h, nb_basis, order=4, regularization_parameter=lbda)
        basis_theta = VectorBSplineSmoothing(self.dim_theta, nb_basis, domain_range=(self.grid[0], self.grid[-1]), order=4, penalization=True)
        grid_theta, raw_theta, weight_theta = LocalApproxFrenetODE(grid, Q=Q).raw_estimates(h)
        basis_theta.fit(grid_theta, raw_theta, weights=weight_theta, regularization_parameter=lbda)
        Q_smooth = Q
        theta  = basis_theta.evaluate(grid)
        Dtheta = theta
        k=0
        while np.linalg.norm(Dtheta)>=epsilon*np.linalg.norm(theta) and k<max_iter:
            # print('iteration smoother n°', k)
            theta_old = theta
            Smoother = TrackingSmootherLinear(grid, Q=Q_smooth)
            Q_smooth = Smoother.fit(lbda_track, basis_theta.evaluate)
            # LS_theta = LocalApproxFrenetODE(grid, Q=Q_smooth)
            # basis_theta = LS_theta.Bspline_smooth_estimates(h, nb_basis, order=4, regularization_parameter=lbda)
            grid_theta, raw_theta, weight_theta = LocalApproxFrenetODE(grid, Q=Q_smooth).raw_estimates(h)
            basis_theta.fit(grid_theta, raw_theta, weights=weight_theta, regularization_parameter=lbda)
            theta  = basis_theta.evaluate(grid)
            Dtheta = theta - theta_old
            del Smoother
            k += 1  

        return basis_theta.evaluate 
    
    # @profile
    def func_CV(self, x, nb_basis, order=4, epsilon=1e-03, max_iter=30, n_splits=5):
        score = np.zeros(n_splits)
        kf = KFold(n_splits=n_splits, shuffle=True)
        grid_split = self.grid[1:-1]
        ind_CV = 0

        for train_index, test_index in kf.split(grid_split):
            # print('-------------- ind_CV:', ind_CV+1)
            train_index = train_index+1
            test_index = test_index+1
            train_index = np.concatenate((np.array([0]), train_index, np.array([len(self.grid[1:-1])+1])))
            grid_train = self.grid[train_index]
            Q_train = self.Q[train_index]
            Q_test = self.Q[test_index]
            lbda = np.array([x[1],x[2]])
            # print(lbda)
            func_basis_theta = self.__fit(grid_train, Q_train, x[3], x[0], lbda, nb_basis, epsilon=epsilon, max_iter=max_iter)
            # print('end fit')
            Q_test_pred = solve_FrenetSerret_ODE_SO(func_basis_theta, self.grid, self.Q[0])
            dist = np.mean(SO3.geodesic_distance(Q_test, Q_test_pred[test_index]))
            # print('end distance')
            score[ind_CV] = dist
            ind_CV += 1 

        return np.mean(score)


    def bayesian_optimization_hyperparameters(self, n_call_bayopt, lambda_track_bounds, lambda_bounds, h_bounds, nb_basis, order=4, epsilon=1e-03, max_iter=30, n_splits=10, verbose=True):

        def func(x):
            # print('hyperparam:', x)
            score = np.zeros(n_splits)
            kf = KFold(n_splits=n_splits, shuffle=True)
            grid_split = self.grid[1:-1]
            ind_CV = 0

            for train_index, test_index in kf.split(grid_split):
                # print('-------------- ind_CV:', ind_CV+1)
                train_index = train_index+1
                test_index = test_index+1
                train_index = np.concatenate((np.array([0]), train_index, np.array([len(self.grid[1:-1])+1])))
                grid_train = self.grid[train_index]
                Q_train = self.Q[train_index]
                Q_test = self.Q[test_index]
                lbda = np.array([x[1],x[2]])
                # print(lbda)
                func_basis_theta = self.__fit(grid_train, Q_train, x[3], x[0], lbda, nb_basis, epsilon=epsilon, max_iter=max_iter)
                # print('end fit')
                Q_test_pred = solve_FrenetSerret_ODE_SO(func_basis_theta, self.grid, self.Q[0])
                dist = np.mean(SO3.geodesic_distance(Q_test, Q_test_pred[test_index]))
                # print('end distance', dist)
                score[ind_CV] = dist
                ind_CV += 1 
            
            # print('FIN CV', np.mean(score))
            return np.mean(score)

        # Do a bayesian optimisation and return the optimal parameter (lambda_kappa, lambda_tau)
        
        bounds = np.array([[h_bounds[0], h_bounds[1]], [lambda_bounds[0,0], lambda_bounds[0,1]], [lambda_bounds[1,0], lambda_bounds[1,1]], [lambda_track_bounds[0], lambda_track_bounds[1]]])
        res_bayopt = gp_minimize(func,               # the function to minimize
                        bounds,        # the bounds on each dimension of x
                        acq_func="EI",        # the acquisition function
                        n_calls=n_call_bayopt,       # the number of evaluations of f
                        n_initial_points=2,    # the number of random initialization points
                        random_state=2,       # the random seed
                        n_jobs=1,            # use all the cores for parallel calculation
                        verbose=verbose)
        param_opt = res_bayopt.x
        h_opt = param_opt[0]
        lbda_opt = np.array([param_opt[1], param_opt[2]])
        lbda_track_opt = param_opt[3]

        return h_opt, lbda_opt, lbda_track_opt


    

    