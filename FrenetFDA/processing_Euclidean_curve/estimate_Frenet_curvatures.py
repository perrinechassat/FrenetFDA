import numpy as np
from FrenetFDA.utils.smoothing_utils import VectorBSplineSmoothing, grid_search_GCV_optimization_Bspline_hyperparameters, LocalPolynomialSmoothing
from FrenetFDA.utils.Frenet_Serret_utils import solve_FrenetSerret_ODE_SE, solve_FrenetSerret_ODE_SO, find_best_rotation, centering, Euclidean_dist_cent_rot
from FrenetFDA.utils.Lie_group.SE3_utils import SE3
from FrenetFDA.utils.Lie_group.SO3_utils import SO3
from scipy import interpolate, optimize
from scipy.integrate import cumtrapz
from sklearn.model_selection import KFold
from joblib import Parallel, delayed



class ExtrinsicFormulas:
 

    def __init__(self, Y, time, arc_length, deg_polynomial=4):
        self.N, self.dim = Y.shape
        if self.dim < 2:
            raise Exception("The Frenet Serret framework is defined only for curves in R^d with d >= 2.")
        if self.N != len(time):
            raise Exception("Number of sample points in attribute Y and time must be equal.")
        self.time = (time - time.min()) / (time.max() - time.min())
        self.Y = Y
        self.deg = deg_polynomial
        self.grid_arc_s = arc_length 
        self.dim_theta = len(np.diag(np.eye(self.dim), k=1))


    def raw_estimates(self, h):

        theta = self.__raw_estimates(self.time, self.Y, h)
        self.raw_data = theta

        return self.raw_data


    def Bspline_smooth_estimates(self,  h, nb_basis, order=4, regularization_parameter=None):

        theta = self.raw_estimates(h)
        if regularization_parameter is None:
            penalization = False
        else:
            penalization = True
        Bspline_repres = VectorBSplineSmoothing(self.dim_theta, nb_basis, domain_range=(self.grid_arc_s[0], self.grid_arc_s[-1]), order=order, penalization=penalization)
        Bspline_repres.fit(self.grid_arc_s, theta, weights=None, regularization_parameter=regularization_parameter)
        cf_limits = Bspline_repres.compute_confidence_limits(0.95)
        return Bspline_repres


    def __raw_estimates(self, grid, data, h):
        
        time_derivatives = LocalPolynomialSmoothing(deg_polynomial=self.deg).fit(data, grid, grid, h)
        N = len(grid)

        if self.dim==3:
            theta = np.zeros((N, self.dim_theta))
            crossvect = np.zeros((N,self.dim))
            norm_crossvect = np.zeros(N)
            for i in range(N):
                crossvect[i,:] = np.cross(time_derivatives[1,i,:],time_derivatives[2,i,:])
                norm_crossvect[i] = np.linalg.norm(crossvect[i,:])
                theta[i,0] = norm_crossvect[i]/np.power(np.linalg.norm(time_derivatives[1,i,:]),3)
                theta[i,1] = (np.dot(crossvect[i,:],np.transpose(time_derivatives[3,i,:])))/(norm_crossvect[i]**2)

        elif self.dim==2:
            theta = np.zeros((N, self.dim_theta))
            crossvect = np.zeros((N))
            norm_crossvect = np.zeros(N)
            for i in range(N):
                crossvect[i] = np.squeeze(np.cross(time_derivatives[1,i,:],time_derivatives[2,i,:]))
                norm_crossvect[i] = crossvect[i]
                theta[i,0] = norm_crossvect[i]/np.power(np.linalg.norm(time_derivatives[1,i,:]),3)

        else:
            theta = np.zeros((N,1))
            crossvect = np.zeros((N,self.dim))
            norm_crossvect = np.zeros(N)
            for i in range(N):
                crossvect[i,:] = np.inner(time_derivatives[1,i,:],time_derivatives[2,i,:])
                norm_crossvect[i] = np.linalg.norm(crossvect[i,:])
                theta[i,0] = norm_crossvect[i]/np.power(np.linalg.norm(time_derivatives[1,i,:]),3)


            print("As we are in dimension >= 3, we compute here only the first frenet curvatures. ")
        
        return theta
        


    def __step_cross_val(self, train_index, test_index, h, lbda, Bspline_repres):
        train_index = train_index+1
        test_index = test_index+1
        train_index = np.concatenate((np.array([0]), train_index, np.array([len(self.time[1:-1])+1])))
        Y_train = self.Y[train_index]
        Y_test = self.Y[test_index]
        raw_theta_train = self.__raw_estimates(self.time[train_index], Y_train, h)
        Bspline_repres.fit(self.grid_arc_s[train_index], raw_theta_train, regularization_parameter=lbda)
        Z_test_pred = solve_FrenetSerret_ODE_SE(Bspline_repres.evaluate, self.grid_arc_s[test_index])
        X_test_pred = Z_test_pred[:,:self.dim,self.dim]
        dist = Euclidean_dist_cent_rot(Y_test, X_test_pred)
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

            print('Begin grid search optimisation with', N_param_basis*N_param_smoothing*N_param_bandwidth, 'combinations of parameters...')

            error_bandwidth = np.zeros(N_param_bandwidth)
            tab_GCV_scores = np.zeros((N_param_basis, N_param_bandwidth, N_param_smoothing, self.dim_theta))
            for i in range(N_param_basis):
                nb_basis = nb_basis_list[i]
                Bspline_repres = VectorBSplineSmoothing(self.dim_theta, nb_basis, domain_range=(self.grid_arc_s[0], self.grid_arc_s[-1]), order=order, penalization=penalization)
                V = np.expand_dims(self.grid_arc_s, 1)
                basis_matrix = Bspline_repres.basis(V,).reshape((Bspline_repres.basis.n_basis, -1)).T # shape (self.dim*N, np.sum(self.nb_basis))
                for j in range(N_param_bandwidth):
                    raw_theta = self.raw_estimates(bandwidth_list[j])
                    data, weights_matrix = Bspline_repres.check_data(self.grid_arc_s, raw_theta)
                    for k in range(N_param_smoothing):
                        tab_GCV_scores[i,j,k] = Bspline_repres.GCV_score(basis_matrix, data, weights_matrix, regularization_parameter_list[k])
            for j in range(N_param_bandwidth):
                raw_theta = self.raw_estimates(bandwidth_list[j])
                nb_basis_opt = np.zeros((self.dim_theta))
                regularization_parameter_opt = np.zeros((self.dim_theta))
                for i in range(self.dim_theta):
                    ind = np.unravel_index(np.argmin(tab_GCV_scores[:,j,:,i], axis=None), tab_GCV_scores[:,j,:,i].shape)
                    nb_basis_opt[i] = nb_basis_list[ind[0],i]
                    regularization_parameter_opt[i] = regularization_parameter_list[ind[1],i]
                Bspline_repres = VectorBSplineSmoothing(self.dim_theta, nb_basis_opt, domain_range=(self.grid_arc_s[0], self.grid_arc_s[-1]), order=order, penalization=penalization)
                Bspline_repres.fit(self.grid_arc_s, raw_theta, regularization_parameter=regularization_parameter_opt)
                Z_pred = solve_FrenetSerret_ODE_SE(Bspline_repres.evaluate, self.grid_arc_s)
                X_pred = Z_pred[:,:self.dim,self.dim]
                error_bandwidth[j] = Euclidean_dist_cent_rot(self.Y, X_pred)

            ind_h = np.argmin(error_bandwidth)
            h_opt = bandwidth_list[ind_h]
            nb_basis_opt = np.zeros((self.dim_theta))
            regularization_parameter_opt = np.zeros((self.dim_theta))
            for i in range(self.dim_theta):
                ind = np.unravel_index(np.argmin(tab_GCV_scores[:,ind_h,:,i], axis=None), tab_GCV_scores[:,ind_h,:,i].shape)
                nb_basis_opt[i] = nb_basis_list[ind[0],i]
                regularization_parameter_opt[i] = regularization_parameter_list[ind[1],i]

            # ind = np.unravel_index(np.argmin(tab_GCV_scores[:,ind,:], axis=None), tab_GCV_scores[:,ind,:].shape)
            # nb_basis_opt = nb_basis_list[ind[0]]
            # regularization_parameter_opt = regularization_parameter_list[ind[1]]

            print('Optimal parameters selected by grid search optimisation: ', 'bandwidth =', h_opt, 'nb_basis =', nb_basis_opt, 'regularization_parameter =', regularization_parameter_opt)
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
            grid_split = self.time[1:-1]

            CV_error_tab = np.zeros((N_param_basis, N_param_bandwidth, N_param_smoothing))
            for i in range(N_param_basis):
                nb_basis = nb_basis_list[i]
                Bspline_repres = VectorBSplineSmoothing(self.dim-1, nb_basis, domain_range=(self.grid_arc_s[0], self.grid_arc_s[-1]), order=order, penalization=penalization)
                for j in range(N_param_bandwidth):
                    h = bandwidth_list[j]
                    for k in range(N_param_smoothing):
                        lbda = regularization_parameter_list[k]
                        # print('nb_basis:', nb_basis, 'h:', h, 'lbda:', lbda)
                        if parallel:
                            func = lambda train_ind, test_ind : self.__step_cross_val(train_ind, test_ind, h, lbda, Bspline_repres)
                            CV_err = Parallel(n_jobs=10)(delayed(func)(train_index, test_index) for train_index, test_index in kf.split(grid_split))
                            CV_error_tab[i,j,k] = np.mean(CV_err)
                        else:
                            CV_err = []
                            for train_index, test_index in kf.split(grid_split):
                                CV_err.append(self.__step_cross_val(train_index, test_index, h, lbda, Bspline_repres))
                            CV_error_tab[i,j,k] = np.mean(CV_err)

            ind = np.unravel_index(np.argmin(CV_error_tab, axis=None), CV_error_tab.shape)
            nb_basis_opt = nb_basis_list[ind[0]]
            h_opt = bandwidth_list[ind[1]]
            regularization_parameter_opt = regularization_parameter_list[ind[2]]
            
            # print('Optimal parameters selected by grid search optimisation: ', 'bandwidth =', h_opt, 'nb_basis =', nb_basis_opt, 'regularization_parameter =', regularization_parameter_opt)
            return h_opt, nb_basis_opt, regularization_parameter_opt, CV_error_tab
        

    def bayesian_optimization_hyperparameters(self):
        """
            Not implemented yet.
        """
        pass