import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from skfda.representation.basis import VectorValued, BSpline, Constant
from skfda.misc.regularization import TikhonovRegularization, compute_penalty_matrix
from skfda.misc.operators import LinearDifferentialOperator
from scipy.linalg import block_diag
from skopt import gp_minimize
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
import plotly.graph_objs as go
import plotly.express as px
import plotly.colors as pc
import plotly.io as pio
from plotly.subplots import make_subplots


def kernel(t): # Epanechnikov kernel
    return (3/4)*(1-np.power(t,2))*(np.abs(t)<1)


def adaptive_kernel(t, delta):
    return np.power((1 - np.power((np.abs(t)/delta),3)), 3)


def compute_weight_neighbors_local_smoothing(grid_in, grid_out, h, adaptive=False):

    neighbor_obs = []
    weight = []
    grid_double = []
    delta = []
    nb_grid_out = len(grid_out)

    if adaptive:
        for q in range(nb_grid_out):
            t_q = grid_out[q]
            delta_s = abs(grid_in-t_q)
            D = 1.0001*np.sort(delta_s)[h-1]
            neighbor_obs.append(np.argsort(delta_s)[:h]) # index of observations in the neighborhood of t_q
            weight.append((1/D)*adaptive_kernel((t_q - grid_in[neighbor_obs[q]]), D)) # K_h(t_q-s_j, D)
            grid_double.append((t_q + grid_in[neighbor_obs[q]])/2) # (t_q+s_j)/2
            delta.append(t_q - grid_in[neighbor_obs[q]])  # t_q-s_j
    else:
        val_min = np.min(grid_in)
        val_max = np.max(grid_in)
        for q in range(nb_grid_out):
            t_q = grid_out[q]
            if t_q-val_min < h and q!=0:
                h_bis = np.abs(t_q-val_min) + 10e-10
                neighbor_obs.append(np.where(abs(grid_in - t_q) <= h_bis)[0])
                weight.append((1/h)*kernel((t_q - grid_in[neighbor_obs[q]])/h))
                grid_double.append((t_q + grid_in[neighbor_obs[q]])/2) # (t_q+s_j)/2
                delta.append(t_q - grid_in[neighbor_obs[q]])
            elif val_max-t_q < h and q!=nb_grid_out-1:
                h_bis = np.abs(val_max-t_q) + 10e-10
                neighbor_obs.append(np.where(abs(grid_in - t_q) <= h_bis)[0])
                weight.append((1/h)*kernel((t_q - grid_in[neighbor_obs[q]])/h))
                grid_double.append((t_q + grid_in[neighbor_obs[q]])/2) # (t_q+s_j)/2
                delta.append(t_q - grid_in[neighbor_obs[q]])
            elif q==0:
                neighbor_obs.append(np.array([0,1]))
                weight.append((1/h)*kernel((t_q - grid_in[neighbor_obs[q]])/h))
                grid_double.append((t_q + grid_in[neighbor_obs[q]])/2) # (t_q+s_j)/2
                delta.append(t_q - grid_in[neighbor_obs[q]])
            elif q==nb_grid_out-1:
                neighbor_obs.append(np.array([len(grid_in)-2,len(grid_in)-1]))
                weight.append((1/h)*kernel((t_q - grid_in[neighbor_obs[q]])/h))
                grid_double.append((t_q + grid_in[neighbor_obs[q]])/2) # (t_q+s_j)/2
                delta.append(t_q - grid_in[neighbor_obs[q]])
            else:
                neighbor_obs.append(np.where(abs(grid_in - t_q) <= h)[0]) # index of observations in the neighborhood of t_q
                weight.append((1/h)*kernel((t_q - grid_in[neighbor_obs[q]])/h)) # K_h(t_q-s_j)
                grid_double.append((t_q + grid_in[neighbor_obs[q]])/2) # (t_q+s_j)/2
                delta.append(t_q - grid_in[neighbor_obs[q]])  # t_q-s_j

    neighbor_obs = np.squeeze(neighbor_obs)
    weight = np.squeeze(np.asarray(weight, dtype=object))
    grid_double = np.squeeze(np.asarray(grid_double, dtype=object))
    delta = np.squeeze(np.asarray(delta, dtype=object))
    
    return neighbor_obs, weight, grid_double, delta



class LocalPolynomialSmoothing:

    def __init__(self, deg_polynomial=4):
        self.deg = deg_polynomial


    def fit(self, data, grid_in, grid_out, bandwidth):

        N, dim = data.shape
        if N != len(grid_in):
            raise Exception("Number of sample points in attribute data and grid_in must be equal.")
        pre_process = PolynomialFeatures(degree=self.deg)
        N_out = len(grid_out)
        deriv_estim = np.zeros((N_out,(self.deg+1)*dim))
        for i in range(N_out):
            T = grid_in - grid_out[i]
            W = kernel(T/bandwidth)
            T_poly = pre_process.fit_transform(T.reshape(-1,1))
            for j in range(self.deg+1):
                T_poly[:,j] = T_poly[:,j]/np.math.factorial(j)
            pr_model = LinearRegression(fit_intercept = False)
            pr_model.fit(T_poly, data, W)
            B = pr_model.coef_
            deriv_estim[i,:] = B.reshape(1,(self.deg+1)*dim, order='F')
        derivatives = np.zeros((dim+1, N_out, dim))
        for k in range(dim+1):
            derivatives[k] = deriv_estim[:,k*dim:(k+1)*dim]

        return derivatives

    def grid_search_CV_optimization_bandwidth(self, data, grid_in, grid_out, bandwidth_grid=np.array([]), K_split=10):
        if len(bandwidth_grid)==0:
            raise Exception("You must give a grid for search of optimal h")
        else:
            N_param = len(bandwidth_grid)
            err_h = np.zeros(N_param)
            kf = KFold(n_splits=K_split, shuffle=True)
            for j in range(N_param):
                err = []
                for train_index, test_index in kf.split(grid_in):
                    t_train, t_test = grid_in[train_index], grid_in[test_index]
                    data_train, data_test = data[train_index,:], data[test_index,:]
                    derivatives = self.fit(data_train, t_train, t_test, bandwidth_grid[j])
                    diff = derivatives[0] - data_test
                    err.append(np.linalg.norm(diff)**2)
                err_h[j] = np.mean(err)
            if isinstance(np.where(err_h==np.min(err_h)), int):
                h_opt = bandwidth_grid[np.where(err_h==np.min(err_h))]
            else:
                h_opt = bandwidth_grid[np.where(err_h==np.min(err_h))][0]
            # print('Optimal smoothing parameter h find by cross-validation:', h_opt)
            return h_opt, err_h
        

    def bayesian_optimization_hyperparameters(self,  data, grid_in, grid_out, n_call_bayopt, h_bounds, n_splits=10, verbose=True):

        def func(x):
            score = np.zeros(n_splits)
            kf = KFold(n_splits=n_splits, shuffle=True)
            ind_CV = 0

            for train_index, test_index in kf.split(grid_in):
                t_train, t_test = grid_in[train_index], grid_in[test_index]
                data_train, data_test = data[train_index,:], data[test_index,:]
                derivatives = self.fit(data_train, t_train, t_test, x[0])
                diff = derivatives[0] - data_test
                score[ind_CV] = np.linalg.norm(diff)**2
                ind_CV += 1 

            return np.mean(score)

        # Do a bayesian optimisation and return the optimal parameter (lambda_kappa, lambda_tau)
        res_bayopt = gp_minimize(func,                  # the function to minimize
                                [h_bounds],                 # the bounds on each dimension of x
                                acq_func="EI",          # the acquisition function
                                n_calls=n_call_bayopt,  # the number of evaluations of f
                                n_random_starts=2,      # the number of random initialization points
                                random_state=1,         # the random seed
                                n_jobs=1,               # use all the cores for parallel calculation
                                verbose=verbose)
        h_opt = res_bayopt.x[0]
        return h_opt


# def local_polynomial_smoothing(deg, data, grid_in, grid_out, h=None, CV_optimization_h={"flag":False, "h_grid":np.array([]), "K":5}):
#     """
#         Local polynomial smoothing of "data" and computation of its n+1 first derivatives.

#     """
#     if h is None:
#         if CV_optimization_h["flag"]==False:
#             raise Exception("You must choose a parameter h or set the optimization flag to true and give a grid for search of optimal h.")
#         else:
#             if len(CV_optimization_h["h_grid"])==0:
#                 raise Exception("You must give a grid for search of optimal h")
#             else:
#                 h_grid = CV_optimization_h["h_grid"]
#                 err_h = np.zeros(len(h_grid))
#                 kf = KFold(n_splits=CV_optimization_h["K"], shuffle=True)
#                 for j in range(len(h_grid)):
#                     err = []
#                     for train_index, test_index in kf.split(grid_in):
#                         t_train, t_test = grid_in[train_index], grid_in[test_index]
#                         data_train, data_test = data[train_index,:], data[test_index,:]
#                         derivatives = __local_polynomial_smoothing(deg, data_train, t_train, t_test, h_grid[j])
#                         diff = derivatives[0] - data_test
#                         err.append(np.linalg.norm(diff)**2)
#                     err_h[j] = np.mean(err)
#                 print('error_h:', err_h)
#                 if isinstance(np.where(err_h==np.min(err_h)), int):
#                     h_opt = h_grid[np.where(err_h==np.min(err_h))]
#                 else:
#                     h_opt = h_grid[np.where(err_h==np.min(err_h))][0]
#                 print('Optimal smoothing parameter h find by cross-validation:', h_opt)
#                 derivatives = __local_polynomial_smoothing(deg, data, grid_in, grid_out, h_opt)
#     else:   
#         derivatives = __local_polynomial_smoothing(deg, data, grid_in, grid_out, h)

#     return derivatives




# def local_polynomial_smoothing(deg, data, grid_in, grid_out, h):

#     N, dim = data.shape
#     if N != len(grid_in):
#         raise Exception("Number of sample points in attribute data and grid_in must be equal.")
#     pre_process = PolynomialFeatures(degree=deg)
#     N_out = len(grid_out)
#     deriv_estim = np.zeros((N_out,(deg+1)*dim))
#     for i in range(N_out):
#         T = grid_in - grid_out[i]
#         W = kernel(T/h)
#         T_poly = pre_process.fit_transform(T.reshape(-1,1))
#         for j in range(deg+1):
#             T_poly[:,j] = T_poly[:,j]/np.math.factorial(j)
#         pr_model = LinearRegression(fit_intercept = False)
#         pr_model.fit(T_poly, data, W)
#         B = pr_model.coef_
#         deriv_estim[i,:] = B.reshape(1,(deg+1)*dim, order='F')
#     derivatives = np.zeros((dim+1, N_out, dim))
#     for k in range(dim+1):
#         derivatives[k] = deriv_estim[:,k*dim:(k+1)*dim]

#     return derivatives



def grid_search_GCV_optimization_Bspline_hyperparameters(dim, grid, data_init, nb_basis_list, regularization_parameter_list, order=4, weights=None, parallel=False):
    """
        Optimization of smoothing parameters: number of basis and regularization parameter.

    """
    N_param_basis = len(nb_basis_list)
    N_param_smoothing = len(regularization_parameter_list)

    if N_param_basis==0:
        raise Exception("nb_basis_list cannot be empty.")
    
    if N_param_smoothing==0:
        penalization = False
        N_param_smoothing = 1
        regularization_parameter_list = np.zeros((1,dim))
        # print('Begin grid search optimisation with', N_param_basis, 'combinations of parameters...')
    else:
        penalization = True
        # print('Begin grid search optimisation with', N_param_basis*N_param_smoothing, 'combinations of parameters...')

    if regularization_parameter_list.ndim == 1:
        regularization_parameter_list = np.stack([regularization_parameter_list for i in range(dim)], axis=-1)
    if nb_basis_list.ndim == 1:
        nb_basis_list = np.stack([nb_basis_list for i in range(dim)], axis=-1)
    
    V = np.expand_dims(grid, 1)
    tab_GCV_scores = np.zeros((N_param_basis, N_param_smoothing, dim))

    for i in range(N_param_basis):
        nb_basis = nb_basis_list[i]
        Bspline_repres = VectorBSplineSmoothing(dim, nb_basis, domain_range=(grid[0], grid[-1]), order=order, penalization=penalization)
        basis_matrix = Bspline_repres.basis(V,).reshape((Bspline_repres.basis.n_basis, -1)).T # shape (self.dim*N, np.sum(self.nb_basis))
        data, weights_matrix = Bspline_repres.check_data(grid, data_init, weights)
        if parallel:
            func = lambda lbda: Bspline_repres.GCV_score(basis_matrix, data, weights_matrix, lbda)
            out = Parallel(n_jobs=-1)(delayed(func)(regularization_parameter_list[j]) for j in range(N_param_smoothing))
            tab_GCV_scores[i] = np.array(out)
        else:
            for j in range(N_param_smoothing):
                tab_GCV_scores[i,j] = Bspline_repres.GCV_score(basis_matrix, data, weights_matrix, regularization_parameter_list[j])

    nb_basis_opt = np.zeros((dim))
    regularization_parameter_opt = np.zeros((dim))
    for i in range(dim):
        ind = np.unravel_index(np.argmin(tab_GCV_scores[:,:,i], axis=None), tab_GCV_scores[:,:,i].shape)
        nb_basis_opt[i] = nb_basis_list[ind[0],i]
        regularization_parameter_opt[i] = regularization_parameter_list[ind[1],i]
    
    # print('Optimal parameters selected by grid search optimisation: ', 'nb_basis =', nb_basis_opt, 'regularization_parameter =', regularization_parameter_opt)
    return nb_basis_opt, regularization_parameter_opt, tab_GCV_scores



# class VectorKernelSmoothing:

#     def __init__(self, dim, kernel, penalization=True):

#         if isinstance(dim, int) and dim > 0:
#             self.dim = dim
#         else:
#             raise Exception("Invalide value of dim.")
        
#         self.kernel = kernel

#         V = np.expand_dims(self.v, 1)
#         def basis_fct(s):
#             if isinstance(s, int) or isinstance(s, float):
#                 return np.kron(self.kernel(V, np.array([s])[:,np.newaxis]), np.eye(2))
#             elif isinstance(s, np.ndarray):
#                 return np.reshape(np.kron(self.kernel(V, s[:,np.newaxis]), np.eye(2)), (-1,len(s),2))
#             else:
#                 raise ValueError('Variable is not a float, a int or a NumPy array.')
#         self.nb_basis = self.N
#         self.basis_fct = basis_fct
#         self.basis_matrix = np.kron(self.kernel(V,V),np.eye(2))
#         if penalization:
#             self.penalty_matrix = np.kron(self.kernel(V,V),np.eye(2))
#         else:
#             self.penalty_matrix = np.zeros((self.basis_matrix.shape))


class VectorBSplineSmoothing:

    def __init__(self, dim, nb_basis=None, domain_range=(0,1), order=4, penalization=True, knots=None):

        if isinstance(dim, int) and dim > 0:
            self.dim = dim
        else:
            raise Exception("Invalide value of dim.")
        
        if (nb_basis is None) and (knots is None):
            raise Exception("Either nb_basis or knots must be set.")
        
        if knots is not None:
            nb_basis = len(knots)+2
        
        if isinstance(nb_basis, int) or isinstance(nb_basis, float) or isinstance(nb_basis, np.int64) or isinstance(nb_basis, np.int32):
            self.nb_basis = np.repeat(int(nb_basis), dim)
        elif len(nb_basis)==dim:
            self.nb_basis = nb_basis.astype(int)
        else:
            raise Exception("Invalide value of nb_basis.")

        self.domain_range = domain_range
        self.order = order
        if knots is not None:
            list_basis = [BSpline(domain_range=self.domain_range, order=order, knots=knots) for i in range(self.dim)]
        else:
            list_basis = [BSpline(domain_range=self.domain_range, n_basis=self.nb_basis[i], order=order) for i in range(self.dim)]
        self.basis = VectorValued(list_basis)
        def basis_fct(s):
            return np.squeeze(self.basis.evaluate(s))
        self.basis_fct = basis_fct
        if penalization:
            self.penalty_matrix = compute_penalty_matrix(basis_iterable=(self.basis,),regularization_parameter=1,regularization=TikhonovRegularization(LinearDifferentialOperator(2)))
        else:
            self.penalty_matrix = np.zeros((nb_basis*2, nb_basis*2))


    def fit(self, grid, data=None, weights=None, regularization_parameter=None):
        """
            grid: dimension N
            data: dimension (N, self.dim), or (N,) if dim=1
            weights: dimension (N, self.dim) or (N*self.dim, N*self.dim) or (N,) if dim=1

        """
        N = len(grid)
        V = np.expand_dims(grid, 1)
        self.basis_matrix = self.basis(V,).reshape((self.basis.n_basis, -1)).T # shape (self.dim*N, np.sum(self.nb_basis))
        if data is not None: 
            self.data, self.weights_matrix = self.check_data(grid, data, weights)
            self.grid = grid
            self.regularization_parameter, self.regularization_parameter_matrix = self.check_regularization_parameter(regularization_parameter)
            left = self.basis_matrix.T @ self.weights_matrix @ self.basis_matrix + self.regularization_parameter_matrix @ self.penalty_matrix
            right = self.basis_matrix.T @ self.weights_matrix @ np.reshape(self.data, (N*self.dim,))
            self.coefs = np.linalg.solve(left, right)


    def evaluate(self, s):
        if isinstance(s, int) or isinstance(s, float):
            return np.squeeze(self.basis_fct(s).T @ self.coefs)
        elif isinstance(s, np.ndarray):
            return np.squeeze((self.basis_fct(s).T @ self.coefs).T)
        else:
            raise ValueError('Variable is not a float, a int or a NumPy array.')
        
    def evaluate_coefs(self, coefs_test):
        def func(s):
            if isinstance(s, int) or isinstance(s, float):
                return np.squeeze(self.basis_fct(s).T @ coefs_test)
            elif isinstance(s, np.ndarray):
                return np.squeeze((self.basis_fct(s).T @ coefs_test).T)
            else:
                raise ValueError('Variable is not a float, a int or a NumPy array.')
        return func

    def GCV_score(self, basis_matrix, data, weights_matrix, regularization_parameter):
        
        regularization_parameter, regularization_parameter_matrix = self.check_regularization_parameter(regularization_parameter)
        N = data.shape[0]
        left = basis_matrix.T @ weights_matrix @ basis_matrix + regularization_parameter_matrix @ self.penalty_matrix
        right = basis_matrix.T @ weights_matrix @ np.reshape(data, (N*self.dim,))
        coefs = np.linalg.solve(left, right)
        err = np.reshape((np.reshape(basis_matrix @ coefs, (-1,self.dim)) - data), (N*self.dim,))
        SSE = np.diag(np.reshape(err @ weights_matrix, (-1,self.dim)).T @ (np.reshape(basis_matrix @ coefs, (-1,self.dim)) - data))
        df_lbda = basis_matrix @ np.linalg.inv(basis_matrix.T @ weights_matrix @ basis_matrix + regularization_parameter_matrix @ self.penalty_matrix) @ basis_matrix.T @ weights_matrix
        df_lbda = np.sum(np.reshape(np.diag(df_lbda), (-1,self.dim)), axis=0)
        GCV_score = np.array([(N*SSE[i])/((N - df_lbda[i])**2) for i in range(self.dim)])
        # GCV_score = np.sum(np.array([(N*SSE[i])/((N - df_lbda[i])**2) for i in range(self.dim)]))
    
        return GCV_score
    

    def check_data(self, grid, data, weights=None):
    
        N = len(grid)
        if N!=data.shape[0]:
            raise Exception("Dimensions of grid and data do not match.")
        if data.ndim > 2:
            raise Exception("Data must be a 1d or 2d numpy array.")
        if data.ndim==1:
            data = data[:,np.newaxis]
        if self.dim!=1 and self.dim!=data.shape[1]:
            raise Exception("Invalide second dimension of data, do not correspond to the dimension of the basis defined.")

        if weights is None:
            weights_matrix = np.eye(N*self.dim)
        elif weights.ndim==1 and len(weights)==N:
            if self.dim==1:
                weights_matrix = np.diag(weights)
            else:
                weights = np.stack([weights for i in range(self.dim)], axis=-1)
                weights_matrix = block_diag(*np.apply_along_axis(np.diag, 1, weights))
        elif weights.ndim==2 and (weights.shape[0]==self.dim*N and weights.shape[1]==self.dim*N):
            weights_matrix = weights
        elif weights.ndim==2 and (weights.shape[0]==N and weights.shape[1]==self.dim):
            weights_matrix = block_diag(*np.apply_along_axis(np.diag, 1, weights))

        return data, weights_matrix
    

    def check_regularization_parameter(self, regularization_parameter):

        if regularization_parameter is None:
                regularization_parameter = np.repeat(1, self.dim)
        else:
            regularization_parameter = np.squeeze(regularization_parameter)
            if isinstance(regularization_parameter, int) or isinstance(regularization_parameter, float):
                regularization_parameter = np.repeat(regularization_parameter, self.dim)
            elif regularization_parameter.ndim==0 and (isinstance(regularization_parameter.item(), int) or isinstance(regularization_parameter.item(), float)):
                regularization_parameter = np.repeat(regularization_parameter.item(), self.dim)
            elif len(regularization_parameter)==self.dim:
                regularization_parameter = regularization_parameter
            else:
                raise Exception("Invalide value of regularization parameter.")
        regularization_parameter_matrix = np.diag(np.concatenate([np.repeat(regularization_parameter[i], self.nb_basis[i]) for i in range(self.dim)]))

        return regularization_parameter, regularization_parameter_matrix
    

    
    def grid_search_optimization_regularization_parameter(self, grid, data, regularization_parameter_list, weights=None, parallel=False):
        
        V = np.expand_dims(grid, 1)
        basis_matrix = self.basis(V,).reshape((self.basis.n_basis, -1)).T # shape (self.dim*N, np.sum(self.nb_basis))
        data, weights_matrix = self.check_data(grid, data, weights)

        n_param = len(regularization_parameter_list)
        if regularization_parameter_list.ndim==1:
            regularization_parameter_list = np.stack([regularization_parameter_list for i in range(self.dim)], axis=-1)
        # print('Begin grid search optimisation with', n_param, 'combinations of parameters...')

        if parallel:
            func = lambda lbda: self.GCV_score(basis_matrix, data, weights_matrix, lbda)
            out = Parallel(n_jobs=-1)(delayed(func)(regularization_parameter_list[i]) for i in range(n_param))
            res = np.zeros((self.dim))
            for i in range(self.dim):
                ind = np.argmin(out[:,i])
                res[i] = regularization_parameter_list[ind][i]
            # ind = np.argmin(out, axis=0)
            # res = regularization_parameter_list[ind]
        else:
            GCV_scores = np.zeros((n_param, self.dim))
            for i in range(n_param):
                GCV_scores[i] = self.GCV_score(basis_matrix, data, weights_matrix, regularization_parameter_list[i])
            res = np.zeros((self.dim))
            for i in range(self.dim):
                ind = np.argmin(GCV_scores[:,i])
                res[i] = regularization_parameter_list[ind][i]
            # res = regularization_parameter_list[ind] 
        
        # print('Optimal regularization parameter selected by grid search optimisation: ', res)
        return res



    def bayesian_optimization_regularization_parameter(self, grid, data, n_calls, regularization_parameter_bounds, weights=None, verbose=True):
        """
            Do a bayesian optimisation and return the optimal parameter (lambda_1, ..., lambda_dim)
        ...
        """

        V = np.expand_dims(grid, 1)
        basis_matrix = self.basis(V,).reshape((self.basis.n_basis, -1)).T # shape (self.dim*N, np.sum(self.nb_basis))
        data, weights_matrix = self.check_data(grid, data, weights)

        func = lambda lbda: self.GCV_score(basis_matrix, data, weights_matrix, lbda)

        res = gp_minimize(func,               # the function to minimize
                    regularization_parameter_bounds,    # the bounds on each dimension of x
                    acq_func="EI",        # the acquisition function
                    n_calls=n_calls,       # the number of evaluations of f
                    n_random_starts=2,    # the number of random initialization points
                    random_state=1,       # the random seed
                    n_jobs=-1,            # use all the cores for parallel calculation
                    verbose=verbose)
        x = res.x
        # print('Optimal regularization parameter selected by bayesian optimization: ', x)
        return x


    def compute_confidence_limits(self, pourcentage):

        self.sampling_variance()

        def confidence_limits(s):
            val = self.evaluate(s)
            basis_fct_arr = self.basis_fct(s).T 
            if self.dim==1:
                if isinstance(s, int) or isinstance(s, float):
                    error = 0
                elif isinstance(s, np.ndarray):
                    error = np.zeros((len(s)))
                else:
                    raise ValueError('Variable is not a float, a int or a NumPy array.')
                error = pourcentage*np.sqrt(np.diag(basis_fct_arr @ self.sampling_variance_coeffs @ basis_fct_arr.T))
            else:
                if isinstance(s, int) or isinstance(s, float):
                    error = np.zeros((self.dim))
                elif isinstance(s, np.ndarray):
                    error = np.zeros((self.dim, len(s)))
                else:
                    raise ValueError('Variable is not a float, a int or a NumPy array.')
                for i in range(self.dim):
                    error[i] = pourcentage*np.sqrt(np.diag(basis_fct_arr[i] @ self.sampling_variance_coeffs @ basis_fct_arr[i].T))
            upper_limit = val + error.T
            lower_limit = val - error.T
            return lower_limit, upper_limit
        
        self.confidence_limits = confidence_limits
        return confidence_limits
    

    def sampling_variance(self):
        N = len(self.grid)
        y_hat = self.evaluate(self.grid)
        if self.dim==1:
            err = np.reshape(y_hat - self.data[:,0], (N*self.dim,))
            self.residuals_error = np.squeeze((1/(N-self.nb_basis[0]))*err[np.newaxis,:] @ self.weights_matrix @ err[:,np.newaxis])
            res_mat = np.diag(np.repeat(self.residuals_error, N)) 
        else:
            err = np.reshape((y_hat - self.data), (N*self.dim,))
            SSE = np.diag(np.reshape(err @ self.weights_matrix, (-1,self.dim)).T @ (y_hat - self.data))
            self.residuals_error = np.array([SSE[i]/(N-self.nb_basis[i]) for i in range(self.dim)])
            res_mat = block_diag(*np.apply_along_axis(np.diag, 1, np.array([self.residuals_error for i in range(N)])))
        S = np.linalg.inv(self.basis_matrix.T @ self.weights_matrix @ self.basis_matrix + self.regularization_parameter_matrix @ self.penalty_matrix) @ self.basis_matrix.T @ self.weights_matrix
        self.sampling_variance_coeffs = S @ res_mat @ S.T 
        self.sampling_variance_yhat = self.basis_matrix @ self.sampling_variance_coeffs @ self.basis_matrix.T
        

    def plot(self):
        
        if hasattr(self, 'confidence_limits'):
            lower_limits, upper_limits = self.confidence_limits(self.grid)
        y_hat = self.evaluate(self.grid)
        layout = go.Layout(plot_bgcolor='rgba(0,0,0,0)')
        if self.dim==1:
            fig = go.Figure(layout=layout)
            fig.add_trace(go.Scatter(x=self.grid, y=y_hat, mode='lines', name='Smooth estimate'))
            fig.add_trace(go.Scatter(x=self.grid, y=self.data[:,0], mode='lines', name='Observations'))
            if hasattr(self, 'confidence_limits'):
                fig.add_trace(go.Scatter(x=self.grid, y=upper_limits, mode='lines', line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=self.grid, y=lower_limits, mode='lines', line=dict(width=0), showlegend=False, fillcolor='rgba(99, 110, 250, 0.2)', fill='tonexty'))
        
            fig.update_layout(title='Smoothing results')
            fig.update_xaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
            fig.update_yaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
            fig.show()
        else:
            for i in range(self.dim):
                fig = go.Figure(layout=layout)
                fig.add_trace(go.Scatter(x=self.grid, y=y_hat[:,i], mode='lines', name='Smooth estimate'))
                fig.add_trace(go.Scatter(x=self.grid, y=self.data[:,i], mode='lines', name='Observations'))
                if hasattr(self, 'confidence_limits'):
                    fig.add_trace(go.Scatter(x=self.grid, y=upper_limits[:,i], mode='lines', line=dict(width=0), showlegend=False))
                    fig.add_trace(go.Scatter(x=self.grid, y=lower_limits[:,i], mode='lines', line=dict(width=0), showlegend=False, fillcolor='rgba(99, 110, 250, 0.2)', fill='tonexty'))
            
                fig.update_layout(title='Smoothing results dimension '+str(i+1))
                fig.update_xaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
                fig.update_yaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
                fig.show()






class VectorConstantBSplineSmoothing:

    def __init__(self, nb_basis=None, domain_range=(0,1), order=4, penalization=True, knots=None):

        self.dim = 2
        
        if (nb_basis is None) and (knots is None):
            raise Exception("Either nb_basis or knots must be set.")
        
        if knots is not None:
            nb_basis = len(knots)+2
        self.nb_basis = int(nb_basis)
        self.domain_range = domain_range
        self.order = order
        if knots is not None:
            list_basis = [Constant(domain_range=self.domain_range), BSpline(domain_range=self.domain_range, order=order, knots=knots)]
        else:
            list_basis = [Constant(domain_range=self.domain_range), BSpline(domain_range=self.domain_range, n_basis=self.nb_basis, order=order)]
        self.basis = VectorValued(list_basis)
        def basis_fct(s):
            return np.squeeze(self.basis.evaluate(s))
        self.basis_fct = basis_fct
        if penalization:
            self.penalty_matrix = compute_penalty_matrix(basis_iterable=(self.basis,),regularization_parameter=1,regularization=TikhonovRegularization(LinearDifferentialOperator(2)))
        else:
            self.penalty_matrix = np.zeros((nb_basis+1, nb_basis+1))


    def fit(self, grid, data=None, weights=None, regularization_parameter=None):
        """
            grid: dimension N
            data: dimension (N, self.dim), or (N,) if dim=1
            weights: dimension (N, self.dim) or (N*self.dim, N*self.dim) or (N,) if dim=1

        """
        N = len(grid)
        V = np.expand_dims(grid, 1)
        self.basis_matrix = self.basis(V,).reshape((self.basis.n_basis, -1)).T # shape (self.dim*N, np.sum(self.nb_basis))
        if data is not None: 
            self.data, self.weights_matrix = self.check_data(grid, data, weights)
            self.grid = grid
            self.regularization_parameter, self.regularization_parameter_matrix = self.check_regularization_parameter(regularization_parameter)
            left = self.basis_matrix.T @ self.weights_matrix @ self.basis_matrix + self.regularization_parameter_matrix @ self.penalty_matrix
            right = self.basis_matrix.T @ self.weights_matrix @ np.reshape(self.data, (N*self.dim,))
            self.coefs = np.linalg.solve(left, right)


    def evaluate(self, s):
        if isinstance(s, int) or isinstance(s, float):
            return np.squeeze(self.basis_fct(s).T @ self.coefs)
        elif isinstance(s, np.ndarray):
            return np.squeeze((self.basis_fct(s).T @ self.coefs).T)
        else:
            raise ValueError('Variable is not a float, a int or a NumPy array.')
        
    def evaluate_coefs(self, coefs_test):
        def func(s):
            if isinstance(s, int) or isinstance(s, float):
                return np.squeeze(self.basis_fct(s).T @ coefs_test)
            elif isinstance(s, np.ndarray):
                return np.squeeze((self.basis_fct(s).T @ coefs_test).T)
            else:
                raise ValueError('Variable is not a float, a int or a NumPy array.')
        return func

    def GCV_score(self, basis_matrix, data, weights_matrix, regularization_parameter):
        
        regularization_parameter, regularization_parameter_matrix = self.check_regularization_parameter(regularization_parameter)
        N = data.shape[0]
        left = basis_matrix.T @ weights_matrix @ basis_matrix + regularization_parameter_matrix @ self.penalty_matrix
        right = basis_matrix.T @ weights_matrix @ np.reshape(data, (N*self.dim,))
        coefs = np.linalg.solve(left, right)
        err = np.reshape((np.reshape(basis_matrix @ coefs, (-1,self.dim)) - data), (N*self.dim,))
        SSE = np.diag(np.reshape(err @ weights_matrix, (-1,self.dim)).T @ (np.reshape(basis_matrix @ coefs, (-1,self.dim)) - data))
        df_lbda = basis_matrix @ np.linalg.inv(basis_matrix.T @ weights_matrix @ basis_matrix + regularization_parameter_matrix @ self.penalty_matrix) @ basis_matrix.T @ weights_matrix
        df_lbda = np.sum(np.reshape(np.diag(df_lbda), (-1,self.dim)), axis=0)
        GCV_score = np.array([(N*SSE[i])/((N - df_lbda[i])**2) for i in range(self.dim)])
        # GCV_score = np.sum(np.array([(N*SSE[i])/((N - df_lbda[i])**2) for i in range(self.dim)]))
    
        return GCV_score
    

    def check_data(self, grid, data, weights=None):
    
        N = len(grid)
        if N!=data.shape[0]:
            raise Exception("Dimensions of grid and data do not match.")
        if data.ndim > 2:
            raise Exception("Data must be a 1d or 2d numpy array.")
        if data.ndim==1:
            data = data[:,np.newaxis]
        if self.dim!=1 and self.dim!=data.shape[1]:
            raise Exception("Invalide second dimension of data, do not correspond to the dimension of the basis defined.")

        if weights is None:
            weights_matrix = np.eye(N*self.dim)
        elif weights.ndim==1 and len(weights)==N:
            if self.dim==1:
                weights_matrix = np.diag(weights)
            else:
                weights = np.stack([weights for i in range(self.dim)], axis=-1)
                weights_matrix = block_diag(*np.apply_along_axis(np.diag, 1, weights))
        elif weights.ndim==2 and (weights.shape[0]==self.dim*N and weights.shape[1]==self.dim*N):
            weights_matrix = weights
        elif weights.ndim==2 and (weights.shape[0]==N and weights.shape[1]==self.dim):
            weights_matrix = block_diag(*np.apply_along_axis(np.diag, 1, weights))

        return data, weights_matrix
    
    def check_regularization_parameter(self, regularization_parameter):

        if regularization_parameter is None:
                regularization_parameter = np.repeat(1, self.dim)
        else:
            regularization_parameter = np.squeeze(regularization_parameter)
            if isinstance(regularization_parameter, int) or isinstance(regularization_parameter, float):
                regularization_parameter = np.repeat(regularization_parameter, self.dim)
            elif regularization_parameter.ndim==0 and (isinstance(regularization_parameter.item(), int) or isinstance(regularization_parameter.item(), float)):
                regularization_parameter = np.repeat(regularization_parameter.item(), self.dim)
            elif len(regularization_parameter)==self.dim:
                regularization_parameter = regularization_parameter
            else:
                raise Exception("Invalide value of regularization parameter.")
        regularization_parameter_matrix = np.diag(np.concatenate([np.repeat(regularization_parameter[i], self.nb_basis[i]) for i in range(self.dim)]))

        return regularization_parameter, regularization_parameter_matrix
    
    def check_regularization_parameter(self, regularization_parameter):

        if regularization_parameter is None:
                regularization_parameter = np.array([0,1])
        else:
            regularization_parameter = np.squeeze(regularization_parameter)
            if isinstance(regularization_parameter, int) or isinstance(regularization_parameter, float):
                regularization_parameter = np.array([0,regularization_parameter])
            elif regularization_parameter.ndim==0 and (isinstance(regularization_parameter.item(), int) or isinstance(regularization_parameter.item(), float)):
                regularization_parameter = np.array([0,regularization_parameter.item()]) 
            elif regularization_parameter.ndim > 0:
                regularization_parameter = np.array([0,regularization_parameter[1]]) 
            else:
                raise Exception("Invalide value of regularization parameter.")
        regularization_parameter_matrix = np.diag(np.concatenate([np.array([0]), np.repeat(regularization_parameter[1], self.nb_basis)]))

        return regularization_parameter, regularization_parameter_matrix
    

    
    def grid_search_optimization_regularization_parameter(self, grid, data, regularization_parameter_list, weights=None, parallel=False):
        
        V = np.expand_dims(grid, 1)
        basis_matrix = self.basis(V,).reshape((self.basis.n_basis, -1)).T # shape (self.dim*N, np.sum(self.nb_basis))
        data, weights_matrix = self.check_data(grid, data, weights)

        n_param = len(regularization_parameter_list)
        if regularization_parameter_list.ndim==1:
            regularization_parameter_list = np.stack([regularization_parameter_list for i in range(self.dim)], axis=-1)
        # print('Begin grid search optimisation with', n_param, 'combinations of parameters...')

        if parallel:
            func = lambda lbda: self.GCV_score(basis_matrix, data, weights_matrix, lbda)
            out = Parallel(n_jobs=-1)(delayed(func)(regularization_parameter_list[i]) for i in range(n_param))
            res = np.zeros((self.dim))
            for i in range(self.dim):
                ind = np.argmin(out[:,i])
                res[i] = regularization_parameter_list[ind][i]
            # ind = np.argmin(out, axis=0)
            # res = regularization_parameter_list[ind]
        else:
            GCV_scores = np.zeros((n_param, self.dim))
            for i in range(n_param):
                GCV_scores[i] = self.GCV_score(basis_matrix, data, weights_matrix, regularization_parameter_list[i])
            res = np.zeros((self.dim))
            for i in range(self.dim):
                ind = np.argmin(GCV_scores[:,i])
                res[i] = regularization_parameter_list[ind][i]
            # res = regularization_parameter_list[ind] 
        
        # print('Optimal regularization parameter selected by grid search optimisation: ', res)
        return res



    def bayesian_optimization_regularization_parameter(self, grid, data, n_calls, regularization_parameter_bounds, weights=None, verbose=True):
        """
            Do a bayesian optimisation and return the optimal parameter (lambda_1, ..., lambda_dim)
        ...
        """

        V = np.expand_dims(grid, 1)
        basis_matrix = self.basis(V,).reshape((self.basis.n_basis, -1)).T # shape (self.dim*N, np.sum(self.nb_basis))
        data, weights_matrix = self.check_data(grid, data, weights)

        func = lambda lbda: self.GCV_score(basis_matrix, data, weights_matrix, lbda)

        res = gp_minimize(func,               # the function to minimize
                    regularization_parameter_bounds,    # the bounds on each dimension of x
                    acq_func="EI",        # the acquisition function
                    n_calls=n_calls,       # the number of evaluations of f
                    n_random_starts=2,    # the number of random initialization points
                    random_state=1,       # the random seed
                    n_jobs=-1,            # use all the cores for parallel calculation
                    verbose=verbose)
        x = res.x
        # print('Optimal regularization parameter selected by bayesian optimization: ', x)
        return x


    # def compute_confidence_limits(self, pourcentage):

    #     self.sampling_variance()

    #     def confidence_limits(s):
    #         val = self.evaluate(s)
    #         basis_fct_arr = self.basis_fct(s).T 
    #         if self.dim==1:
    #             if isinstance(s, int) or isinstance(s, float):
    #                 error = 0
    #             elif isinstance(s, np.ndarray):
    #                 error = np.zeros((len(s)))
    #             else:
    #                 raise ValueError('Variable is not a float, a int or a NumPy array.')
    #             error = pourcentage*np.sqrt(np.diag(basis_fct_arr @ self.sampling_variance_coeffs @ basis_fct_arr.T))
    #         else:
    #             if isinstance(s, int) or isinstance(s, float):
    #                 error = np.zeros((self.dim))
    #             elif isinstance(s, np.ndarray):
    #                 error = np.zeros((self.dim, len(s)))
    #             else:
    #                 raise ValueError('Variable is not a float, a int or a NumPy array.')
    #             for i in range(self.dim):
    #                 error[i] = pourcentage*np.sqrt(np.diag(basis_fct_arr[i] @ self.sampling_variance_coeffs @ basis_fct_arr[i].T))
    #         upper_limit = val + error.T
    #         lower_limit = val - error.T
    #         return lower_limit, upper_limit
        
    #     self.confidence_limits = confidence_limits
    #     return confidence_limits
    

    # def sampling_variance(self):
    #     N = len(self.grid)
    #     y_hat = self.evaluate(self.grid)
    #     if self.dim==1:
    #         err = np.reshape(y_hat - self.data[:,0], (N*self.dim,))
    #         self.residuals_error = np.squeeze((1/(N-self.nb_basis[0]))*err[np.newaxis,:] @ self.weights_matrix @ err[:,np.newaxis])
    #         res_mat = np.diag(np.repeat(self.residuals_error, N)) 
    #     else:
    #         err = np.reshape((y_hat - self.data), (N*self.dim,))
    #         SSE = np.diag(np.reshape(err @ self.weights_matrix, (-1,self.dim)).T @ (y_hat - self.data))
    #         self.residuals_error = np.array([SSE[i]/(N-self.nb_basis[i]) for i in range(self.dim)])
    #         res_mat = block_diag(*np.apply_along_axis(np.diag, 1, np.array([self.residuals_error for i in range(N)])))
    #     S = np.linalg.inv(self.basis_matrix.T @ self.weights_matrix @ self.basis_matrix + self.regularization_parameter_matrix @ self.penalty_matrix) @ self.basis_matrix.T @ self.weights_matrix
    #     self.sampling_variance_coeffs = S @ res_mat @ S.T 
    #     self.sampling_variance_yhat = self.basis_matrix @ self.sampling_variance_coeffs @ self.basis_matrix.T
        

    # def plot(self):
        
    #     if hasattr(self, 'confidence_limits'):
    #         lower_limits, upper_limits = self.confidence_limits(self.grid)
    #     y_hat = self.evaluate(self.grid)
    #     layout = go.Layout(plot_bgcolor='rgba(0,0,0,0)')
    #     if self.dim==1:
    #         fig = go.Figure(layout=layout)
    #         fig.add_trace(go.Scatter(x=self.grid, y=y_hat, mode='lines', name='Smooth estimate'))
    #         fig.add_trace(go.Scatter(x=self.grid, y=self.data[:,0], mode='lines', name='Observations'))
    #         if hasattr(self, 'confidence_limits'):
    #             fig.add_trace(go.Scatter(x=self.grid, y=upper_limits, mode='lines', line=dict(width=0), showlegend=False))
    #             fig.add_trace(go.Scatter(x=self.grid, y=lower_limits, mode='lines', line=dict(width=0), showlegend=False, fillcolor='rgba(99, 110, 250, 0.2)', fill='tonexty'))
        
    #         fig.update_layout(title='Smoothing results')
    #         fig.update_xaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
    #         fig.update_yaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
    #         fig.show()
    #     else:
    #         for i in range(self.dim):
    #             fig = go.Figure(layout=layout)
    #             fig.add_trace(go.Scatter(x=self.grid, y=y_hat[:,i], mode='lines', name='Smooth estimate'))
    #             fig.add_trace(go.Scatter(x=self.grid, y=self.data[:,i], mode='lines', name='Observations'))
    #             if hasattr(self, 'confidence_limits'):
    #                 fig.add_trace(go.Scatter(x=self.grid, y=upper_limits[:,i], mode='lines', line=dict(width=0), showlegend=False))
    #                 fig.add_trace(go.Scatter(x=self.grid, y=lower_limits[:,i], mode='lines', line=dict(width=0), showlegend=False, fillcolor='rgba(99, 110, 250, 0.2)', fill='tonexty'))
            
    #             fig.update_layout(title='Smoothing results dimension '+str(i+1))
    #             fig.update_xaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
    #             fig.update_yaxes(showline=True, showgrid=False, linewidth=1, linecolor='black')
    #             fig.show()