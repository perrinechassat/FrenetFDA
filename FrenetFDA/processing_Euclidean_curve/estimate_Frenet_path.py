import numpy as np
from scipy import interpolate, optimize
from scipy.integrate import cumtrapz
from FrenetFDA.utils.smoothing_utils import LocalPolynomialSmoothing, kernel, adaptive_kernel
from FrenetFDA.processing_Euclidean_curve.preprocessing import compute_arc_length, compute_derivatives
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from skopt import gp_minimize


class GramSchmidtOrthogonalization:

    def __init__(self, Y, arc_length, deg=4):
        self.N, self.dim = Y.shape
        if self.dim < 2:
            raise Exception("The Frenet Serret framework is defined only for curves in R^d with d >= 2.")
        if self.N != len(arc_length):
            raise Exception("Number of sample points in attribute Y and time must be equal.")
        self.Y = Y
        self.grid_arc_s = arc_length
        self.deg = deg


    def __compute_arc_length_derivatives(self, h):
        
        arc_length_derivatives = LocalPolynomialSmoothing(self.deg).fit(self.Y, self.grid_arc_s, self.grid_arc_s, h)

        return arc_length_derivatives


    def __GramSchmidt(self, derivatives):
        """ 
            Do the Gram-Schmidt orthogonalization 
        """
        vect_frenet_frame = np.zeros((self.dim, self.dim))
        vect_frenet_frame[0] = derivatives[0]/np.linalg.norm(derivatives[0])
        for k in range(1,self.dim):
            vect_frenet_frame[k] = derivatives[k] 
            for j in range(0,k):
                vect_frenet_frame[k] = vect_frenet_frame[k] - np.dot(np.transpose(vect_frenet_frame[j]),derivatives[k])*vect_frenet_frame[j]
            if np.linalg.norm(vect_frenet_frame[k]) < 1e-30:
                vect_frenet_frame[k] = np.ones(vect_frenet_frame[k].shape)
            else:
                vect_frenet_frame[k] = vect_frenet_frame[k]/np.linalg.norm(vect_frenet_frame[k])
        Q = np.stack(vect_frenet_frame)
        if np.linalg.det(Q)<0:
            vect_frenet_frame[-1] = - vect_frenet_frame[-1]
            Q = np.stack(vect_frenet_frame)
        return np.transpose(Q)
    

    def fit(self, h, reparam_grid=None):

        self.arc_length_derivatives = self.__compute_arc_length_derivatives(h)
        Q = np.zeros((self.N, self.dim, self.dim)) 
        for i in range(self.N):
            Qi = self.__GramSchmidt(self.arc_length_derivatives[1:,i,:])
            Q[i,:,:]= Qi
        if reparam_grid is None:
            reparam_grid = self.grid_arc_s
            reparam_Q = Q
            reparam_X = self.arc_length_derivatives[0]
        else:
            reparam_Q = np.zeros((len(reparam_grid), self.dim, self.dim))
            for k in range(self.dim):
                reparam_Q[:,:,k] = interpolate.griddata(self.grid_arc_s, Q[:,:,k], reparam_grid, method='cubic')
            reparam_X = interpolate.griddata(self.grid_arc_s, self.arc_length_derivatives[0], reparam_grid, method='cubic')
        Z = np.concatenate((reparam_Q, reparam_X[:,:,np.newaxis]), axis=-1)
        vec = np.zeros((len(reparam_grid),1,4))
        vec[:,:,3] = np.ones((len(reparam_grid),1))
        self.Z = np.concatenate((Z, vec), axis=1)
        self.Q = reparam_Q
        self.X = reparam_X

        return self.Z, self.Q, self.X
    

    def grid_search_CV_optimization_bandwidth(self, bandwidth_grid=np.array([]), K_split=10):
        
        h_opt, err_h = LocalPolynomialSmoothing(self.deg).grid_search_CV_optimization_bandwidth(self.Y, self.grid_arc_s, self.grid_arc_s, bandwidth_grid=bandwidth_grid, K_split=K_split)

        return h_opt, err_h
    

    def bayesian_optimization_hyperparameters(self, n_call_bayopt, h_bounds, n_splits=10, verbose=True):

        h_opt = LocalPolynomialSmoothing(self.deg).bayesian_optimization_hyperparameters(self.Y, self.grid_arc_s, self.grid_arc_s, n_call_bayopt, h_bounds, n_splits=n_splits, verbose=verbose)

        return h_opt



class ConstrainedLocalPolynomialRegression:


    def __init__(self, Y, arc_length, adaptative=False, deg_polynomial=3, ibound=0):
        self.N, self.dim = Y.shape
        if self.dim < 2:
            raise Exception("The Frenet Serret framework is defined only for curves in R^d with d >= 2.")
        if self.N != len(arc_length):
            raise Exception("Number of sample points in attribute Y and time must be equal.")
        self.Y = Y
        self.grid_arc_s = arc_length
        self.adaptative = adaptative
        self.ibound = ibound
        self.deg = deg_polynomial


    def fit(self, h, reparam_grid=None):

        if self.dim != 3:
            raise Exception("Only available in dimension 3.")
        
        if reparam_grid is None:
            reparam_grid = self.grid_arc_s
        # Q_LP, X_LP, vkappa, Param, Param0, vparam, success = self.__constrained_local_polynomial_regression(self.Y, self.grid_arc_s, reparam_grid, h)
        Q_LP, X_LP = self.__constrained_local_polynomial_regression(self.Y, self.grid_arc_s, reparam_grid, h)
        Z = np.concatenate((Q_LP, X_LP[:,:,np.newaxis]), axis=-1) 
        vec = np.zeros((len(reparam_grid),1,4))
        vec[:,:,3] = np.ones((len(reparam_grid),1))
        self.Z = np.concatenate((Z, vec), axis=1)
        self.Q = Q_LP
        self.X = X_LP

        return self.Z, self.Q, self.X
    

    def grid_search_CV_optimization_bandwidth(self, bandwidth_grid=np.array([]), K_split=10):
        if len(bandwidth_grid)==0:
            raise Exception("You must give a grid for search of optimal h")
        else:
            N_param = len(bandwidth_grid)
            err_h = np.zeros(N_param)
            kf = KFold(n_splits=K_split, shuffle=True)
            for j in range(N_param):
                err = []
                for train_index, test_index in kf.split(self.grid_arc_s):
                    t_train, t_test = self.grid_arc_s[train_index], self.grid_arc_s[test_index]
                    data_train, data_test = self.Y[train_index,:], self.Y[test_index,:]
                    # Q_LP, X_LP, vkappa, Param, Param0, vparam, success = self.__constrained_local_polynomial_regression(data_train, t_train, t_test, bandwidth_grid[j])
                    Q_LP, X_LP = self.__constrained_local_polynomial_regression(data_train, t_train, t_test, bandwidth_grid[j])
                    diff = X_LP - data_test
                    err.append(np.linalg.norm(diff)**2)
                err_h[j] = np.mean(err)
            if isinstance(np.where(err_h==np.min(err_h)), int):
                h_opt = bandwidth_grid[np.where(err_h==np.min(err_h))]
            else:
                h_opt = bandwidth_grid[np.where(err_h==np.min(err_h))][0]
            # print('Optimal smoothing parameter h find by cross-validation:', h_opt)

            return h_opt, err_h
    

    def __constrained_local_polynomial_regression(self, data, grid_in, grid_out, h):
        """ 
            Solve the following constrained local polynomial regression problem: (|T|=1, <T,N>=0)
                b0 + b1(t-t_0)+b2(t-t0)^2/2 + b3(t-t0)^3/6 + ... + bp(t-t0)^p/p!
                |b1|=1, <b1,b2>=0
            Minimize: (Y-XB)'W(Y-XB) -la*(|b1|^2-1) - mu(2*<b1,b2>)
            Inputs:
               vtout - output grid, length(vtout)=nout
               Y     - J x 3 matrix
               vt    - input grid
               h     - scalar
               p     - degree of polynomial (defaul = 3)
               iflag - [1,1] for both constraints 
                   [1,0] for |b1|=1
                   [0,1] for <b1,b2>=0
               ibound - 1 for boundary correction 
                        0 by default 
            Outputs:
                Q
                X
               --------
        """
        if self.adaptative:
            N_in = len(grid_in)
            T_end = grid_in[-1]
            step_unif = T_end/N_in
            kNN = int(h/step_unif)

        N_out = len(grid_out)
        pre_process = PolynomialFeatures(degree=self.deg)
        Param = np.zeros((N_out,(self.deg+1)*self.dim))
        U = np.zeros((self.deg+1,self.deg+1))
        U[1,1] = 1
        V = np.zeros((self.deg+1,self.deg+1))
        V[1,2] = 1
        V[2,1] = 1
        for i in range(N_out):
            T = grid_in - grid_out[i]

            if self.adaptative==True:
                adapt_h = 1.0001*np.sort(abs(T))[kNN-1]*2
                # print(adapt_h)
                W = kernel(T/adapt_h)
                # delta = 1.0001*np.maximum(T[-1], -T[0])
                # W = adaptive_kernel(T,adapt_h)
            else:
                W = kernel(T/h)
            
            T_poly = pre_process.fit_transform(T.reshape(-1,1))
            for j in range(self.deg+1):
                T_poly[:,j] = T_poly[:,j]/np.math.factorial(j)
       
            Si = T_poly.T @ np.diag(W) @ T_poly # p+1 x p+1
            Ti = T_poly.T @ np.diag(W) @ data # p+1 x 3

            # try:
            # estimates with constraints
            if self.deg==1: # local linear
                tb0 = np.array([-Si[0,1], Si[0,0]]) @ Ti
                la_m = (np.linalg.det(Si) - np.linalg.norm(tb0))/Si[0,0]
                Param[i,:] = np.reshape(np.linalg.solve(Si-la_m*np.array([[0,0],[0,1]]), Ti),(1,(self.deg+1)*self.dim))

            elif self.deg>1:
                la0 = 0
                mu0 = 0
                # tol = 1e-4
                param0 = np.array([la0, mu0])
                res = optimize.root(fun=self.__get_loc_param, x0=param0, args=(Si, Ti), method='hybr')
                parami = res.x
                itr = 0
                epsilon_vect = np.array([10e-6,10e-6])
                while res.success==False and itr<30:
                    parami += epsilon_vect
                    res = optimize.root(fun=self.__get_loc_param, x0=parami, args=(Si, Ti), method='hybr')
                    parami = res.x
                    itr += 1

                la0 = parami[0]
                mu0 = parami[1]
                Bi = np.linalg.inv(Si-la0*U-mu0*V) @ Ti
                Param[i,:] = np.reshape(Bi,(1,(self.deg+1)*self.dim))
            # except:
            #     raise Exception("The bandwidth parameter is too small.")

        # output
        Gamma = Param[:,:3]
        T = Param[:,3:6]
        if (self.deg>1):
            Ntilde = Param[:,6:9]
            kappa = np.sqrt(np.sum(np.power(Ntilde,2),1))
            N = np.diag(1/kappa)@Ntilde
            Bi = np.cross(T, N)

        Q = np.zeros((N_out, self.dim, self.dim))
        Q[:,:,0] = T
        Q[:,:,1] = N
        Q[:,:,2] = Bi
        smooth_trajectory = Gamma

        # success = True
        # if len(list_error) > 0:
        #     success = False

        return Q, smooth_trajectory
    

    def __get_loc_param(self,param,S,T):
        """
            param - 1 x 2 vector
            S  - pp x pp
            T  - pp x d
        """
        pp = S.shape[0]
        U = np.zeros((pp,pp))
        U[1,1] = 1
        V = np.zeros((pp,pp))
        V[1,2] = 1
        V[2,1] = 1
        B = np.linalg.inv(S-param[0]*U-param[1]*V) @ T
        # B = np.linalg.solve(S-param[0]*U-param[1]*V,T)
        out = [B[1,:] @ B[1,:].T - 1, B[1,:] @ B[2,:].T]
        return out


    def bayesian_optimization_hyperparameters(self, n_call_bayopt, h_bounds, n_splits=10, verbose=True):

        def func(x):
            score = np.zeros(n_splits)
            kf = KFold(n_splits=n_splits, shuffle=True)
            grid_split = self.grid_arc_s[1:-1]
            ind_CV = 0

            for train_index, test_index in kf.split(grid_split):
                train_index = train_index+1
                test_index = test_index+1
                train_index = np.concatenate((np.array([0]), train_index, np.array([len(self.grid_arc_s[1:-1])+1])))
                Y_train = self.Y[train_index]
                Y_test = self.Y[test_index]
                t_train, t_test = self.grid_arc_s[train_index], self.grid_arc_s[test_index]
                try:
                    Q_LP, X_LP = self.__constrained_local_polynomial_regression(Y_train, t_train, t_test, x[0])
                    error = np.linalg.norm(X_LP - Y_test)**2
                except:
                    error = 1e3
                score[ind_CV] = error
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




class GramSchmidtOrthogonalizationSphericalCurves:

    def __init__(self, Y, arc_length, L=None, deg=4):
        self.N, self.dim = Y.shape
        if self.dim < 2:
            raise Exception("The Frenet Serret framework is defined only for curves in R^d with d >= 2.")
        if self.N != len(arc_length):
            raise Exception("Number of sample points in attribute Y and time must be equal.")
        self.Y = Y
        self.grid_arc_s = arc_length
        self.deg = deg

    def __compute_arc_length_derivatives(self, h):
        
        arc_length_derivatives = LocalPolynomialSmoothing(self.deg).fit(self.Y, self.grid_arc_s, self.grid_arc_s, h)

        return arc_length_derivatives    

    def fit(self, h, reparam_grid=None):

        self.arc_length_derivatives = self.__compute_arc_length_derivatives(h)
        Q = np.zeros((self.N, self.dim, self.dim))
        for i in range(self.N):
            Q[i,:,0] = self.arc_length_derivatives[0][i]/np.linalg.norm(self.arc_length_derivatives[0][i])
            Q[i,:,1] = self.arc_length_derivatives[1][i]/np.linalg.norm(self.arc_length_derivatives[1][i])
            Q[i,:,2] = np.cross(Q[i,:,0], Q[i,:,1])
        if reparam_grid is None:
            reparam_grid = self.grid_arc_s
            reparam_Q = Q
            reparam_X = self.arc_length_derivatives[0]
        else:
            reparam_Q = np.zeros((len(reparam_grid), self.dim, self.dim))
            for k in range(self.dim):
                reparam_Q[:,:,k] = interpolate.griddata(self.grid_arc_s, Q[:,:,k], reparam_grid, method='cubic')
            reparam_X = interpolate.griddata(self.grid_arc_s, self.arc_length_derivatives[0], reparam_grid, method='cubic')
        self.Q = reparam_Q
        self.X = reparam_X

        return self.Q, self.X
    

    def grid_search_CV_optimization_bandwidth(self, bandwidth_grid=np.array([]), K_split=10):
        
        h_opt, err_h = LocalPolynomialSmoothing(self.deg).grid_search_CV_optimization_bandwidth(self.Y, self.grid_arc_s, self.grid_arc_s, bandwidth_grid=bandwidth_grid, K_split=K_split)

        return h_opt, err_h
    

    def bayesian_optimization_hyperparameters(self, n_call_bayopt, h_bounds, n_splits=10, verbose=True):

        h_opt = LocalPolynomialSmoothing(self.deg).bayesian_optimization_hyperparameters(self.Y, self.grid_arc_s, self.grid_arc_s, n_call_bayopt, h_bounds, n_splits=n_splits, verbose=verbose)

        return h_opt