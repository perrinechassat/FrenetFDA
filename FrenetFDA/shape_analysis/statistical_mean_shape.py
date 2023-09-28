import numpy as np
from FrenetFDA.utils.smoothing_utils import compute_weight_neighbors_local_smoothing, VectorBSplineSmoothing, grid_search_GCV_optimization_Bspline_hyperparameters
from FrenetFDA.utils.Frenet_Serret_utils import solve_FrenetSerret_ODE_SE, solve_FrenetSerret_ODE_SO
from FrenetFDA.utils.Lie_group.SO3_utils import SO3
from FrenetFDA.utils.Lie_group.SE3_utils import SE3
from FrenetFDA.utils.alignment_utils import align_vect_curvatures_fPCA, warp_curvatures, align_vect_SRC_fPCA
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline


class StatisticalMeanShape:

    def __init__(self, list_grids, list_Frenet_paths, adaptive=False):
        if len(list_Frenet_paths) != len(list_grids):
             raise Exception("Number of grids and Frenet paths data must be equals.") 
        if len(list_Frenet_paths) < 2:
             raise Exception("You must pass at least two curves.")
        self.N_samples = len(list_Frenet_paths)
        self.list_Q = list_Frenet_paths
        self.list_grids = list_grids
        self.adaptive_ind = adaptive
        self.N, self.dim, _ = list_Frenet_paths[0].shape
        self.dim_theta = len(np.diag(np.eye(self.dim), k=1))


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
    

    def raw_estimates(self, h):
        return np.zeros(self.N), np.zeros((self.N,self.dim_theta)), np.zeros(self.N)

    
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




class StatisticalMeanShapeV1(StatisticalMeanShape):

    def __init__(self, list_grids, list_Frenet_paths, adaptive=False):
        super().__init__(list_grids, list_Frenet_paths, adaptive=False)


    def raw_estimates(self, h):
        grid_theta, raw_theta, weight_theta = self.__raw_estimates(h, self.list_grids, self.list_Q)
        return grid_theta, raw_theta, weight_theta

    
    def __raw_estimates(self, h, list_grids, list_Q):

        N_samples = len(list_Q)

        Omega, S, Kappa, Tau = [], [], [], []
        for i in range(N_samples):
            neighbor_obs, weight, grid_double, delta = compute_weight_neighbors_local_smoothing(list_grids[i], list_grids[i], h, self.adaptive_ind)

            for q in range(len(list_grids[i])):
                if q==0:
                    s = list_grids[i][0]*np.ones(len(neighbor_obs[q]))
                elif q==len(list_grids[i])-1:
                    s = list_grids[i][-1]*np.ones(len(neighbor_obs[q]))
                else:
                    s = grid_double[q]
                S += list(s)

                N_q = len(neighbor_obs[q])
                Obs_q = list_Q[i][neighbor_obs[q]]
                w_q = weight[q]
                u_q = np.copy(delta[q])
                omega_q = np.multiply(w_q,np.power(u_q,2))
                if q!=0 and q!=len(list_grids[i])-1:
                    v_q = np.where(u_q==0)[0]
                    u_q[u_q==0] = 1
                R_q = np.zeros((N_q, self.dim))
                for j in range(N_q):
                    if (q!=0 or j!=0) and (q!=len(list_grids[i])-1 or j!=N_q-1):
                        R_q[j] = -SO3.log(np.transpose(np.ascontiguousarray(list_Q[i][q]))@np.ascontiguousarray(Obs_q[j]))/u_q[j]
                if q!=0 and q!=len(list_grids[i])-1:
                    R_q[v_q] = np.abs(0*R_q[v_q])
                kappa = np.squeeze(R_q[:,2])
                tau = np.squeeze(R_q[:,0])

                Omega = np.append(Omega, omega_q.tolist())
                Kappa = np.append(Kappa, kappa.tolist())
                Tau = np.append(Tau, tau.tolist())     

        Ms, Momega, Mkappa, Mtau = self.__compute_sort_unique_val(np.around(S, 8), Omega, Kappa, Tau)
        Mkappa = abs(Mkappa)

        # Test pour enlever les valeurs à zeros.
        Momega = np.asarray(Momega)
        ind_nozero = np.where(Momega!=0.)
        weight_theta = np.squeeze(Momega[ind_nozero])
        grid_theta = Ms[ind_nozero]
        raw_theta = np.stack((np.squeeze(np.asarray(Mkappa)[ind_nozero]),np.squeeze(np.asarray(Mtau)[ind_nozero])), axis=1)

        return grid_theta, raw_theta, weight_theta         

    
    
    

class StatisticalMeanShapeV2(StatisticalMeanShape):

    def __init__(self, grid, list_Frenet_paths, adaptive=False):
        list_grids = np.array([grid for i in range(len(list_Frenet_paths))])
        super().__init__(list_grids, list_Frenet_paths, adaptive=False)
        self.grid = grid


    def raw_estimates(self, h):
        grid_theta, raw_theta, weight_theta, gam_theta, ind_conv, theta_align = self.__raw_estimates(h, self.list_grids, self.list_Q)
        self.gam = gam_theta
        self.theta_align = theta_align
        self.ind_conv = ind_conv
        return grid_theta, raw_theta, weight_theta

    
    def __raw_estimates(self, h, grid, list_Q, sigma=1.0):

        N_samples = len(list_Q)
        
        # Compute individual raw estimates 
        neighbor_obs, weight, grid_double, delta = compute_weight_neighbors_local_smoothing(grid, grid, h, self.adaptive_ind)
        Omega, S, Kappa, Tau = [], [], [], []
        for i in range(N_samples):
            Omega_i, S_i, Kappa_i, Tau_i = [], [], [], []

            for q in range(len(grid)):
                if q==0:
                    s = grid[0]*np.ones(len(neighbor_obs[q]))
                elif q==len(grid)-1:
                    s = grid[i][-1]*np.ones(len(neighbor_obs[q]))
                else:
                    s = grid_double[q]
                S_i += list(s)

                N_q = len(neighbor_obs[q])
                Obs_q = list_Q[i][neighbor_obs[q]]
                w_q = weight[q]
                u_q = np.copy(delta[q])
                omega_q = np.multiply(w_q,np.power(u_q,2))
                if q!=0 and q!=len(grid)-1:
                    v_q = np.where(u_q==0)[0]
                    u_q[u_q==0] = 1
                R_q = np.zeros((N_q, self.dim))
                for j in range(N_q):
                    if (q!=0 or j!=0) and (q!=len(grid)-1 or j!=N_q-1):
                        R_q[j] = -SO3.log(np.transpose(np.ascontiguousarray(list_Q[i][q]))@np.ascontiguousarray(Obs_q[j]))/u_q[j]
                if q!=0 and q!=len(grid)-1:
                    R_q[v_q] = np.abs(0*R_q[v_q])
                kappa = np.squeeze(R_q[:,2])
                tau = np.squeeze(R_q[:,0])

                Omega_i = np.append(Omega_i, omega_q.tolist())
                Kappa_i = np.append(Kappa_i, kappa.tolist())
                Tau_i = np.append(Tau_i, tau.tolist())     

            Ms_i, Momega_i, Mkappa_i, Mtau_i = self.__compute_sort_unique_val(np.around(S_i, 8), Omega_i, Kappa_i, Tau_i)

            S.append(Ms_i)
            Omega.append(Momega_i)
            Kappa.append(Mkappa_i)
            Tau.append(Mtau_i)
        
        Omega = np.asarray(Omega)
        sum_omega = np.sum(Omega, axis=0)
        ind_nozero = np.where(sum_omega!=0.)
        sum_omega = sum_omega[ind_nozero]
        Omega = np.squeeze(Omega[:,ind_nozero])
        Kappa = np.squeeze(np.asarray(Kappa)[:,ind_nozero])
        Tau = np.squeeze(np.asarray(Tau)[:,ind_nozero])
        S_grid = np.squeeze(S[0][ind_nozero])
        

        # Alignement

        theta = np.stack((np.transpose(Kappa), np.abs(np.transpose(Tau))))
        theta[np.isnan(theta)] = 0.0
        theta[np.isinf(theta)] = 0.0

        res = align_vect_curvatures_fPCA(theta, S_grid, np.transpose(Omega), num_comp=3, cores=-1, smoothdata=False, MaxItr=20, lam=sigma)
        theta_align, gam_theta, weighted_mean_theta = res.fn, res.gamf, res.mfn
        # shape theta_align (dim_theta, N, N_samples)
        # shape weighted_mean_theta (dim_theta, N)
        # shape gam_theta (N, N_samples)
        ind_conv = res.convergence

        gam_fct = np.empty((gam_theta.shape[1]), dtype=object)
        for i in range(gam_theta.shape[1]):
            gam_fct[i] = interp1d(S_grid, gam_theta[:,i])

        weighted_mean_kappa = np.squeeze(weighted_mean_theta[0])
        tau_align, weighted_mean_tau = warp_curvatures(Tau, gam_fct, S_grid, Omega)

        # remplacer dans theta_align le tau par tau_align
        theta_align = theta_align.T
        theta_align[:,:,1] = tau_align

        raw_theta = np.stack((np.squeeze(weighted_mean_kappa),np.squeeze(weighted_mean_tau)), axis=1)

        return S_grid, raw_theta, sum_omega, gam_theta.T, ind_conv, theta_align 
    


class StatisticalMeanShapeV3(StatisticalMeanShape):

    def __init__(self, grid, list_Frenet_paths, adaptive=False):
        list_grids = np.array([grid for i in range(len(list_Frenet_paths))])
        super().__init__(list_grids, list_Frenet_paths, adaptive=False)
        self.grid = grid


    def raw_estimates(self, h):
        grid_theta, raw_theta, weight_theta, gam_SRC, ind_conv, SRC_align, theta_align = self.__raw_estimates(h, self.list_grids, self.list_Q)
        self.gam = gam_SRC
        self.theta_align = theta_align
        self.ind_conv = ind_conv
        self.SRC_align = SRC_align
        return grid_theta, raw_theta, weight_theta

    
    def __raw_estimates(self, h, grid, list_Q, sigma=1.0):
        
        N_samples = len(list_Q)
        
        # Compute individual raw estimates 
        neighbor_obs, weight, grid_double, delta = compute_weight_neighbors_local_smoothing(grid, grid, h, self.adaptive_ind)
        Omega, S, Kappa, Tau = [], [], [], []
        for i in range(N_samples):
            Omega_i, S_i, Kappa_i, Tau_i = [], [], [], []

            for q in range(len(grid)):
                if q==0:
                    s = grid[0]*np.ones(len(neighbor_obs[q]))
                elif q==len(grid)-1:
                    s = grid[i][-1]*np.ones(len(neighbor_obs[q]))
                else:
                    s = grid_double[q]
                S_i += list(s)

                N_q = len(neighbor_obs[q])
                Obs_q = list_Q[i][neighbor_obs[q]]
                w_q = weight[q]
                u_q = np.copy(delta[q])
                omega_q = np.multiply(w_q,np.power(u_q,2))
                if q!=0 and q!=len(grid)-1:
                    v_q = np.where(u_q==0)[0]
                    u_q[u_q==0] = 1
                R_q = np.zeros((N_q, self.dim))
                for j in range(N_q):
                    if (q!=0 or j!=0) and (q!=len(grid)-1 or j!=N_q-1):
                        R_q[j] = -SO3.log(np.transpose(np.ascontiguousarray(list_Q[i][q]))@np.ascontiguousarray(Obs_q[j]))/u_q[j]
                if q!=0 and q!=len(grid)-1:
                    R_q[v_q] = np.abs(0*R_q[v_q])
                kappa = np.squeeze(R_q[:,2])
                tau = np.squeeze(R_q[:,0])

                Omega_i = np.append(Omega_i, omega_q.tolist())
                Kappa_i = np.append(Kappa_i, kappa.tolist())
                Tau_i = np.append(Tau_i, tau.tolist())     

            Ms_i, Momega_i, Mkappa_i, Mtau_i = self.__compute_sort_unique_val(np.around(S_i, 8), Omega_i, Kappa_i, Tau_i)

            S.append(Ms_i)
            Omega.append(Momega_i)
            Kappa.append(Mkappa_i)
            Tau.append(Mtau_i)
        
        Omega = np.asarray(Omega)
        sum_omega = np.sum(Omega, axis=0)
        ind_nozero = np.where(sum_omega!=0.)
        sum_omega = sum_omega[ind_nozero]
        Omega = np.squeeze(Omega[:,ind_nozero])
        Kappa = np.squeeze(np.asarray(Kappa)[:,ind_nozero])
        Tau = np.squeeze(np.asarray(Tau)[:,ind_nozero])
        S_grid = np.squeeze(S[0][ind_nozero])


        # Alignement

        theta = np.stack((np.transpose(Kappa), np.transpose(Tau)))
        theta[np.isnan(theta)] = 0.0
        theta[np.isinf(theta)] = 0.0
        # taille (2, N, N_samples)
        SRC_tab = np.zeros((theta.shape))
        for i in range(N_samples):
            SRC_tab[:,:,i] = theta[:,:,i]/np.sqrt(np.linalg.norm(theta[:,:,i], axis=0))
        
        res = align_vect_SRC_fPCA(SRC_tab, S_grid, np.transpose(Omega), num_comp=3, cores=-1, smoothdata=False, MaxItr=20, lam=sigma)
        SRC_align, gam_SRC, weighted_mean_SRC = res.fn.T, res.gamf.T, res.mfn.T
        # shape SRC_align (N_samples, N, dim_theta)
        # shape weighted_mean_SRC (N, dim_theta)
        # shape gam_SRC (N_samples, N)
        ind_conv = res.convergence

        weighted_mean_theta = np.zeros(weighted_mean_SRC.shape)
        for i in range(weighted_mean_SRC.shape[0]):
            weighted_mean_theta[i] = np.linalg.norm(weighted_mean_SRC[i])*weighted_mean_SRC[i]

        theta_align = np.zeros(SRC_align.shape)
        for i in range(N_samples):
            for j in range(SRC_align.shape[1]):
                theta_align[i,j,:] = np.linalg.norm(SRC_align[i,j])*SRC_align[i,j]


        return S_grid, weighted_mean_theta, sum_omega, gam_SRC, ind_conv, SRC_align, theta_align
    