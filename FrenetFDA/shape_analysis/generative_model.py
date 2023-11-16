import fdasrsf as fs
import numpy as np 
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
import collections
from geomstats.learning.pca import TangentPCA
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.learning.frechet_mean import FrechetMean
from FrenetFDA.utils.Frenet_Serret_utils import solve_FrenetSerret_ODE_SO, solve_FrenetSerret_ODE_SE
from scipy.interpolate import UnivariateSpline

class GenerativeModel:

    def __init__(self, mtheta, aligned_theta, tab_gamma, tab_arclength=None, tab_L=None, tab_Q0=None, tab_X0=None, tab_T=None, init="init", s_smooth=0.0):
        self.mtheta = mtheta
        self.N = len(aligned_theta)
        self.n = len(tab_gamma[0])
        self.Ntheta = aligned_theta
        self.s_smooth = s_smooth
        if s_smooth == 0.0:
            self.Ngamma = tab_gamma
        else:
            grid = np.linspace(0,1,self.n)
            for i in range(self.N):
                tmp_spline = UnivariateSpline(grid, tab_gamma[i], s=self.s_smooth)
                tab_gamma[i] = tmp_spline(grid)
            self.Ngamma = tab_gamma
        if tab_arclength is None:
            self.Narclength = np.array([np.linspace(0,1,self.n) for i in range(self.N)])
            self.onlyshape = True
        else:
            self.Narclength = tab_arclength
            self.onlyshape = False
        if tab_X0 is None:
            self.Nx0 = np.zeros((self.N,3))
        else:
            self.Nx0 = tab_X0
        if tab_Q0 is None:
            self.NQ0 = np.stack([np.eye(3) for i in range(self.N)])
        else: 
            self.NQ0 = tab_Q0
        if tab_L is None:
            self.NL = np.ones(self.N)
        else:
            self.NL = tab_L
        if tab_T is None:
            self.NT = np.ones(self.N)
        else:
            self.NT = tab_T
        self.init = init
  
      
    def fit(self):
        
        time = np.linspace(0,1,self.n)

        res_fpca_s = horizontal_pca(self.Narclength.T, time)
        self.fpca_s = res_fpca_s
        c_s = res_fpca_s.coef

        res_fpca_gam = horizontal_pca(self.Ngamma.T, time)
        self.fpca_gam = res_fpca_gam
        c_gam = res_fpca_gam.coef

        res_fpca_theta = vertical_pca(self.mtheta, self.Ntheta, time)
        self.fpca_theta = res_fpca_theta
        c_theta = res_fpca_theta.coef

        res_pca_Z0 = pca_init_position(self.NQ0, self.Nx0)
        self.pca_Z0 = res_pca_Z0
        c_Z0 = res_pca_Z0.coef 

        res_pca_LT = pca(self.NL, self.NT)
        self.pca_LT = res_pca_LT
        c_LT = res_pca_LT.coef

        self.mat_coef = np.concatenate((c_s,c_gam,c_theta,c_Z0,c_LT), axis=1).T

        self.cov = np.cov(self.mat_coef)
        
    
    def generate_sample(self, num, smoothing=False, nb_basis=10, smoothing_param=0.0):
        self.num = num
        array_mean = np.zeros(len(self.mat_coef))
        # array_mean[-1] = self.m_L

        random_coef = np.random.multivariate_normal(array_mean, self.cov, size=num)
        self.random_coef = random_coef

        no = np.cumsum([0, self.fpca_s.no, self.fpca_gam.no, self.fpca_theta.no, self.pca_Z0.no, self.pca_LT.no])

        r_s_scale = inv_horizontal_pca(random_coef[:,no[0]:no[1]], self.fpca_s.U, self.fpca_s.psi_mu)
        r_s_scale = r_s_scale.T
        r_gamma = inv_horizontal_pca(random_coef[:,no[1]:no[2]], self.fpca_gam.U, self.fpca_gam.psi_mu) # n+1 x num
        r_gamma = r_gamma.T 
        r_theta = inv_vertical_pca(random_coef[:,no[2]:no[3]], self.fpca_theta.U, self.fpca_theta.mflatten_theta) # num x n x 2
        r_Z0 = inv_pca_init_position(self.pca_Z0.tpca, random_coef[:,no[3]:no[4]])
        r_LT = inv_pca(random_coef[:,no[4]:no[5]], self.pca_LT.U, self.pca_LT.mu)

        time = np.linspace(0,1,self.n)
        time_bis = np.linspace(0,1,r_gamma[0].shape[0])

        r_theta_warp = np.zeros((num,r_gamma[0].shape[0],2))
        r_theta_warp_func = np.empty((num), dtype=object)
        for k in range(num):
            if self.s_smooth == 0.0:
                time0 = (time[-1] - time[0]) * r_gamma[k] + time[0]
                grad = np.gradient(r_gamma[k], 1 / float(len(r_gamma[k]) - 1))
                r_theta_warp[k,:,0] = np.interp(time0, time, r_theta[k,:,0]) * grad
                r_theta_warp[k,:,1] = np.interp(time0, time, r_theta[k,:,1]) * grad
            else:
                time_bis = np.linspace(0,1,len(r_gamma[k]))
                tmp_spline = UnivariateSpline(time_bis, r_gamma[k], s=self.s_smooth)
                gam = tmp_spline(time_bis)
                gam_der = tmp_spline(time_bis, 1)
                r_theta_warp[k,:,0] = np.interp(gam, time, r_theta[k,:,0]) * gam_der
                r_theta_warp[k,:,1] = np.interp(gam, time, r_theta[k,:,1]) * gam_der

            # ajouter une étape de smoothing ici éventuellement
            # if smoothing:
            #     pass
            # else:  
            #     r_theta_warp_func[k] = lambda x: (interp1d(time_bis, r_theta_warp[k].T)(x)).T

        r_L = r_LT[:,0]
        r_T = r_LT[:,1]
        r_s = np.zeros(r_s_scale.shape)
        r_time = np.zeros((num, self.n))
        for k in range(num):
            r_s[k] = r_s_scale[k]*r_L[k]
            r_time[k] = time*r_T[k]

        rand_Z = np.zeros((num,self.n+1,4,4))
        rand_Q = np.zeros((num,self.n+1,3,3))
        rand_X_scale = np.zeros((num,self.n+1,3))
        rand_X = np.zeros((num,self.n+1,3))
       
        if self.init=='mean':
            so3 = SpecialOrthogonal(n=3)
            for k in range(num):
                Q = solve_FrenetSerret_ODE_SO(theta=lambda x: (interp1d(time_bis, r_theta_warp[k].T)(x)).T, t_eval=r_s_scale[k], Q0=np.eye(3))
                print(Q.shape)
                Q = so3.projection(Q)
                print(Q.shape)
                Q_mean = FrechetMean(metric=so3.metric).fit(Q).estimate_
                init_Q = r_Z0[k,:3,:3]@Q_mean.T
                init_Z = np.eye(4)
                init_Z[:3,:3] = init_Q
                Z_random =  solve_FrenetSerret_ODE_SE(theta=lambda x: (interp1d(time_bis, r_theta_warp[k].T)(x)).T, t_eval=r_s_scale[k], Z0=init_Z)
                rand_Z[k] = Z_random
                rand_Q[k] = Z_random[:,:3,:3]
                rand_X_scale[k] = Z_random[:,:3,3] + r_Z0[k,:3,3]
                rand_X[k] = rand_X_scale[k]*r_L[k]

        elif self.init=='init':
            for k in range(num):
                Z = solve_FrenetSerret_ODE_SE(theta=lambda x: (interp1d(time_bis, r_theta_warp[k].T)(x)).T, t_eval=r_s_scale[k], Z0=r_Z0[k])
                rand_Z[k] = Z
                rand_Q[k] = Z[:,:3,:3]
                rand_X_scale[k] = Z[:,:3,3]
                rand_X[k] = rand_X_scale[k]*r_L[k]
        else: 
            raise Exception("Not implemented")
        
        res_samples = collections.namedtuple('res_samples', ['rand_Z', 'rand_X', 'rand_Q', 'rand_X_scale', 'rand_lengths', 'rand_time', 'rand_arclengths', 'rand_theta', 'rand_gam', 'rand_Z0', 'rand_theta_warp', 'rand_theta_warp_func'])
        out = res_samples(rand_Z, rand_X, rand_Q, rand_X_scale, r_L, r_time, r_s_scale, r_theta, r_gamma, r_Z0, r_theta_warp, r_theta_warp_func)

        return out
    


def pca(list_L, list_T):

    N = len(list_L)
    data = np.stack((list_L, list_T)) # shape 2 x N
    mean_data = np.mean(data, axis=1)
    data_cent = (data.T - mean_data).T
    K = np.cov(data)
    U, s, V = np.linalg.svd(K)
    no = len(np.where(np.cumsum(s)/np.sum(s)<0.9)[0]) + 1

    c = np.zeros((N, no))
    for k in range(0, no):
        for l in range(0, N):
            c[l, k] = sum(data_cent[:,l] * U[:, k])

    res_pca = collections.namedtuple('PCA', ['U', 'coef', 'latent', 'no', 'mu'])
    out = res_pca(U[:,0:no], c, s[0:no], no, mean_data)
    return out 


def inv_pca(coef, U, mu):
    return mu + (U @ coef.T).T
    

def vertical_pca(mtheta, Ntheta, grid, stds = np.arange(-1, 2)):
    N = len(Ntheta)
    flat_mtheta = mtheta(grid).flatten(order='F')
    flat_theta_aligned = np.zeros((N, len(flat_mtheta)))
    for i in range(N):
        flat_theta_aligned[i] = Ntheta[i].flatten(order='F')
    
    Nstd = stds.shape[0]

    K = np.cov(flat_theta_aligned.T)
    U, s, V = np.linalg.svd(K)

    no = len(np.where(np.cumsum(s)/np.sum(s)<0.9)[0]) + 1
    
    stdS = np.sqrt(s)
       
    theta_pca = np.ndarray(shape=(len(flat_mtheta), Nstd, no), dtype=float)
    for k in range(0, no):
        for l in range(0, Nstd):
            theta_pca[:, l, k] = flat_mtheta + stds[l] * stdS[k] * U[:, k]

    c = np.zeros((N, no))
    for k in range(0, no):
        for l in range(0, N):
            c[l, k] = sum(((flat_theta_aligned.T)[:, l] - flat_mtheta.T) * U[:, k])

    res_fpca = collections.namedtuple('res_fPCA', ['theta_pca', 'U', 'coef', 'latent', 'no', 'stds', 'mflatten_theta'])
    out = res_fpca(theta_pca, U[:,0:no], c, s[0:no], no, stds, flat_mtheta)

    return out



def inv_vertical_pca(coef, U, mu):

    v = U @ coef.T
    flat_theta = mu + v.T

    theta = np.zeros((flat_theta.shape[0], int(flat_theta.shape[1]/2), 2))
    theta[:,:,0] = flat_theta[:,:int(flat_theta.shape[1]/2)]
    theta[:,:,1] = flat_theta[:,int(flat_theta.shape[1]/2):]

    return theta 



def multivariate_horizontal_pca(list_tab_gam, time, stds = np.arange(-1, 2)):
    """ Joint PCA of multiple gam """
    pass 


def horizontal_pca(gam, time, stds = np.arange(-1, 2)):
    """
    This function calculates horizontal functional principal component analysis on aligned data

    :param stds: number of standard deviations along gedoesic to compute (default = -1,0,1)
    :type no: int

    """
    # Calculate Shooting Vectors
    mu, gam_mu, psi, vec = fs.utility_functions.SqrtMean(gam)
    TT = time.shape[0]

    # TFPCA
    K = np.cov(vec)

    U, s, V = np.linalg.svd(K)

    # explained_variance_ = (s ** 2) / (TT - 1)
    # total_var = explained_variance_.sum()
    # explained_variance_ratio_ = explained_variance_ / total_var
    # no = len(np.where(np.cumsum(explained_variance_ratio_)<0.9)[0]) + 1

    no = len(np.where(np.cumsum(s)/np.sum(s)<0.9)[0]) + 1
    vm = vec.mean(axis=1)

    gam_pca = np.ndarray(shape=(stds.shape[0], mu.shape[0], no), dtype=float)
    psi_pca = np.ndarray(shape=(stds.shape[0], mu.shape[0], no), dtype=float)
    for j in range(0, no):
        cnt = 0
        for k in stds:
            v = k * np.sqrt(s[j]) * U[:, j]
            vn = np.linalg.norm(v) / np.sqrt(TT)
            if vn < 0.0001:
                psi_pca[cnt, :, j] = mu
            else:
                psi_pca[cnt, :, j] = np.cos(vn) * mu + np.sin(vn) * v / vn

            tmp = cumtrapz(psi_pca[cnt, :, j] * psi_pca[cnt, :, j], np.linspace(0,1,TT), initial=0)
            gam_pca[cnt, :, j] = (tmp - tmp[0]) / (tmp[-1] - tmp[0])
            cnt += 1

    N2 = gam.shape[1]
    c = np.zeros((N2,no))
    for k in range(0,no):
        for i in range(0,N2):
            c[i,k] = np.sum(np.dot(vec[:,i]-vm,U[:,k]))


    res_fpca = collections.namedtuple('res_fPCA', ['gam_pca', 'psi_pca', 'U', 'coef', 'latent', 'gam_mu', 'psi_mu', 'vec', 'no', 'stds'])
    out = res_fpca(gam_pca, psi_pca, U[:,0:no], c, s[0:no], gam_mu, mu, vec, no, stds)

    return out


def inv_horizontal_pca(coef, U, mu):

    v = U @ coef.T
    num = coef.shape[0]
    TT = U.shape[0] + 1
    gam = np.zeros((TT, num))

    for k in range(0, num):
        vn = np.linalg.norm(v[:,k]) / np.sqrt(TT)
        psi = np.cos(vn) * mu + np.sin(vn) * v[:,k] / vn
        tmp = np.zeros(TT)
        tmp[1:TT] = np.cumsum(psi * psi) / TT
        gam[:, k] = (tmp - tmp[0]) / (tmp[-1] - tmp[0])

    return gam



def pca_init_position(Q0, X0):

    d = Q0[0].shape[0]
    Z0 = np.zeros((Q0.shape[0], d+1, d+1))
    Z0[:,:d,:d] = Q0
    Z0[:,:d,d] = X0
    Z0[:,d,d] = np.ones(Q0.shape[0])
    se = SpecialEuclidean(n=d)
    mean = FrechetMean(metric=se.metric)
    mean.fit(Z0)
    mean_Z0 = se.projection(mean.estimate_)
    # metric = se.bi_invariant_metric
    # tpca = TangentPCA(metric=se.metric, n_components='mle')
    tpca = TangentPCA(metric=se.metric, n_components=3)
    tpca = tpca.fit(Z0, base_point=mean_Z0)
    tangent_projected_data = tpca.transform(Z0)

    res_rpca= collections.namedtuple('res_rPCA', ['tpca', 'U', 'coef', 'latent', 'mu', 'no'])
    out = res_rpca(tpca, tpca.components_, tangent_projected_data, tpca.singular_values_, mean_Z0, tpca.n_components_)

    return out


def inv_pca_init_position(tpca, coef):
    res = tpca.inverse_transform(coef)
    return res