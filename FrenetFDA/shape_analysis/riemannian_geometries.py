import numpy as np
import fdasrsf as fs
from scipy import interpolate
from scipy.integrate import cumtrapz, solve_ivp
import optimum_reparamN2 as orN2
from FrenetFDA.utils.Frenet_Serret_utils import solve_FrenetSerret_ODE_SO, solve_FrenetSerret_ODE_SE
import FrenetFDA.utils.visualization as visu
from FrenetFDA.utils.smoothing_utils import VectorBSplineSmoothing
from joblib import Parallel, delayed
import time as tt

class SRVF:

    def __init__(self, dim):
        """ 
            SRVF: square root velocity functions

            Inputs:
                - dim: int 
                dimension of the space considered
        """
        self.dim = dim
    
    def align(self, x0, x1):
        """
            Compute the optimal reparameterization under the SRVF framework of x1 to match x0. 

            Inputs:
            - x0, x1: numpy array of shape (N, dim).
        """
        x1_bis, q1_bis, R_opt, h_opt = fs.curve_functions.find_rotation_and_seed_coord(x0.T, x1.T, rotation=True)
        return x1_bis, q1_bis, R_opt, h_opt

    def geodesic(self, x0, x1, k, align=True):
        """
            Compute the geodesic path under the SRVF framework bewteen x0 and x1 with k steps. 

            Inputs:
            - x0, x1: numpy array of shape (N, dim).
            - k: int. Number of steps in the geodesic path.
            - align: if True the optimal reparameterization is computed before computed the geodesic. 
        """
        N = len(x0)
        q0, len0, lenq0 = fs.curve_functions.curve_to_q(x0.T)
        if align:
            x1_align, q1_align, R_opt, h_opt = self.align(x0, x1)
            x1_align = x1_align.T
        else:
            R_opt = np.eye(self.dim)
            h_opt = np.linspace(0,1,N)
            x1_align = x1
            q1_align, len1, lenq1 = fs.curve_functions.curve_to_q(x1.T)
        dist = Sphere.dist(q0, q1_align)

        path_q = np.zeros((k, self.dim, N))
        path_x = np.zeros((k, N, self.dim))
        for tau in range(0, k):
            if tau == 0:
                tau1 = 0
            else:
                tau1 = tau / (k - 1.)
            s = dist * tau1
            if dist > 0:
                path_q[tau, :, :] = (np.sin(dist-s)*q0+np.sin(s)*q1_align)/np.sin(dist)
            elif dist == 0:
                path_q[tau, :, :] = (1 - tau1)*q0 + (tau1)*q1_align
            else:
                raise Exception("geod_sphere computed a negative distance")
            path_x[tau, :, :] =  fs.curve_functions.q_to_curve(path_q[tau, :, :]).T

        return dist, path_x, path_q, R_opt, h_opt
    
    def deform(self, x, gam, T=None):
        """
            Compute the deformed curve in the SRVF registration problem. 

            Inputs:
            - x: numpy array of shape (N,dim); Euclidean curve
            - gam: numpy array of shape (N,); warping function
            
        """
        N = x.shape[0]
        time = np.linspace(0,1,N)
        if T is None:
            dot_x = np.gradient(x, 1. / (N - 1))
            dot_x = dot_x[0]
            T = np.zeros((N,self.dim))
            for i in range(N):
                L = np.linalg.norm(dot_x[i,:])
                if L > 0.0001:
                    T[i] = dot_x[i] / L
                else:
                    T[i] = dot_x[i] * 0.0001
        gam_warp_fct = interpolate.UnivariateSpline(time, gam, s=0.0001)
        gam_smooth = lambda t: (gam_warp_fct(t) - gam_warp_fct(time).min())/ (gam_warp_fct(time).max() - gam_warp_fct(time).min())
        T_warp = np.sqrt(gam_warp_fct(time,1))*(interpolate.interp1d(np.linspace(0,1,N), T.T)(gam_smooth(time)))
        x_def = cumtrapz(T_warp, np.linspace(0,1,N), initial=0).T
        return x_def
    
    def dist(self, x0, x1, align=True):
        """
            Compute the SRVF distance bewteen x0 and x1. 

            Inputs:
            - x0, x1: numpy array of shape (N, dim).
            - align: if True the optimal reparameterization is computed before computed the distance. 
        """
        N = len(x0)
        q0, len0, lenq0 = fs.curve_functions.curve_to_q(x0.T)
        if align:
            x1_align, q1_align, R_opt, h_opt = self.align(x0, x1)
            x1_align = x1_align.T
        else:
            R_opt = np.eye(self.dim)
            h_opt = np.linspace(0,1,N)
            x1_align = x1
            q1_align, len1, lenq1 = fs.curve_functions.curve_to_q(x1.T)
        dist = Sphere.dist(q0, q1_align)
        return dist
    
    def karcher_mean(self, arr_x, new_N=None):
        """
            Compute the Karcher mean under the SRVF framework. 
            
            Input:
            - arr_x: numpy array of shape (K, N, dim) of K curves in dimension 'dim' with N samples points 
              Set of Euclidean curves.

        """
        if new_N is None:
            N = arr_x[0].shape[0]
        else:
            N = new_N
        beta = np.zeros((self.dim, arr_x[0].shape[0], len(arr_x)))
        for k in range(len(arr_x)):
            beta[:,:,k] = arr_x[k].T
        obj = fs.curve_stats.fdacurve(beta, N=N)
        obj.karcher_mean()
        return obj.beta_mean.T



class SRC:

    def __init__(self, dim):
        """ 
        SRC Transforms: square root curvature transforms

        Inputs:
            - dim: int 
              dimension of the space considered
        """
        self.dim = dim

    def warp(self, c, gam, smooth=True):
        """
            Compute the group action of a warping function 'gam' of the SRC 'c'
        """
        time = np.linspace(0,1,len(gam))
        if smooth:
            gam_warp_fct = interpolate.UnivariateSpline(time, gam, s=0.0001)
            gam_smooth = lambda t: (gam_warp_fct(t) - gam_warp_fct(time).min())/ (gam_warp_fct(time).max() - gam_warp_fct(time).min())
            c_warp_fct = lambda t: c(gam_smooth(t))*np.sqrt(gam_warp_fct(t,1))
        else:
            g = gam
            g_dev = np.gradient(gam, 1. / (len(gam)-1))
            c_warp = c(g)*np.sqrt(g_dev)
            c_warp_fct = interpolate.interp1d(time, c_warp)
        return c_warp_fct
    
    def align(self, c0, c1, time, lam=1, smooth=False):
        """
            Compute the optimal alignment between two SRC representations.

            Inputs:
            - c0, c1: functions such that c0(t) is a numpy array of shape (dim-1); the two SRCs
            - time: numpy array of shape (N,); grid of time points bewteen 0 and 1.
            - lam: float; ponderation coefficient in SRC distance.
            - smooth: if True an additional smoothing of the optimal warping function is made.
        """
        if np.linalg.norm(c0(time)-c1(time),'fro') > 0.0001:
            gam = orN2.coptimum_reparam_curve(np.ascontiguousarray(c0(time)), time, np.ascontiguousarray(c1(time)), lam)
            c1_align = self.warp(c1, gam, smooth)
        else:
            gam = time
            c1_align = c1
        return c1_align, gam
    
    def geodesic(self, theta0, theta1, s0, s1, time, k, align=True, smooth=True, lam=1, arr_Q0=None):
        """
            Compute the geodesic path under the SRC framework.

            Inputs:
            - theta0, theta1: functions such that theta_i(t) is a numpy array of shape (dim-1); Frenet curvatures of the two curves from which we want to compute the geodesic.
            - s0, s1: function on [0,1]; Arc-length function of x0 and x1.
            - time: numpy array of shape (N,); grid of N time points between 0 and 1.
            - k: int; number of steps along the geodesic. 
            - lam: float; ponderation coefficient in SRC distance.
            - smooth: if True an additional smoothing of the optimal warping function is made.
            - align: if True the optimal reparameterization is computed before computed the geodesic. 
            - arr_Q0: numpy array of shape (k, dim, dim); possibility to use a list of initial rotation to reconstruct the curves along the geodesic. 
        """
        c0_theta = lambda t: theta0(t)/np.sqrt(np.linalg.norm(theta0(t)))
        c1_theta = lambda t: theta1(t)/np.sqrt(np.linalg.norm(theta1(t)))
        if align:
            _, gamma_opt = self.align(c0_theta, c1_theta, time, lam=lam, smooth=smooth)
            tmp_spline = interpolate.UnivariateSpline(time, gamma_opt, s=0.0001)
            gam_smooth = lambda t: (tmp_spline(t) - tmp_spline(time).min())/ (tmp_spline(time).max() - tmp_spline(time).min())
            c1_theta_align = lambda t: c1_theta(gam_smooth(t))*np.sqrt(tmp_spline(t,1))
            s1_h = gam_smooth(s0(time))
        else:
            gamma_opt = time
            s1_h = s1(time)
            c1_theta_align = c1_theta

        dist_gam, path_arc_s, path_psi = Diff.geodesic(s0(time), s1_h, k)

        f_s0_dot = interpolate.UnivariateSpline(time, s0(time), s=0.00001)
        s0_dot = lambda t: f_s0_dot(t, 1)

        dist = np.linalg.norm(np.sqrt(s0_dot(time))*(c0_theta(s0(time))-c1_theta_align(s0(time)))) + lam*dist_gam

        path_theta = np.zeros((k, len(time), self.dim-1))
        path_Q = np.zeros((k, len(time), self.dim, self.dim))
        path_curves = np.zeros((k, len(time), self.dim))
        for tau in range(0, k):
            if tau == 0:
                tau1 = 0
            else:
                tau1 = tau / (k - 1.)
            theta_tau = lambda s: ((1-tau1)*c0_theta(s) + tau1*c1_theta_align(s))*np.linalg.norm(((1-tau1)*c0_theta(s) + tau1*c1_theta_align(s)))
            sdot_theta_tau = lambda t: theta_tau(s0(t))*s0_dot(t)
            for i in range(len(time)):
                path_theta[tau, i, :] = sdot_theta_tau(time[i])/(path_psi[tau][i]**2)

            if arr_Q0 is None:
                Q = solve_FrenetSerret_ODE_SO(sdot_theta_tau, time)
            else:
                Q = solve_FrenetSerret_ODE_SO(sdot_theta_tau, time, Q0=arr_Q0[tau])
            path_curves[tau] = cumtrapz(Q[:,:,0].T, path_arc_s[tau], initial=0).T
            path_Q[tau] = Q

        return dist, path_curves, path_Q, path_theta, path_arc_s, path_psi, gamma_opt
    

    def deform(self, theta, gam):
        """
            Compute the deformed curve in the SRC registration problem. 

            Inputs:
            - theta: function such that theta(t) is a numpy array of shape (dim-1); Frenet curvatures
            - gam: numpy array of shape (N,); warping function
            
        """ 
        time = np.linspace(0,1,len(gam))
        gam_warp_fct = interpolate.UnivariateSpline(time, gam, s=0.0001)
        gam_smooth = lambda t: (gam_warp_fct(t) - gam_warp_fct(time).min())/ (gam_warp_fct(time).max() - gam_warp_fct(time).min())
        theta_warp = lambda t: theta(gam_smooth(t))*gam_warp_fct(t,1)
        time = np.linspace(0,1,len(gam))
        Q = solve_FrenetSerret_ODE_SO(theta_warp, time)
        X = cumtrapz(Q[:,:,0].T, time, initial=0).T
        return X
    
    def dist(self, theta0, theta1, s0, s1, time, align=True, smooth=True, lam=1):
        """
            Compute the geodesic distance under the SRC framework.

            Inputs:
            - theta0, theta1: functions such that theta_i(t) is a numpy array of shape (dim-1); Frenet curvatures of the two curves from which we want to compute the geodesic distance.
            - s0, s1: function on [0,1]; Arc-length function of x0 and x1.
            - time: numpy array of shape (N,); grid of N time points between 0 and 1.
            - lam: float; ponderation coefficient in SRC distance.
            - smooth: if True an additional smoothing of the optimal warping function is made.
            - align: if True the optimal reparameterization is computed before computed the geodesic. 
        """
        c0_theta = lambda t: theta0(t)/np.sqrt(np.linalg.norm(theta0(t)))
        c1_theta = lambda t: theta1(t)/np.sqrt(np.linalg.norm(theta1(t)))
        if align:
            _, gamma_opt = self.align(c0_theta, c1_theta, time, lam=lam, smooth=smooth)
            gamma_opt = (gamma_opt - gamma_opt.min())/(gamma_opt.max() - gamma_opt.min())
            binsize = np.mean(np.diff(time))
            psi_gamma_opt = np.sqrt(np.gradient(gamma_opt,binsize))
            s1_h = np.interp(s0, time, gamma_opt)
            s1_h = (s1_h - s1_h.min())/(s1_h.max() - s1_h.min())
            c1_theta_align_s0 = c1_theta(s1_h)*psi_gamma_opt
        else:
            gamma_opt = time
            s1_h = s1
            c1_theta_align_s0 = c1_theta(s0)
        
        binsize = np.mean(np.diff(time))
        psi0 = np.sqrt(np.gradient(s0,binsize))
        psi1 = np.sqrt(np.gradient(s1_h,binsize))
        dist_gam = Sphere.dist(psi0, psi1)
        dist = np.linalg.norm(psi0*(c0_theta(s0)-c1_theta_align_s0)) + lam*dist_gam    
        return dist


    def dist_bis(self, theta0, theta1, s0, s1, time, lam=1):
        """
            Compute the geodesic distance under the SRC framework.

            Inputs:
            - theta0, theta1: functions such that theta_i(t) is a numpy array of shape (dim-1); Frenet curvatures of the two curves from which we want to compute the geodesic distance.
            - s0, s1: function on [0,1]; Arc-length function of x0 and x1.
            - time: numpy array of shape (N,); grid of N time points between 0 and 1.
            - lam: float; ponderation coefficient in SRC distance.
            - smooth: if True an additional smoothing of the optimal warping function is made.
            - align: if True the optimal reparameterization is computed before computed the geodesic. 
        """
        T = len(time)
        c0 = np.zeros((len(s0),self.dim-1))
        c1 = np.zeros((len(s1),self.dim-1))
        for j in range(T):
            c0[j,:] = theta0(s0[j])/np.sqrt(np.linalg.norm(theta0(s0[j])))
            c1[j,:] = theta1(s1[j])/np.sqrt(np.linalg.norm(theta1(s1[j])))
        h_opt = orN2.coptimum_reparam_curve(np.ascontiguousarray(c0.T), time, np.ascontiguousarray(c1.T), lam)
        h_opt = (h_opt - h_opt.min())/(h_opt.max() - h_opt.min())
        grad_h_opt = np.gradient(h_opt,time)
        s1_h = np.interp(h_opt, time, s1)
        c1_h = np.zeros(c1.shape)
        for j in range(T):
            c1_h[j] = np.sqrt(grad_h_opt[j])*theta1(s1_h[j])/np.sqrt(np.linalg.norm(theta1(s1_h[j])))

        psi0 = np.sqrt(np.gradient(s0,time))
        psi1 = np.sqrt(np.gradient(s1_h,time))
        dist_gam = Sphere.dist(psi0, psi1)
        dist = np.linalg.norm((c0-c1_h)) + dist_gam  
        return dist
    
    
    def karcher_mean_old(self, arr_theta, arr_arc_s, tol, max_iter, lam=1):
        """
            Karcher mean under the square-root curvature transform framework. 

            Input:
            - arr_theta: array of K Frenet curvatures functions such that arr_theta[k](s) is a numpy array of size (dim-1) (the number of Frenet curvatures)
              Set of Frenet curvature functions of the arc-length parameter (not of the time) of the Euclidean curves considered for computing the mean.
            - arr_arc_s: numpy array of size (K,N) of the K arc-length functions of the considered curves, with N samples points. 
            - tol: float 
              tolerance for the difference between iterative error. If |error_k - error_{k-1}| < tol, the iteration is stop. 
            - max_iter: int
              Number of maximum iterations to find the optimal mean.
            - lam: float
              Parameter used to add a ponteration in the SRC distance. 

        """
        n = len(arr_theta)
        print('Computing Karcher Mean of %d curves in SRC space.. \n' % (n))
        T = len(arr_arc_s[0])
        time = np.linspace(0,1,T)
        binsize = np.mean(np.diff(time))

        arr_c = np.zeros((n,self.dim-1,T))
        arr_psi = np.zeros(arr_arc_s.shape)
        for i in range(n):
            arr_psi[i] = np.sqrt(np.gradient(arr_arc_s[i],binsize))
            for j in range(T):
                arr_c[i,:,j] = arr_psi[i,j]*arr_theta[i](arr_arc_s[i,j])/np.sqrt(np.linalg.norm(arr_theta[i](arr_arc_s[i,j])))
        mean_c = np.mean(arr_c, axis=0)
        mean_psi = np.mean(arr_psi, axis=0)
        
        visu.plot_array_2D(time, arr_c[:,0,:], 'c 0')
        visu.plot_array_2D(time, arr_c[:,1,:], 'c 1')

        dist_arr = np.zeros(n)
        for i in range(n):
            dist_arr[i] = np.linalg.norm(mean_c - arr_c[i]) + lam*Sphere.dist(mean_psi, arr_psi[i]) 
        ind = np.argmin(dist_arr)
        
        temp_mean_psi = arr_psi[ind]
        temp_mean_c = arr_c[ind]
        temp_error = np.linalg.norm((mean_c - temp_mean_c)) + lam*Sphere.dist(mean_psi, temp_mean_psi) 
        up_err = temp_error
        k = 0
        arr_h = []
        print('Iteration ', k, '/', max_iter, ': error ', temp_error)
        while up_err > tol and k < max_iter:
            arr_c_align = np.zeros((n,self.dim-1,T))
            arr_arc_align = np.zeros((n,T))
            arr_psi_align = np.zeros((n,T))
            for i in range(n):
                if np.linalg.norm(temp_mean_c - arr_c[i],'fro') > 0.0001:
                    h_opt = orN2.coptimum_reparam_curve(np.ascontiguousarray(temp_mean_c), time, np.ascontiguousarray(arr_c[i]), lam)
                else:
                    h_opt = time
                h_opt = (h_opt - h_opt.min())/(h_opt.max() - h_opt.min())
                arr_h.append(h_opt)
                si_h = np.interp(h_opt, time, arr_arc_s[i])
                arr_arc_align[i] = (si_h - si_h.min())/(si_h.max() - si_h.min())
                arr_psi_align[i] = np.sqrt(np.gradient(arr_arc_align[i],binsize))
                for j in range(T):
                    arr_c_align[i,:,j] = arr_psi_align[i,j]*arr_theta[i](arr_arc_align[i,j])/np.sqrt(np.linalg.norm(arr_theta[i](arr_arc_align[i,j])))

            mean_c = np.mean(arr_c_align, axis=0)
            mean_psi = np.mean(arr_psi_align, axis=0)
            error = np.linalg.norm((mean_c - temp_mean_c)) + lam*Sphere.dist(mean_psi, temp_mean_psi) 
            up_err = abs(temp_error - error)
            temp_error = error
            k += 1
            print('Iteration ', k, '/', max_iter, ': error ', temp_error)
            temp_mean_psi = mean_psi
            temp_mean_c = mean_c
        
        print('Number of iterations', k, '\n')

        visu.plot_2D(time, mean_psi)
        visu.plot_array_2D(time, arr_c_align[:,0,:], 'c 0')
        visu.plot_array_2D(time, arr_c_align[:,1,:], 'c 1')
        
        mean_s = cumtrapz(temp_mean_psi*temp_mean_psi, time, initial=0)
        mean_s = (mean_s - mean_s.min())/(mean_s.max() - mean_s.min())
        mean_theta = np.zeros(temp_mean_c.shape)
        for j in range(T):
            x = mean_c[:,j]/mean_psi[j]
            mean_theta[:,j] = x*np.linalg.norm(x)

        theta = lambda t: interpolate.griddata(mean_s, mean_theta.T, t, method='cubic')
        Q = solve_FrenetSerret_ODE_SO(theta, mean_s)
        mean_x = cumtrapz(Q[:,:,0].T, mean_s, initial=0).T

        return mean_x, theta, mean_s, mean_c, arr_h

    def karcher_mean(self, arr_theta, arr_arc_s, tol, max_iter, lam=1, parallel=False):
        """
            Karcher mean under the square-root curvature transform framework. 

            Input:
            - arr_theta: array of K Frenet curvatures functions such that arr_theta[k](s) is a numpy array of size (dim-1) (the number of Frenet curvatures)
              Set of Frenet curvature functions of the arc-length parameter (not of the time) of the Euclidean curves considered for computing the mean.
            - arr_arc_s: numpy array of size (K,N) of the K arc-length functions of the considered curves, with N samples points. 
            - tol: float 
              tolerance for the difference between iterative error. If |error_k - error_{k-1}| < tol, the iteration is stop. 
            - max_iter: int
              Number of maximum iterations to find the optimal mean.
            - lam: float
              Parameter used to add a ponteration in the SRC distance. 

        """
        N_samples = len(arr_theta)
        print("Computing Karcher Mean of %d curves in SRC space.. \n" % (N_samples))
        T = len(arr_arc_s[0])
        time = np.linspace(0,1,T)

        arr_src_theta = np.zeros((N_samples,T,self.dim-1))
        for i in range(N_samples):
            for j in range(T):
                arr_src_theta[i,j,:] = arr_theta[i](time[j])/np.sqrt(np.linalg.norm(arr_theta[i](time[j])))
        mean_src_theta = np.mean(arr_src_theta, axis=0)

        dist_arr = np.zeros(N_samples)
        for i in range(N_samples):
            dist_arr[i] = np.linalg.norm(mean_src_theta - arr_src_theta[i])
        ind = np.argmin(dist_arr)

        temp_mean_src_theta = arr_src_theta[ind]
        temp_error = np.linalg.norm((mean_src_theta - temp_mean_src_theta)) 
        up_err = temp_error
        k = 0

        arr_gam = np.array([np.linspace(0,1,T) for i in range(N_samples)])
        
        print("Aligning %d functions in maximum %d iterations..."
            % (N_samples, max_iter))
        while up_err > tol and k < max_iter:
            arr_src_align = np.zeros((N_samples, T, self.dim-1))
            arr_gam = np.zeros((N_samples, T))

            if parallel:

                def to_run(m_src, src, grid_ptn, param):
                    if np.linalg.norm(m_src - src,'fro') > 0.0001:
                        gam = orN2.coptimum_reparam_curve(np.ascontiguousarray(m_src.T), grid_ptn, np.ascontiguousarray(src.T), param)
                    else:
                        gam = grid_ptn
                    return gam
                
                out = Parallel(n_jobs=-1)(delayed(to_run)(temp_mean_src_theta, arr_src_theta[i], time, lam) for i in range(N_samples))
                gam_t = np.array(out)
                for i in range(N_samples):
                    arr_gam[i] = (gam_t[i] - gam_t[i][0])/(gam_t[i][-1] - gam_t[i][0])
                    for j in range(0, self.dim-1):
                        time0 = (time[-1] - time[0]) * arr_gam[i] + time[0]
                        arr_src_align[i,:,j] = np.interp(time0, time, arr_src_theta[i, :, j]) * np.sqrt(np.gradient(arr_gam[i], time))

            else:
                for i in range(N_samples):
                    if np.linalg.norm(temp_mean_src_theta - arr_src_theta[i],'fro') > 0.0001:
                        gam = orN2.coptimum_reparam_curve(np.ascontiguousarray(temp_mean_src_theta.T), time, np.ascontiguousarray(arr_src_theta[i].T), lam)
                    else:
                        gam = time
                    gam = (gam - gam.min())/(gam.max() - gam.min())
                    arr_gam[i] = gam

                    for j in range(0, self.dim-1):
                        time0 = (time[-1] - time[0]) * gam + time[0]
                        arr_src_align[i,:,j] = np.interp(time0, time, arr_src_theta[i, :, j]) * np.sqrt(np.gradient(gam, time))

            mean_src_theta = np.mean(arr_src_align, axis=0)
            error = np.linalg.norm((mean_src_theta - temp_mean_src_theta))  
            up_err = abs(temp_error - error)
            temp_error = error
            k += 1
            temp_mean_src_theta = mean_src_theta
            
        print("Alignment in %d iterations" % (k))
        
        # mean_s = mean of gam_inv_i circ s_i 
        gami_inv_si = np.zeros(arr_gam.shape)
        for k in range(N_samples):
            gami_inv_si[k] = interpolate.interp1d(arr_gam[k], time)(arr_arc_s[k])
            gami_inv_si[k] = (gami_inv_si[k] - gami_inv_si[k,0])/(gami_inv_si[k,-1] - gami_inv_si[k,0])
        mean_psi, mean_s, psi_tab, vec = fs.utility_functions.SqrtMean(gami_inv_si.T)
        mean_s = (mean_s - mean_s.min())/(mean_s.max() - mean_s.min())

        mean_theta = np.zeros(temp_mean_src_theta.shape)
        for j in range(T):
            mean_theta[j] = mean_src_theta[j]*np.linalg.norm(mean_src_theta[j])

        theta = lambda t: interpolate.griddata(time, mean_theta, t, method='cubic')
        Z = solve_FrenetSerret_ODE_SE(theta, mean_s)
        mean_x = Z[:,:3,3]

        return mean_x, theta, mean_s, mean_src_theta, arr_gam
    

    def karcher_mean_bspline(self, arr_theta_coefs, arr_arc_s, tol, max_iter, nb_basis=None, lam=1, parallel=False, knots=None):
        """
            Karcher mean under the square-root curvature transform framework. 

            Input:
            - arr_theta: array of K Frenet curvatures functions such that arr_theta[k](s) is a numpy array of size (dim-1) (the number of Frenet curvatures)
              Set of Frenet curvature functions of the arc-length parameter (not of the time) of the Euclidean curves considered for computing the mean.
            - arr_arc_s: numpy array of size (K,N) of the K arc-length functions of the considered curves, with N samples points. 
            - tol: float 
              tolerance for the difference between iterative error. If |error_k - error_{k-1}| < tol, the iteration is stop. 
            - max_iter: int
              Number of maximum iterations to find the optimal mean.
            - lam: float
              Parameter used to add a ponteration in the SRC distance. 

        """
        N_samples = len(arr_theta_coefs)

        bspline_decomp = VectorBSplineSmoothing(2, nb_basis, domain_range=(0, 1), order=4, penalization=False, knots=knots)
        
        print("Computing Karcher Mean of %d curves in SRC space.. \n" % (N_samples))
        T = len(arr_arc_s[0])
        time = np.linspace(0,1,T)

        # eps = 1e-03
        arr_src_theta = np.zeros((N_samples,T,self.dim-1))
        for i in range(N_samples):
            theta_i = np.squeeze((bspline_decomp.basis_fct(time).T @ arr_theta_coefs[i]).T)
            for j in range(T):
                arr_src_theta[i,j,:] = theta_i[j]/np.sqrt(np.linalg.norm(theta_i[j]))
                # if arr_src_theta[i,j,0] <= 0:
                #     arr_src_theta[i,j,0] = eps
        mean_src_theta = np.mean(arr_src_theta, axis=0)

        dist_arr = np.zeros(N_samples)
        for i in range(N_samples):
            dist_arr[i] = np.linalg.norm(mean_src_theta - arr_src_theta[i])
        ind = np.argmin(dist_arr)

        temp_mean_src_theta = arr_src_theta[ind]
        temp_error = np.linalg.norm((mean_src_theta - temp_mean_src_theta)) 
        up_err = temp_error
        k = 0

        arr_gam = np.array([np.linspace(0,1,T) for i in range(N_samples)])
        
        # visu.plot_array_2D(time, arr_src_theta[:,:,0], 'curv')
        # visu.plot_array_2D(time, arr_src_theta[:,:,1], 'tors')

        print("Aligning %d functions in maximum %d iterations..."
            % (N_samples, max_iter))
        while up_err > tol and k < max_iter:
            arr_src_align = np.zeros((N_samples, T, self.dim-1))
            arr_gam = np.zeros((N_samples, T))

            if parallel:

                def to_run(m_src, src, grid_ptn, param):
                    if np.linalg.norm(m_src - src,'fro') > 0.0001:
                        gam = orN2.coptimum_reparam_curve(np.ascontiguousarray(m_src.T), grid_ptn, np.ascontiguousarray(src.T), param)
                    else:
                        gam = grid_ptn
                    return gam
                
                out = Parallel(n_jobs=-1)(delayed(to_run)(temp_mean_src_theta, arr_src_theta[i], time, lam) for i in range(N_samples))
                gam_t = np.array(out)
                for i in range(N_samples):
                    arr_gam[i] = (gam_t[i] - gam_t[i][0])/(gam_t[i][-1] - gam_t[i][0])
                    for j in range(0, self.dim-1):
                        time0 = (time[-1] - time[0]) * arr_gam[i] + time[0]
                        arr_src_align[i,:,j] = np.interp(time0, time, arr_src_theta[i, :, j]) * np.sqrt(np.gradient(arr_gam[i], time))

            else:
                for i in range(N_samples):
                    if np.linalg.norm(temp_mean_src_theta - arr_src_theta[i],'fro') > 0.0001:
                        gam = orN2.coptimum_reparam_curve(np.ascontiguousarray(temp_mean_src_theta.T), time, np.ascontiguousarray(arr_src_theta[i].T), lam)
                    else:
                        gam = time
                    gam = (gam - gam.min())/(gam.max() - gam.min())
                    arr_gam[i] = gam

                    for j in range(0, self.dim-1):
                        time0 = (time[-1] - time[0]) * gam + time[0]
                        arr_src_align[i,:,j] = np.interp(time0, time, arr_src_theta[i, :, j]) * np.sqrt(np.gradient(gam, time))

                # visu.plot_array_2D(time, arr_src_align[:,:,0], 'curv')
                # visu.plot_array_2D(time, arr_src_align[:,:,1], 'tors')

            mean_src_theta = np.mean(arr_src_align, axis=0)
            error = np.linalg.norm((mean_src_theta - temp_mean_src_theta))  
            up_err = abs(temp_error - error)
            temp_error = error
            k += 1
            temp_mean_src_theta = mean_src_theta
            
        print("Alignment in %d iterations" % (k))
        
        # mean_s = mean of gam_inv_i circ s_i 
        gami_inv_si = np.zeros(arr_gam.shape)
        for k in range(N_samples):
            gami_inv_si[k] = interpolate.interp1d(arr_gam[k], time)(arr_arc_s[k])
            gami_inv_si[k] = (gami_inv_si[k] - gami_inv_si[k,0])/(gami_inv_si[k,-1] - gami_inv_si[k,0])
        mean_psi, mean_s, psi_tab, vec = fs.utility_functions.SqrtMean(gami_inv_si.T)
        mean_s = (mean_s - mean_s.min())/(mean_s.max() - mean_s.min())

        mean_theta = np.zeros(temp_mean_src_theta.shape)
        for j in range(T):
            mean_theta[j] = mean_src_theta[j]*np.linalg.norm(mean_src_theta[j])

        theta = lambda t: interpolate.griddata(time, mean_theta, t, method='cubic')
        Z = solve_FrenetSerret_ODE_SE(theta, mean_s)
        mean_x = Z[:,:3,3]

        return mean_x, theta, mean_s, mean_src_theta, arr_gam

       



class Frenet_Curvatures:

    def __init__(self, dim):
        """ 
        Frenet curvatures (FC)

        Inputs:
            - dim: int 
              dimension of the space considered
        """
        self.dim = dim

    
    def geodesic(self, theta0, theta1, s0, s1, time, k, arr_Q0=None):
        """
            Compute the geodesic path under the Frenet curvatures framework.

            Inputs:
            - theta0, theta1: functions such that theta_i(t) is a numpy array of shape (dim-1); Frenet curvatures of the two curves from which we want to compute the geodesic.
            - s0, s1: function on [0,1]; Arc-length function of x0 and x1.
            - time: numpy array of shape (N,); grid of N time points between 0 and 1.
            - k: int; number of steps along the geodesic. 
            - arr_Q0: numpy array of shape (k, dim, dim); possibility to use a list of initial rotation to reconstruct the curves along the geodesic. 
        """
        s1_inv = fs.utility_functions.invertGamma(s1)
        h_opt = np.interp(s0, time, s1_inv)
        path_theta = np.zeros((k, len(time), self.dim-1))
        path_theta_fct = np.empty((k), dtype=object)
        path_Q = np.zeros((k, len(time), self.dim, self.dim))
        path_curves = np.zeros((k, len(time), self.dim))

        dist = np.linalg.norm((theta0(time)-theta1(time)))
        for tau in range(0, k):
            if tau == 0:
                tau1 = 0
            else:
                tau1 = tau / (k - 1.)
            theta_tau = lambda t: (1-tau1)*theta0(t) + tau1*theta1(t)
            path_theta_fct[tau] = theta_tau
            path_theta[tau] = theta_tau(time)

            if arr_Q0 is None:
                Q = solve_FrenetSerret_ODE_SO(theta_tau, time)
            else:
                Q = solve_FrenetSerret_ODE_SO(theta_tau, time, Q0=arr_Q0[tau])
            path_curves[tau] = cumtrapz(Q[:,:,0].T, time, initial=0).T
            path_Q[tau] = Q

        return dist, path_curves, path_Q, path_theta, path_theta_fct, h_opt
    

    def dist(self, theta0, theta1, time):
        """
            Compute the geodesic distance under the Frenet curvatures framework.

            Inputs:
            - theta0, theta1: functions such that theta_i(t) is a numpy array of shape (dim-1); Frenet curvatures of the two curves from which we want to compute the geodesic.
            - time: numpy array of shape (N,); grid of N time points between 0 and 1.
        """
        dist = np.linalg.norm((theta0(time)-theta1(time)))
        return dist


    def karcher_mean(self, arr_theta, arr_arc_s):
        """
            Compute the Karcher mean under the Frenet curvatures representation. 

            Input:
            - arr_theta: array of K Frenet curvatures functions such that arr_theta[k](s) is a numpy array of size (dim-1) (the number of Frenet curvatures)
              Set of Frenet curvature functions of the arc-length parameter (not of the time) of the Euclidean curves considered for computing the mean.
            - arr_arc_s: numpy array of size (K,N) of the K arc-length functions of the considered curves, with N samples points. 

        """
        n = len(arr_theta)
        mean_theta = lambda s: np.mean([arr_theta[i](s) for i in range(n)], axis=0)
        psi_mu, gam_mu, psi_arr, vec = fs.utility_functions.SqrtMean(arr_arc_s.T)
        Q = solve_FrenetSerret_ODE_SO(mean_theta, gam_mu)
        mean_x = cumtrapz(Q[:,:,0].T, gam_mu, initial=0).T
        return mean_x, mean_theta, gam_mu
    


    def karcher_mean_bspline(self, arr_theta_coefs, arr_arc_s, nb_basis=None, knots=None):
        """
            Compute the Karcher mean under the Frenet curvatures representation. 

            Input:
            - arr_theta: array of K Frenet curvatures functions such that arr_theta[k](s) is a numpy array of size (dim-1) (the number of Frenet curvatures)
              Set of Frenet curvature functions of the arc-length parameter (not of the time) of the Euclidean curves considered for computing the mean.
            - arr_arc_s: numpy array of size (K,N) of the K arc-length functions of the considered curves, with N samples points. 

        """
        bspline_decomp = VectorBSplineSmoothing(2, nb_basis, domain_range=(0, 1), order=4, penalization=False, knots=knots)
        n = len(arr_theta_coefs)
        mean_theta = lambda s: np.mean([bspline_decomp.evaluate_coefs(arr_theta_coefs[i])(s) for i in range(n)], axis=0)
        psi_mu, gam_mu, psi_arr, vec = fs.utility_functions.SqrtMean(arr_arc_s.T)
        Q = solve_FrenetSerret_ODE_SO(mean_theta, gam_mu)
        mean_x = cumtrapz(Q[:,:,0].T, gam_mu, initial=0).T
        return mean_x, mean_theta, gam_mu
    


class Diff:
    
    @classmethod
    def geodesic(self, gam0, gam1, k):
        """
            Compute the geodesic bewteen two diffeomorphisms in Diff_+([0,1]) in the space Psi([0,1]).

            Inputs:
            - gam0, gam1: numpy arrays of shape (N,); two diffeomorphisms in Diff_+([0,1])
            - k: int; number of steps along the geodesic path.
        """
        N = gam0.shape[0]
        time = np.linspace(0,1,N)
        binsize = np.mean(np.diff(time))
        psi0 = np.sqrt(np.gradient(gam0,binsize))
        psi1 = np.sqrt(np.gradient(gam1,binsize))
        dist = Sphere.dist(psi0, psi1)
        path_psi = np.zeros((k, N))
        path_gam = np.zeros((k, N))
        for tau in range(0, k):
            if tau == 0:
                tau1 = 0
            else:
                tau1 = tau / (k - 1.)
            s = dist * tau1
            if dist > 0:
                path_psi[tau, :] = (np.sin(dist-s)*psi0+np.sin(s)*psi1)/np.sin(dist)
            elif dist == 0:
                path_psi[tau, :] = (1 - tau1)*psi0 + (tau1)*psi1
            else:
                raise Exception("geod_sphere computed a negative distance")
            gam = cumtrapz(path_psi[tau, :]*path_psi[tau, :], time, initial=0)
            path_gam[tau,:] = (gam - gam.min()) / (gam.max() - gam.min())
        return dist, path_gam, path_psi
    
    @classmethod
    def smooth(self, time, gam, lam):
        tmp_spline = interpolate.UnivariateSpline(time, gam, s=lam)
        gam_smooth = tmp_spline(time)
        gam_smooth = (gam_smooth - gam_smooth.min()) / (gam_smooth.max() - gam_smooth.min())
        gam_smooth_dev = tmp_spline(time, 1)
        return gam_smooth, gam_smooth_dev 
    
    @classmethod
    def gamma_to_h(self, gamma, s0, s1):
        time = np.linspace(0,1,len(s0))
        s1_inv = fs.utility_functions.invertGamma(s1)
        gam_s0 = np.interp(s0, time, gamma)
        h = np.interp(gam_s0, time, s1_inv)
        return h

    @classmethod
    def h_to_gamma(self, h, s0, s1):
        time = np.linspace(0,1,len(s0))
        s0_inv = fs.utility_functions.invertGamma(s0)
        s1_h = np.interp(h, time, s1)
        gamma = np.interp(s0_inv, time, s1_h)
        return gamma
    


class Sphere:

    @classmethod
    def dist(self, obj0, obj1):
        """
            Geodesic distance on the sphere. 
        """
        N = obj0.shape[-1]
        val = np.sum(np.sum(obj0 * obj1))/N
        if val > 1:
            if val < 1.001: # assume numerical error
                # import warnings
                # warnings.warn(f"Corrected a numerical error in geod_sphere: rounded {val} to 1")
                val = 1
            else:
                raise Exception(f"innerpod_q2 computed an inner product of {val} which is much greater than 1")
        elif val < -1:
            if val > -1.001: # assume numerical error
                # import warnings
                # warnings.warn(f"Corrected a numerical error in geod_sphere: rounded {val} to -1")
                val = -1
            else:
                raise Exception(f"innerpod_q2 computed an inner product of {val} which is much less than -1")
        dist = np.arccos(val)
        if np.isnan(dist):
            raise Exception("geod_sphere computed a dist value which is NaN")
        return dist
