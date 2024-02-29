import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm, block_diag
from FrenetFDA.utils.Lie_group.SE3_utils import SE3
from FrenetFDA.utils.Lie_group.SO3_utils import SO3
import time as ttime 
# from numba import njit


class IEKFilterSmootherFrenetState():

    """Class for continuous-discrete invariant extended kalman filter for the tracking and smoothing of the Frenet Serret frame on SE(3).

    i.e. we consider elements of the Lie group of rigid transformations SE(n).

    Parameters
    ----------
    n : int
        Dimension for the Lie group SE(n)
    dim_g : int
            Dimension of the Lie Algebra se(n)
    Sigma : matrix of size (dim_g,dim_g)
            Covariance matrix of the white Gaussian noise
    N : matrix of size (n,n)
        Covariance matrix of the noisy observations
    Z : matrix in SE(n), size (n+1,n+1)
        Actual state of the system: (Q  X
                                     0  1)
    Q : matrix in SO(n), size (n,n)
        Actual state of rotation matrix
    X : vector in R^n, size (n,1)
        Actual state of translation vector
    xi : vector of size ((m+1)*dim_g,1)
            Lie Algebraic error and gaussian process noise
    P : matrix of size ((m+1)*dim_g,(m+1)*dim_g)
        Actual state of the covariance
    """

    def __init__(self, n, Gamma, Sigma, theta, Z0 = None, P0 = None, K_pts=3):

        self.n = n # SE(n)
        self.dim_g = int((1/2)*n*(n+1)) # dimension of the Lie Algebra se(n)
        self.Sigma = Sigma # fct of matrix (n-1 x n-1)
        self.Gamma = Gamma 
        # self.xi = np.zeros((self.dim_g))
        self.P0 = np.zeros((self.dim_g,self.dim_g)) if P0 is None else P0
        self.P = self.P0
        self.C = np.eye(self.dim_g)
        self.Q = np.eye(self.n) if Z0 is None else Z0[:self.n,:self.n]
        self.X = np.zeros((self.n)) if Z0 is None else Z0[:self.n,self.n]
        self.Z = np.vstack((np.hstack((self.Q,self.X[:,np.newaxis])),np.hstack((np.zeros((self.n)), np.array((1)))))) if Z0 is None else Z0
        self.U = np.eye(self.dim_g)
        self.list_U = [self.U]
        self.K_pts = K_pts

        self.F = lambda s: -SE3.Ad(np.array([theta(s)[1], 0*s, theta(s)[0], 1+0*s, 0*s, 0*s]))
        self.A = lambda s: -SO3.wedge(np.array([theta(s)[1], 0*s, theta(s)[0]]))
        self.H = np.hstack((np.zeros((self.n, self.n)), np.eye(self.n)))
        self.L = np.array([[0,1],[0,0],[1,0],[0,0],[0,0],[0,0]])


    def __propagation_Z(self, t_span):
        """
        
        """
        X0 = self.X
        Q0 = self.Q
        A22 = lambda t: np.kron(self.A(t), np.eye(self.n))
        A11 = np.zeros((self.n,self.n))
        A21 = np.zeros((self.n*self.n,self.n))
        A12 = np.concatenate((np.eye(self.n), np.zeros((self.n,self.n*(self.n-1)))), axis=1)
        Az  = lambda t: np.concatenate((np.concatenate((A11, A12), axis=1), np.concatenate((A21, A22(t)), axis=1)))
        Z0  = np.concatenate((X0, Q0[:,0], Q0[:,1], Q0[:,2]))
        ode_func = lambda t,z: np.matmul(Az(t),z)
        sol = solve_ivp(ode_func, t_span=t_span, y0=Z0, t_eval=np.array([t_span[-1]])) #, method='Radau')
        Z = sol.y[:,0]
        self.X = Z[0:self.n]
        self.Q = np.vstack((Z[self.n:2*self.n], Z[2*self.n:3*self.n], Z[3*self.n:4*self.n])).T
        self.Z = np.vstack((np.hstack((self.Q,self.X[:,np.newaxis])),np.hstack((np.zeros((self.n)), np.array((1))))))


    # def __propagation_xi(self, t_span):
    #     """
        
    #     """
    #     ode_func = lambda t,x: np.matmul(self.F(t),x)
    #     sol = solve_ivp(ode_func, t_span=t_span, y0=self.xi, t_eval=np.array([t_span[-1]])) #, method='Radau')
    #     self.xi = sol.y[:,0]

    def __propagation_P(self, t_span):
        """
        
        """
        ode_func = lambda t,p: (np.matmul(self.F(t),p.reshape(self.dim_g, self.dim_g)) + np.matmul(p.reshape(self.dim_g, self.dim_g),self.F(t).T) + self.L @ self.Sigma(t) @ self.L.T).flatten()
        sol = solve_ivp(ode_func, t_span=t_span, y0=self.P.flatten(), t_eval=np.array([t_span[-1]])) #, method='Radau')
        self.P = sol.y[:,-1].reshape(self.dim_g, self.dim_g)

    def __propagation_C(self, t_span):
        """
        
        """
        ode_func = lambda t,c: np.matmul(self.F(t), c.reshape(self.dim_g, self.dim_g)).flatten()
        sol = solve_ivp(ode_func, t_span=t_span, y0=(np.eye(self.dim_g)).flatten(), t_eval=np.array([t_span[-1]])) #, method='Radau')
        self.C = sol.y[:,0].reshape(self.dim_g, self.dim_g)


    def __propagation_U(self, t_span):
        """
        
        """
        x_eval = np.linspace(t_span[0],t_span[-1],self.K_pts)
        u_eval = x_eval[1:] - x_eval[:-1]
        v_eval = (x_eval[1:] + x_eval[:-1])/2
        ode_func = lambda t,p: (np.matmul(self.F(t),p.reshape(self.dim_g, self.dim_g)) + np.matmul(p.reshape(self.dim_g, self.dim_g),self.F(t).T) + self.L @ self.Sigma(t) @ self.L.T).flatten()
        sol = solve_ivp(ode_func, t_span=t_span, y0=self.P.flatten(), t_eval=v_eval) #, method='Radau')
        self.list_P = sol.y[:,:].reshape(self.dim_g, self.dim_g, self.K_pts-1)
        for i in range(self.K_pts-1):
            self.U = expm(u_eval[i]*(self.F(v_eval[i]) + self.L @ self.Sigma(v_eval[i]) @ self.L.T @np.linalg.inv(self.list_P[:,:,i])))@self.U
            self.list_U.append(self.U)


    def __predict(self, ti, tf):
        """
        
        """
        if self.n == 3:
            t_span = np.array([ti, tf])
            self.__propagation_Z(t_span)
            # self.__propagation_xi(t_span)
            self.__propagation_C(t_span)
            self.__propagation_U(t_span)
            self.__propagation_P(t_span)
        else:
            raise NotImplementedError("Propagation step not implemented for n != 3.")
        

    def __update(self, y):
        """
        
        """
        S = self.H @ self.P @ self.H.T + (self.Q).T @ self.Gamma @ self.Q
        if S.any() == np.zeros((self.n,self.n)).any():
            self.K = np.zeros((self.dim_g,self.n))
        else:
            self.K = self.P @ self.H.T @ np.linalg.inv(S)
        I = np.eye(self.dim_g)
        self.P = (I - self.K @ self.H) @ self.P
        self.P = (self.P+self.P.T)/2
        eps = self.K @ (self.Q).T @ (y - self.X)
        # self.xi = self.xi - self.K @ self.H @ self.xi + eps
        self.Z = self.Z @ SE3.exp(eps[:self.dim_g])
        self.Q = self.Z[:self.n,:self.n]
        self.X = self.Z[:self.n,self.n]
        # print('before:', self.P)
        # J = SE3.left_jacobian(eps[:self.dim_g])
        # self.P = J @ self.P @ J.T
        # print('after:', self.P)


    def tracking(self, grid, Y):
        """
        
        """
        self.N = len(grid)
        self.grid = grid
        pred_Z, pred_P, pred_C = [], [], []
        track_Z, track_Q, track_X, track_P = [self.Z], [self.Q], [self.X], [self.P]

        for si, sf, y in zip(grid[:-1], grid[1:], Y):
            ''' Propagation step '''
            self.__predict(si, sf)
            pred_Z.append(self.Z)
            # pred_xi.append(self.xi)
            pred_P.append(self.P)
            pred_C.append(self.C)
            ''' Update step '''
            self.__update(y)
            track_Z.append(self.Z)
            track_Q.append(self.Q)
            track_X.append(self.X)
            # track_xi.append(self.xi)
            track_P.append(self.P)

        self.pred_Z = pred_Z
        # self.pred_xi = pred_xi
        self.pred_P = pred_P
        self.pred_C = pred_C
        self.track_Z = track_Z
        self.track_Q = np.array(track_Q)
        self.track_X = np.array(track_X)
        # self.track_xi = np.array(track_xi)
        self.track_P = track_P
        self.list_U = np.array(self.list_U)

        return
    
    
    def smoothing(self, grid, Y):
        """
        
        """
        # Do the tracking if it hasn't been done
        if hasattr(self, "track_Z")==False:
            self.tracking(grid, Y)

        self.Z_S = self.track_Z[-1]
        self.P_S = self.track_P[-1]
        I = np.eye(self.dim_g)
        smooth_Z = [self.Z_S]
        smooth_P_bis = [self.P_S]
        smooth_gain = []

        for Z_t, Z_p, P_t, P_p, C in zip(reversed(self.track_Z[:-1]), reversed(self.pred_Z),reversed(self.track_P[:-1]),reversed(self.pred_P),reversed(self.pred_C)):
            if P_p.any() == np.zeros((self.dim_g,self.dim_g)).any():
                D = np.zeros((self.dim_g,self.dim_g))
                smooth_gain.append(D)
            else:
                D = P_t @ C.T @ np.linalg.inv(P_p)
                smooth_gain.append(D)

            self.Z_S = Z_t@SE3.exp(D@SE3.log(np.linalg.inv(Z_p)@self.Z_S))
            smooth_Z.append(self.Z_S)

            self.P_S = P_t + D @ (self.P_S - P_p) @ D.T
            smooth_P_bis.append(self.P_S)

        I = np.eye(self.dim_g)
        self.P_dble_S = (I - self.K @ self.H) @ self.pred_C[-1] @ self.track_P[-2]
        smooth_P_dble_bis = [self.P_dble_S.T]
        for P_t, C, D, D_m1 in zip(reversed(self.track_P[:-1]), reversed(self.pred_C), smooth_gain, smooth_gain[1:]):
            self.P_dble_S = P_t @ D_m1.T + D @ (self.P_dble_S - C @ P_t) @ D_m1.T
            smooth_P_dble_bis.append(self.P_dble_S.T)

        # self.smooth_P_bis = np.array(list(reversed(smooth_P_bis)))
        # self.smooth_P_dble_bis = np.array(list(reversed(smooth_P_dble_bis)))

        self.P_full, self.P_full_mat = self.__compute_full_P_smooth()
        print(self.P_full.shape, self.P_full_mat.shape)

        # smooth_P_dble = np.zeros((len(P_full)-1,self.dim_g,self.dim_g))
        # smooth_P = np.zeros((len(P_full),self.dim_g,self.dim_g))
        # for i in range(len(P_full)):
        #     smooth_P[i] = P_full[i,i,:,:]
        #     if i > 0:
        #         smooth_P_dble[i-1] = P_full[i-1,i,:,:]

        self.smooth_Z = np.array(list(reversed(smooth_Z)))
        self.smooth_Q = self.smooth_Z[:,0:self.n,0:self.n]
        self.smooth_X = self.smooth_Z[:,0:self.n,self.n]
        # self.smooth_P_dble = smooth_P_dble
        # self.smooth_P = smooth_P
        # self.smooth_P_full = P_full_mat
        self.smooth_P_dble = np.array(list(reversed(smooth_P_dble_bis)))
        self.smooth_P = np.array(list(reversed(smooth_P_bis)))

        return


    def __compute_full_P_smooth(self):
        """
        
        """
        C = np.zeros((self.dim_g*self.N,self.dim_g*self.N))
        W_appox = np.zeros((self.N,self.dim_g,self.dim_g))
        W_appox[-1] = self.track_P[-1]
        for j in range(self.N):
            for i in range(j+1):
                if i==j:
                    C[i*self.dim_g:(i+1)*self.dim_g,j*self.dim_g:(j+1)*self.dim_g] = np.eye(self.dim_g)
                else:
                    C[i*self.dim_g:(i+1)*self.dim_g,j*self.dim_g:(j+1)*self.dim_g] = self.list_U[i*(self.K_pts-1),:,:]@np.linalg.inv(self.list_U[j*(self.K_pts-1),:,:])

            if j < self.N-1:
                x = np.linspace(self.grid[j],self.grid[j+1],self.K_pts)
                w_to_int = np.zeros((self.dim_g,self.dim_g,self.K_pts))
                for k in range(self.K_pts):
                    phi = self.list_U[j*(self.K_pts-1),:,:]@np.linalg.inv(self.list_U[j*(self.K_pts-1)+k,:,:])
                    w_to_int[:,:,k] = phi @ self.L @ self.Sigma(x[k]) @ self.L.T @ phi.T
                W_appox[j] = np.trapz(w_to_int, x=x)
        W_mat = block_diag(*W_appox)
        P_mat = C@W_mat@C.T
        P = np.reshape(P_mat, (self.N,self.dim_g,self.N,self.dim_g))
        P = np.moveaxis(P, (1,2), (2,1))
        return P, P_mat 
    







class LatentForceIEKFilterSmootherFrenetState():

    """Class for continuous-discrete invariant extended kalman filter for the tracking and smoothing of the Frenet Serret frame on SE(3).

    i.e. we consider elements of the Lie group of rigid transformations SE(n).

    Parameters
    ----------
    n : int
        Dimension for the Lie group SE(n)
    dim_g : int
            Dimension of the Lie Algebra se(n)
    Sigma : matrix of size (dim_g,dim_g)
            Covariance matrix of the white Gaussian noise
    N : matrix of size (n,n)
        Covariance matrix of the noisy observations
    Z : matrix in SE(n), size (n+1,n+1)
        Actual state of the system: (Q  X
                                     0  1)
    Q : matrix in SO(n), size (n,n)
        Actual state of rotation matrix
    X : vector in R^n, size (n,1)
        Actual state of translation vector
    xi : vector of size ((m+1)*dim_g,1)
            Lie Algebraic error and gaussian process noise
    P : matrix of size ((m+1)*dim_g,(m+1)*dim_g)
        Actual state of the covariance
    """

    def __init__(self, n, Gamma, Sigma, theta, Z0 = None, P0 = None, K_pts=3):

        self.n = n # SE(n)
        self.dim_theta = n-1
        self.dim_g = int((1/2)*n*(n+1)) # dimension of the Lie Algebra se(n)
        self.dim_state = self.dim_g + self.dim_theta
        self.Sigma = Sigma # fct of matrix (n-1 x n-1)
        self.Gamma = Gamma 
        self.xa = np.zeros((self.dim_state))
        self.eta = np.zeros((self.dim_theta))
        self.P0 = np.zeros((self.dim_state,self.dim_state)) if P0 is None else P0
        self.P = self.P0
        self.C = np.eye(self.dim_state)
        self.Q = np.eye(self.n) if Z0 is None else Z0[:self.n,:self.n]
        self.X = np.zeros((self.n)) if Z0 is None else Z0[:self.n,self.n]
        self.Z = np.vstack((np.hstack((self.Q,self.X[:,np.newaxis])),np.hstack((np.zeros((self.n)), np.array((1)))))) if Z0 is None else Z0
        self.U = np.eye(self.dim_state)
        self.list_U = [self.U]
        self.K_pts = K_pts
        
        self.L = np.array([[0,1],[0,0],[1,0],[0,0],[0,0],[0,0]])
        self.F = lambda s: np.vstack((np.hstack((-SE3.Ad(np.array([theta(s)[1], 0*s, theta(s)[0], 1+0*s, 0*s, 0*s])), self.L)), np.zeros((self.dim_theta, self.dim_state))))
        self.A = lambda s: -SO3.wedge(np.array([theta(s)[1], 0*s, theta(s)[0]]))
        self.H = np.hstack((np.hstack((np.zeros((self.n, self.n)), np.eye(self.n))), np.zeros((self.n,self.dim_theta))))
        self.L_a = np.vstack((np.zeros((self.dim_g, self.dim_theta)), np.eye(self.dim_theta)))
        


    def __propagation_Z(self, t_span):
        """
        
        """
        X0 = self.X
        Q0 = self.Q
        A22 = lambda t: np.kron(self.A(t), np.eye(self.n))
        A11 = np.zeros((self.n,self.n))
        A21 = np.zeros((self.n*self.n,self.n))
        A12 = np.concatenate((np.eye(self.n), np.zeros((self.n,self.n*(self.n-1)))), axis=1)
        Az  = lambda t: np.concatenate((np.concatenate((A11, A12), axis=1), np.concatenate((A21, A22(t)), axis=1)))
        Z0  = np.concatenate((X0, Q0[:,0], Q0[:,1], Q0[:,2]))
        ode_func = lambda t,z: np.matmul(Az(t),z)
        sol = solve_ivp(ode_func, t_span=t_span, y0=Z0, t_eval=np.array([t_span[-1]])) #, method='Radau')
        Z = sol.y[:,0]
        self.X = Z[0:self.n]
        self.Q = np.vstack((Z[self.n:2*self.n], Z[2*self.n:3*self.n], Z[3*self.n:4*self.n])).T
        self.Z = np.vstack((np.hstack((self.Q,self.X[:,np.newaxis])),np.hstack((np.zeros((self.n)), np.array((1))))))


    def __propagation_xa(self, t_span):
        """
        
        """
        ode_func = lambda t,x: np.matmul(self.F(t),x)
        sol = solve_ivp(ode_func, t_span=t_span, y0=self.xa, t_eval=np.array([t_span[-1]])) #, method='Radau')
        self.xa = sol.y[:,0]

    def __propagation_P(self, t_span):
        """
        
        """
        ode_func = lambda t,p: (np.matmul(self.F(t),p.reshape(self.dim_state, self.dim_state)) + np.matmul(p.reshape(self.dim_state, self.dim_state),self.F(t).T) + self.L_a @ self.Sigma(t) @ self.L_a.T).flatten()
        sol = solve_ivp(ode_func, t_span=t_span, y0=self.P.flatten(), t_eval=np.array([t_span[-1]])) #, method='Radau')
        self.P = sol.y[:,-1].reshape(self.dim_state, self.dim_state)

    def __propagation_C(self, t_span):
        """
        
        """
        ode_func = lambda t,c: np.matmul(self.F(t), c.reshape(self.dim_state, self.dim_state)).flatten()
        sol = solve_ivp(ode_func, t_span=t_span, y0=(np.eye(self.dim_state)).flatten(), t_eval=np.array([t_span[-1]])) #, method='Radau')
        self.C = sol.y[:,0].reshape(self.dim_state, self.dim_state)


    def __propagation_U(self, t_span):
        """
        
        """
        x_eval = np.linspace(t_span[0],t_span[-1],self.K_pts)
        u_eval = x_eval[1:] - x_eval[:-1]
        v_eval = (x_eval[1:] + x_eval[:-1])/2
        ode_func = lambda t,p: (np.matmul(self.F(t),p.reshape(self.dim_state, self.dim_state)) + np.matmul(p.reshape(self.dim_state, self.dim_state),self.F(t).T) + self.L_a @ self.Sigma(t) @ self.L_a.T).flatten()
        sol = solve_ivp(ode_func, t_span=t_span, y0=self.P.flatten(), t_eval=v_eval) #, method='Radau')
        self.list_P = sol.y[:,:].reshape(self.dim_state, self.dim_state, self.K_pts-1)
        for i in range(self.K_pts-1):
            self.U = expm(u_eval[i]*(self.F(v_eval[i]) + self.L_a @ self.Sigma(v_eval[i]) @ self.L_a.T @np.linalg.inv(self.list_P[:,:,i])))@self.U
            self.list_U.append(self.U)


    def __predict(self, ti, tf):
        """
        
        """
        if self.n == 3:
            t_span = np.array([ti, tf])
            self.__propagation_Z(t_span)
            self.__propagation_xa(t_span)
            self.__propagation_C(t_span)
            self.__propagation_U(t_span)
            self.__propagation_P(t_span)
        else:
            raise NotImplementedError("Propagation step not implemented for n != 3.")


    def __update(self, y):
        """
        
        """
        S = self.H @ self.P @ self.H.T + (self.Q).T @ self.Gamma @ self.Q
        if S.any() == np.zeros((self.n,self.n)).any():
            self.K = np.zeros((self.dim_state,self.n))
        else:
            self.K = self.P @ self.H.T @ np.linalg.inv(S)
        I = np.eye(self.dim_state)
        self.P = (I - self.K @ self.H) @ self.P
        self.P = (self.P+self.P.T)/2
        eps = self.K @ (self.Q).T @ (y - self.X) 
        self.xa = self.xa - self.K @ self.H @ self.xa + eps
        self.eta = self.xa[self.dim_g:]
        self.Z = self.Z @ SE3.exp(eps[:self.dim_g])
        self.Q = self.Z[:self.n,:self.n]
        self.X = self.Z[:self.n,self.n]
        # print('before:', self.P)
        # J = SE3.left_jacobian(eps[:self.dim_g])
        # self.P = J @ self.P @ J.T
        # print('after:', self.P)


    def tracking(self, grid, Y):
        """
        
        """
        self.N = len(grid)
        self.grid = grid
        pred_Z, pred_xa, pred_P, pred_C = [], [], [], []
        track_Z, track_Q, track_X, track_xa, track_P, track_eta = [self.Z], [self.Q], [self.X], [self.xa], [self.P], [self.eta]

        for si, sf, y in zip(grid[:-1], grid[1:], Y):
            ''' Propagation step '''
            self.__predict(si, sf)
            pred_Z.append(self.Z)
            pred_xa.append(self.xa)
            pred_P.append(self.P)
            pred_C.append(self.C)
            ''' Update step '''
            self.__update(y)
            track_Z.append(self.Z)
            track_Q.append(self.Q)
            track_X.append(self.X)
            track_xa.append(self.xa)
            track_eta.append(self.eta)
            track_P.append(self.P)

        self.pred_Z = pred_Z
        self.pred_xa = pred_xa
        self.pred_P = pred_P
        self.pred_C = pred_C
        self.track_Z = track_Z
        self.track_Q = np.array(track_Q)
        self.track_X = np.array(track_X)
        self.track_xa = np.array(track_xa)
        self.track_eta = np.array(track_eta)
        self.track_P = track_P
        self.list_U = np.array(self.list_U)

        return


    def smoothing(self, grid, Y):
        """
        
        """
        # Do the tracking if it hasn't been done
        if hasattr(self, "track_Z")==False:
            self.tracking(grid, Y)

        self.Z_S = self.track_Z[-1]
        self.Z_S_2 = self.track_Z[-1]
        self.Z_S_3 = self.track_Z[-1]

        self.xa_S = self.track_xa[-1]
        I = np.eye(self.dim_g)
        smooth_Z = [self.Z_S]
        smooth_Z_2 = [self.Z_S_2]
        smooth_Z_3 = [self.Z_S_3]

        smooth_xa = [self.xa_S]
        smooth_gain = []

        for xa_t, xa_p, Z_t, Z_p, P_t, P_p, C in zip(reversed(self.track_xa[:-1]), reversed(self.pred_xa), reversed(self.track_Z[:-1]), reversed(self.pred_Z),reversed(self.track_P[:-1]),reversed(self.pred_P),reversed(self.pred_C)):
            if P_p.any() == np.zeros((self.dim_state,self.dim_state)).any():
                D = np.zeros((self.dim_state,self.dim_state))
                smooth_gain.append(D)
            else:
                D = P_t @ C.T @ np.linalg.inv(P_p)
                smooth_gain.append(D)

            self.xa_S = xa_t + D@(self.xa_S - xa_p)

            self.Z_S = Z_t@SE3.exp(D[:self.dim_g,:self.dim_g]@SE3.log(np.linalg.inv(Z_p)@self.Z_S))

            err = (self.xa_S - xa_p)
            eps = D@err
            self.Z_S_2 = Z_t@SE3.exp(eps[:self.dim_g])

            err = (self.xa_S - xa_p)
            err[:self.dim_g] = SE3.log(np.linalg.inv(Z_p)@self.Z_S)
            eps = D@err
            self.Z_S_3 = Z_t@SE3.exp(eps[:self.dim_g])

            smooth_Z.append(self.Z_S)
            smooth_Z_2.append(self.Z_S_2)
            smooth_Z_3.append(self.Z_S_3)

            smooth_xa.append(self.xa_S)

        P_full, P_full_mat = self.__compute_full_P_smooth()
        smooth_P_dble = np.zeros((len(P_full)-1,self.dim_state,self.dim_state))
        smooth_P = np.zeros((len(P_full),self.dim_state,self.dim_state))
        for i in range(len(P_full)):
            smooth_P[i] = P_full[i,i,:,:]
            if i > 0:
                smooth_P_dble[i-1] = P_full[i-1,i,:,:]

        self.smooth_Z = np.array(list(reversed(smooth_Z)))
        self.smooth_Z_2 = np.array(list(reversed(smooth_Z_2)))
        self.smooth_Z_3 = np.array(list(reversed(smooth_Z_3)))

        self.smooth_xa = np.array(list(reversed(smooth_xa)))
        self.smooth_eta = self.smooth_xa[:,self.dim_g:]
        self.smooth_Q = self.smooth_Z[:,0:self.n,0:self.n]
        self.smooth_X = self.smooth_Z[:,0:self.n,self.n]
        self.smooth_P_dble = smooth_P_dble
        self.smooth_P = smooth_P
        self.smooth_P_full = P_full_mat

        return


    def __compute_full_P_smooth(self):
        """
        
        """
        C = np.zeros((self.dim_state*self.N,self.dim_state*self.N))
        W_appox = np.zeros((self.N,self.dim_state,self.dim_state))
        W_appox[-1] = self.track_P[-1]
        for j in range(self.N):
            for i in range(j+1):
                if i==j:
                    C[i*self.dim_state:(i+1)*self.dim_state,j*self.dim_state:(j+1)*self.dim_state] = np.eye(self.dim_state)
                else:
                    C[i*self.dim_state:(i+1)*self.dim_state,j*self.dim_state:(j+1)*self.dim_state] = self.list_U[i*(self.K_pts-1),:,:]@np.linalg.inv(self.list_U[j*(self.K_pts-1),:,:])

            if j < self.N-1:
                x = np.linspace(self.grid[j],self.grid[j+1],self.K_pts)
                w_to_int = np.zeros((self.dim_state,self.dim_state,self.K_pts))
                for k in range(self.K_pts):
                    phi = self.list_U[j*(self.K_pts-1),:,:]@np.linalg.inv(self.list_U[j*(self.K_pts-1)+k,:,:])
                    w_to_int[:,:,k] = phi @ self.L_a @ self.Sigma(x[k]) @ self.L_a.T @ phi.T
                W_appox[j] = np.trapz(w_to_int, x=x)
        W_mat = block_diag(*W_appox)
        P_mat = C@W_mat@C.T
        P = np.reshape(P_mat, (self.N,self.dim_state,self.N,self.dim_state))
        P = np.moveaxis(P, (1,2), (2,1))
        return P, P_mat