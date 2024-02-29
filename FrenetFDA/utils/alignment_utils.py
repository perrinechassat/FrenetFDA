import optimum_reparam_N as orN
import optimum_reparamN2 as orN2
import numpy as np
from joblib import Parallel, delayed
from scipy.integrate import trapz
from scipy.linalg import svd
from scipy.interpolate import interp1d
import collections
import fdasrsf.utility_functions as uf
import optimum_reparam_N_curvatures as orNC


def optimum_reparam_curvature_1d(theta1, time, theta2, lam=0.0, grid_dim=7):

    theta1 = np.array([theta1]).T
    theta2 = np.array([theta2]).T

    gam = orNC.coptimum_reparamN2(np.ascontiguousarray(theta1), time, np.ascontiguousarray(theta2), lam, grid_dim)
    gam = np.squeeze(gam)
    
    return gam


def optimum_reparam_curvatures(theta1, time, theta2, lam=0.0, grid_dim=7):
    """
    calculates the warping to align theta2 to theta1

    Param:
        theta1: matrix of size 2xM (curvature, torsion)
        time: vector of size M describing the sample points
        theta2: matrix of size 2xM (curvature, torsion)
        lam: controls the amount of elasticity (default = 0.0)
        grid_dim: size of the grid (default = 7)

    Return:
        gam: describing the warping function used to align theta2 with theta1

    """
    gam = orNC.coptimum_reparamN2(np.ascontiguousarray(theta1), time,
                                          np.ascontiguousarray(theta2), lam, grid_dim)

    return gam



def optimum_reparam_vect_curvatures(theta1, time, theta2, lam=0.0, grid_dim=7):
    """
    calculates the warping to align theta2 to theta1

    Param:
        theta1: matrix of size 2xM (curvature, torsion)
        time: vector of size M describing the sample points
        theta2: matrix of size 2xM (curvature, torsion)
        lam: controls the amount of elasticity (default = 0.0)
        grid_dim: size of the grid, for the DP2 method only (default = 7)

    Return:
        gam: describing the warping function used to align theta2 with theta1

    """

    gam = orNC.coptimum_reparam_curve(np.ascontiguousarray(theta1), time,
                                         np.ascontiguousarray(theta2), lam, grid_dim)

    return gam


def align_vect_SRC_fPCA(f, time, weights=None, num_comp=3, cores=-1, MaxItr=1, init_cost=0, lam=0.0):
    
    if len(f.shape)==2:
        f = f[np.newaxis, :, :]

    n = f.shape[0]
    M = f.shape[1]
    N = f.shape[2]
    parallel = True

    eps = np.finfo(np.double).eps

    f0 = f

    if weights is not None:
        mf0 = weighted_mean_vect(f, weights)
    else:
        mf0 = f.mean(axis=2)
    a = mf0.repeat(N)
    d1 = a.reshape(n, M, N)
    d = (f - d1) ** 2
    dqq = np.sqrt(d.sum(axis=1).sum(axis=0))
    min_ind = dqq.argmin()

    itr = 0
    mf = np.zeros((n, M, MaxItr + 1))
    mf_cent = np.zeros((n, M, MaxItr + 1))
    mf[:, :, itr] = f[:, :, min_ind]
    mf_cent[:, :, itr] = f[:, :, min_ind]
    # mf[:, itr] = mf0
    fi = np.zeros((n, M, N, MaxItr + 1))
    fi_cent = np.zeros((n, M, N, MaxItr + 1))
    fi[:, :, :, 0] = f
    fi_cent[:, :, :, 0] = f
    gam = np.zeros((M, N, MaxItr + 1))
    cost = np.zeros(MaxItr + 1)
    cost[itr] = init_cost

    MS_phase = (trapz(f[:, :, min_ind] ** 2, time) - trapz(mf0 ** 2, time)).mean()
    # print('MS_phase :', MS_phase)
    if np.abs(MS_phase) < 0.01:
        print('MS_phase :', MS_phase)
        print("%d functions already aligned..."
              % (N))
        mfn = mf0
        fn = f0
        gamf = np.zeros((M,N))
        for k in range(0, N):
            gamf[:, k] = time

        align_results = collections.namedtuple('align_fPCA', ['fn', 'gamf', 'mfn', 'nb_itr', 'convergence'])

        out = align_results(fn, gamf, mfn, 0, True)

        return out

    print("Aligning %d functions to %d fPCA components..."
          % (N, num_comp))


    while itr < MaxItr:
        # print("updating step: r=%d" % (itr + 1))

        # PCA Step
        fhat = np.zeros((n,M,N))
        for k in range(n):
            a = mf[k, :, itr].repeat(N)
            d1 = a.reshape(M, N)
            fhat_cent = fi[k, :, :, itr] - d1
            K = np.cov(fi[k, :, :, itr])
            if True in np.isnan(K) or True in np.isinf(K):
                mfn = mf0
                fn = f0
                gamf = np.zeros((M,N))
                for k in range(0, N):
                    gamf[:, k] = time
                align_results = collections.namedtuple('align_fPCA', ['fn', 'gamf', 'mfn', 'nb_itr', 'convergence'])
                out = align_results(f0, gamf, mfn, MaxItr, False)
                return out

            U, s, V = svd(K)

            alpha_i = np.zeros((num_comp, N))
            for ii in range(0, num_comp):
                for jj in range(0, N):
                    alpha_i[ii, jj] = trapz(fhat_cent[:, jj] * U[:, ii], time)

            U1 = U[:, 0:num_comp]
            tmp = U1.dot(alpha_i)
            fhat[k,:,:] = d1 + tmp
            
            # plt.figure()
            # for i in range(N):
            #     plt.plot(time, fhat[k,:,i])
            # plt.show()

        cost_init = np.zeros(N)

        # Matching Step

        if parallel:
            out = Parallel(n_jobs=cores)(delayed(orN2.coptimum_reparam_curve)(np.ascontiguousarray(fhat[:, :, k]), time, np.ascontiguousarray(fi[:, :, k, itr]), lam) for k in range(N))
            gam_t = np.array(out)
            for k in range(N):
                gam[:,k,itr] = (gam_t[k] - gam_t[k][0])/(gam_t[k][-1] - gam_t[k][0])
            # gam[:, :, itr] = gam_t.transpose()
        else:
            for k in range(N):
                gam[:, k, itr] = orN2.coptimum_reparam_curve(np.ascontiguousarray(fhat[:, :, k]), time, np.ascontiguousarray(fi[:, :, k, itr]), lam)

        for kk in range(n):
            for k in range(0, N):
                time0 = (time[-1] - time[0]) * gam[:, k, itr] + time[0]
                fi[kk, :, k, itr + 1] = np.interp(time0, time, fi[kk, :, k, itr]) * np.sqrt(np.gradient(gam[:, k, itr], 1 / float(M - 1)))

        fi[np.isnan(fi)] = 0.0
        fi[np.isinf(fi)] = 0.0

        ftemp = fi[:, :, :, itr + 1]
        if weights is not None:
            mf[:, :, itr + 1] = weighted_mean_vect(ftemp, weights)
        else:
            mf[:, :, itr + 1] = ftemp.mean(axis=2)

        # plt.figure()
        # for i in range(N):
        #     plt.plot(time, gam[:, i, itr])
        # plt.show()
  
        # plt.figure()
        # for i in range(N):
        #     plt.plot(time, ftemp[0, :, i])
        # plt.show()
        # plt.figure()
        # plt.plot(time, mf[0, :, itr + 1])
        # plt.show()
        
        # plt.figure()
        # for i in range(N):
        #     plt.plot(time, ftemp[1, :, i])
        # plt.show()
        # plt.figure()
        # plt.plot(time, mf[1, :, itr + 1])
        # plt.show()

        fi_cent[:, :, :, itr + 1], mf_cent[:, :, itr + 1] = align_and_center_src(np.copy(gam), np.copy(mf[:, :, itr + 1]), np.copy(ftemp), itr+1, np.copy(time))

        cost_temp = np.zeros(N)

        for ii in range(0, N):
            cost_temp[ii] = np.linalg.norm(fi[:,:,ii,itr] - ftemp[:,:,ii], 'fro')

        cost[itr + 1] = cost_temp.mean()

        if abs(cost[itr + 1] - cost[itr]) < 1:
            break

        itr += 1

    print("Alignment in %d iterations" % (itr))
    if itr >= MaxItr:
        itrf = MaxItr
    else:
        itrf = itr+1
    cost = cost[1:(itrf+1)]

    # Aligned data & stats
    fn = fi[:, :, :, itrf]
    mfn = mf[:, :, itrf]
    gamf = gam[:, :, 0]
    for k in range(1, itrf):
        gam_k = gam[:, :, k]
        for l in range(0, N):
            time0 = (time[-1] - time[0]) * gam_k[:, l] + time[0]
            gamf[:, l] = np.interp(time0, time, gamf[:, l])

    ## Center Mean
    gamI = uf.SqrtMeanInverse(gamf)
    gamI_dev = np.gradient(gamI, 1 / float(M - 1))
    time0 = (time[-1] - time[0]) * gamI + time[0]
    for kk in range(n):
        mfn[kk] = np.interp(time0, time, mfn[kk]) * np.sqrt(gamI_dev)
        for k in range(0, N):
            fn[kk, :, k] = np.interp(time0, time, fn[kk, :, k]) * np.sqrt(gamI_dev)

    for k in range(0, N):
        gamf[:, k] = np.interp(time0, time, gamf[:, k])
        gamf[:,k] = (gamf[:,k] - gamf[0,k])/(gamf[-1,k] - gamf[0,k])

    gamf_inv = np.zeros(gamf.shape)
    for k in range(N):
        gamf_inv[:,k] = interp1d(gamf[:,k], time)(time)
        gamf_inv[:,k] = (gamf_inv[:,k] - gamf_inv[0,k])/(gamf_inv[-1,k] - gamf_inv[0,k])
    
    # plt.figure()
    # plt.plot(time, mfn[0])
    # plt.show()
    # plt.figure()
    # plt.plot(time, mfn[1])
    # plt.show()
    
    # plt.figure()
    # for i in range(N):
    #     plt.plot(time, gamf[:, i])
    # plt.show()

    # plot_array_2D(time, fn[0].T, '')
    # plot_array_2D(time, fn[1].T, '')

    # plt.figure()
    # for i in range(N):
    #     plt.plot(time, fn[0, :, i])
    # plt.show()
    # plt.figure()
    # for i in range(N):
    #     plt.plot(time, fn[1, :, i])
    # plt.show()

    align_results = collections.namedtuple('align_fPCA', ['fn', 'gamf', 'mfn', 'fi', 'gam', 'mf', 'nb_itr', 'convergence', 'gamf_inv'])

    if itr==MaxItr:
        out = align_results(fn, gamf, mfn, fi_cent[:,:,:,0:itrf+1], gam[:,:,0:itrf+1], mf_cent[:,:,0:itrf+1], itr, False, gamf_inv)
    else:
        out = align_results(fn, gamf, mfn, fi_cent[:,:,:,0:itrf+1], gam[:,:,0:itrf+1], mf_cent[:,:,0:itrf+1], itr, True, gamf_inv)

    return out



def align_vect_curvatures_fPCA(f, time, weights=None, num_comp=3, cores=-1, MaxItr=1, init_cost=0, lam=0.0):
    """
    aligns a collection of functions while extracting principal components.
    The functions are aligned to the principal components

    ...

    Param:
        f: numpy ndarray of shape (n,M,N) of N functions with M samples of 2 dimensions (kappa and tau)
        time: vector of size M describing the sample points
        weights: numpy ndarray of shape (M,N) of N functions with M samples
        num_comp: number of fPCA components
        number of cores for parallel (default = -1 (all))
        smooth_data: bool, smooth the data using a box filter (default = F)
        MaxItr: maximum number of iterations (default = 1)
        init_cost: (default = 0)
        lam: coef of alignment (default = 0)

    Return:
        fn: numpy array of aligned functions (n,M,N)
        gamf: numpy array of warping functions used to align the data (M,N)
        mfn: weighted mean of the functions algned (2,M)
        fi: aligned functions at each iterations (n,M,N,nb_itr)
        gam: estimated warping functions at each iterations (M,N,nb_itr)
        mf: estimated weighted mean at each iterations (2,M,nb_itr)
        nb_itr: number of iterations needed to align curves
        convergence: True if nb_itr < MaxItr, False otherwise

    """
    if len(f.shape)==2:
        f = f[np.newaxis, :, :]

    n = f.shape[0]
    M = f.shape[1]
    N = f.shape[2]
    parallel = True

    eps = np.finfo(np.double).eps

    f0 = f

    if weights is not None:
        mf0 = weighted_mean_vect(f, weights)
    else:
        mf0 = f.mean(axis=2)
    a = mf0.repeat(N)
    d1 = a.reshape(n, M, N)
    d = (f - d1) ** 2
    dqq = np.sqrt(d.sum(axis=1).sum(axis=0))
    min_ind = dqq.argmin()

    itr = 0
    mf = np.zeros((n, M, MaxItr + 1))
    mf_cent = np.zeros((n, M, MaxItr + 1))
    mf[:, :, itr] = f[:, :, min_ind]
    mf_cent[:, :, itr] = f[:, :, min_ind]
    # mf[:, itr] = mf0
    fi = np.zeros((n, M, N, MaxItr + 1))
    fi_cent = np.zeros((n, M, N, MaxItr + 1))
    fi[:, :, :, 0] = f
    fi_cent[:, :, :, 0] = f
    gam = np.zeros((M, N, MaxItr + 1))
    cost = np.zeros(MaxItr + 1)
    cost[itr] = init_cost

    MS_phase = (trapz(f[:, :, min_ind] ** 2, time) - trapz(mf0 ** 2, time)).mean()
    # print('MS_phase :', MS_phase)
    if np.abs(MS_phase) < 0.01:
        print('MS_phase :', MS_phase)
        print("%d functions already aligned..."
              % (N))
        mfn = mf0
        fn = f0
        gamf = np.zeros((M,N))
        for k in range(0, N):
            gamf[:, k] = time

        align_results = collections.namedtuple('align_fPCA', ['fn', 'gamf', 'mfn', 'nb_itr', 'convergence'])

        out = align_results(fn, gamf, mfn, 0, True)

        return out

    print("Aligning %d functions to %d fPCA components..."
          % (N, num_comp))

    while itr < MaxItr:
        # print("updating step: r=%d" % (itr + 1))

        # PCA Step
        fhat = np.zeros((n,M,N))
        for k in range(n):
            a = mf[k, :, itr].repeat(N)
            d1 = a.reshape(M, N)
            fhat_cent = fi[k, :, :, itr] - d1
            K = np.cov(fi[k, :, :, itr])
            if True in np.isnan(K) or True in np.isinf(K):
                mfn = mf0
                fn = f0
                gamf = np.zeros((M,N))
                for k in range(0, N):
                    gamf[:, k] = time
                align_results = collections.namedtuple('align_fPCA', ['fn', 'gamf', 'mfn', 'nb_itr', 'convergence'])
                out = align_results(f0, gamf, mfn, MaxItr, False)
                return out

            U, s, V = svd(K)

            alpha_i = np.zeros((num_comp, N))
            for ii in range(0, num_comp):
                for jj in range(0, N):
                    alpha_i[ii, jj] = trapz(fhat_cent[:, jj] * U[:, ii], time)

            U1 = U[:, 0:num_comp]
            tmp = U1.dot(alpha_i)
            fhat[k,:,:] = d1 + tmp
            
            # plt.figure()
            # for i in range(N):
            #     plt.plot(time, fhat[k,:,i])
            # plt.show()

        cost_init = np.zeros(N)

        # Matching Step

        if parallel:
            out = Parallel(n_jobs=cores)(delayed(orNC.coptimum_reparam_curve)(np.ascontiguousarray(fhat[:, :, k]), time, np.ascontiguousarray(fi[:, :, k, itr]), lam) for k in range(N))
            gam_t = np.array(out)
            for k in range(N):
                gam[:,k,itr] = (gam_t[k] - gam_t[k][0])/(gam_t[k][-1] - gam_t[k][0])
            # gam[:, :, itr] = gam_t.transpose()
        else:
            for k in range(N):
                gam[:, k, itr] = orNC.coptimum_reparam_curve(np.ascontiguousarray(fhat[:, :, k]), time, np.ascontiguousarray(fi[:, :, k, itr]), lam)

        for kk in range(n):
            for k in range(0, N):
                time0 = (time[-1] - time[0]) * gam[:, k, itr] + time[0]
                fi[kk, :, k, itr + 1] = np.interp(time0, time, fi[kk, :, k, itr]) * np.gradient(gam[:, k, itr], 1 / float(M - 1))

        fi[np.isnan(fi)] = 0.0
        fi[np.isinf(fi)] = 0.0

        ftemp = fi[:, :, :, itr + 1]
        if weights is not None:
            mf[:, :, itr + 1] = weighted_mean_vect(ftemp, weights)
        else:
            mf[:, :, itr + 1] = ftemp.mean(axis=2)

        # plt.figure()
        # for i in range(N):
        #     plt.plot(time, gam[:, i, itr])
        # plt.show()
  
        # plt.figure()
        # for i in range(N):
        #     plt.plot(time, ftemp[0, :, i])
        # plt.show()
        # plt.figure()
        # plt.plot(time, mf[0, :, itr + 1])
        # plt.show()
        
        # plt.figure()
        # for i in range(N):
        #     plt.plot(time, ftemp[1, :, i])
        # plt.show()
        # plt.figure()
        # plt.plot(time, mf[1, :, itr + 1])
        # plt.show()

        fi_cent[:, :, :, itr + 1], mf_cent[:, :, itr + 1] = align_and_center(np.copy(gam), np.copy(mf[:, :, itr + 1]), np.copy(ftemp), itr+1, np.copy(time))

        cost_temp = np.zeros(N)

        for ii in range(0, N):
            cost_temp[ii] = np.linalg.norm(fi[:,:,ii,itr] - ftemp[:,:,ii], 'fro')

        cost[itr + 1] = cost_temp.mean()

        if abs(cost[itr + 1] - cost[itr]) < 1:
            break

        itr += 1

    print("Alignment in %d iterations" % (itr))
    if itr >= MaxItr:
        itrf = MaxItr
    else:
        itrf = itr+1
    cost = cost[1:(itrf+1)]

    # Aligned data & stats
    fn = fi[:, :, :, itrf]
    mfn = mf[:, :, itrf]
    gamf = gam[:, :, 0]
    for k in range(1, itrf):
        gam_k = gam[:, :, k]
        for l in range(0, N):
            time0 = (time[-1] - time[0]) * gam_k[:, l] + time[0]
            gamf[:, l] = np.interp(time0, time, gamf[:, l])

    ## Center Mean
    gamI = uf.SqrtMeanInverse(gamf)
    gamI_dev = np.gradient(gamI, 1 / float(M - 1))
    time0 = (time[-1] - time[0]) * gamI + time[0]
    for kk in range(n):
        mfn[kk] = np.interp(time0, time, mfn[kk]) * gamI_dev
        for k in range(0, N):
            fn[kk, :, k] = np.interp(time0, time, fn[kk, :, k]) * gamI_dev

    for k in range(0, N):
        gamf[:, k] = np.interp(time0, time, gamf[:, k])
        gamf[:,k] = (gamf[:,k] - gamf[0,k])/(gamf[-1,k] - gamf[0,k])

    gamf_inv = np.zeros(gamf.shape)
    for k in range(N):
        gamf_inv[:,k] = interp1d(gamf[:,k], time)(time)
        gamf_inv[:,k] = (gamf_inv[:,k] - gamf_inv[0,k])/(gamf_inv[-1,k] - gamf_inv[0,k])
    
    # plt.figure()
    # plt.plot(time, mfn[0])
    # plt.show()
    # plt.figure()
    # plt.plot(time, mfn[1])
    # plt.show()
    
    # plt.figure()
    # for i in range(N):
    #     plt.plot(time, gamf[:, i])
    # plt.show()

    # plot_array_2D(time, fn[0].T, '')
    # plot_array_2D(time, fn[1].T, '')

    # plt.figure()
    # for i in range(N):
    #     plt.plot(time, fn[0, :, i])
    # plt.show()
    # plt.figure()
    # for i in range(N):
    #     plt.plot(time, fn[1, :, i])
    # plt.show()

    align_results = collections.namedtuple('align_fPCA', ['fn', 'gamf', 'mfn', 'fi', 'gam', 'mf', 'nb_itr', 'convergence', 'gamf_inv'])

    if itr==MaxItr:
        out = align_results(fn, gamf, mfn, fi_cent[:,:,:,0:itrf+1], gam[:,:,0:itrf+1], mf_cent[:,:,0:itrf+1], itr, False, gamf_inv)
    else:
        out = align_results(fn, gamf, mfn, fi_cent[:,:,:,0:itrf+1], gam[:,:,0:itrf+1], mf_cent[:,:,0:itrf+1], itr, True, gamf_inv)

    return out



def compose(f, g, time):
    """
    Compose functions f by functions g on the grid time.
    ...

    Param:
        f: array of N functions evaluted on time (M,N)
        g: array of N warping functions evaluted on time (M,N)
        time: array of time points (M)

    Return:
        f_g: array of functions f evaluted on g(time)

    """
    N = f.shape[1]
    f_g = np.zeros((time.shape[0], N))
    for n in range(N):
        f_g[:,n] = np.interp((time[-1] - time[0]) * g[:,n] + time[0], time, f[:,n])

    return f_g



def weighted_mean_vect(f, weights):
    """
    Compute the weighted mean of a vector of functions
    ...
    """
    sum_weights = np.sum(weights, axis=1)
    n = f.shape[0]
    N = f.shape[2]
    M = f.shape[1]
    mfw = np.zeros((n,M))
    for i in range(n):
        for j in range(M):
            if sum_weights[j]>0:
                mfw[i,j] = (np.ascontiguousarray(f[i,j,:]) @ np.ascontiguousarray(weights[j,:]))/sum_weights[j]
            else:
                mfw[i,j] = 0
    return mfw



def align_and_center(gam, mf, f, itr, time):
    """
    Utility functions for the alignment function, used to aligned functions at the end of the iterations.
    ...

    """
    n = f.shape[0]
    N = gam.shape[1]
    M = gam.shape[0]
    gamf = gam[:, :, 0]
    for k in range(1, itr):
        gam_k = gam[:, :, k]
        for l in range(0, N):
            time0 = (time[-1] - time[0]) * gam_k[:, l] + time[0]
            gamf[:, l] = np.interp(time0, time, gamf[:, l])

    ## Center Mean
    gamI = uf.SqrtMeanInverse(gamf)
    gamI_dev = np.gradient(gamI, 1 / float(M - 1))
    time0 = (time[-1] - time[0]) * gamI + time[0]
    for kk in range(n):
        mf[kk] = np.interp(time0, time, mf[kk]) * gamI_dev
        for k in range(0, N):
            f[kk, :, k] = np.interp(time0, time, f[kk, :, k]) * gamI_dev

    return f, mf



def align_and_center_src(gam, mf, f, itr, time):
    """
    Utility functions for the alignment function, used to aligned functions at the end of the iterations.
    ...

    """
    n = f.shape[0]
    N = gam.shape[1]
    M = gam.shape[0]
    gamf = gam[:, :, 0]
    for k in range(1, itr):
        gam_k = gam[:, :, k]
        for l in range(0, N):
            time0 = (time[-1] - time[0]) * gam_k[:, l] + time[0]
            gamf[:, l] = np.interp(time0, time, gamf[:, l])

    ## Center Mean
    gamI = uf.SqrtMeanInverse(gamf)
    gamI_dev = np.gradient(gamI, 1 / float(M - 1))
    time0 = (time[-1] - time[0]) * gamI + time[0]
    for kk in range(n):
        mf[kk] = np.interp(time0, time, mf[kk]) * np.sqrt(gamI_dev)
        for k in range(0, N):
            f[kk, :, k] = np.interp(time0, time, f[kk, :, k]) * np.sqrt(gamI_dev)

    return f, mf


def weighted_mean(f, weights):
    """
    Compute the weighted mean of set of functions
    ...
    """
    sum_weights = np.sum(weights, axis=1)
    N = f.shape[1]
    M = f.shape[0]
    mfw = np.zeros(M)
    for j in range(M):
        if sum_weights[j]>0:
            mfw[j] = (np.ascontiguousarray(f[j,:]) @ np.ascontiguousarray(weights[j,:]))/sum_weights[j]
        else:
            mfw[j] = 0
    return mfw


def warp_curvatures(theta_i, gam_fct, time, weights):
    """
    Apply warping on curvatures: theta_align = theta(gam_fct(time))*grad(gam_fct)(time)
    and compute the weighted mean of the aligned functions
    ...

    Param:
        theta: array of curvatures or torsions (N, M)
        gam_fct: functions, array of N warping functions
        time: array of time points (M)
        weights: array of weights (M)

    Return:
        theta align: array of functions theta aligned
        weighted_mean_theta: weighted mean of the aligned functions (M)

    """
    theta = theta_i.T
    M = theta.shape[0]
    N = theta.shape[1]
    theta_align = np.zeros(theta.shape)
    gam = np.zeros((time.shape[0], N))
    for n in range(N):
        gam[:,n] = gam_fct[n](time)
        time0 = (time[-1] - time[0]) * gam[:, n] + time[0]
        theta_align[:,n] = np.interp(time0, time, theta[:,n]) * np.gradient(gam[:, n], time)
    weighted_mean_theta = weighted_mean(theta_align, weights.T)

    return theta_align.T, weighted_mean_theta


def warp_src(src, gam_fct, time, weights):
    """
    Apply warping on curvatures: theta_align = theta(gam_fct(time))*grad(gam_fct)(time)
    and compute the weighted mean of the aligned functions
    ...

    Param:
        theta: array of curvatures or torsions (N, M, d)
        gam_fct: functions, array of N warping functions
        time: array of time points (M)
        weights: array of weights (M)

    Return:
        theta align: array of functions theta aligned
        weighted_mean_theta: weighted mean of the aligned functions (M)

    """
    N = src.shape[1]
    K = src.shape[0]
    d = src.shape[2]
    src_align = np.zeros(src.shape)
    gam = np.zeros((time.shape[0], K))
    for k in range(K):
        gam[:,k] = gam_fct[k](time)
        time0 = (time[-1] - time[0]) * gam[:, k] + time[0]
        sqrt_grad = np.sqrt(np.gradient(gam[:, k], time))
        for i in range(d):
            src_align[k,:,i] = np.interp(time0, time, src[k,:,i]) * sqrt_grad
    
    weighted_mean_src = []
    for i in range(d):
        weighted_mean_src.append(weighted_mean(src_align[:,:,i].T, weights.T))
    weighted_mean_src = np.stack(weighted_mean_src, axis=-1)

    return src_align, weighted_mean_src


def align_src(arr_src, grid, weights=None, max_iter=20, tol=0.01, lam=1.0, parallel=True):

    N_samples, N, d = arr_src.shape
    if weights is None:
        mean_src = np.mean(arr_src, axis=0)
    else:
        mean_src = weighted_mean_vect(arr_src.T, weights.T).T

    dist_arr = np.zeros(N_samples)
    for i in range(N_samples):
        dist_arr[i] = np.linalg.norm(mean_src - arr_src[i])
    ind = np.argmin(dist_arr)

    temp_mean_src = arr_src[ind]
    temp_error = np.linalg.norm((mean_src - temp_mean_src)) 
    up_err = temp_error
    k = 0
    
    # print('Iteration ', k, '/', max_iter, ': error ', temp_error)
    print("Aligning %d functions in maximum %d iterations..."
          % (N_samples, max_iter))
    while up_err > tol and k < max_iter:
        # arr_c_align = np.zeros((n,self.dim-1,T))
        arr_src_align = np.zeros((N_samples, N, d))
        arr_gam = np.zeros((N_samples, N))

        if parallel:

            def to_run(m_src, src, grid_ptn, param):
                if np.linalg.norm(m_src - src,'fro') > 0.0001:
                    gam = orN2.coptimum_reparam_curve(np.ascontiguousarray(m_src.T), grid_ptn, np.ascontiguousarray(src.T), param)
                else:
                    gam = grid_ptn
                return gam
            
            out = Parallel(n_jobs=-1)(delayed(to_run)(temp_mean_src, arr_src[i], grid, lam) for i in range(N_samples))
            gam_t = np.array(out)
            for i in range(N_samples):
                arr_gam[i] = (gam_t[i] - gam_t[i][0])/(gam_t[i][-1] - gam_t[i][0])
                for j in range(0, d):
                    time0 = (grid[-1] - grid[0]) * arr_gam[i] + grid[0]
                    arr_src_align[i,:,j] = np.interp(time0, grid, arr_src[i, :, j]) * np.sqrt(np.gradient(arr_gam[i], grid))

        else:
            for i in range(N_samples):
                if np.linalg.norm(temp_mean_src - arr_src[i],'fro') > 0.0001:
                    gam = orN2.coptimum_reparam_curve(np.ascontiguousarray(temp_mean_src.T), grid, np.ascontiguousarray(arr_src[i].T), lam)
                else:
                    gam = grid
                gam = (gam - gam.min())/(gam.max() - gam.min())
                arr_gam[i] = gam

                for j in range(0, d):
                    time0 = (grid[-1] - grid[0]) * gam + grid[0]
                    arr_src_align[i,:,j] = np.interp(time0, grid, arr_src[i, :, j]) * np.sqrt(np.gradient(gam, grid))

        if weights is None:
            mean_src = np.mean(arr_src_align, axis=0)
        else:
            mean_src = weighted_mean_vect(arr_src_align.T, weights.T).T
        error = np.linalg.norm((mean_src - temp_mean_src))  
        up_err = abs(temp_error - error)
        temp_error = error
        k += 1
        # print('Iteration ', k, '/', max_iter, ': error ', temp_error)
        temp_mean_src = mean_src
        
    print("Alignment in %d iterations" % (k))

    gamf_inv = np.zeros(arr_gam.shape)
    for k in range(N_samples):
        gamf_inv[k] = interp1d(arr_gam[k], grid)(grid)
        gamf_inv[k] = (gamf_inv[k] - gamf_inv[k,0])/(gamf_inv[k,-1] - gamf_inv[k,0])

    # visu.plot_array_2D(grid, arr_src_align[:,:,0], 'c 0')
    # visu.plot_array_2D(grid, arr_src_align[:,:,1], 'c 1')

    align_results = collections.namedtuple('align_src', ['fn', 'gamf', 'mfn', 'nb_itr', 'gamf_inv'])
    out = align_results(arr_src_align, arr_gam, mean_src, k, gamf_inv.T)

    return out




def align_curvatures(arr_theta, grid, weights=None, max_iter=20, tol=0.01, lam=1.0, parallel=True):

    N_samples, N, d = arr_theta.shape
    if weights is None:
        mean_theta = np.mean(arr_theta, axis=0)
    else:
        mean_theta = weighted_mean_vect(arr_theta.T, weights.T).T

    dist_arr = np.zeros(N_samples)
    for i in range(N_samples):
        dist_arr[i] = np.linalg.norm(mean_theta - arr_theta[i])
    ind = np.argmin(dist_arr)
    temp_mean_theta = arr_theta[ind]
    temp_error = np.linalg.norm((mean_theta - temp_mean_theta)) 
    up_err = temp_error
    k = 0
    
    # print('Iteration ', k, '/', max_iter, ': error ', temp_error)
    print("Aligning %d functions in maximum %d iterations..."
          % (N_samples, max_iter))
    while up_err > tol and k < max_iter:
        
        arr_theta_align = np.zeros((N_samples, N, d))
        arr_gam = np.zeros((N_samples, N))

        if parallel:

            def to_run(m_theta, theta, grid_ptn, param):
                if np.linalg.norm(m_theta - theta,'fro') > 0.0001:
                    gam = orN2.coptimum_reparam_curve(np.ascontiguousarray(m_theta.T), grid_ptn, np.ascontiguousarray(theta.T), param)
                else:
                    gam = grid_ptn
                return gam
            
            out = Parallel(n_jobs=-1)(delayed(to_run)(temp_mean_theta, arr_theta[i], grid, lam) for i in range(N_samples))
            gam_t = np.array(out)
            for i in range(N_samples):
                arr_gam[i] = (gam_t[i] - gam_t[i][0])/(gam_t[i][-1] - gam_t[i][0])
                for j in range(0, d):
                    time0 = (grid[-1] - grid[0]) * arr_gam[i] + grid[0]
                    arr_theta_align[i,:,j] = np.interp(time0, grid, arr_theta[i, :, j]) * np.sqrt(np.gradient(arr_gam[i], grid))

        else:
            for i in range(N_samples):
                if np.linalg.norm(temp_mean_theta - arr_theta[i],'fro') > 0.0001:
                    gam = orN2.coptimum_reparam_curve(np.ascontiguousarray(temp_mean_theta.T), grid, np.ascontiguousarray(arr_theta[i].T), lam)
                else:
                    gam = grid
                gam = (gam - gam.min())/(gam.max() - gam.min())
                arr_gam[i] = gam

                for j in range(0, d):
                    time0 = (grid[-1] - grid[0]) * gam + grid[0]
                    arr_theta_align[i,:,j] = np.interp(time0, grid, arr_theta[i, :, j]) * np.sqrt(np.gradient(gam, grid))

        if weights is None:
            mean_theta = np.mean(arr_theta_align, axis=0)
        else:
            mean_theta = weighted_mean_vect(arr_theta_align.T, weights.T).T
        error = np.linalg.norm((mean_theta - temp_mean_theta))  
        up_err = abs(temp_error - error)
        temp_error = error
        k += 1
        # print('Iteration ', k, '/', max_iter, ': error ', temp_error)
        temp_mean_theta = mean_theta
        
    print("Alignment in %d iterations" % (k))

    gamf_inv = np.zeros(arr_gam.shape)
    for k in range(N_samples):
        gamf_inv[k] = interp1d(arr_gam[k], grid)(grid)
        gamf_inv[k] = (gamf_inv[k] - gamf_inv[k,0])/(gamf_inv[k,-1] - gamf_inv[k,0])

    # visu.plot_array_2D(grid, arr_theta_align[:,:,0], '0')
    # visu.plot_array_2D(grid, arr_theta_align[:,:,1], '1')

    align_results = collections.namedtuple('align_theta', ['fn', 'gamf', 'mfn', 'nb_itr', 'gamf_inv'])
    out = align_results(arr_theta_align, arr_gam, mean_theta, k, gamf_inv.T)

    return out