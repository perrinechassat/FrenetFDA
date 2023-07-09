import numpy as np
from scipy import interpolate, optimize
from scipy.integrate import cumtrapz
from FrenetFDA.utils.smoothing_utils import LocalPolynomialSmoothing

    
def compute_arc_length(Y, time, scale=True, smooth=True, smoothing_param=None, CV_optimization={"flag":False, "h_grid":np.array([]), "K":10}):

    """ 
            Compute the arc length function and its derivative. (In all case the attribute "grid_arc_s" is scaled.)

    """
    
    N, dim = Y.shape
    if dim < 2:
        raise Exception("The Frenet Serret framework is defined only for curves in R^d with d >= 2.")
    if N != len(time):
        raise Exception("Number of sample points in attribute Y and time must be equal.")
    time = (time - time.min()) / (time.max() - time.min())
    
    if smooth==True:
        derivatives, h_opt = compute_derivatives(Y, time, deg=3, h=smoothing_param, CV_optimization_h=CV_optimization)
        sdot = np.linalg.norm(derivatives[1], axis=1)
    else:
        Ydot = np.gradient(Y, 1./N)
        Ydot = Ydot[0]
        sdot  = np.linalg.norm(Ydot, axis=1)

    s_int = cumtrapz(sdot, time, initial=0)
    # s_int = s_int - s_int.min()
    L = s_int[-1]
    if scale:
        arc_s_dot = interpolate.interp1d(time, sdot/L)
        grid_arc_s = (s_int - s_int.min()) / (s_int.max() - s_int.min())
        arc_s = interpolate.interp1d(time, grid_arc_s)
        Y = Y/L
    else:
        arc_s_dot = interpolate.interp1d(time, sdot)
        arc_s = interpolate.interp1d(time, s_int)
        grid_arc_s = (s_int - s_int.min()) / (s_int.max() - s_int.min())

    return grid_arc_s, L, arc_s, arc_s_dot
    


def compute_derivatives(Y, time, deg, h=None, CV_optimization_h={"flag":False, "h_grid":np.array([]), "K":10, "method":'bayesian', "n_call":10, "verbose":True}):

    N, dim = Y.shape
    if dim < 2:
        raise Exception("The Frenet Serret framework is defined only for curves in R^d with d >= 2.")
    if N != len(time):
        raise Exception("Number of sample points in attribute Y and time must be equal.")
    time = (time - time.min()) / (time.max() - time.min())

    if h is None:
        if CV_optimization_h["flag"]==False:
            raise Exception("You must choose a parameter h or set the optimization flag to true and give a grid for search of optimal h.")
        else:
            if len(CV_optimization_h["h_grid"])==0:
                raise Exception("You must give a grid for search of optimal h")
            else:
                if CV_optimization_h["method"]=='gridsearch':
                    h_grid = CV_optimization_h["h_grid"]
                    LP = LocalPolynomialSmoothing(deg)
                    h_opt, err_h = LP.grid_search_CV_optimization_bandwidth(Y, time, time, h_grid, K_split=CV_optimization_h["K"])
                    derivatives = LP.fit(Y, time, time, h_opt)
                else:
                    h_bounds = np.array([CV_optimization_h["h_grid"][0], CV_optimization_h["h_grid"][-1]])
                    LP = LocalPolynomialSmoothing(deg)
                    h_opt = LP.bayesian_optimization_hyperparameters(Y, time, time, CV_optimization_h["n_call"], h_bounds, n_splits=CV_optimization_h["K"], verbose=CV_optimization_h["verbose"])
                    derivatives = LP.fit(Y, time, time, h_opt)
    else:   
        h_opt = h
        derivatives = LocalPolynomialSmoothing(deg).fit(Y, time, time, h)

    return derivatives, h_opt




    
