import numpy as np
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.special_orthogonal import SpecialOrthogonal

sphere = Hypersphere(dim=2)

def phi(t):
    return 5*(t+1)
    # return 10*(np.sin(t)+2)

def theta(t):
    return 10*t+0.5
    # return 5*t + 10

def exponential_map(v1,v2):
    tangent_vector = sphere.to_tangent(v2, base_point=v1)
    return sphere.metric.exp(tangent_vector, v1)

def rotation_matrix_from_vectors(vec1,vec2):
    if np.linalg.norm(vec2)==0:
        a,b = (vec1 / np.linalg.norm(vec1)).reshape(3), vec2
    else:
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    SO3 = SpecialOrthogonal(3)
    r_proj = SO3.projection(rotation_matrix)
    return r_proj

def mu(t):
    return exponential_map(np.array([0,0,1]), np.array([np.cos(theta(t))*phi(t), np.sin(theta(t))*phi(t), 0]))

def R(t):
    return rotation_matrix_from_vectors(np.array([0,0,1]), mu(t))

def Ei(k,N):
    return np.random.normal(0,np.sqrt(np.power(0.007,k/2)),N)

def V(t,f):
    return np.squeeze(R(t) @ np.array([f(t/2), f((t+1)/2), 0])[:,np.newaxis])

def X(t,f):
    return exponential_map(mu(t), V(t,f))

def generative_model_spherical_curves(n_curves, K, n_points, domain_range):
    T = np.linspace(domain_range[0], domain_range[1], n_points)
    Mu = np.zeros((n_points,3))
    for j in range(n_points):
        Mu[j] = mu(T[j])

    Ei_tab = np.zeros((K,n_curves))
    # for k in range(K):
    #     Ei_tab[k,:] = Ei(k,n_curves)
    #     print(np.mean(Ei_tab[k,:]))
    # Ei_tab = np.random.normal(0,1,(K,n_curves))
    # print(np.mean(Ei_tab))

    # for i in range(n_curves):
    #     Ei_tab[:,i] = np.random.normal(0,1,K)
    #     # Ei_tab[:,i] = np.random.uniform(-np.pi/4,np.pi/4,K)
    #     for k in range(K):
    #         Ei_tab[k,i] = np.sqrt(np.power(0.07,(k+1)/2))*Ei_tab[k,i]
    #         # Ei_tab[k,i] = np.sqrt(2*np.power(k+1,-1/2))*Ei_tab[k,i]
    
    for i in range(n_curves):
        for k in range(K):
            Ei_tab[k,i] = np.random.normal(0,np.power(0.07,(k+1)/2))

    X_tab = np.zeros((n_curves,n_points,3))
    V_tab = np.zeros((n_curves,n_points,3))
    for i in range(n_curves):
        # list_coef = [Ei_tab[k,i] for k in range(K)] + [0]
        list_coef = [Ei_tab[k,i] for k in range(K)]
        f = np.polynomial.legendre.Legendre(list_coef, domain=[0,1], window=[0,1])
        # Fourier_basis = Fourier((0,1), n_basis=K)
        # fd_basis = FDataBasis(
        #     basis=Fourier_basis,
        #     coefficients=[list_coef,  # First (and unique) observation
        #     ],
        # )
        # f = lambda t: np.squeeze(fd_basis.evaluate(t))
        for j in range(n_points):
            X_tab[i,j,:] = X(T[j], f)
            V_tab[i,j,:] = V(T[j], f)

    return Mu, X_tab, V_tab
