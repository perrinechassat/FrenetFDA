import numpy as np
import os
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from geomstats.geometry.special_orthogonal import SpecialOrthogonal
from geomstats.geometry.discrete_curves import DiscreteCurves, SRVMetric
from geomstats.learning.frechet_mean import FrechetMean
from geomstats.geometry.matrices import Matrices
sys.stderr = stderr

class SO3:
    """Rotation matrix in :math:`SO(3)`
    .. math::
        SO(3) &= \\left\\{ \\mathbf{C} \\in \\mathbb{R}^{3 \\times 3}
        ~\\middle|~ \\mathbf{C}\\mathbf{C}^T = \\mathbf{1}, \\det
            \\mathbf{C} = 1 \\right\\} \\\\
        \\mathfrak{so}(3) &= \\left\\{ \\boldsymbol{\\Phi} =
        \\boldsymbol{\\phi}^\\wedge \\in \\mathbb{R}^{3 \\times 3}
        ~\\middle|~ \\boldsymbol{\\phi} = \\phi \\mathbf{a} \\in \\mathbb{R}
        ^3, \\phi = \\Vert \\boldsymbol{\\phi} \\Vert \\right\\}
    """

    #  tolerance criterion
    TOL = 1e-8
    Id_3 = np.eye(3)

    @classmethod
    def Ad(self, Rot):
        """Adjoint matrix of the transformation.
        .. math::
            \\text{Ad}(\\mathbf{C}) = \\mathbf{C}
            \\in \\mathbb{R}^{3 \\times 3}
        """
        return Rot

    @classmethod
    def exp(self, phi):
        """Exponential map for :math:`SO(3)`, which computes a transformation
        from a tangent vector:
        .. math::
            \\mathbf{C}(\\boldsymbol{\\phi}) =
            \\exp(\\boldsymbol{\\phi}^\wedge) =
            \\begin{cases}
                \\mathbf{1} + \\boldsymbol{\\phi}^\wedge,
                & \\text{if } \\phi \\text{ is small} \\\\
                \\cos \\phi \\mathbf{1} +
                (1 - \\cos \\phi) \\mathbf{a}\\mathbf{a}^T +
                \\sin \\phi \\mathbf{a}^\\wedge, & \\text{otherwise}
            \\end{cases}
        This is the inverse operation to :meth:`~ukfm.SO3.log`.
        """
        angle = np.linalg.norm(phi)
        if angle < self.TOL:
            # Near |phi|==0, use first order Taylor expansion
            Rot = self.Id_3 + SO3.wedge(phi)
        else:
            axis = phi / angle
            c = np.cos(angle)
            s = np.sin(angle)
            Rot = c * self.Id_3 + (1-c)*np.outer(axis,axis) + s * self.wedge(axis)
        return Rot

    @classmethod
    def inv_left_jacobian(self, phi):
        """:math:`SO(3)` inverse left Jacobian
        .. math::
            \\mathbf{J}^{-1}(\\boldsymbol{\\phi}) =
            \\begin{cases}
                \\mathbf{1} - \\frac{1}{2} \\boldsymbol{\\phi}^\wedge, &
                    \\text{if } \\phi \\text{ is small} \\\\
                \\frac{\\phi}{2} \\cot \\frac{\\phi}{2} \\mathbf{1} +
                \\left( 1 - \\frac{\\phi}{2} \\cot \\frac{\\phi}{2}
                \\right) \\mathbf{a}\\mathbf{a}^T -
                \\frac{\\phi}{2} \\mathbf{a}^\\wedge, &
                \\text{otherwise}
            \\end{cases}
        """
        angle = np.linalg.norm(phi)
        if angle < self.TOL:
            # Near |phi|==0, use first order Taylor expansion
            J = np.eye(3) - 1/2 * self.wedge(phi)
        else:
            axis = phi / angle
            half_angle = angle/2
            cot = 1 / np.tan(half_angle)
            J = half_angle * cot * self.Id_3 + \
                (1 - half_angle * cot) * np.outer(axis, axis) -\
                half_angle * self.wedge(axis)
        return J

    @classmethod
    def left_jacobian(self, phi):
        """:math:`SO(3)` left Jacobian.
        .. math::
            \\mathbf{J}(\\boldsymbol{\\phi}) =
            \\begin{cases}
                \\mathbf{1} + \\frac{1}{2} \\boldsymbol{\\phi}^\wedge, &
                    \\text{if } \\phi \\text{ is small} \\\\
                \\frac{\\sin \\phi}{\\phi} \\mathbf{1} +
                \\left(1 - \\frac{\\sin \\phi}{\\phi} \\right)
                \\mathbf{a}\\mathbf{a}^T +
                \\frac{1 - \\cos \\phi}{\\phi} \\mathbf{a}^\\wedge, &
                \\text{otherwise}
            \\end{cases}
        """
        angle = np.linalg.norm(phi)
        if angle < self.TOL:
            # Near |phi|==0, use first order Taylor expansion
            J = self.Id_3 - 1/2 * SO3.wedge(phi)
        else:
            axis = phi / angle
            s = np.sin(angle)
            c = np.cos(angle)
            J = (s / angle) * self.Id_3 + \
                (1 - s / angle) * np.outer(axis, axis) +\
                ((1 - c) / angle) * self.wedge(axis)
        return J

    @classmethod
    def log(self, Rot):
        """Logarithmic map for :math:`SO(3)`, which computes a tangent vector
        from a transformation:
        .. math::
            \\phi &= \\frac{1}{2}
            \\left( \\mathrm{Tr}(\\mathbf{C}) - 1 \\right) \\\\
            \\boldsymbol{\\phi}(\\mathbf{C}) &=
            \\ln(\\mathbf{C})^\\vee =
            \\begin{cases}
                \\mathbf{1} - \\boldsymbol{\\phi}^\wedge,
                & \\text{if } \\phi \\text{ is small} \\\\
                \\left( \\frac{1}{2} \\frac{\\phi}{\\sin \\phi}
                \\left( \\mathbf{C} - \\mathbf{C}^T \\right)
                \\right)^\\vee, & \\text{otherwise}
            \\end{cases}
        This is the inverse operation to :meth:`~ukfm.SO3.log`.
        """
        cos_angle = 0.5 * np.trace(Rot) - 0.5
        # Clip np.cos(angle) to its proper domain to avoid NaNs from rounding
        # errors
        cos_angle = np.min([np.max([cos_angle, -1]), 1])
        angle = np.arccos(cos_angle)

        # If angle is close to zero, use first-order Taylor expansion
        if np.linalg.norm(angle) < self.TOL:
            phi = self.vee(Rot - self.Id_3)
        else:
            # Otherwise take the matrix logarithm and return the rotation vector
            phi = self.vee((0.5 * angle / np.sin(angle)) * (Rot - Rot.T))
        return phi

    @classmethod
    def to_rpy(self, Rot):
        """Convert a rotation matrix to RPY Euler angles
        :math:`(\\alpha, \\beta, \\gamma)`."""

        pitch = np.arctan2(-Rot[2, 0], np.sqrt(Rot[0, 0]**2 + Rot[1, 0]**2))

        if np.linalg.norm(pitch - np.pi/2) < self.TOL:
            yaw = 0
            roll = np.arctan2(Rot[0, 1], Rot[1, 1])
        elif np.linalg.norm(pitch + np.pi/2.) < 1e-9:
            yaw = 0.
            roll = -np.arctan2(Rot[0, 1], Rot[1, 1])
        else:
            sec_pitch = 1. / np.cos(pitch)
            yaw = np.arctan2(Rot[1, 0] * sec_pitch, Rot[0, 0] * sec_pitch)
            roll = np.arctan2(Rot[2, 1] * sec_pitch, Rot[2, 2] * sec_pitch)

        rpy = np.array([roll, pitch, yaw])
        return rpy

    @classmethod
    def vee(self, Phi):
        """:math:`SO(3)` vee operator as defined by
        :cite:`barfootAssociating2014`.
        .. math::
            \\phi = \\boldsymbol{\\Phi}^\\vee
        This is the inverse operation to :meth:`~ukfm.SO3.wedge`.
        """
        phi = np.array([Phi[2, 1], Phi[0, 2], Phi[1, 0]])
        return phi

    @classmethod
    def wedge(self, phi):
        """:math:`SO(3)` wedge operator as defined by
        :cite:`barfootAssociating2014`.
        .. math::
            \\boldsymbol{\\Phi} =
            \\boldsymbol{\\phi}^\\wedge =
            \\begin{bmatrix}
                0 & -\\phi_3 & \\phi_2 \\\\
                \\phi_3 & 0 & -\\phi_1 \\\\
                -\\phi_2 & \\phi_1 & 0
            \\end{bmatrix}
        This is the inverse operation to :meth:`~ukfm.SO3.vee`.
        """
        Phi = np.array([[0, -phi[2], phi[1]],
                        [phi[2], 0, -phi[0]],
                        [-phi[1], phi[0], 0]])
        return Phi

    @classmethod
    def from_rpy(self, roll, pitch, yaw):
        """Form a rotation matrix from RPY Euler angles
        :math:`(\\alpha, \\beta, \\gamma)`.
        .. math::

            \\mathbf{C} = \\mathbf{C}_z(\\gamma) \\mathbf{C}_y(\\beta)
            \\mathbf{C}_x(\\alpha)
        """
        return self.rotz(yaw).dot(self.roty(pitch).dot(self.rotx(roll)))

    @classmethod
    def rotx(self, angle_in_radians):
        """Form a rotation matrix given an angle in rad about the x-axis.
        .. math::
            \\mathbf{C}_x(\\phi) =
            \\begin{bmatrix}
                1 & 0 & 0 \\\\
                0 & \\cos \\phi & -\\sin \\phi \\\\
                0 & \\sin \\phi & \\cos \\phi
            \\end{bmatrix}
        """
        c = np.cos(angle_in_radians)
        s = np.sin(angle_in_radians)

        return np.array([[1., 0., 0.],
                         [0., c, -s],
                         [0., s,  c]])

    @classmethod
    def roty(self, angle_in_radians):
        """Form a rotation matrix given an angle in rad about the y-axis.
        .. math::
            \\mathbf{C}_y(\\phi) =
            \\begin{bmatrix}
                \\cos \\phi & 0 & \\sin \\phi \\\\
                0 & 1 & 0 \\\\
                \\sin \\phi & 0 & \\cos \\phi
            \\end{bmatrix}
        """
        c = np.cos(angle_in_radians)
        s = np.sin(angle_in_radians)

        return np.array([[c,  0., s],
                         [0., 1., 0.],
                         [-s, 0., c]])

    @classmethod
    def rotz(self, angle_in_radians):
        """Form a rotation matrix given an angle in rad about the z-axis.
        .. math::

            \\mathbf{C}_z(\\phi) =
            \\begin{bmatrix}
                \\cos \\phi & -\\sin \\phi & 0 \\\\
                \\sin \\phi  & \\cos \\phi & 0 \\\\
                0 & 0 & 1
            \\end{bmatrix}
        """
        c = np.cos(angle_in_radians)
        s = np.sin(angle_in_radians)

        return np.array([[c, -s,  0.],
                         [s,  c,  0.],
                         [0., 0., 1.]])


    @classmethod
    def geodesic_distance(self, Q1, Q2):
        """ pointwise distance 
        """
        so3 = SpecialOrthogonal(3)
        gdist = so3.metric.dist(Q1, Q2)
        return gdist
    

    @classmethod
    def srv_distance(self, Q1, Q2):
        so3 = SpecialOrthogonal(3, point_type='vector')
        Q1_vec = so3.rotation_vector_from_matrix(Q1)
        Q2_vec = so3.rotation_vector_from_matrix(Q2)
        N = Q1.shape[0]
        dc = DiscreteCurves(so3, k_sampling_points=N, start_at_the_origin=False)
        srv_Q1 = SRVMetric(dc).f_transform(Q1_vec)
        srv_Q2 = SRVMetric(dc).f_transform(Q2_vec)
        dist = np.sqrt(self.geodesic_distance(Q1[0],Q2[0])**2 + np.linalg.norm(srv_Q1-srv_Q2)**2)
        return dist 


    @classmethod
    def frechet_mean(self, arr_R, weights=None):
        """ pointwise distance 
        """
        # try:
        so3 = SpecialOrthogonal(3)
        mean = FrechetMean(metric=so3.metric, method='adaptive')
        mean.fit(arr_R, weights=weights)
        return mean.estimate_

        # except:
        #     print('mean with projections')
        #     so3 = SpecialOrthogonal(3)
        #     mean = FrechetMean(metric=so3.metric)
        #     arr_R_bis = so3.projection(arr_R)
        #     mean.fit(arr_R_bis, weights=weights)
        #     return mean.estimate_


    @classmethod
    def random_point_uniform(self, n_samples, bound=1):
        so3 = SpecialOrthogonal(3)
        return so3.random_point(n_samples, bound=bound)


    @classmethod
    def random_point_fisher(self, n_samples, K, mean_directions=None):
        random_points = np.zeros((n_samples, 3, 3))
        if mean_directions is None:
            for i in range(n_samples):
                random_points[i] = self.__matrix_rnd(K*np.eye(3))
        else:
            for i in range(n_samples):
                random_points[i] = self.__matrix_rnd(K*np.eye(3)) @ mean_directions[i]
        return random_points

    @classmethod
    def __matrix_rnd(self, F):
        '''
        Simulate one observation from Matrix Fisher Distribution in SO(3) parametrized
        with matrix F = K* Omega, where K is the concentration matrix and Omega is the mean direction.
        Algorithm proposed by Michael Habeck,  "Generation of three dimensional random rotations in fitting
        and matching problems", Computational Statistics, 2009.
        ...
        '''
        U,L,V = np.linalg.svd(F, hermitian=True)
        alpha,beta,gamma,_,_,_ = self.__euler_gibbs_sampling(L,600)
        S = np.zeros((3,3))
        cosa = np.cos(alpha)
        sina = np.sin(alpha)
        cosb = np.cos(beta)
        sinb = np.sin(beta)
        cosg = np.cos(gamma)
        sing = np.sin(gamma)

        S[0,0] = cosa*cosb*cosg-sina*sing
        S[0,1] = sina*cosb*cosg+cosa*sing
        S[0,2] = -sinb*cosg
        S[1,0] = -cosa*cosb*sing-sina*cosg
        S[1,1] = -sina*cosb*sing+cosa*cosg
        S[1,2] = sinb*sing
        S[2,0] = cosa*sinb
        S[2,1] = sina*sinb
        S[2,2] = cosb

        R = U @ S @ V
        return R


    @classmethod
    def __euler_gibbs_sampling(self, Lambda, n):
        '''
        Gibbs sampling by Habeck 2009 for the simulation of random matrices
        ...
        '''
        beta = 0
        Alpha = np.zeros(n)
        Beta  = np.zeros(n)
        Gamma = np.zeros(n)
        for i in range(n):
            kappa_phi = (Lambda[0]+Lambda[1])*(np.cos(0.5*beta))**2
            kappa_psi = (Lambda[0]-Lambda[1])*(np.sin(0.5*beta))**2
            if kappa_phi < 1e-6:
                phi = 2*np.pi*np.random.random()
            else:
                phi = np.random.vonmises(0,kappa_phi)
            psi = 2*np.pi*np.random.random()
            u  = int(np.random.random()<0.5)
            alpha = (phi+psi)/2 + np.pi*u
            gamma = (phi-psi)/2 + np.pi*u
            Alpha[i] = alpha
            Gamma[i] = gamma
            kappa_beta = (Lambda[0]+Lambda[1])*np.cos(phi)+(Lambda[0]-Lambda[1])*np.cos(psi)+2*Lambda[2]
            r = np.random.random()
            x = 1+2*np.log(r+(1-r)*np.exp(-kappa_beta))/kappa_beta
            beta = np.arccos(x)
            Beta[i] = beta

        return alpha,beta,gamma,Alpha,Beta,Gamma