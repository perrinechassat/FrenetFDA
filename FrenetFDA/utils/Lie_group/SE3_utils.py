import numpy as np
from FrenetFDA.utils.Lie_group.SO3_utils import SO3
import os
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from geomstats.geometry.special_euclidean import SpecialEuclidean
sys.stderr = stderr

class SE3:
    """Homogeneous transformation matrix in :math:`SE(3)`.
    .. math::
        SE(3) &= \\left\\{ \\mathbf{T}=
                \\begin{bmatrix}
                    \\mathbf{C} & \\mathbf{r} \\\\
                        \\mathbf{0}^T & 1
                \\end{bmatrix} \\in \\mathbb{R}^{4 \\times 4} ~\\middle|~
                \\mathbf{C} \\in SO(3), \\mathbf{r} \\in \\mathbb{R}^3
                \\right\\} \\\\
        \\mathfrak{se}(3) &= \\left\\{ \\boldsymbol{\\Xi} =
        \\boldsymbol{\\xi}^\\wedge \\in \\mathbb{R}^{4 \\times 4} ~\\middle|~
         \\boldsymbol{\\xi}=
            \\begin{bmatrix}
                \\boldsymbol{\\phi} \\\\ \\boldsymbol{\\rho}
            \\end{bmatrix} \\in \\mathbb{R}^6, \\boldsymbol{\\rho} \\in
            \\mathbb{R}^3, \\boldsymbol{\\phi} \in \\mathbb{R}^3 \\right\\}
    """
    TOL = 1e-8

    @classmethod
    def wedge(self, xi):
        chi = np.zeros((4,4))
        chi[:3,:3] = SO3.wedge(xi[:3])
        chi[:3,3] = xi[3:]
        return chi

    @classmethod
    def vee(self, chi):
        """:math:`SE(3)` vee operator as defined by
        :cite:`barfootAssociating2014`.
        .. math::
            \\phi = \\boldsymbol{\\Phi}^\\vee
        This is the inverse operation to :meth:`~ukfm.SO3.wedge`.
        """
        xi = np.zeros((6))
        xi[:3] = SO3.vee(chi[:3,:3])
        xi[3:] = chi[:3,3]
        return xi
    
    @classmethod
    def Ad(self, xi):
        """Adjoint matrix in Lie Algebra.
        .. math::
            \\text{Ad}(\\mathbf{C}) = \\mathbf{C}
            \\in \\mathbb{R}^{3 \\times 3}
        """
        Ad_mat = np.zeros((6,6))
        Ad_mat[:3,:3] = SO3.wedge(xi[:3])
        Ad_mat[3:,3:] = SO3.wedge(xi[:3])
        Ad_mat[3:,:3] = SO3.wedge(xi[3:])
        return Ad_mat
    
    @classmethod
    def Ad_group(self, chi):
        """Adjoint matrix in Lie group.
        """
        Ad_mat = np.zeros((6,6))
        Ad_mat[:3,:3] = chi[:3, :3]
        Ad_mat[3:,3:] = chi[:3, :3]
        Ad_mat[3:,:3] = SO3.wedge(SO3.left_jacobian(chi[:3, 3]).dot(chi[:3, 3]))@chi[:3, :3]
        return Ad_mat

    @classmethod
    def exp(self, xi):
        """Exponential map for :math:`SE(3)`, which computes a transformation
        from a tangent vector:
        .. math::
            \\mathbf{T}(\\boldsymbol{\\xi}) =
            \\exp(\\boldsymbol{\\xi}^\\wedge) =
            \\begin{bmatrix}
                \\exp(\\boldsymbol{\\phi}^\\wedge) & \\mathbf{J}
                \\boldsymbol{\\rho}  \\\\
                \\mathbf{0} ^ T & 1
            \\end{bmatrix}
        This is the inverse operation to :meth:`~ukfm.SE3.log`.
        """
        chi = np.eye(4)
        chi[:3, :3] = SO3.exp(xi[:3])
        chi[:3, 3] = SO3.left_jacobian(xi[:3]).dot(xi[3:])
        return chi

    @classmethod
    def inv(self, chi):
        """Inverse map for :math:`SE(3)`.
        .. math::
            \\mathbf{T}^{-1} =
            \\begin{bmatrix}
                \\mathbf{C}^T  & -\\mathbf{C}^T \\boldsymbol{\\rho}
                    \\\\
                \\mathbf{0} ^ T & 1
            \\end{bmatrix}
        """
        chi_inv = np.eye(4)
        chi_inv[:3, :3] = chi[:3, :3].T
        chi_inv[:3, 3] = -chi[:3, :3].T.dot(chi[:3, 3])
        return chi_inv

    @classmethod
    def log(self, chi):
        """Logarithmic map for :math:`SE(3)`, which computes a tangent vector
        from a transformation:
        .. math::

            \\boldsymbol{\\xi}(\\mathbf{T}) =
            \\ln(\\mathbf{T})^\\vee =
            \\begin{bmatrix}
                \\mathbf{J} ^ {-1} \\mathbf{r} \\\\
                \\ln(\\boldsymbol{C}) ^\\vee
            \\end{bmatrix}
        This is the inverse operation to :meth:`~ukfm.SE3.exp`.
        """
        phi = SO3.log(chi[:3, :3])
        xi = np.hstack([phi, SO3.inv_left_jacobian(phi).dot(chi[:3, 3])])
        return xi

    @classmethod
    def left_jacobian(self, xi):
        """:math:`SE(3)` left Jacobian.
        """
        phi = xi[:3]
        rho = xi[3:]

        J_phi = SO3.left_jacobian(phi)
        J = np.zeros((6,6))
        J[:3,:3] = J_phi
        J[3:,3:] = J_phi
        angle = np.linalg.norm(phi)
        if angle < self.TOL:
            c1 = 1/6
            c2 = 1/24
            c3 = 1/120
        else:
            s = np.sin(angle)
            c = np.cos(angle)
            c1 = (angle - s)/np.power(angle, 3)
            c2 = (angle**2 + 2*c - 2)/(2*np.power(angle, 4))
            c3 = (2*angle - 3*s + angle*c)/(2*np.power(angle, 5))

        phi_mat = SO3.wedge(phi)
        rho_mat = SO3.wedge(rho)
        phi_rho = phi_mat @ rho_mat
        rho_phi = rho_mat @ phi_mat
        phi_phi = phi_mat @ phi_mat
        Q = (1/2)*SO3.wedge(rho) + c1*(phi_rho + rho_phi + phi_rho @ phi_mat) + c2*(phi_mat @ phi_rho + rho_phi @ phi_mat - 3*phi_rho @ phi_mat) + c3*(phi_rho @ phi_phi + phi_phi @ rho_phi)
        J[3:,:3] = Q
        return J

    @classmethod
    def compute_L(self, i, j):
        L = np.zeros(6)
        # u3
        if (i==0 and j==1) or (i==3 and j==4):
            L[2] = -1
        elif (i==1 and j==0) or (i==4 and j==3):
            L[2] = 1
        # u2
        elif (i==0 and j==2) or (i==3 and j==5):
            L[1] = 1
        elif (i==2 and j==0) or (i==5 and j==3):
            L[1] = -1
        # u1
        elif (i==1 and j==2) or (i==4 and j==5):
            L[0] = -1
        elif (i==2 and j==1) or (i==5 and j==4):
            L[0] = 1
        # a3
        elif (i==3 and j==1):
            L[5] = -1
        elif (i==4 and j==0):
            L[5] = 1
        # a2
        elif (i==5 and j==0):
            L[4] = -1
        elif (i==3 and j==2):
            L[4] = 1
        # a1
        elif (i==4 and j==2):
            L[3] = -1
        elif (i==5 and j==1):
            L[3] = 1
        return L


    @classmethod
    def geodesic_distance(self, Z1, Z2):
        SE3 = SpecialEuclidean(3)
        gdist = SE3.metric.dist(Z1, Z2)
        return gdist