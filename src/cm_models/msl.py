from __future__ import annotations

import abc
import functools

import typing

import numpy as np
import math

from . import abstracts, util

if typing.TYPE_CHECKING:
    import numpy.typing as npt


class _DetailsSuperlattice(typing.NamedTuple):
    """Cached details of superlattice"""

    num_vectors: int
    """Number of Bragg vectors in the basis"""

    m: npt.NDArray[np.integer]
    """Indices of each Bragg vector in the basis

    Array of integers of shape ``(num_vectors, msl_dims)``.

    Each :math:`m = (m_1, ...)`, gives a general vector, 
    :math:`m_1 \\boldsymbol{G}_1 + ...`, summed over the basis of Bragg 
    vectors, :math:`(\\boldsymbol{G}_1, ...)` (see ``G_basis``).
    
    """

    G: npt.NDArray[np.floating]
    """Each Bragg vector in the basis

    Array of floats of shape ``(num_vectors, msl_dims)``.
    
    """

    num_partial_vectors: int
    """Number of Bragg vectors in the partial basis
    
    Removes the zero Bragg vector and vectors related by symmetry operations 
    such as rotation and inversion.
    
    """

    m_partial: npt.NDArray[np.integer]
    """Indices of Bragg vectors in the partial basis
    
    Removes the zero Bragg vector and vectors related by symmetry operations 
    such as rotation and inversion.

    Array of integers of shape ``(num_partial_vectors, msl_dims)``.

    Each :math:`m = (m_1, ...)`, gives a general vector, 
    :math:`m_1 \\boldsymbol{G}_1 + ...`, summed over the basis of Bragg 
    vectors, :math:`(\\boldsymbol{G}_1, ...)` (see ``G_basis``).

    Equal to ``m[1:num_partial_vectors+1]``.
    
    """

    G_partial: npt.NDArray[np.floating]
    """Each Bragg vector in the partial basis
    
    Removes the zero Bragg vector and vectors related by symmetry operations 
    such as rotation and inversion.

    Array of floats of shape ``(num_partial_vectors, msl_dims)``.

    Equal to ``G[1:num_partial_vectors+1]``.
    
    """


class Superlattice(abstracts.Base):
    """Base superlattice class"""

    msl_dims: int
    """Number of superlattice dimensions."""

    A_basis: npt.NDArray[np.floating]
    """basis of lattice vectors, shape = (dims of superlattice, dims of lattice)"""

    G_basis: npt.NDArray[np.floating]
    """basis of Bragg vectors, shape = (dims of superlattice, dims of lattice)"""

    m_first: npt.NDArray[np.integer]
    """First star Bragg vector indices"""

    def get_A(self, n: npt.ArrayLike) -> npt.NDArray[np.floating]:
        """Convert indices to lattice vectors"""

        return np.matmul(n, self.A_basis)

    def get_G(self, m: npt.ArrayLike) -> npt.NDArray[np.floating]:
        """Convert indices to Bragg vectors"""

        return np.matmul(m, self.G_basis)

    @abc.abstractmethod
    @functools.cache
    @util.process_args
    def details_superlattice(self, **model_kwargs) -> _DetailsSuperlattice:
        """Get the superlattice details"""

    @property
    def G_first(self) -> npt.NDArray[np.floating]:
        """First star Bragg vectors"""

        return self.get_G(self.m_first)


class Superlattice1D2D(Superlattice):
    """1D superlattice in 2D space"""

    msl_dims = 1

    extra_model_parameters = dict(
        period=(float, "Superlattice period [nm]"),
        phi=(0.0, "Superlattice angle [rad]"),
    )

    m_first = np.array([[1], [-1]])

    def fill_namespace(self):
        # length of shortest non-zero Bragg vector
        self.G0 = 2.0 * math.pi / self.period

        self.A_basis = self.period * np.array(
            [[math.cos(self.phi), math.sin(self.phi)]]
        )
        self.G_basis = (self.G0 / self.period) * self.A_basis

    @functools.cache
    @util.process_args
    def details_superlattice(self, *, M: int, **model_kwargs) -> _DetailsSuperlattice:
        """Get the superlattice details with ``M`` stars"""

        # non-zero positive vectors within number of stars
        m1_p = np.arange(1, M + 1)

        # full array of vectors
        num_vectors = 2 * M + 1
        m = np.empty((num_vectors, 1), dtype=int)
        m[0] = 0  # first vector is zero vector
        m[1 : M + 1, 0] = m1_p  # positive vectors
        m[M + 1 :] = -m[1 : M + 1]  # negative vectors

        G = self.get_G(m)

        return _DetailsSuperlattice(
            num_partial_vectors=M,
            # views to save memory
            m_partial=m[1 : M + 1],
            G_partial=G[1 : M + 1],
            num_vectors=num_vectors,
            m=m,
            G=G,
        )


class SuperlatticeSquare2D(Superlattice):
    """2D square superlattice in 2D space"""

    msl_dims = 2

    extra_model_parameters = dict(
        period=(float, "Superlattice period [nm]"),
        phi=(0.0, "Superlattice angle [rad]"),
    )

    m_first = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])

    def fill_namespace(self):
        # length of shortest non-zero Bragg vector
        self.G0 = 2.0 * math.pi / self.period

        angle = np.array([self.phi, self.phi + 0.5 * math.pi])
        self.A_basis = self.period * np.stack([np.cos(angle), np.sin(angle)], axis=1)
        self.G_basis = (self.G0 / self.period) * self.A_basis

    @functools.cache
    @util.process_args
    def details_superlattice(self, *, M: int, **model_kwargs) -> _DetailsSuperlattice:
        """Get the superlattice details with ``M`` stars"""

        # non-zero vectors in first quadrant within number of stars
        m1_q, m2_q = map(
            np.ndarray.flatten,
            np.broadcast_arrays(np.arange(1, M + 1)[:, None], np.arange(M + 1)),
        )
        discriminant = m1_q**2 + m2_q**2  # integer proportional to square length

        # sort by discriminant and truncate
        index = discriminant.argsort()
        m1_q, m2_q, discriminant = (
            array[index] for array in (m1_q, m2_q, discriminant)
        )
        unique_discriminant = np.unique(discriminant)
        mask = discriminant < unique_discriminant[M]
        npv = num_partial_vectors = mask.sum()

        # full array of vectors
        num_vectors = 4 * npv + 1
        m = np.empty((num_vectors, 2), dtype=int)

        m[0] = 0  # first vector is zero vector

        # first quadrant
        m[1 : npv + 1, 0] = m1_q[mask]
        m[1 : npv + 1, 1] = m2_q[mask]

        # second quadrant is rotated by 90 degrees
        m[npv + 1 : 2 * npv + 1, 0] = -m[1 : npv + 1, 1]
        m[npv + 1 : 2 * npv + 1, 1] = m[1 : npv + 1, 0]

        # final quadrants are 180 degree rotations of previous quadrants
        m[2 * npv + 1 :] = -m[1 : 2 * npv + 1]

        G = self.get_G(m)

        return _DetailsSuperlattice(
            num_partial_vectors=num_partial_vectors,
            # views to save memory
            m_partial=m[1 : npv + 1],
            G_partial=G[1 : npv + 1],
            num_vectors=num_vectors,
            m=m,
            G=G,
        )


class SuperlatticeTriangular2D(Superlattice):
    """2D triangular superlattice in 2D space"""

    msl_dims = 2

    extra_model_parameters = dict(
        period=(float, "Superlattice period [nm]"),
        phi=(0.0, "Superlattice angle [rad]"),
    )

    m_first = np.array([[1, -1], [1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1]])

    def fill_namespace(self):
        # length of shortest non-zero Bragg vector
        self.G0 = 4.0 * math.pi / (math.sqrt(3.0) * self.period)

        # note that we use G1 and G2 as basis (not G0 and G1), by convention
        angle = np.array([self.phi + math.pi / 3.0, self.phi + 2.0 * math.pi / 3.0])
        cos = np.cos(angle)
        sin = np.sin(angle)
        self.A_basis = self.period * np.stack([cos, sin], axis=1)
        self.G_basis = self.G0 * np.stack([-sin, cos], axis=1)

    @functools.cache
    @util.process_args
    def details_superlattice(self, *, M: int, **model_kwargs) -> _DetailsSuperlattice:
        """Get the superlattice details with ``M`` stars"""

        # non-zero vectors in first sextant within number of stars
        m1_s, m2_s = map(
            np.ndarray.flatten,
            np.broadcast_arrays(np.arange(1, M + 1)[:, None], np.arange(M + 1)),
        )
        # integer proportional to square length
        discriminant = m1_s**2 + m1_s * m2_s + m2_s**2

        # sort by discriminant and truncate
        index = discriminant.argsort()
        m1_s, m2_s, discriminant = (
            array[index] for array in (m1_s, m2_s, discriminant)
        )
        unique_discriminant = np.unique(discriminant)
        mask = discriminant < unique_discriminant[M]
        npv = num_partial_vectors = mask.sum()

        # full array of vectors
        num_vectors = 6 * npv + 1
        m = np.empty((num_vectors, 2), dtype=int)

        m[0] = 0  # first vector is zero vector

        # first sextant
        m[1 : npv + 1, 0] = m1_s[mask]
        m[1 : npv + 1, 1] = m2_s[mask]

        # second and third sextants are rotated by 60 and 120 degrees
        m[npv + 1 : 2 * npv + 1, 0] = -m[1 : npv + 1, 1]
        m[npv + 1 : 2 * npv + 1, 1] = m[1 : npv + 1, 0] + m[1 : npv + 1, 1]
        m[2 * npv + 1 : 3 * npv + 1, 0] = -m[npv + 1 : 2 * npv + 1, 1]
        m[2 * npv + 1 : 3 * npv + 1, 1] = (
            m[npv + 1 : 2 * npv + 1, 0] + m[npv + 1 : 2 * npv + 1, 1]
        )

        # final sextants are 180 degree rotations of previous sextants
        m[3 * npv + 1 :] = -m[1 : 3 * npv + 1]

        G = self.get_G(m)

        return _DetailsSuperlattice(
            num_partial_vectors=num_partial_vectors,
            # views to save memory
            m_partial=m[1 : npv + 1],
            G_partial=G[1 : npv + 1],
            num_vectors=num_vectors,
            m=m,
            G=G,
        )
