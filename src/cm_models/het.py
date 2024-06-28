from __future__ import annotations

import functools
import itertools

import typing
import numbers
from collections.abc import Iterable, Iterator

import numpy as np

from . import abstracts, msl, util

if typing.TYPE_CHECKING:
    import numpy.typing as npt


class ModelSuperlattice(abstracts.Model, msl.Superlattice):
    """Arbitrary dimensional electronic system with superlattice"""


class _DetailsSuper(typing.NamedTuple):
    """Cached details of zone-folded calculation"""

    num_extended_vectors: int
    """Number of Bragg vectors in the extended basis
    
    Includes Bragg vectors outside the finite basis, ``G``, which appear as 
    intermediate vectors in the superlattice matrix elements.
    
    """

    m_extended: npt.NDArray[np.integer]
    """Indices of Bragg vectors in the extended basis
    
    Includes Bragg vectors outside the finite basis, ``G``, which appear as 
    intermediate vectors in the superlattice matrix elements.

    Array of integers of shape ``(num_extended_vectors, msl_dims)``.

    Each :math:`m = (m_1, ...)`, gives a general vector, 
    :math:`m_1 \\boldsymbol{G}_1 + ...`, summed over the basis of Bragg 
    vectors, :math:`(\\boldsymbol{G}_1, ...)` (see ``G_basis``).
    
    """

    G_extended: npt.NDArray[np.floating]
    """Each Bragg vector in the extended basis
    
    Includes Bragg vectors outside the finite basis, ``G``, which appear as 
    intermediate vectors in the superlattice matrix elements.

    Array of floats of shape ``(num_partial_vectors, msl_dims)``.
    
    """

    basis_super: tuple
    """Basis of zone-folded Hamiltonian
    
    Corresponds to sublattices ``\\otimes`` Bragg vector indices, ``m``.
    
    """

    shape_super: tuple[int]
    """Shape of zone-folded Hamiltonian
    
    Equal to ``(dim, num_vectors)``.
    
    """

    dim_super: int
    """Dimensionality of zone-folded Hamiltonian
    
    Equal to ``dim * num_vectors``.
    
    """

    processed_super_elements: tuple[int, int, int, int, numbers.Number, tuple, bool]
    """Processed zone-folded elements

    Added to expanded zone-folded Hamiltonian, ``H`` of shape 
    ``(*shape_super, *shape_super)``, according to
    ``H[n1, ind1, n2, ind2] += value * processed_super_operators``.
    TODO: elaborate on meaning of operators
    
    Yields
    -------
    n1 : array of integers
        Component 1 in sublattice basis (first component of ``shape_super``).
    ind1 : array of integers
        Component 1 in Bragg vector basis (second component of 
        ``shape_super``).
    n2 : array of integers
        Component 2 in sublattice basis (first component of ``shape_super``).
    ind2 : array of integers
        Component 2 in Bragg vector basis (second component of 
        ``shape_super``).
    value : Number
        Scalar value that operators are multiplied by.
    processed_super_operators : tuple
        Tuple of ``(is_conjugate, m)`` pairs where ``is_conjugate=False`` 
        is :math:`\\hat{\kappa}` and True is :math:`\\hat{\kappa}^\dagger` and 
        ``m`` is the index of a Bragg vector in the basis - see 
        ``details_super().m``.
    is_off_diagonal : bool
        Whether the element is off the diagonal. Equal to ``n1!=n2``.
    
    """


class ModelSuperlattice2D(abstracts.Model2D, ModelSuperlattice):
    """2D electronic system with superlattice"""

    @staticmethod
    def get_element_super(
        value: numbers.Number,
        operators: Iterable[bool],
        k: npt.NDArray[np.complexfloating],
        kc: npt.NDArray[np.complexfloating],
    ) -> npt.ArrayLike:
        """Evaluate a zone-folded matrix element at each momentum point"""

        if operators:
            k_arrays = (
                kc[..., ind] if is_conjugate else k[..., ind]
                for (is_conjugate, ind) in operators
            )

            element = value * next(k_arrays)
            for k_array in k_arrays:
                element *= k_array

            return element
        else:
            return value

    def extra_super_elements(self, **function_kwargs) -> Iterator[Iterable]:
        """Iterable of extra super matrix elements.

        See ``super_elements(**function_kwargs)``.

        """

        yield from ()

    @util.process_args
    def super_elements(self, **function_kwargs) -> Iterator[Iterable]:
        """Iterator of superlattice matrix elements.

        Assembled from all classes in inheritance tree with
        ``extra_super_elements(**function_kwargs)`` generator function defined.

        Added to Hamiltonian ``H`` according to
        ``H[i1, i2] += value * operators``.

        warning::
            Only includes values on or below the diagonal (``i1 >= i2``) to
            avoid double counting.

        Parameters
        ----------
        **function_kwargs
            Model-specific keyword arguments - see ``function_kwargs()``.

        Yields
        ------
        i1
            Sublattice component 1 - see ``basis(**function_kwargs)``.
        i2
            Sublattice component 2.
        value : Number
            Scalar value that operators are multiplied by.
        operators : iterable of {"k", "kc", int, iterable of ints}
            Iterable of operators from left to right. "k" is
            :math:`\\hat{\kappa}`, "kc" is :math:`\\hat{\kappa}^\dagger`,
            :math:`m` gives a first star vector,
            :math:`e^{i \\boldsymbol{G}_m \cdot \\boldsymbol{r}}`, and
            an iterable of ints, :math:`(m_1, ...)`, gives a general vector,
            :math:`e^{i (m_1 \\boldsymbol{G}_1 + ...) \cdot \\boldsymbol{r}}`,
            summed over the basis vectors, :math:`(\\boldsymbol{G}_1, ...)`.

        """

        yield from itertools.chain.from_iterable(
            extra_super_elements(self, **function_kwargs)
            for extra_super_elements in self.hierarchy("extra_super_elements")
        )

    @functools.cache
    @util.process_args
    def details_super(self, *, M: int, **function_kwargs) -> _DetailsSuper:
        """Get the model's zone-folded details"""

        if self.msl_dims > self.dims:
            raise ValueError("number of superlattice dimensions exceeds dimensions")

        details = self.details(**function_kwargs)
        dim = details.dim

        details_superlattice = self.details_superlattice(M=M)
        num_vectors = details_superlattice.num_vectors

        initial_m = details_superlattice.m
        valid_final_m = frozenset([tuple(end_point) for end_point in initial_m])

        # partial matrix elements
        partial_details_super = []
        points = set()
        for n1, n2, value, operators, is_off_diagonal in self.process_elements(
            self.super_elements(**function_kwargs), details.inverse_basis
        ):
            details_super = []

            m = initial_m.copy()

            # iterate through operators from right to left (backward)
            for op in operators[::-1]:
                if isinstance(op, bool):
                    # momentum operator
                    details_super.append([op, m.copy()])
                elif isinstance(op, numbers.Integral):
                    # Bragg vector in first star
                    m += self.m_first[op]
                else:
                    # Bragg vector indices
                    m += op

            mask = np.fromiter(
                (tuple(end_point) in valid_final_m for end_point in m),
                bool,
                num_vectors,
            )
            if mask.any():
                final_m = m[mask]
                points.update((tuple(index) for index in final_m))
                operators = []
                for op, m in details_super:
                    m_ = m[mask]
                    points.update((tuple(point_) for point_ in m_))
                    operators.append([op, m_])
                partial_details_super.append(
                    [
                        n1,
                        n2,
                        final_m,
                        *np.nonzero(mask),  # unpack tuple with single element
                        value,
                        operators,
                        is_off_diagonal,
                    ]
                )

        # array of full points
        extra_points = [point for point in points if point not in valid_final_m]
        if extra_points:
            m_extended = np.concatenate([initial_m, extra_points])
        else:
            m_extended = initial_m

        conversion = {tuple(m_): ind for ind, m_ in enumerate(m_extended)}

        processed_super_elements = []
        for (
            n1,
            n2,
            m,
            ind2,
            value,
            compound_operators,
            is_off_diagonal,
        ) in partial_details_super:
            details_super = []

            ind1 = [conversion[tuple(m_)] for m_ in m]

            if compound_operators:
                processed_operators = tuple(
                    (op, [conversion[tuple(m_)] for m_ in indices])
                    for op, indices in compound_operators
                )
            else:
                processed_operators = ()

            processed_super_elements.append(
                (n1, ind1, n2, ind2, value, processed_operators, is_off_diagonal)
            )

        return _DetailsSuper(
            num_extended_vectors=num_vectors,
            m_extended=m_extended,
            G_extended=self.get_G(m_extended),
            basis_super=tuple(itertools.product(details.basis, map(tuple, initial_m))),
            shape_super=(dim, num_vectors),
            dim_super=dim * num_vectors,
            processed_super_elements=tuple(processed_super_elements),
        )

    def _H_super(
        self,
        kx: npt.NDArray,
        ky: npt.NDArray,
        fill_upper: bool = False,
        *,
        M: int,
        **function_kwargs,
    ) -> npt.NDArray[np.complexfloating]:
        """Internal zero magnetic field mSL-reconstructed Hamiltonian"""

        details = self.details(**function_kwargs)

        details_super = self.details_super(M=M, **function_kwargs)
        shape_super = details_super.shape_super
        dim_super = details_super.dim_super
        G_extended = details_super.G_extended

        k = (kx + 1j * ky)[..., None] + (G_extended[:, 0] + 1j * G_extended[:, 1])
        kc = k.conj()
        shape = kx.shape

        H_flattened = np.zeros([*shape, dim_super, dim_super], dtype=complex)
        H = H_flattened.reshape(*shape, *shape_super, *shape_super)

        # intrinsic elements which are diagonal in zone-folded points
        num_vectors = shape_super[1]
        ind = np.arange(num_vectors)
        k0 = k[..., :num_vectors]
        kc0 = kc[..., :num_vectors]
        for n1, n2, value, processed_operators, is_off_diagonal in itertools.chain(
            details.processed_major_elements, details.processed_minor_elements
        ):
            if is_off_diagonal and fill_upper:
                temp = self.get_element(value, processed_operators, k0, kc0)
                H[..., n1, ind, n2, ind] += temp
                H[..., n2, ind, n1, ind] += temp.conjugate()
            else:
                H[..., n1, ind, n2, ind] += self.get_element(
                    value, processed_operators, k0, kc0
                )

        # interaction elements which are off-diagonal in zone-folded points
        for (
            n1,
            ind1,
            n2,
            ind2,
            value,
            processed_operators,
            is_off_diagonal,
        ) in details_super.processed_super_elements:
            if is_off_diagonal and fill_upper:
                temp = self.get_element_super(value, processed_operators, k, kc)
                H[..., n1, ind1, n2, ind2] += temp
                H[..., n2, ind2, n1, ind1] += temp.conjugate()
            else:
                H[..., n1, ind1, n2, ind2] += self.get_element_super(
                    value, processed_operators, k, kc
                )

        return H.reshape(*shape, dim_super, dim_super)

    @util.process_args
    def H_super(
        self, kx: npt.ArrayLike, ky: npt.ArrayLike, M: int = 3, **function_kwargs
    ) -> npt.NDArray[np.complexfloating]:
        """Zero magnetic field mSL-reconstructed Hamiltonian.

        Hamiltonian has dimensionality ``dim_super`` - see ``dim_super(**function_kwargs)``.

        Parameters
        ----------
        kx : array_like
            x-component of wavevector. Must be real and broadcast to common
            shape with ``ky``.
        ky : array_like
            x-component of wavevector. Must be real and broadcast to common
            shape with ``kx``.
        M : int, optional
            Number of stars of non-zero Bragg vectors. (default is 3)
        **function_kwargs
            Model-specific keyword arguments - see ``function_kwargs()``.

        Returns
        -------
        H : ndarray
            Hamiltonian.

        """

        return self._H_super(kx, ky, M=M, **function_kwargs)

    @util.process_args
    def e_super(
        self,
        kx: npt.ArrayLike,
        ky: npt.ArrayLike,
        M: int = 3,
        bands: typing.Optional[npt.ArrayLike] = None,
        chunksize: int = 0,
        verbose: bool = False,
        **function_kwargs,
    ) -> npt.NDArray[np.floating]:
        """Zero magnetic field mSL-reconstructed energy eigenvalues.

        Hamiltonian has dimensionality ``dim_super`` - see ``dim_super(**function_kwargs)``.

        Parameters
        ----------
        kx : array_like
            x-component of wavevector. Must be real and broadcast to common
            shape with ``ky``.
        ky : array_like
            x-component of wavevector. Must be real and broadcast to common
            shape with ``kx``.
        M : int, optional
            Number of stars of non-zero Bragg vectors. (default is 3)
        bands : {None, array_like of ints}, optional
            Indices of bands to calculate. If None, then all bands are
            calculated. (default is None)
        chunksize : int, optional
            Chunksize for calculation. If 0, a single chunk is used. (default
            is 0)
        verbose : bool, optional
            Show progressbar for calculation if ``chunksize``>1. (default is
            False)
        **function_kwargs
            Model-specific keyword arguments - see ``function_kwargs()``.

        Returns
        -------
        e : ndarray
            Energy eigenvalues.

        Raises
        ------
        LinAlgError
            If the eigenvalue computation does not converge.

        """

        return util.chunker(
            util.e_func,
            chunksize,
            verbose,
            (kx, ky),
            self._H_super,
            bands,
            M=M,
            **function_kwargs,
        )

    @util.process_args
    def bands_super(
        self,
        kx: npt.ArrayLike,
        ky: npt.ArrayLike,
        M: int = 3,
        bands: typing.Optional[npt.ArrayLike] = None,
        chunksize: int = 0,
        verbose: bool = False,
        **function_kwargs,
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.complexfloating]]:
        """Zero magnetic field mSL-reconstructed energy eigenvalues and eigenvectors.

        Hamiltonian has dimensionality ``dim_super`` - see ``dim_super(**function_kwargs)``.

        Parameters
        ----------
        kx : array_like
            x-component of wavevector. Must be real and broadcast to common
            shape with ``ky``.
        ky : array_like
            x-component of wavevector. Must be real and broadcast to common
            shape with ``kx``.
        M : int, optional
            Number of stars of non-zero Bragg vectors. (default is 3)
        bands : {None, array_like of ints}, optional
            Indices of bands to calculate. If None, then all bands are
            calculated. (default is None)
        chunksize : int, optional
            Chunksize for calculation. If 0, a single chunk is used. (default
            is 0)
        verbose : bool, optional
            Show progressbar for calculation if ``chunksize``>1. (default is
            False)
        **function_kwargs
            Model-specific keyword arguments - see ``function_kwargs()``.

        Returns
        -------
        e : ndarray
            Energy eigenvalues.
        psi : ndarray
            Eigenvectors.

        Raises
        ------
        LinAlgError
            If the eigenvalue computation does not converge.

        """

        return util.chunker(
            util.bands_func,
            chunksize,
            verbose,
            (kx, ky),
            self._H_super,
            bands,
            M=M,
            **function_kwargs,
        )

    @util.process_args
    def dim_super(self, M: int = 3, **function_kwargs):
        """Dimensionality of zero magnetic field mSL-reconstructed Hamiltonian.

        Parameters
        ----------
        M : int, optional
            Number of stars of non-zero Bragg vectors. (default is 3)
        **function_kwargs
            Model-specific keyword arguments - see ``function_kwargs()``.

        Returns
        -------
        dim : int
            Hamiltonian dimensionality.

        """

        return self.details_super(M=M, **function_kwargs).dim_super

    @util.process_args
    def basis_super(self, M: int = 3, **function_kwargs):
        """Basis of zero magnetic field mSL-reconstructed Hamiltonian.

        Parameters
        ----------
        M : int, optional
            Number of stars of non-zero Bragg vectors. (default is 3)
        **function_kwargs
            Model-specific keyword arguments - see ``function_kwargs()``.

        Returns
        -------
        basis : tuple
            Hamiltonian basis. Corresponds to sublattices ``\\otimes`` Bragg
            vector indices.

        """

        return self.details_super(M=M, **function_kwargs).basis_super
