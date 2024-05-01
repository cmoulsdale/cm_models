from __future__ import annotations

import abc
import functools
import itertools

import typing
import numbers
from collections.abc import Iterable, Iterator, Mapping

P = typing.ParamSpec("P")
T = typing.TypeVar("T")

import math
import numpy as np
from scipy import constants

from . import util

if typing.TYPE_CHECKING:
    import numpy.typing as npt

# raise Exception("reject harmonics with Bragg vectors outside basis")

"""
    In functions, keyword arguments generally describe large changes which 
    are evaluated separately and must be scalar quantities. Conversely, 
    arguments are array-like.
"""

# TODO: type hint optional functions without declaration
# TODO: have separate type hint (pyi) files?

UNKNOWN = None
"""Alias for unknown variables"""


class Base(abc.ABC):
    extra_model_parameters: dict
    """Extra default model parameters.

    See ``default_model_parameters()``.

    Returns
    -------
    dict
        Each item has signature ``name: (default_value, description)``.
        Type is inferred from from ``default_value``, with no default
        value if ``default_value`` is itself a ``type``.

    """

    remove_model_parameters_names: Iterable
    """Names of model parameters to remove.

    Used for reducing the size of the parameter space, e.g. determining the 
    values of two parameters by a single parameter. The removed parameters 
    must be restored to the model with the corresponding class method 
    ``resolve_model_parameters()``.

    See ``default_model_parameters()``.

    Returns
    -------
    iterable
    
    """

    def resolve_model_parameters(self):
        """Restores removed model parameters.

        Restores all parameters removed from the
        ``default_model_parameters()`` dictionary by the corresponding
        ``remove_model_parameters_names`` class attribute.

        See ``default_model_parameters()``.

        """

    def fill_namespace(self):
        """Fill model namespace with secondary attributes."""

    @classmethod
    def hierarchy(cls, name: str, reverse: bool = True) -> Iterator:
        """Iterator of hierarchy of class attribute.

        Parameters
        ----------
        name : str
            Name of class attribute.
        reverse : bool
            Iterate from bottom to top of class hierarchy (method resolution
            order) if ``reverse=True``, and top to bottom otherwise.

        Yields
        ------
        object
            Attribute for each class

        """

        for cls_ in cls.__mro__[:: -1 if reverse else 1]:
            attr = cls_.__dict__.get(name, NotImplemented)
            if attr is not NotImplemented:
                yield attr

    @classmethod
    def default_model_parameters(cls) -> dict:
        """Default model parameters.

        Defines the default keyword arguments, ``kwargs``, for the class
        initialiser, ``__init__(**kwargs)``.

        Built up from the class hierarchy of optional
        ``extra_model_parameters()`` dictionary attributes, with sub classes
        overriding super classes.

        Entries can be removed for each class in the hierarchy with an
        optional ``remove_model_parameters_names`` iterable attribute, and
        these must be restored with the corresponding
        ``resolve_model_parameters`` method.

        Returns
        -------
        dict
            Each item has signature ``name: (default_value, description)``.
            Type is inferred from from ``default_value``, with no default
            value if ``default_value`` is itself a ``type``.

        """

        out = {}

        for extra_model_paramaters in cls.hierarchy("extra_model_parameters"):
            out.update(extra_model_paramaters)

        for name in itertools.chain.from_iterable(
            cls.hierarchy("remove_model_parameters_names")
        ):
            del out[name]

        return out

    @classmethod
    def model_parameters_doc(cls) -> str:
        """Pseudo-docstring for model parameters.

        Returns
        -------
        str

        """

        return util.pseudo_docstring(cls.default_model_parameters())

    extra_function_parameters: dict
    """Extra default function parameters - see ``function_parameters()``."""

    @classmethod
    def default_function_parameters(cls) -> dict:
        """Default function parameters.

        Defines the model-specific keyword arguments, ``model_kwargs``, with
        scalar values for user-facing functions,
        ``func(*args, **method_kwargs, **model_kwargs)``,
        where ``*args`` are the array-like arguments and ``**method_kwargs``
        are scalar-valued keyword arguments which determine evaluation of
        that function specifically.

        Built up from the class hierarchy of optional
        ``extra_function_parameters`` dictionary attributes, with sub classes
        overriding super classes.

        Returns
        -------
        dict
            Each item has signature ``name: (default_value, description)``.
            Type is inferred from from ``default_value``, with no default
            value if ``default_value`` is itself a ``type``.

        """

        out = {}

        for params in cls.hierarchy("extra_function_parameters"):
            out.update(params)

        return out

    @classmethod
    def function_parameters_doc(cls) -> str:
        """Pseudo-docstring for model-specific function parameters.

        Returns
        -------
        str

        """

        return util.pseudo_docstring(cls.default_function_parameters())

    default_model_kwargs: dict
    """Default model-specific keyword arguments for functions"""

    def __init__(self, **kwargs):
        """Initialize the model.

        Parameters
        ----------
        **kwargs
            Model parameters - see ``default_model_parameters()``.

        Raises
        ------
        ValueError
            If a parameter is not recognized or if keyword-only parameter(s)
            are missing.

        """

        default_model_parameters = self.default_model_parameters()

        unrecognized_parameters = [
            name for name in kwargs if name not in default_model_parameters
        ]
        if unrecognized_parameters:
            raise ValueError(
                f"unrecognized parameters: {unrecognized_parameters} - "
                "see ``default_model_parameters()``."
            )

        full_kwargs = {
            name: kwargs.get(name, default_value)
            for name, (default_value, description) in default_model_parameters.items()
        }

        missing_parameters = [
            name for name, value in full_kwargs.items() if isinstance(value, type)
        ]
        if missing_parameters:
            raise ValueError(
                f"missing keyword-only parameters: {missing_parameters} - "
                "see ``default_model_parameters()``."
            )

        self.__dict__.update(full_kwargs)

        self.default_model_kwargs = {
            name: default_value
            for name, (
                default_value,
                description,
            ) in self.default_function_parameters().items()
        }

        for resolve_model_parameters in self.hierarchy("resolve_model_parameters"):
            resolve_model_parameters(self)

        for fill_namespace in self.hierarchy("fill_namespace"):
            fill_namespace(self)


class Model(Base):
    """Arbitrary dimensional electronic system"""

    dims: int
    """Number of dimensions"""

    @abc.abstractmethod
    def sublattices(self, **model_kwargs) -> Iterable:
        """Sublattices of model.

        Returns
        -------
        iterable

        """


class _Details(typing.NamedTuple):
    """Cached details of calculation"""

    basis: tuple
    """Basis of sublattices.
    
    Equal to ``sublattices(**model_kwargs)``.
    
    """

    dim: int
    """Dimensionality of Hamiltonian"""

    inverse_basis: typing.Dict[typing.Any, int]
    """Inverse basis lookup"""

    processed_major_elements: tuple[int, int, numbers.Number, tuple, bool]
    """Processed major elements

    See ``major_elements(**model_kwargs)``.
    
    Yields
    -------
    n1 : int
        Sublattice component 1 is ``i1=basis[n1]``.
    n2 : int
        Sublattice component 2 is ``i2=basis[n2]``.
    value : Number
        Scalar value that operators are multiplied by.
    processed_operators : tuple
        Iterable of processed operators from left to right. False is
        :math:`\hat{\kappa}` and True is :math:`\hat{\kappa}^\dagger`.
    is_off_diagonal : bool
        Whether the element is off the diagonal. Equal to ``n1!=n2``.
    
    """

    processed_minor_elements: tuple[int, int, numbers.Number, tuple, bool]
    """Processed minor elements

    See ``minor_elements(**model_kwargs)``.
    
    Yields
    -------
    n1 : int
        Sublattice component 1 is ``i1=basis[n1]``.
    n2 : int
        Sublattice component 2 is ``i2=basis[n2]``.
    value : Number
        Scalar value that operators are multiplied by.
    processed_operators : tuple
        Iterable of processed operators from left to right. False is
        :math:`\hat{\kappa}` and True is :math:`\hat{\kappa}^\dagger`.
    is_off_diagonal : bool
        Whether the element is off the diagonal. Equal to ``n1!=n2``.
    
    """


class _DetailsLevels(typing.NamedTuple):
    """Cached details of Landeu level calculation"""

    basis_levels: tuple
    """Basis of finite magnetic field Hamiltonian.
    
    Direct sum of the finite basis of Landau levels for each sublattice.

    E.g. for two sublattices, ``(A, B)``, with steps, ``(N, N-1)``, the 
    basis is ``(A0, ..., AN-1, B0, ..., BN-2)``.
    
    """

    dim_levels: int
    """Dimensionality of finite magnetic field Hamiltonian."""

    processed_levels_elements: tuple[int, int, float, numbers.Number, bool]
    """Processed elements.

    Added to Hamiltonian ``H`` according to
    ``H[n1, n2] += value * B**B_exponent * value``.
    
    Returns
    -------
    n1 : array of integers
        Component 1 in flattened ``basis``.
    n2 : array of integers
        Component 2 in flattened ``basis``.
    B_exponent : float
        Exponent of magnetic field and element.
    value : Number
        Scalar value that operators are multiplied by.
    is_off_diagonal : bool
        Whether the element is off the diagonal. Equal to ``n1!=n2``.
    
    """

    step: npt.NDArray[np.integer]
    """Number of Landau level basis vectors (step) for each sublattice"""

    offset: npt.NDArray[np.integer]
    """Flattened index at which sector of that sublattice begins.
    
    Each element is equal to ``offset[i]=step[0]+...+step[i-1]``.

    """


class Model2D(Model):
    """2D electronic system"""

    dims = 2
    """Number of dimensions"""

    optimize_levels = True
    """Whether to optimize the number of basis levels in each sublattice
    (True), or consider a fixed number for each.

    warning::
        Setting this to False may result in false annihilation where the 
        index is raised above the cutoff.

    Returns
    -------
    bool
    
    """

    levels_phase: complex
    """Phase of `pi` operator
    
    Assumed to be irrelevant if not set
    
    """

    def extra_minor_elements(self, **model_kwargs) -> Iterator[Iterable]:
        """Iterable of extra minor matrix elements.

        See ``minor_elements(**model_kwargs)``.

        """

        yield from ()

    @util.process_args
    def minor_elements(self, **model_kwargs) -> Iterator[Iterable]:
        """Iterator of minor matrix elements.

        Assembled from all classes in inheritance tree with
        ``extra_minor_elements(**model_kwargs)`` generator function defined.

        Added to Hamiltonian ``H`` according to
        ``H[i1, i2] += value * operators``.

        Unlike the major matrix elements from
        ``major_elements(**model_kwargs)``, these are not used to determine
        the basis of Landau levels.

        warning::
            Only includes values on or below the diagonal (``i1 >= i2``) to
            avoid double counting.

        Parameters
        ----------
        **model_kwargs
            Model-specific keyword arguments - see ``function_parameters()``.

        Yields
        ------
        i1
            Sublattice component 1 - see ``basis(**model_kwargs)``.
        i2
            Sublattice component 2.
        value : Number
            Scalar value that operators are multiplied by.
        operators : iterable of {"k", "kc"}
            Iterable of operators from left to right. "k" is
            :math:`\hat{\kappa}` and "kc" is :math:`\hat{\kappa}^\dagger`.

        """

        yield from itertools.chain.from_iterable(
            extra_minor_elements(self, **model_kwargs)
            for extra_minor_elements in self.hierarchy("extra_minor_elements")
        )

    def extra_major_elements(self, **model_kwargs) -> Iterator[Iterable]:
        """Iterable of extra major matrix elements.

        See ``major_elements(**model_kwargs)``.

        """

        yield from ()

    @util.process_args
    def major_elements(self, **model_kwargs) -> Iterator[Iterable]:
        """Iterator of major matrix elements.

        Assembled from all classes in inheritance tree with
        ``extra_minor_elements(**model_kwargs)`` generator function defined.

        The major matrix elements define the cutoff of the Landau level basis
        in each sublattice such that the conjugate momentum operator,
        :math:`\hat{\kappa}^\dagger`, cannot raise the index above any
        cutoff. Hence, the major elements must
        constitute the vertices of a spanning tree whose nodes are the
        components of the basis, i.e. ``dim-1`` vertices connecting every
        nodes to a single graph (equivalently no loops). Future releases may
        allow for sub-bases with separate spanning trees.

        warning::
            Only includes values on or below the diagonal (``i1 >= i2``) to
            avoid double counting.

        Parameters
        ----------
        **model_kwargs
            Model-specific keyword arguments - see ``function_parameters()``.

        Yields
        ------
        i1
            Sublattice component 1 - see ``basis(**model_kwargs)``.
        i2
            Sublattice component 2.
        value : Number
            Scalar value that operators are multiplied by.
        operators : iterable of {"k", "kc"}
            Iterable of operators from left to right. "k" is
            :math:`\hat{\kappa}` and "kc" is :math:`\hat{\kappa}^\dagger`.

        """

        yield from itertools.chain.from_iterable(
            extra_major_elements(self, **model_kwargs)
            for extra_major_elements in self.hierarchy("extra_major_elements")
        )

    @staticmethod
    def process_elements(elements: Iterable, inverse_sublattices: Mapping) -> Iterator:
        """Process elements, yielding only non-zero elements"""

        for i1, i2, value, operators in elements:
            n1 = inverse_sublattices[i1]
            n2 = inverse_sublattices[i2]

            if n2 > n1:
                raise ValueError(
                    f"element '{(i1, i2, value, operators)}' is above the diagonal"
                )

            if value:
                yield (
                    n1,
                    n2,
                    value,
                    tuple(
                        True if op == "kc" else False if op == "k" else op
                        for op in operators
                    ),
                    n1 != n2,
                )

    @functools.cache
    @util.process_args
    def details(self, **model_kwargs) -> _Details:
        """Get the model details"""

        sublattices = tuple(self.sublattices(**model_kwargs))
        dim = len(sublattices)
        inverse_basis = dict(zip(sublattices, range(dim)))

        return _Details(
            dim=dim,
            basis=sublattices,
            inverse_basis=inverse_basis,
            processed_major_elements=tuple(
                self.process_elements(
                    self.major_elements(**model_kwargs), inverse_basis
                )
            ),
            processed_minor_elements=tuple(
                self.process_elements(
                    self.minor_elements(**model_kwargs), inverse_basis
                )
            ),
        )

    @staticmethod
    def get_element(
        value: numbers.Number,
        operators: Iterable[bool],
        k: npt.NDArray[np.complexfloating],
        kc: npt.NDArray[np.complexfloating],
    ) -> npt.ArrayLike:
        """Evaluate a matrix element at each momentum point"""

        if operators:
            k_arrays = (kc if is_conjugate else k for is_conjugate in operators)

            element = value * next(k_arrays)
            for k_array in k_arrays:
                element *= k_array

            return element
        else:
            return value

    def _H(
        self, kx: npt.NDArray, ky: npt.NDArray, fill_upper: bool = False, **model_kwargs
    ) -> npt.NDArray[np.complexfloating]:
        k = kx + 1j * ky
        kc = k.conjugate()

        details = self.details(**model_kwargs)
        dim = details.dim

        H = np.zeros([*k.shape, dim, dim], dtype=complex)
        for n1, n2, value, processed_operators, is_off_diagonal in itertools.chain(
            details.processed_major_elements, details.processed_minor_elements
        ):
            if is_off_diagonal and fill_upper:
                temp = self.get_element(value, processed_operators, k, kc)
                H[..., n1, n2] += temp
                H[..., n2, n1] += temp.conjugate()
            else:
                H[..., n1, n2] += self.get_element(value, processed_operators, k, kc)

        return H

    @util.process_args
    def H(
        self, kx: npt.ArrayLike, ky: npt.ArrayLike, **model_kwargs
    ) -> npt.NDArray[np.complexfloating]:
        """Zero magnetic field Hamiltonian.

        Hamiltonian has dimensionality ``dim`` - see ``dim(**model_kwargs)``.

        Parameters
        ----------
        kx : array_like
            x-component of wavevector. Must be real and broadcast to common
            shape with ``ky``.
        ky : array_like
            x-component of wavevector. Must be real and broadcast to common
            shape with ``kx``.
        **model_kwargs
            Model-specific keyword arguments - see ``function_parameters()``.

        Returns
        -------
        H : ndarray
            Hamiltonian.

        """

        return self._H(kx, ky, fill_upper=True, **model_kwargs)

    @util.process_args
    def e(
        self,
        kx: npt.ArrayLike,
        ky: npt.ArrayLike,
        bands: typing.Optional[npt.ArrayLike] = None,
        chunksize: int = 0,
        verbose: bool = False,
        **model_kwargs,
    ) -> npt.NDArray[np.floating]:
        """Zero magnetic field energy eigenvalues.

        Hamiltonian has dimensionality ``dim`` - see ``dim(**model_kwargs)``.

        Parameters
        ----------
        kx : array_like
            x-component of wavevector. Must be real and broadcast to common
            shape with ``ky``.
        ky : array_like
            x-component of wavevector. Must be real and broadcast to common
            shape with ``kx``.
        bands : {None, array_like of ints}, optional
            Indices of bands to calculate. If None, then all bands are
            calculated. (default is None)
        chunksize : int, optional
            Chunksize for calculation. If 0, a single chunk is used. (default
            is 0)
        verbose : bool, optional
            Show progressbar for calculation if ``chunksize``>1. (default is
            False)
        **model_kwargs
            Model-specific keyword arguments - see ``function_parameters()``.

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
            util.e_func, chunksize, verbose, (kx, ky), self._H, bands, **model_kwargs
        )

    @util.process_args
    def bands(
        self,
        kx: npt.ArrayLike,
        ky: npt.ArrayLike,
        bands: typing.Optional[npt.ArrayLike] = None,
        chunksize: int = 0,
        verbose: bool = False,
        **model_kwargs,
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.complexfloating]]:
        """Zero magnetic field energy eigenvalues and eigenvectors.

        Hamiltonian has dimensionality ``dim`` - see ``dim(**model_kwargs)``.

        Parameters
        ----------
        kx : array_like
            x-component of wavevector. Must be real and broadcast to common
            shape with ``ky``.
        ky : array_like
            x-component of wavevector. Must be real and broadcast to common
            shape with ``kx``.
        bands : {None, array_like of ints}, optional
            Indices of bands to calculate. If None, then all bands are
            calculated. (default is None)
        chunksize : int, optional
            Chunksize for calculation. If 0, a single chunk is used. (default
            is 0)
        verbose : bool, optional
            Show progressbar for calculation if ``chunksize``>1. (default is
            False)
        **model_kwargs
            Model-specific keyword arguments - see ``function_parameters()``.

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
            self._H,
            bands,
            **model_kwargs,
        )

    @util.process_args
    def dim(self, **model_kwargs) -> int:
        """Dimensionality of zero magnetic field Hamiltonian.

        Parameters
        ----------
        **model_kwargs
            Model-specific keyword arguments - see
            py:class:`models.abstracts.Model.function_parameters`.

        Returns
        -------
        dim : int
            Hamiltonian dimensionality.

        """

        return self.details(**model_kwargs).dim

    @util.process_args
    def basis(self, **model_kwargs) -> tuple:
        """Basis of zero magnetic field Hamiltonian.

        Parameters
        ----------
        **model_kwargs
            Model-specific keyword arguments - see ``function_parameters()``.

        Returns
        -------
        basis : tuple
            Hamiltonian basis. Corresponds to sublattices.

        """

        return self.details(**model_kwargs).basis

    @staticmethod
    def process_levels_elements(
        offset: npt.NDArray[np.integer],
        step: npt.NDArray[np.integer],
        levels_phase: complex,
        partial_elements,
    ) -> Iterator[tuple[int, int, float, numbers.Number, bool]]:
        """Process elements for Landau level calculation"""

        x = -1e-9 * math.sqrt(2.0 * constants.e / constants.hbar) * levels_phase
        x_conjugate = x.conjugate()

        for n1, n2, value, signature, is_off_diagonal in partial_elements:
            if signature:
                # combination of momentum matrices

                B_exponent = 0.5 * len(signature)

                shift = sum(signature)
                off1 = max(0, shift)
                off2 = -min(0, shift)

                P = min(step[n1] - off1, step[n2] - off2)
                off1 += offset[n1]
                off2 += offset[n2]

                squared_factor = np.ones(P)
                shift = max(0, shift)  # finish at zero shift
                intermediate_value = value

                for op in signature:
                    if op == -1:
                        squared_factor *= np.arange(shift + 1, shift + 1 + P)
                        shift += 1
                        intermediate_value *= x
                    else:
                        squared_factor *= np.arange(shift, shift + P)
                        shift -= 1
                        intermediate_value *= x_conjugate

                final_value = intermediate_value * np.sqrt(squared_factor)
            else:
                # operator is identity matrix

                B_exponent = 0.0

                off1 = offset[n1]
                off2 = offset[n2]

                P = min(step[n1], step[n2])

                final_value = value

            n1 = np.arange(off1, off1 + P)
            n2 = np.arange(off2, off2 + P)

            yield (n1, n2, B_exponent, final_value, is_off_diagonal)

    @functools.cache
    @util.process_args
    def details_levels(self, *, N: int, **model_kwargs) -> _DetailsLevels:
        """Get the model's finite magnetic field details"""
        
        details = self.details(**model_kwargs)
        dim = details.dim

        partial_details_levels = []
        graph_vertices = []
        for elements, is_major in [
            [details.processed_major_elements, True],
            [details.processed_minor_elements, False],
        ]:
            for n1, n2, value, operators, is_off_diagonal in elements:
                # 1 if +K, 0 if -K
                signature = [1 if op else -1 for op in operators]

                partial_details_levels.append(
                    (n1, n2, value, signature, is_off_diagonal)
                )
                if is_major:
                    graph_vertices.append((n1, n2, sum(signature)))

        if self.optimize_levels:
            # determine difference of number of levels in each sublattice
            # guarantees that the momentum operators in the major elements
            # cannot raise the basis index above the cutoff, which causes
            # false annihilation

            # TODO: allow for isolated sub-bases with their own spanning trees

            num_graph_vertices = len(graph_vertices)
            required_num_graph_vertices = dim - 1
            if num_graph_vertices != required_num_graph_vertices:
                raise ValueError(
                    "Determination of Landau level basis with "
                    f"{dim} sublattices requires "
                    f"{required_num_graph_vertices} non-zero major elements "
                    f"(received {num_graph_vertices})"
                )

            # Initially unknown number of levels in each sublattice offset
            # from some constant, to be determined by major elements
            offsets = [UNKNOWN for _ in range(dim)]

            n1, n2, relative_offset = graph_vertices.pop(0)
            offsets[n1] = 0
            offsets[n2] = -relative_offset

            while graph_vertices:
                for i, (n1, n2, relative_offset) in enumerate(graph_vertices):
                    offset1 = offsets[n1]
                    offset2 = offsets[n2]

                    if offset1 is UNKNOWN:
                        offsets[n1] = offset2 + relative_offset
                    elif offset2 is UNKNOWN:
                        offsets[n2] = offset1 - relative_offset
                    else:
                        raise ValueError("loop in Landau level basis offset graph")

                    graph_vertices.pop(i)
                    break

            # maximum step is N
            step_constant = N - max(offsets)
            step = tuple(step + step_constant for step in offsets)
            offset = tuple(itertools.accumulate([0, *step[:-1]]))
        else:
            # all steps are N
            step = tuple(itertools.repeat(N, dim))
            offset = tuple(range(0, dim * N, N))

        # LL phase doesn't matter if it hasn't been set by user
        levels_phase = getattr(self, "levels_phase", 1j)

        return _DetailsLevels(
            basis_levels=tuple(
                itertools.chain.from_iterable(
                    (f"{sublattice},{level}" for level in range(step))
                    for sublattice, step in zip(details.basis, step)
                )
            ),
            dim_levels=sum(step),
            processed_levels_elements=tuple(
                self.process_levels_elements(
                    offset, step, levels_phase, partial_details_levels
                )
            ),
            step=step,
            offset=offset,
        )

    def _H_levels(
        self, B: npt.NDArray, fill_upper: bool = False, *, N: int, **model_kwargs
    ) -> npt.NDArray[np.complexfloating]:
        """Internal finite magnetic field Hamiltonian."""

        B_padded = B[..., None]

        details_levels = self.details_levels(N=N, **model_kwargs)
        dim_levels = details_levels.dim

        H = np.zeros([*B.shape, dim_levels, dim_levels], dtype=complex)

        for (
            n1,
            n2,
            B_exponent,
            value,
            is_off_diagonal,
        ) in details_levels.processed_elements:
            if is_off_diagonal and fill_upper:
                temp = B_padded**B_exponent * value
                H[..., n1, n2] += temp
                H[..., n2, n1] += temp.conjugate()
            else:
                H[..., n1, n2] += B_padded**B_exponent * value

        return H

    @util.process_args
    def H_levels(
        self, B: npt.ArrayLike, N: int = 100, **model_kwargs
    ) -> npt.NDArray[np.complexfloating]:
        """Finite magnetic field Hamiltonian.

        Hamiltonian has dimensionality ``dim_levels`` - see ``dim_levels(**model_kwargs)``.

        Parameters
        ----------
        B : array_like
            Magnetic field. Must be real.
        N : int, optional
            Maximum number of basis levels in each sublattice. (default is 100)
        **model_kwargs
            Model-specific keyword arguments - see
            ``default_function_parameters()``.

        Returns
        -------
        H : ndarray
            Hamiltonian.

        """

        return self._H_levels(B, fill_upper=True, N=N, **model_kwargs)

    @util.process_args
    def e_levels(
        self,
        B: npt.ArrayLike,
        N: int = 100,
        bands: typing.Optional[npt.ArrayLike] = None,
        chunksize: int = 0,
        verbose: bool = False,
        **model_kwargs,
    ) -> npt.NDArray[np.floating]:
        """Finite magnetic field energy eigenvalues.

        Hamiltonian has dimensionality ``dim_levels`` - see ``dim_levels(**model_kwargs)``.

        Parameters
        ----------
        B : array_like
            Magnetic field. Must be real.
        N : int, optional
            Maximum number of basis levels in each sublattice. (default is 100)
        bands : {None, array_like of ints}, optional
            Indices of bands to calculate. If None, then all bands are
            calculated. (default is None)
        chunksize : int, optional
            Chunksize for calculation. If 0, a single chunk is used. (default
            is 0)
        verbose : bool, optional
            Show progressbar for calculation if ``chunksize``>1. (default is
            False)
        **model_kwargs
            Model-specific keyword arguments - see ``function_parameters()``.

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
            (B,),
            self._H_levels,
            bands,
            N=N,
            **model_kwargs,
        )

    @util.process_args
    def bands_levels(
        self,
        B: npt.ArrayLike,
        N: int = 100,
        bands: typing.Optional[npt.ArrayLike] = None,
        chunksize: int = 0,
        verbose: bool = False,
        **model_kwargs,
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.complexfloating]]:
        """Finite magnetic field energy eigenvalues and eigenvectors.

        Hamiltonian has dimensionality ``dim_levels`` - see ``dim_levels(**model_kwargs)``.

        Parameters
        ----------
        B : array_like
            Magnetic field. Must be real.
        N : int, optional
            Maximum number of basis levels in each sublattice. (default is 100)
        bands : {None, array_like of ints}, optional
            Indices of bands to calculate. If None, then all bands are
            calculated. (default is None)
        chunksize : int, optional
            Chunksize for calculation. If 0, a single chunk is used. (default
            is 0)
        verbose : bool, optional
            Show progressbar for calculation if ``chunksize``>1. (default is
            False)
        **model_kwargs
            Model-specific keyword arguments - see ``function_parameters()``.

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
            (B,),
            self._H_levels,
            bands,
            N=N,
            **model_kwargs,
        )

    @util.process_args
    def dim_levels(self, N: int = 100, **model_kwargs) -> int:
        """Dimensionality of finite magnetic field Hamiltonian.

        Parameters
        ----------
        N : int, optional
            Maximum number of basis levels in each sublattice. (default is 100)
        **model_kwargs
            Model-specific keyword arguments - see
            py:class:`models.abstracts.Model.function_parameters`.

        Returns
        -------
        dim : int
            Hamiltonian dimensionality.

        """

        return self.details_levels(N=N, **model_kwargs).dim

    @util.process_args
    def basis_levels(self, N: int = 100, **model_kwargs) -> tuple:
        """Basis of finite magnetic field Hamiltonian.

        Parameters
        ----------
        N : int, optional
            Maximum number of basis levels in each sublattice. (default is 100)
        **model_kwargs
            Model-specific keyword arguments - see ``function_parameters()``.

        Returns
        -------
        basis : tuple
            Hamiltonian basis. Corresponds to sublattices ``\otimes`` basis levels.

        """

        return self.details_levels(N=N, **model_kwargs).basis
