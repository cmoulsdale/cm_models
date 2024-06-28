from __future__ import annotations

import typing
from collections.abc import Callable, Iterable

import functools

import numpy as np

from tqdm import tqdm as Progressbar

from . import abstracts

if typing.TYPE_CHECKING:
    import numpy.typing as npt

    P = typing.ParamSpec("P")
    T = typing.TypeVar("T")  # generic return value type for decorator


def pseudo_docstring(default_parameters: dict) -> str:
    """Make pseudo-docstring for a given defaults parameters dictionary."""

    return f"Parameters\n----------\n\n" + "\n".join(
        (
            f"{name} : {default_value.__name__}\n    {description}"
            if isinstance(default_value, type)
            else f"{name} : {type(default_value).__name__}, optional\n"
            f"    {description} (default is {default_value})"
        )
        for name, (default_value, description) in default_parameters.items()
    )


def chunker(
    func: Callable[..., tuple[npt.ArrayLike]],
    chunksize: int,
    verbose: bool,
    array_args: Iterable[npt.ArrayLike],
    *args,
    **kwargs,
) -> typing.Union[npt.ArrayLike, tuple[npt.ArrayLike]]:
    """Evaluate function over array-like inputs in chunks.

    Parameters
    ----------
    func : callable
        The function to evaluate, with signature
        ``func(array_args, *args, **kwargs) -> results``, returning a tuple
        of array-like ``results``.
    chunksize : int
        Chunksize for calculation. If 0, a single chunk is used.
    verbose : bool
        Show progressbar for calculation if ``chunksize``>1.
    array_args: iterable of arrays
        Array arguments to be broadcasted and split into chunks.
    *args
        Additional (scalar) arguments to be passed to function.
    **kwargs
        Additional (scalar) keyword arguments to be passed to function.

    Returns
    -------
    array-like or tuple of array-likes
        Single array-like result if ``len(results)==1``, otherwise tuple of
        array-like results.

    warning::
        ``func`` must return tuple of ``results``, even if there is only one.

    """

    array_args = tuple(map(np.asarray, array_args))

    broadcasted_args = np.broadcast_arrays(*array_args)
    shape = broadcasted_args[0].shape
    size = broadcasted_args[0].size

    if chunksize == 0 or size == 1:
        results = func(array_args, *args, **kwargs)

        if len(results) == 1:
            return results[0]
        else:
            return results
    else:
        flattened_args = tuple(arg.flatten() for arg in broadcasted_args)

        num_chunks, num_remaining_elements = divmod(size, chunksize)
        if num_remaining_elements != 0:
            num_chunks += 1  # extra incomplete chunk
        chunks = zip(
            range(0, num_chunks * chunksize, chunksize),
            range(chunksize, (num_chunks + 1) * chunksize, chunksize),
        )

        zipped_results_generator = (
            func((arg[start:stop] for arg in flattened_args), *args, **kwargs)
            for start, stop in chunks
        )
        if verbose:
            zipped_results_generator = Progressbar(
                zipped_results_generator, total=num_chunks
            )

        results = tuple(map(np.concatenate, zip(*zipped_results_generator)))

        if len(results) == 1:
            result = results[0]
            return result.reshape(*shape, *result.shape[1:])
        else:
            return tuple(
                result.reshape(*shape, *result.shape[1:]) for result in results
            )


def e_func(
    array_args: Iterable[npt.NDArray],
    H_func: Callable[..., npt.NDArray[np.complexfloating]],
    bands: typing.Union[None, npt.ArrayLike],
    **kwargs,
) -> tuple[npt.NDArray[np.floating]]:
    """Returns energy eigenvalues"""

    e_full = np.linalg.eigvalsh(H_func(*array_args, **kwargs))

    if bands is None:
        return (e_full,)
    else:
        return (e_full[..., bands],)


def bands_func(
    array_args: Iterable[npt.NDArray],
    H_func: Callable[..., npt.NDArray[np.complexfloating]],
    bands: typing.Union[None, npt.ArrayLike],
    **kwargs,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.complexfloating]]:
    """Returns energy eigenvalues and eigenvectors"""

    e_full, psi_full = np.linalg.eigh(H_func(*array_args, **kwargs))

    if bands is None:
        return (e_full, psi_full)
    else:
        return (e_full[..., bands], psi_full[..., bands])


def process_args(
    func: Callable[typing.Concatenate[abstracts.Base, P], T]
) -> Callable[typing.Concatenate[abstracts.Base, P], T]:
    """

    Convenience function to ensure decoratored functions have arguments
    converted to numpy arrays and missing model-specific function parameters
    are substituted for their default values - see
    ``default_function_parameters()``.

    Raises
    ------
    ValueError
        If keyword-only parameter(s) are missing.

    """

    @functools.wraps(func)
    def wrapper(model: type[abstracts.Base], *args: P.args, **kwargs: P.kwargs) -> T:
        full_kwargs = model.default_function_kwargs.copy()
        full_kwargs.update(kwargs)

        missing_parameters = [
            name for name, value in full_kwargs.items() if isinstance(value, type)
        ]
        if missing_parameters:
            raise ValueError(
                f"missing keyword-only arguments: {missing_parameters} - "
                "see ``default_function_parameters()``."
            )

        return func(model, *map(np.asarray, args), **full_kwargs)

    return wrapper
