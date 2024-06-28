import math
import numpy as np
from cm_models import abstracts, het, msl
from scipy import constants

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError('install matplotlib with "pip install matplotlib"')

"""
    How to create and use a model of the form
    :math:``H = \pm k_x \sigma_x + k_y \sigma_y + \Delta \sigma_y / 2``,
    in the ``(A, B)`` basis 
    for electrons in the :math:``K ^ \pm`` valley, where the velocity is 
    :math:``v`` and the band gap is :math:``\Delta``.
"""


class Model(abstracts.Model2D):
    """
    Model class inherits from a base model class, e.g. for a 2D material
    either use graphene.Graphene for a graphene structure or
    abstracts.Model2D for a  generic material
    """

    extra_model_parameters = dict(v=(1e6, "Velocity [m/s]"), Delta=(0.0, "Gap [eV]"))
    """
    Add model parameters as a dictionary of 
    `name: (default_value, description)` entries. If `default_value` is a 
    type, then no default value is provided and the user must provide these 
    instead.
    
    The model parameters the model is built from the class hierarchy, with 
    higher classes overriding lower classes.

    These define the valid keyword arguments, ``kwargs``, at instantiation:
    ``Model(**kwargs)``
    """

    extra_function_parameters = dict(K_plus=(False, "In K+ valley."))
    """
    Add function parameters in the same format as model parameters.

    These are model-specific keyword arguments, ``function_kwargs``, with
    scalar values for user-facing functions,
    ``func(*args, **eval_kwargs, **function_kwargs)``,
    where ``*args`` are the array-like arguments and ``**eval_kwargs``
    are scalar-valued keyword arguments which determine evaluation of
    that function specifically.
    """

    def fill_namespace(self):
        """
        Fill model namespace with secondary attributes, such as the rescaled
        velocity, :math:``\hbar v``, in units of eV.nm.
        """

        self.rescaled_v = (
            constants.hbar
            * self.v  # hbar * v in units of J.m
            / (1e-9 * constants.e)  # convert to eV.nm
        )

    dims = 2
    """Dimensionality of Hamiltonian (number of sublattices)"""

    def sublattices(self, *, K_plus, **function_kwargs):
        """Return the sublattices"""

        return ("A", "B")

    def extra_major_elements(self, *, K_plus, **function_kwargs):
        """
        Yield major matrix elements, of the form
        i1
            Sublattice component 1 - see ``basis(**function_kwargs)``.
        i2
            Sublattice component 2.
        value : Number
            Scalar value that operators are multiplied by.
        operators : iterable of {"k", "kc"}
            Iterable of operators from left to right. "k" is
            :math:`\\hat{\kappa} = kx + i ky` and "kc" is
            :math:`\\hat{\kappa}^\dagger = kx - i ky`.

        Built up from the class hierachy.

        The major matrix elements define the cutoff of the Landau level basis
        in each sublattice such that the conjugate momentum operator,
        :math:`\\hat{\kappa}^\dagger = k_x - i k_y`, cannot raise the index
        above any cutoff, so there should only be ``dim-1`` non-diagonal
        entries.

        warning::
            Only includes values on or below the diagonal (``i1 >= i2``) to
            avoid double counting.

        """

        print(self.rescaled_v)

        if K_plus:
            # + kx + i ky: k
            yield ("B", "A", self.rescaled_v, ("k",))
        else:
            # - kx + i ky: -kc
            yield ("B", "A", -self.rescaled_v, ("kc",))

    def extra_minor_elements(self, *, K_plus, **function_kwargs):
        """
        The minor elements have the same format as the major elements, except
        they don't determine the Landau level basis.
        """

        # Delta sigma_z / 2
        yield ("A", "A", 0.5 * self.Delta, ())
        yield ("B", "B", -0.5 * self.Delta, ())


class GaplessModel(Model):
    """A gapless version of the model (:math:``Delta = 0``)"""

    remove_model_parameters_names = ["Delta"]
    """List of names to remove"""

    def resolve_model_parameters(self):
        """Restores removed model parameters."""

        self.Delta = 0.0


class GaplessModelSquareSuperlattice(
    GaplessModel,  # the gapless model without the superlattice
    msl.SuperlatticeSquare2D,  # the square superlattice class
    het.ModelSuperlattice2D,  # the class which deals with the superlattice
):
    """The gapless model with a square superlattice

    ``msl.SuperlatticeSquare2D`` describes a square superlattice with a mSL
    period, ``period``, and rotated from the x-axis by ``phi``.

    The ```het.ModelSuperlattice2D`` class deals with the electronic effects
    of the superlattice, although we must still define mSL-periodic terms in
    the Hamiltonian via ``extra_super_elements``.

    """

    extra_model_parameters = dict(
        u0=(0.0, "the strength of an mSL-periodic scalar potential")
    )

    def extra_super_elements(self, *, K_plus, **function_kwargs):
        """
        mSL-periodic scalar potential has an equal harmonic, ``u0``,
        for each of the shortest mSL Bragg vectors
        """
        for indices in ([1, 0], [0, 1], [-1, 0], [0, -1]):
            yield ("A", "A", self.u0, [indices])
            yield ("B", "B", self.u0, [indices])


k_max = 0.3  # maximum magnitude of each momentum component
P = 101  # number of momentum points
v = 8e5  # velocity
period = 10.0  # superlattice (mSL) period [nm]
u0 = 0.2  # strength of an mSL-periodic scalar potential [eV]
M = 3  # number of stars in zone folded electronic structure calculation
chunksize = 100  # split calculations into chunks to save memory
verbose = True  # show progress of calculations

# sparse momentum grid
ky = np.linspace(-k_max, k_max, P)
kx = ky[:, None]

model = GaplessModel(v=v)  # instantiate gapless model

bands = 1  # only return results for conduction band (valence is 0)

# calculate energy: automatically broadcasts array inputs
e = model.e(kx, ky, bands=bands, chunksize=chunksize, verbose=verbose)

# plot conical energy
fig, axis = plt.subplots(subplot_kw=dict(aspect=1))  # equal aspect figure
im = axis.pcolormesh(*np.broadcast_arrays(kx, ky, e))  # broadcast sparse arrays
fig.colorbar(im, ax=axis, label=r"$\epsilon\,$[eV]")
axis.set_xlabel(r"$k_x\,$[nm$^{-1}$]")
axis.set_ylabel(r"$k_y\,$[nm$^{-1}$]")
fig.tight_layout()

# instantiate gapless model with a square superlattice
model_superlattice = GaplessModelSquareSuperlattice(period=period, u0=u0)

# momentum in superlattice Brillouin zone
ky = np.linspace(-math.pi / period, math.pi / period, P)

# energies of all bands along kx=0 slice
e0 = model_superlattice.e(0.0, ky, chunksize=chunksize, verbose=verbose)  # without mSL
e1 = model_superlattice.e_super(
    0.0, ky, M=M, chunksize=chunksize, verbose=verbose
)  # with mSL: use e_super

# compare energies with and without superlattice to see zone folding and
# opening of minigaps
fig, axis = plt.subplots()
for e, style, label in [[e0, "C0-", "Off"], [e1, "C1:", "On"]]:
    axis.plot((period / (2.0 * math.pi)) * ky, e[:, 0], style, label=label)
    axis.plot((period / (2.0 * math.pi)) * ky, e[:, 1:], style)
axis.legend()
axis.set_xlim(-0.5, 0.5)
axis.set_ylim(-0.3, 0.3)
axis.set_xlabel(r"$k_x\,$[nm$^{-1}$]")
axis.set_ylabel(r"$\epsilon\,$[eV]")
fig.tight_layout()

plt.show()
