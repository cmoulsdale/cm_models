import numpy as np

from .graphene import (
    Graphene,
    GrapheneHBNSuperlatticeUV,
    uV_extra_model_parameters_names,
    uV_extra_model_parameters,
)


class SLG(Graphene):
    """Single layer graphene.

    2-band model of single layer graphene in the (A, B) sublattice basis.

    Parameters
    ----------

    a : float, optional
        Graphene lattice parameter [nm]. (default is 0.246)
    g0 : float, optional
        gamma0 coupling [eV]. (default is 3.16)
    Delta_AB : float, optional
        External sublattice asymmetry [eV]. (default is 0.0)

    """

    extra_model_parameters = dict(
        g0=(3.16, "Intralayer SWMcC coupling. [eV]"),
        Delta_AB=(0.0, "Sublattice potential asymmetry. [eV]"),
    )

    def sublattices(self, **function_kwargs):
        return ("A", "B")

    def fill_namespace(self):
        # velocities
        self.v0 = 0.5 * np.sqrt(3.0) * self.a * self.g0

    def extra_major_elements(self, *, K_plus, **function_kwargs):
        yield from self.resolve_valley(
            [
                # gamma0
                ("B", "A", self.v0, ("k",))
            ],
            K_plus,
        )

    def extra_minor_elements(self, *, K_plus, **function_kwargs):
        # on site potentials
        yield ("A", "A", 0.5 * self.Delta_AB, ())
        yield ("B", "B", -0.5 * self.Delta_AB, ())


class SLGHBN(GrapheneHBNSuperlatticeUV, SLG):
    # default model parameters
    extra_model_parameters = dict(
        **uV_extra_model_parameters("_b", "bottom"),
        **uV_extra_model_parameters("_t", "top"),
        Delta_h=(0.0, "External sublattice asymmetry [eV]."),
        Delta_u=(0.0, "External sublattice asymmetry [eV]."),
    )

    def fill_namespace(self):
        # resolve potentials in each layer
        self.uV_fill_namespace("_b")
        self.uV_fill_namespace("_t")

        # super position of potentials
        self.WAA = self.WAA_b + self.WAA_t
        self.WBB = self.WBB_b + self.WBB_t
        self.WBA = self.WBA_b + self.WBA_t

    def extra_super_elements(self, *, K_plus, **function_kwargs):
        yield from self.uV_super_elements("A", "B", K_plus)


class SLGOnHBN(SLGHBN):
    # default model parameters
    extra_model_parameters = dict(
        [
            *(
                [name, value_and_help]
                for name, value_and_help in uV_extra_model_parameters().items()
                if name not in ("tau_x", "tau_y")
            ),
            ["top_layer", (False, "hBN layer is on top.")],
        ]
    )

    # model parameter keys to pop
    remove_model_parameters_names = (
        *uV_extra_model_parameters_names("_b"),
        *uV_extra_model_parameters_names("_t"),
        "Delta_u",
        "Delta_h",
    )

    def resolve_model_parameters(self):
        self.tau_x_b = 0.0
        self.tau_y_b = 0.0
        self.tau_x_t = 0.0
        self.tau_y_t = 0.0
        if self.top_layer:
            for name in uV_extra_model_parameters_names()[:-3]:
                setattr(self, f"{name}_b", 0.0)
                setattr(self, f"{name}_t", getattr(self, name))
            self.inverted_b = False
            self.inverted_t = self.inverted
        else:
            for name in uV_extra_model_parameters_names()[:-3]:
                setattr(self, f"{name}_b", getattr(self, name))
                setattr(self, f"{name}_t", 0.0)
            self.inverted_b = self.inverted
            self.inverted_t = False


class SLGEncapsulatedHBN(SLGHBN):
    extra_model_parameters = dict(
        [
            *(
                [name, value_and_help]
                for name, value_and_help in uV_extra_model_parameters().items()
                if name not in ("tau_x", "tau_y", "inverted")
            ),
            ["tau_x", (0.0, "x-component of offset between hBN layers")],
            ["tau_y", (0.0, "x-component of offset between hBN layers")],
            ["inverted_b", (False, "hBN layer is inverted. (bottom)")],
            ["inverted_t", (False, "hBN layer is inverted. (top)")],
        ]
    )

    remove_model_parameters_names = (
        *uV_extra_model_parameters_names("_b")[:-1],
        *uV_extra_model_parameters_names("_t")[:-1],
    )

    def resolve_model_parameters(self):
        for name in uV_extra_model_parameters_names()[:8]:
            setattr(self, f"{name}_b", getattr(self, name))
            setattr(self, f"{name}_t", getattr(self, name))
        self.tau_x_b = 0.5 * self.tau_x
        self.tau_y_b = 0.5 * self.tau_y
        self.tau_x_t = -0.5 * self.tau_x
        self.tau_y_t = -0.5 * self.tau_y
